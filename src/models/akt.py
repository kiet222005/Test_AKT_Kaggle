from enum import IntEnum
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_
import torch.nn.functional as F

# Import các hàm xử lý mask nhóm từ src/masks.py
from src.masks import allowed_group_mask, question_groups

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dim(IntEnum):
  batch = 0
  seq = 1
  feature = 2


class AKT(nn.Module):

  def __init__(
      self,
      n_question,
      n_pid,
      d_model,
      n_blocks,
      dropout,
      d_ff=256,
      kq_same=1,
      final_fc_dim=512,
      num_attn_heads=8,
      separate_qa=False,
      l2=1e-5,
      emb_type="qid",
      **kwargs,
  ):
    super().__init__()
    self.model_name = "akt"
    self.n_question = n_question
    self.dropout = dropout
    self.kq_same = kq_same
    self.n_pid = n_pid
    self.l2 = l2
    self.separate_qa = separate_qa
    self.emb_type = emb_type
    embed_l = d_model

    if self.n_pid > 0:
      self.difficult_param = nn.Embedding(self.n_pid + 1, 1)
      self.q_embed_diff = nn.Embedding(self.n_question + 1, embed_l)
      if self.separate_qa:
        self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)
      else:
        self.qa_embed_diff = nn.Embedding(2, embed_l)

    if emb_type.startswith("qid"):
      self.q_embed = nn.Embedding(self.n_question + 1, embed_l)
      if self.separate_qa:
        self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)
      else:
        self.qa_embed = nn.Embedding(2, embed_l)

    self.model = Architecture(
        n_question=n_question,
        n_blocks=n_blocks,
        n_heads=num_attn_heads,
        dropout=dropout,
        d_model=d_model,
        d_feature=d_model / num_attn_heads,
        d_ff=d_ff,
        kq_same=self.kq_same,
        model_type=self.model_name,
        emb_type=self.emb_type,
    )

    self.out = nn.Sequential(
        nn.Linear(d_model + embed_l, final_fc_dim),
        nn.ReLU(),
        nn.Dropout(self.dropout),
        nn.Linear(final_fc_dim, 256),
        nn.ReLU(),
        nn.Dropout(self.dropout),
        nn.Linear(256, 1),
    )
    self.reset()

  def reset(self):
    for p in self.parameters():
      if p.size(0) == self.n_pid + 1 and self.n_pid > 0:
        constant_(p, 0.0)

  def base_emb(self, q_data, target):
    q_embed_data = self.q_embed(q_data)
    if self.separate_qa:
      qa_data = q_data + self.n_question * target
      qa_embed_data = self.qa_embed(qa_data)
    else:
      qa_embed_data = self.qa_embed(target) + q_embed_data
    return q_embed_data, qa_embed_data

  def forward(self, batch, qtest=False, **kwargs):
    """Linh hoạt nhận cả Dictionary batch lẫn các tham số riêng lẻ."""
    if isinstance(batch, dict):
      q_data = batch["concepts"]
      target = batch["responses"]
      pid_data = batch.get("questions", None)
      is_repeat = batch.get("is_repeat", None)
    else:
      q_data = batch
      target = kwargs.get("target", None)
      pid_data = kwargs.get("pid_data", None)
      is_repeat = kwargs.get("is_repeat", None)

    q_data = q_data.clamp(min=0)
    #  ĐÃ SỬA: Chặn target chuẩn trong khoảng [0, 1] để tránh lỗi IndexError ở qa_embed
    target = target.clamp(0, 1)

    if pid_data is not None:
      pid_data = pid_data.clamp(min=0)

    if self.emb_type.startswith("qid"):
      q_embed_data, qa_embed_data = self.base_emb(q_data, target)

    pid_embed_data = None
    if self.n_pid > 0 and pid_data is not None:
      q_embed_diff_data = self.q_embed_diff(q_data)
      pid_embed_data = self.difficult_param(pid_data)
      q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

      # Xử lý qa_embed_diff_data chính xác theo separate_qa
      if self.separate_qa:
        qa_data = q_data + self.n_question * target
        qa_embed_diff_data = self.qa_embed_diff(qa_data)
        qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
      else:
        qa_embed_diff_data = self.qa_embed_diff(target)
        qa_embed_data = qa_embed_data + pid_embed_data * (
            qa_embed_diff_data + q_embed_diff_data
        )

      c_reg_loss = (pid_embed_data**2.0).sum() * self.l2
    else:
      c_reg_loss = 0.0

    d_output = self.model(
        q_embed_data, qa_embed_data, pid_embed_data, is_repeat=is_repeat
    )

    concat_q = torch.cat([d_output, q_embed_data], dim=-1)
    output = self.out(concat_q).squeeze(-1)
    preds = torch.sigmoid(output)

    if not qtest:
      return preds, c_reg_loss
    else:
      return preds, c_reg_loss, concat_q


class Architecture(nn.Module):

  def __init__(
      self,
      n_question,
      n_blocks,
      d_model,
      d_feature,
      d_ff,
      n_heads,
      dropout,
      kq_same,
      model_type,
      emb_type,
  ):
    super().__init__()
    self.d_model = d_model
    self.blocks_1 = nn.ModuleList([
        TransformerLayer(
            d_model=d_model,
            d_feature=d_model // n_heads,
            d_ff=d_ff,
            dropout=dropout,
            n_heads=n_heads,
            kq_same=kq_same,
            emb_type=emb_type,
        )
        for _ in range(n_blocks)
    ])
    self.blocks_2 = nn.ModuleList([
        TransformerLayer(
            d_model=d_model,
            d_feature=d_model // n_heads,
            d_ff=d_ff,
            dropout=dropout,
            n_heads=n_heads,
            kq_same=kq_same,
            emb_type=emb_type,
        )
        for _ in range(n_blocks * 2)
    ])

  def forward(
      self, q_embed_data, qa_embed_data, pid_embed_data, is_repeat=None
  ):
    y = qa_embed_data
    x = q_embed_data

    # Khối 1: Encoder tương tác QA
    for block in self.blocks_1:
      y = block(
          mask=1,
          query=y,
          key=y,
          values=y,
          pdiff=pid_embed_data,
          is_repeat=is_repeat,
      )

    flag_first = True
    # Khối 2: Decoder / Knowledge Retriever
    for block in self.blocks_2:
      if flag_first:
        x = block(
            mask=1,
            query=x,
            key=x,
            values=x,
            apply_pos=False,
            pdiff=pid_embed_data,
            is_repeat=is_repeat,
        )
        flag_first = False
      else:
        # Knowledge Retriever (CHẶN RÒ RỈ: mask=0 áp dụng Group-Causal Mask)
        x = block(
            mask=0,
            query=x,
            key=x,
            values=y,
            apply_pos=True,
            pdiff=pid_embed_data,
            is_repeat=is_repeat,
        )
        flag_first = True
    return x


class TransformerLayer(nn.Module):

  def __init__(
      self, d_model, d_feature, d_ff, n_heads, dropout, kq_same, emb_type
  ):
    super().__init__()
    kq_same = kq_same == 1
    self.masked_attn_head = MultiHeadAttention(
        d_model,
        d_feature,
        n_heads,
        dropout,
        kq_same=kq_same,
        emb_type=emb_type,
    )

    self.layer_norm1 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.linear1 = nn.Linear(d_model, d_ff)
    self.activation = nn.ReLU()
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(d_ff, d_model)
    self.layer_norm2 = nn.LayerNorm(d_model)
    self.dropout2 = nn.Dropout(dropout)

  def forward(
      self,
      mask,
      query,
      key,
      values,
      apply_pos=True,
      pdiff=None,
      is_repeat=None,
  ):
    seqlen = query.size(1)

    if is_repeat is not None and mask == 0:
      g = question_groups(is_repeat)
      allow = allowed_group_mask(g, g)
      src_mask = allow.unsqueeze(1)

      query2 = self.masked_attn_head(
          query,
          key,
          values,
          mask=src_mask,
          zero_pad=True,
          pdiff=pdiff,
          is_group_mask=True,
      )
    else:
      nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype(
          "uint8"
      )
      src_mask = (torch.from_numpy(nopeek_mask) == 0).to(query.device)
      query2 = self.masked_attn_head(
          query,
          key,
          values,
          mask=src_mask,
          zero_pad=(mask == 0),
          pdiff=pdiff,
          is_group_mask=False,
      )

    query = query + self.dropout1(query2)
    query = self.layer_norm1(query)
    if apply_pos:
      query2 = self.linear2(
          self.dropout(self.activation(self.linear1(query)))
      )
      query = query + self.dropout2(query2)
      query = self.layer_norm2(query)
    return query


class MultiHeadAttention(nn.Module):

  def __init__(
      self,
      d_model,
      d_feature,
      n_heads,
      dropout,
      kq_same,
      bias=True,
      emb_type="qid",
  ):
    super().__init__()
    self.d_model = d_model
    self.emb_type = emb_type
    self.d_k = d_feature
    self.h = n_heads
    self.kq_same = kq_same

    self.v_linear = nn.Linear(d_model, d_model, bias=bias)
    self.k_linear = nn.Linear(d_model, d_model, bias=bias)
    if kq_same is False:
      self.q_linear = nn.Linear(d_model, d_model, bias=bias)
    self.dropout = nn.Dropout(dropout)
    self.out_proj = nn.Linear(d_model, d_model, bias=bias)
    self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
    torch.nn.init.xavier_uniform_(self.gammas)

  def forward(
      self, q, k, v, mask, zero_pad, pdiff=None, is_group_mask=False
  ):
    bs = q.size(0)
    k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
    if self.kq_same is False:
      q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
    else:
      q = self.k_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
    v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)

    gammas = self.gammas
    if self.emb_type.find("pdiff") == -1:
      pdiff = None

    scores = attention(
        q,
        k,
        v,
        self.d_k,
        mask,
        self.dropout,
        zero_pad,
        gammas,
        pdiff,
        is_group_mask=is_group_mask,
    )
    concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
    return self.out_proj(concat)


def attention(
    q,
    k,
    v,
    d_k,
    mask,
    dropout,
    zero_pad,
    gamma=None,
    pdiff=None,
    is_group_mask=False,
):
  scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
  bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

  x1 = torch.arange(seqlen).expand(seqlen, -1).to(q.device)
  x2 = x1.transpose(0, 1).contiguous()

  with torch.no_grad():
    if is_group_mask:
      scores_ = scores.masked_fill(mask == False, -1e32)
    else:
      scores_ = scores.masked_fill(mask == 0, -1e32)

    scores_ = F.softmax(scores_, dim=-1)

    if is_group_mask:
      has_valid_key = mask.any(dim=-1, keepdim=True)
      scores_ = torch.where(has_valid_key, scores_, torch.zeros_like(scores_))

    distcum_scores = torch.cumsum(scores_, dim=-1)
    disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
    position_effect = (
        torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(q.device)
    )
    dist_scores = torch.clamp(
        (disttotal_scores - distcum_scores) * position_effect, min=0.0
    )
    dist_scores = dist_scores.sqrt().detach()

  m = nn.Softplus()
  gamma = -1.0 * m(gamma).unsqueeze(0)

  if pdiff is None:
    total_effect = torch.clamp(
        torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5
    )
  else:
    diff = pdiff.unsqueeze(1).expand(
        pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2]
    )
    diff = diff.sigmoid().exp()
    total_effect = torch.clamp(
        torch.clamp((dist_scores * gamma * diff).exp(), min=1e-5), max=1e5
    )

  scores = scores * total_effect

  if is_group_mask:
    scores.masked_fill_(mask == False, -1e32)
  else:
    scores.masked_fill_(mask == 0, -1e32)

  scores = F.softmax(scores, dim=-1)

  if is_group_mask:
    has_valid_key = mask.any(dim=-1, keepdim=True)
    scores = torch.where(has_valid_key, scores, torch.zeros_like(scores))

  if zero_pad and not is_group_mask:
    pad_zero = torch.zeros(bs, head, 1, seqlen).to(q.device)
    scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)

  scores = dropout(scores)
  return torch.matmul(scores, v)