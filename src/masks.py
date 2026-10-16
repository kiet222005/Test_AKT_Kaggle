import torch


def question_groups(is_repeat):
  """Chuyển cờ is_repeat thành ID nhóm câu hỏi.

  is_repeat (B, T): 0 = câu hỏi mới, 1 = KC tiếp theo cùng câu hỏi.
  Trả về g (B, T) = 0, 1, 2, ... : Chỉ số nhóm câu hỏi.
  """
  return torch.cumsum((is_repeat == 0).long(), dim=1) - 1


def allowed_group_mask(g_query, g_key):
  """True tại vị trí Key thuộc câu hỏi ĐỨNG TRƯỚC Query (được phép nhìn).

  Shape trả về: (B, T_query, T_key)
  """
  return g_key.unsqueeze(1) < g_query.unsqueeze(2)


def build_group_attn_mask(is_repeat, num_heads=1):
  """Tạo mask cho PyTorch MultiheadAttention từ is_repeat.

  Trong PyTorch Attention: True = MASK (CHE), False = GIỮ.
  """
  g = question_groups(is_repeat)
  # query tại t+1 (dịch 1 bước), key/value tại 0..t
  allow = allowed_group_mask(g[:, 1:], g[:, :-1])

  # Thêm token BOS ở đầu để tránh hàng attention đầu tiên bị mask 100% gây NaN
  bos_allow = torch.ones_like(allow[..., :1])
  allow = torch.cat([bos_allow, allow], dim=-1)

  attn_mask = ~allow
  if num_heads > 1:
    attn_mask = attn_mask.repeat_interleave(num_heads, dim=0)

  return attn_mask