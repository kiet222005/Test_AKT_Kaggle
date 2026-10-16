from src.models.akt import AKT

# Khai báo các mô hình mới tại đây sau này (DKT, SAKT, DTransformer,...)
MODEL_REGISTRY = {
    "akt": AKT,
}


def build_model(config, device):
  model_name = config["model"]["name"].lower()
  if model_name not in MODEL_REGISTRY:
    raise ValueError(
        f"Mô hình '{model_name}' chưa được đăng ký trong MODEL_REGISTRY!"
    )

  m_cls = MODEL_REGISTRY[model_name]
  m_cfg = config["model"]

  # Lọc bỏ các trường meta-data không thuộc __init__ của model
  init_args = {
      k: v
      for k, v in m_cfg.items()
      if k not in ["name", "backbone", "head", "description"]
  }

  model = m_cls(**init_args)
  return model.to(device)