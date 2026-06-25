import torch

ckpt = torch.load(
    "best_model.pth",
    map_location="cpu"
)

print(type(ckpt))

if isinstance(ckpt, dict):
    print(list(ckpt.keys())[:20])