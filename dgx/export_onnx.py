"""
export_onnx.py — RSL-RL .pt 체크포인트 → ONNX 변환

실행:
    python dgx/export_onnx.py

출력:
    BHL/WALKING/checkpoints/stage_a_biped.onnx
    BHL/WALKING/checkpoints/stage_d4_hylion_v6.onnx
"""

import os
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

EXPORTS = [
    {
        "name": "stage_a_biped",
        "ckpt": "checkpoints/biped/stage_a_biped/best.pt",
        "obs_dim": 45,
        "act_dim": 12,
    },
    {
        "name": "stage_d4_hylion_v6",
        "ckpt": "checkpoints/biped/stage_d4_hylion_v6/best.pt",
        "obs_dim": 45,
        "act_dim": 12,
    },
]

OUT_DIR = os.path.join(REPO_ROOT, "BHL/WALKING/checkpoints")


class ActorMLP(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ELU(),
            nn.Linear(256, 128),    nn.ELU(),
            nn.Linear(128, 128),    nn.ELU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_actor(ckpt_path, obs_dim, act_dim):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    actor = ActorMLP(obs_dim, act_dim)
    actor.eval()

    if "actor_state_dict" in ckpt:
        raw = ckpt["actor_state_dict"]
        state = {"net." + k[len("mlp."):]: v for k, v in raw.items() if k.startswith("mlp.")}
    elif "model_state_dict" in ckpt:
        raw = ckpt["model_state_dict"]
        state = {"net." + k[len("actor."):]: v for k, v in raw.items() if k.startswith("actor.")}
    else:
        state = ckpt

    missing, unexpected = actor.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] missing keys: {missing}")

    return actor


def verify(actor, onnx_path, obs_dim):
    dummy = torch.zeros(1, obs_dim)
    with torch.no_grad():
        torch_out = actor(dummy).numpy()

    sess = ort.InferenceSession(onnx_path)
    onnx_out = sess.run(None, {"obs": dummy.numpy()})[0]

    max_err = np.abs(torch_out - onnx_out).max()
    ok = max_err < 1e-4
    print(f"  검증: max_err={max_err:.2e}  {'OK' if ok else 'FAIL'}")
    return ok


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for cfg in EXPORTS:
        ckpt_path = os.path.join(REPO_ROOT, cfg["ckpt"])
        out_path  = os.path.join(OUT_DIR, cfg["name"] + ".onnx")

        print(f"\n[{cfg['name']}]")
        print(f"  입력: {ckpt_path}")
        print(f"  출력: {out_path}")

        actor = load_actor(ckpt_path, cfg["obs_dim"], cfg["act_dim"])

        dummy = torch.zeros(1, cfg["obs_dim"])
        torch.onnx.export(
            actor, dummy, out_path,
            input_names=["obs"],
            output_names=["actions"],
            opset_version=18,
        )

        verify(actor, out_path, cfg["obs_dim"])
        print(f"  완료: {out_path}")

    print("\n모든 export 완료.")
    print(f"출력 위치: {OUT_DIR}")


if __name__ == "__main__":
    main()
