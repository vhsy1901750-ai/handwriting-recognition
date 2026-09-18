# 2026-09-18 18:45 KST
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from model import HandwritingCNN
from export_web import export_weights


def test_export_weights_matches_state_dict(tmp_path):
    torch.manual_seed(0)
    model = HandwritingCNN()
    checkpoint_path = tmp_path / "model.pth"
    torch.save(model.state_dict(), checkpoint_path)

    output_path = tmp_path / "weights.json"
    export_weights(str(checkpoint_path), str(output_path))

    with open(output_path) as f:
        data = json.load(f)

    state = model.state_dict()
    assert data["conv1"]["weight_shape"] == list(state["conv1.weight"].shape)
    assert data["conv1"]["weight"] == state["conv1.weight"].flatten().tolist()
    assert data["conv2"]["bias"] == state["conv2.bias"].flatten().tolist()
    assert data["fc"]["weight_shape"] == list(state["fc.weight"].shape)
