# 2026-09-18 17:43 KST
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch

from model import HandwritingCNN


def export_weights(checkpoint_path, output_path):
    model = HandwritingCNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    state = model.state_dict()

    layers = {
        "conv1": (state["conv1.weight"], state["conv1.bias"]),
        "conv2": (state["conv2.weight"], state["conv2.bias"]),
        "fc": (state["fc.weight"], state["fc.bias"]),
    }

    output = {}
    for name, (weight, bias) in layers.items():
        output[name] = {
            "weight": weight.flatten().tolist(),
            "weight_shape": list(weight.shape),
            "bias": bias.flatten().tolist(),
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    export_weights("checkpoints/model.pth", "../web_version/model/weights.json")
