# training/verify_parity.py
# 2026-09-18 18:25 KST
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
from torchvision import datasets, transforms

from model import HandwritingCNN

NUM_SAMPLES = 20


def build_fixture():
    model = HandwritingCNN()
    model.load_state_dict(torch.load("checkpoints/model.pth", map_location="cpu"))
    model.eval()

    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

    images = []
    predictions = []
    with torch.no_grad():
        for i in range(NUM_SAMPLES):
            image, _ = test_set[i]
            logits = model(image.unsqueeze(0))
            predicted = int(logits.argmax(dim=1).item())
            images.append(image.flatten().tolist())
            predictions.append(predicted)

    fixture = {"images": images, "pytorch_predictions": predictions}
    os.makedirs("fixtures", exist_ok=True)
    with open("fixtures/parity_check.json", "w") as f:
        json.dump(fixture, f)


def run_js_predictions():
    result = subprocess.run(
        ["node", "run_web_inference.js"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def main():
    build_fixture()
    js_predictions = run_js_predictions()

    with open("fixtures/parity_check.json") as f:
        fixture = json.load(f)

    matches = sum(
        1 for a, b in zip(fixture["pytorch_predictions"], js_predictions) if a == b
    )
    total = len(fixture["pytorch_predictions"])
    print(f"일치: {matches}/{total}")
    if matches == total:
        print("PASS: PyTorch와 web_version/inference.js 예측이 모두 일치합니다.")
    else:
        print("FAIL: PyTorch와 JS 추론 결과가 다릅니다.")
    sys.exit(0 if matches == total else 1)


if __name__ == "__main__":
    main()
