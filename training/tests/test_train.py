# 2026-09-18 17:39 KST
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader, TensorDataset
from model import HandwritingCNN
from train import evaluate


def test_evaluate_computes_accuracy_from_predictions():
    model = HandwritingCNN()
    model.eval()
    images = torch.randn(8, 1, 28, 28)
    with torch.no_grad():
        logits = model(images)
    labels = logits.argmax(dim=1)  # 모델이 항상 맞히도록 라벨을 예측값과 동일하게 구성
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    acc = evaluate(model, loader, torch.device("cpu"))

    assert acc == 1.0
