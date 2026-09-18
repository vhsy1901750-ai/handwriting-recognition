# 2026-09-18 18:17 KST
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from PIL import Image
from model import HandwritingCNN
from inference import preprocess_image, predict


def test_preprocess_image_shape_and_range():
    image = Image.new("L", (280, 280), color=0)
    tensor = preprocess_image(image)
    assert tensor.shape == (1, 1, 28, 28)
    assert tensor.min().item() >= 0.0
    assert tensor.max().item() <= 1.0


def test_predict_returns_valid_digit_and_probabilities():
    model = HandwritingCNN()
    model.eval()
    tensor = torch.zeros(1, 1, 28, 28)
    digit, probs = predict(model, tensor)
    assert 0 <= digit <= 9
    assert len(probs) == 10
    assert abs(sum(probs) - 1.0) < 1e-4
