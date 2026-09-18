import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from model import HandwritingCNN


def test_forward_output_shape_batch():
    model = HandwritingCNN()
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)


def test_forward_output_shape_single():
    model = HandwritingCNN()
    x = torch.randn(1, 1, 28, 28)
    out = model(x)
    assert out.shape == (1, 10)
