# 2026-09-18 18:17 KST
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from model import HandwritingCNN


def load_model(weights_path):
    model = HandwritingCNN()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


def preprocess_image(pil_image):
    gray = pil_image.convert("L").resize((28, 28))
    array = np.array(gray, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def predict(model, tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    digit = int(probs.argmax().item())
    return digit, probs.tolist()
