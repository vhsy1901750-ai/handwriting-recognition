<!-- 2026-09-18 16:56 KST -->
# 손글씨 숫자 인식 (웹 + 데스크톱) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** MNIST 손글씨 숫자(0-9)를 인식하는 프로그램을 웹 버전(순수 JS 추론, GitHub Pages 정적 배포)과 데스크톱 버전(Python + PyTorch + Tkinter)으로 만들고, 두 버전이 같은 학습된 모델 가중치를 공유하게 한다.

**Architecture:** `training/`에서 PyTorch로 CNN을 1회 학습하고, 가중치를 `desktop_version/model/weights.pth`(그대로 복사)와 `web_version/model/weights.json`(export)으로 배포한다. 데스크톱은 PyTorch로 직접 추론하고, 웹은 `inference.js`에 손으로 구현한 conv2d/maxpool/linear/softmax로 추론한다. 두 구현이 같은 예측을 내는지 `training/verify_parity.py`로 교차검증한다.

**Tech Stack:** Python 3, PyTorch, torchvision, Pillow, Tkinter, pytest / 순수 JavaScript(ES2020), Node.js 내장 `node:test`(개발용 테스트 전용, 배포 페이지에는 포함되지 않음).

**Spec:** [docs/superpowers/specs/2026-09-18-handwriting-recognition-design.md](../specs/2026-09-18-handwriting-recognition-design.md)

## Global Constraints

- web_version은 외부 라이브러리, CDN, 빌드 도구를 쓰지 않는다 — 순수 JS로만 추론하며 `index.html`을 정적으로 서빙하면 그대로 동작해야 한다(GitHub Pages 배포 목표).
- 모델 구조는 `training/model.py`가 단일 진실 소스이며 `desktop_version/model.py`는 동일한 구조의 복사본이다. 구조를 바꾸면 두 파일을 함께 수정한다.
- 가중치 JSON은 `{layer: {weight: number[], weight_shape: number[], bias: number[]}}` 형식이고, `weight`는 PyTorch 텐서를 `.flatten().tolist()`한 row-major 1차원 배열이다.
- 입력 이미지는 28x28 흑백, 0~1 정규화(검은 배경 0, 흰 선 1)로 통일한다.
- 각 하위 폴더(`training/`, `desktop_version/`, `web_version/`)와 루트에 CLAUDE.md를 둔다.
- 새로 만드는 파일 맨 위에는 `TZ=Asia/Seoul date "+%Y-%m-%d %H:%M"` 기준 생성 일시 주석을 남긴다(이 문서 작성 시점: 2026-09-18 16:56 KST — 실제 구현 시점에는 그때의 날짜/시각으로 다시 확인해서 쓴다).
- 데스크톱 GUI(Tkinter)와 웹 캔버스 UI는 자동 유닛테스트 대상이 아니다 — 로직(전처리/추론)만 분리해서 테스트하고, GUI 자체는 실행해서 직접 확인한다(웹은 브라우저로, 데스크톱은 로컬 실행으로).

---

### Task 1: training/model.py — CNN 모델 정의

**Files:**
- Create: `training/model.py`
- Test: `training/tests/test_model.py`

**Interfaces:**
- Produces: `HandwritingCNN` (`torch.nn.Module`), `forward(x: Tensor[N,1,28,28]) -> Tensor[N,10]`. 이후 모든 Python 태스크가 이 클래스를 `from model import HandwritingCNN`으로 사용한다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# training/tests/test_model.py
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
```

- [ ] **Step 2: 테스트 실행해서 실패 확인**

Run: `cd training && pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'model'`

- [ ] **Step 3: 최소 구현 작성**

```python
# training/model.py
# 2026-09-18 16:56 KST
import torch.nn as nn
import torch.nn.functional as F


class HandwritingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

- [ ] **Step 4: 테스트 실행해서 통과 확인**

Run: `cd training && pytest tests/test_model.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: 커밋**

```bash
git add training/model.py training/tests/test_model.py
git commit -m "feat(training): CNN 모델 정의 추가"
```

---

### Task 2: training/train.py — 학습 스크립트

**Files:**
- Create: `training/train.py`
- Create: `training/requirements.txt`
- Test: `training/tests/test_train.py`

**Interfaces:**
- Consumes: `training.model.HandwritingCNN`
- Produces: `evaluate(model, loader, device) -> float`(정확도), `main()`(학습 후 `training/checkpoints/model.pth` 저장). Task 4와 Task 12가 `evaluate`와 저장 경로를 사용한다.

- [ ] **Step 1: requirements.txt 작성**

```
# training/requirements.txt
torch
torchvision
pytest
```

- [ ] **Step 2: evaluate()에 대한 실패하는 테스트 작성**

```python
# training/tests/test_train.py
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
```

- [ ] **Step 3: 테스트 실행해서 실패 확인**

Run: `cd training && pytest tests/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'train'`

- [ ] **Step 4: train.py 구현**

```python
# training/train.py
# 2026-09-18 16:56 KST
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import HandwritingCNN


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predicted = model(images).argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()

    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    model = HandwritingCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch + 1}/{epochs} test accuracy: {acc:.4f}")

    final_acc = evaluate(model, test_loader, device)
    if final_acc < 0.99:
        print(f"경고: 목표 정확도(99%) 미달 — 현재 {final_acc:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/model.pth")
    print("저장 완료: checkpoints/model.pth")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 테스트 실행해서 통과 확인**

Run: `cd training && pytest tests/test_train.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: 커밋**

```bash
git add training/train.py training/requirements.txt training/tests/test_train.py
git commit -m "feat(training): 학습 스크립트와 evaluate 함수 추가"
```

---

### Task 3: training/export_web.py — 웹용 가중치 export

**Files:**
- Create: `training/export_web.py`
- Test: `training/tests/test_export_web.py`

**Interfaces:**
- Consumes: `training.model.HandwritingCNN`
- Produces: `export_weights(checkpoint_path: str, output_path: str) -> None`. Task 4가 실제 체크포인트에 대해 이 함수를 호출한다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# training/tests/test_export_web.py
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
```

- [ ] **Step 2: 테스트 실행해서 실패 확인**

Run: `cd training && pytest tests/test_export_web.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'export_web'`

- [ ] **Step 3: export_web.py 구현**

```python
# training/export_web.py
# 2026-09-18 16:56 KST
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
```

- [ ] **Step 4: 테스트 실행해서 통과 확인**

Run: `cd training && pytest tests/test_export_web.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: 커밋**

```bash
git add training/export_web.py training/tests/test_export_web.py
git commit -m "feat(training): 웹용 가중치 export 스크립트 추가"
```

---

### Task 4: 실제 학습 실행 및 가중치 배포 (통합 단계)

**Files:**
- Create: `training/checkpoints/model.pth` (실행 결과물)
- Create: `desktop_version/model/weights.pth` (복사본)
- Create: `web_version/model/weights.json` (export 결과물)

**Interfaces:**
- Consumes: Task 2의 `train.py`, Task 3의 `export_web.py`
- Produces: 실제 학습된 가중치 파일 2종 — Task 5 이후 desktop, Task 9 이후 web이 이 파일들을 사용한다.

- [ ] **Step 1: 학습 실행 (수 분 소요, MNIST 자동 다운로드)**

Run: `cd training && pip install -r requirements.txt && python train.py`
Expected: 각 epoch마다 정확도 출력, 마지막에 `저장 완료: checkpoints/model.pth`. 최종 정확도가 99% 근처인지 육안 확인(미달 시 경고 메시지가 출력되지만 스크립트는 계속 진행됨).

- [ ] **Step 2: desktop_version으로 가중치 복사**

```bash
mkdir -p desktop_version/model
cp training/checkpoints/model.pth desktop_version/model/weights.pth
```

- [ ] **Step 3: web_version으로 가중치 export**

Run: `cd training && python export_web.py`
Expected: `web_version/model/weights.json` 생성됨. 파일 크기가 수백 KB 수준인지 확인(`ls -la ../web_version/model/weights.json`).

- [ ] **Step 4: 커밋**

```bash
git add training/checkpoints/model.pth desktop_version/model/weights.pth web_version/model/weights.json
git commit -m "chore: 학습된 모델 가중치 생성 및 배포"
```

---

### Task 5: desktop_version/model.py + inference.py — 추론 로직

**Files:**
- Create: `desktop_version/model.py` (training/model.py와 동일 내용)
- Create: `desktop_version/inference.py`
- Create: `desktop_version/requirements.txt`
- Test: `desktop_version/tests/test_inference.py`

**Interfaces:**
- Consumes: Task 4의 `desktop_version/model/weights.pth`
- Produces: `load_model(weights_path) -> HandwritingCNN`, `preprocess_image(pil_image) -> Tensor[1,1,28,28]`, `predict(model, tensor) -> tuple[int, list[float]]`. Task 6의 `app.py`가 이 세 함수를 사용한다.

- [ ] **Step 1: requirements.txt 작성**

```
# desktop_version/requirements.txt
torch
pillow
numpy
pytest
```

- [ ] **Step 2: model.py 복사**

`training/model.py`와 동일한 내용으로 `desktop_version/model.py`를 만든다(내용은 Task 1의 Step 3와 동일, 파일 상단 생성 일시 주석만 이 태스크를 실행하는 시점 기준으로 갱신).

- [ ] **Step 3: 실패하는 테스트 작성**

```python
# desktop_version/tests/test_inference.py
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
```

- [ ] **Step 4: 테스트 실행해서 실패 확인**

Run: `cd desktop_version && pytest tests/test_inference.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'inference'`

- [ ] **Step 5: inference.py 구현**

```python
# desktop_version/inference.py
# 2026-09-18 16:56 KST
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
```

- [ ] **Step 6: 테스트 실행해서 통과 확인**

Run: `cd desktop_version && pytest tests/test_inference.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: 커밋**

```bash
git add desktop_version/model.py desktop_version/inference.py desktop_version/requirements.txt desktop_version/tests/test_inference.py
git commit -m "feat(desktop): 모델 로드/전처리/예측 로직 추가"
```

---

### Task 6: desktop_version/app.py — Tkinter GUI

**Files:**
- Create: `desktop_version/app.py`

**Interfaces:**
- Consumes: Task 5의 `load_model`, `preprocess_image`, `predict`

- [ ] **Step 1: app.py 구현**

```python
# desktop_version/app.py
# 2026-09-18 16:56 KST
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import tkinter as tk
from PIL import Image, ImageDraw

from inference import load_model, preprocess_image, predict

CANVAS_SIZE = 280
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "model", "weights.pth")


class HandwritingApp:
    def __init__(self, root):
        self.model = load_model(WEIGHTS_PATH)
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.on_draw)

        self.result_label = tk.Label(root, text="숫자를 그려주세요", font=("Arial", 16))
        self.result_label.pack()

        button_frame = tk.Frame(root)
        button_frame.pack()
        tk.Button(button_frame, text="인식", command=self.on_recognize).pack(side="left")
        tk.Button(button_frame, text="지우기", command=self.on_clear).pack(side="left")

    def on_draw(self, event):
        r = 8
        x, y = event.x, event.y
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def on_clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="숫자를 그려주세요")

    def on_recognize(self):
        if self.image.getextrema() == (0, 0):
            self.result_label.config(text="그림을 먼저 그려주세요")
            return
        tensor = preprocess_image(self.image)
        digit, probs = predict(self.model, tensor)
        confidence = probs[digit] * 100
        self.result_label.config(text=f"예측: {digit} ({confidence:.1f}%)")


def main():
    root = tk.Tk()
    root.title("손글씨 숫자 인식")
    HandwritingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행해서 확인 (수동, GUI라 자동 테스트 불가)**

Run: `cd desktop_version && python app.py`
Expected: 창이 뜨고 예외 없이 실행됨. 직접 마우스로 숫자를 그린 뒤 "인식" 버튼을 눌러 예측 결과가 표시되는지, "지우기" 버튼이 캔버스를 비우는지, 빈 캔버스에서 "인식"을 누르면 "그림을 먼저 그려주세요"가 뜨는지 확인.

- [ ] **Step 3: 커밋**

```bash
git add desktop_version/app.py
git commit -m "feat(desktop): Tkinter GUI 추가"
```

---

### Task 7: desktop_version/CLAUDE.md

**Files:**
- Create: `desktop_version/CLAUDE.md`

- [ ] **Step 1: 작성**

```markdown
<!-- 2026-09-18 16:56 KST -->
# desktop_version

손글씨 숫자 인식 데스크톱 앱. Python, PyTorch, Tkinter로 만들어졌다.

## 구성

- `model.py`: CNN 모델 정의. `../training/model.py`와 반드시 동일한 구조를 유지해야 한다. 모델 구조를 바꿀 때는 두 파일을 함께 수정해 주세요.
- `inference.py`: 이미지 전처리와 예측 로직. Tkinter에 의존하지 않으므로 GUI 없이 단위 테스트가 가능하다.
- `app.py`: Tkinter GUI. 캔버스에 그린 숫자를 인식해 결과를 보여준다.
- `model/weights.pth`: `../training/train.py`로 학습한 뒤 복사해 온 가중치 파일. 직접 학습하지 않는다.

## 실행

```bash
pip install -r requirements.txt
python app.py
```

## 테스트

```bash
pytest tests/ -v
```

`app.py`(GUI)는 자동 테스트 대상이 아니다. 실행 후 직접 숫자를 그려 인식 결과를 확인해 주세요.

## 가중치 갱신

모델을 다시 학습했다면 `../training/checkpoints/model.pth`를 `model/weights.pth`로 다시 복사해 주세요. 자동 동기화는 없다.
```

- [ ] **Step 2: 커밋**

```bash
git add desktop_version/CLAUDE.md
git commit -m "docs(desktop): CLAUDE.md 추가"
```

---

### Task 8: web_version/inference.js — 순수 JS 추론 함수

**Files:**
- Create: `web_version/inference.js`
- Test: `web_version/tests/inference.test.js`

**Interfaces:**
- Produces: `conv2d(input, inShape, weight, weightShape, bias) -> {data, shape}`, `relu(data) -> Float32Array`, `maxPool2d(input, shape, poolSize) -> {data, shape}`, `linear(input, weight, weightShape, bias) -> Float32Array`, `softmax(logits) -> Float32Array`. Task 9가 이 함수들로 `runInference`를 조립한다.

- [ ] **Step 1: 실패하는 테스트 작성**

```js
// web_version/tests/inference.test.js
const test = require("node:test");
const assert = require("node:assert/strict");
const { conv2d, relu, maxPool2d, linear, softmax } = require("../inference.js");

test("conv2d with identity kernel returns input unchanged", () => {
  const input = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]); // 1x3x3
  const weight = new Float32Array([0, 0, 0, 0, 1, 0, 0, 0, 0]); // 중앙만 1인 3x3 커널
  const bias = new Float32Array([0]);
  const { data, shape } = conv2d(input, [1, 3, 3], weight, [1, 1, 3, 3], bias);
  assert.deepEqual(shape, [1, 3, 3]);
  assert.deepEqual(Array.from(data), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
});

test("relu zeroes out negative values", () => {
  const input = new Float32Array([-2, -1, 0, 1, 2]);
  const output = relu(input);
  assert.deepEqual(Array.from(output), [0, 0, 0, 1, 2]);
});

test("maxPool2d downsamples by taking the max of each 2x2 block", () => {
  const input = new Float32Array([
    1, 2, 5, 6,
    3, 4, 7, 8,
    9, 10, 13, 14,
    11, 12, 15, 16,
  ]); // 1x4x4
  const { data, shape } = maxPool2d(input, [1, 4, 4], 2);
  assert.deepEqual(shape, [1, 2, 2]);
  assert.deepEqual(Array.from(data), [4, 8, 12, 16]);
});

test("linear computes weight * input + bias", () => {
  const input = new Float32Array([1, 2]);
  const weight = new Float32Array([1, 0, 0, 1, 1, 1]); // shape [3,2]
  const bias = new Float32Array([0, 0, 10]);
  const output = linear(input, weight, [3, 2], bias);
  assert.deepEqual(Array.from(output), [1, 2, 13]);
});

test("softmax outputs a probability distribution", () => {
  const logits = new Float32Array([1, 2, 3]);
  const probs = softmax(logits);
  const sum = Array.from(probs).reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-6);
  assert.ok(probs[2] > probs[1] && probs[1] > probs[0]);
});
```

- [ ] **Step 2: 테스트 실행해서 실패 확인**

Run: `cd web_version && node --test tests/inference.test.js`
Expected: FAIL — `Cannot find module '../inference.js'`

- [ ] **Step 3: inference.js 구현 (1/2 — 순수 함수)**

```js
// web_version/inference.js
// 2026-09-18 16:56 KST
function conv2d(input, inShape, weight, weightShape, bias) {
  const [inC, inH, inW] = inShape;
  const [outC, wInC, kH, kW] = weightShape;
  const pad = 1;
  const outH = inH;
  const outW = inW;
  const output = new Float32Array(outC * outH * outW);

  for (let oc = 0; oc < outC; oc++) {
    for (let oy = 0; oy < outH; oy++) {
      for (let ox = 0; ox < outW; ox++) {
        let sum = bias[oc];
        for (let ic = 0; ic < inC; ic++) {
          for (let ky = 0; ky < kH; ky++) {
            const iy = oy + ky - pad;
            if (iy < 0 || iy >= inH) continue;
            for (let kx = 0; kx < kW; kx++) {
              const ix = ox + kx - pad;
              if (ix < 0 || ix >= inW) continue;
              const inIdx = ic * inH * inW + iy * inW + ix;
              const wIdx = ((oc * wInC + ic) * kH + ky) * kW + kx;
              sum += input[inIdx] * weight[wIdx];
            }
          }
        }
        output[oc * outH * outW + oy * outW + ox] = sum;
      }
    }
  }
  return { data: output, shape: [outC, outH, outW] };
}

function relu(data) {
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = data[i] > 0 ? data[i] : 0;
  return out;
}

function maxPool2d(input, shape, poolSize) {
  const [c, h, w] = shape;
  const outH = Math.floor(h / poolSize);
  const outW = Math.floor(w / poolSize);
  const output = new Float32Array(c * outH * outW);

  for (let ch = 0; ch < c; ch++) {
    for (let oy = 0; oy < outH; oy++) {
      for (let ox = 0; ox < outW; ox++) {
        let max = -Infinity;
        for (let py = 0; py < poolSize; py++) {
          for (let px = 0; px < poolSize; px++) {
            const iy = oy * poolSize + py;
            const ix = ox * poolSize + px;
            const val = input[ch * h * w + iy * w + ix];
            if (val > max) max = val;
          }
        }
        output[ch * outH * outW + oy * outW + ox] = max;
      }
    }
  }
  return { data: output, shape: [c, outH, outW] };
}

function linear(input, weight, weightShape, bias) {
  const [outF, inF] = weightShape;
  const output = new Float32Array(outF);
  for (let o = 0; o < outF; o++) {
    let sum = bias[o];
    for (let i = 0; i < inF; i++) {
      sum += input[i] * weight[o * inF + i];
    }
    output[o] = sum;
  }
  return output;
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return Float32Array.from(exps, (v) => v / sum);
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { conv2d, relu, maxPool2d, linear, softmax };
}
```

- [ ] **Step 4: 테스트 실행해서 통과 확인**

Run: `cd web_version && node --test tests/inference.test.js`
Expected: PASS (5 passed)

- [ ] **Step 5: 커밋**

```bash
git add web_version/inference.js web_version/tests/inference.test.js
git commit -m "feat(web): conv2d/relu/maxPool2d/linear/softmax 순수 JS 구현"
```

---

### Task 9: web_version/inference.js — runInference 파이프라인

**Files:**
- Modify: `web_version/inference.js` (Task 8에서 만든 파일에 `runInference` 추가)
- Modify: `web_version/tests/inference.test.js`

**Interfaces:**
- Consumes: Task 8의 `conv2d`, `relu`, `maxPool2d`, `linear`, `softmax`; Task 4의 `web_version/model/weights.json`
- Produces: `runInference(weights, image: Float32Array[784]) -> {predicted: number, probabilities: number[]}`. Task 10의 `app.js`가 이 함수를 사용한다.

- [ ] **Step 1: 실패하는 테스트 추가**

```js
// web_version/tests/inference.test.js 에 추가
const fs = require("node:fs");
const path = require("node:path");
const { runInference } = require("../inference.js");

test("runInference produces a valid probability distribution for a blank image", () => {
  const weightsPath = path.join(__dirname, "../model/weights.json");
  const weights = JSON.parse(fs.readFileSync(weightsPath, "utf8"));
  const blankImage = new Float32Array(28 * 28); // 전부 0 (빈 캔버스)

  const { predicted, probabilities } = runInference(weights, blankImage);

  assert.ok(Number.isInteger(predicted) && predicted >= 0 && predicted <= 9);
  assert.equal(probabilities.length, 10);
  const sum = probabilities.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-4);
});
```

- [ ] **Step 2: 테스트 실행해서 실패 확인**

Run: `cd web_version && node --test tests/inference.test.js`
Expected: FAIL — `runInference is not a function` (Task 4에서 `model/weights.json`이 이미 생성되어 있어야 함)

- [ ] **Step 3: runInference 구현 추가**

`web_version/inference.js`의 `if (typeof module ...` 블록 바로 위에 추가:

```js
function runInference(weights, image) {
  let x = { data: image, shape: [1, 28, 28] };

  x = conv2d(
    x.data,
    x.shape,
    new Float32Array(weights.conv1.weight),
    weights.conv1.weight_shape,
    new Float32Array(weights.conv1.bias)
  );
  x.data = relu(x.data);
  x = maxPool2d(x.data, x.shape, 2);

  x = conv2d(
    x.data,
    x.shape,
    new Float32Array(weights.conv2.weight),
    weights.conv2.weight_shape,
    new Float32Array(weights.conv2.bias)
  );
  x.data = relu(x.data);
  x = maxPool2d(x.data, x.shape, 2);

  const logits = linear(
    x.data,
    new Float32Array(weights.fc.weight),
    weights.fc.weight_shape,
    new Float32Array(weights.fc.bias)
  );
  const probabilities = softmax(logits);

  let predicted = 0;
  for (let i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > probabilities[predicted]) predicted = i;
  }

  return { predicted, probabilities: Array.from(probabilities) };
}
```

그리고 `module.exports`에 `runInference` 추가:

```js
if (typeof module !== "undefined" && module.exports) {
  module.exports = { conv2d, relu, maxPool2d, linear, softmax, runInference };
}
```

- [ ] **Step 4: 테스트 실행해서 통과 확인**

Run: `cd web_version && node --test tests/inference.test.js`
Expected: PASS (6 passed)

- [ ] **Step 5: 커밋**

```bash
git add web_version/inference.js web_version/tests/inference.test.js
git commit -m "feat(web): runInference 파이프라인 추가"
```

---

### Task 10: web_version/app.js + index.html — 캔버스 UI

**Files:**
- Create: `web_version/index.html`
- Create: `web_version/app.js`

**Interfaces:**
- Consumes: Task 9의 `runInference` (전역 함수로 `inference.js`가 먼저 로드됨)

- [ ] **Step 1: index.html 작성**

```html
<!-- web_version/index.html -->
<!-- 2026-09-18 16:56 KST -->
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>손글씨 숫자 인식</title>
<style>
  body { font-family: sans-serif; text-align: center; }
  canvas { border: 2px solid #333; background: black; touch-action: none; }
  #result { font-size: 20px; margin-top: 12px; }
  button { font-size: 16px; margin: 8px; padding: 6px 16px; }
</style>
</head>
<body>
  <h1>손글씨 숫자 인식</h1>
  <canvas id="canvas" width="280" height="280"></canvas>
  <div>
    <button id="recognize">인식</button>
    <button id="clear">지우기</button>
  </div>
  <div id="result">숫자를 그려주세요</div>
  <script src="inference.js"></script>
  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: app.js 작성**

```js
// web_version/app.js
// 2026-09-18 16:56 KST
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const resultEl = document.getElementById("result");

ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = "white";
ctx.lineWidth = 16;
ctx.lineCap = "round";

let drawing = false;
let lastX = 0;
let lastY = 0;

function getPos(event) {
  const rect = canvas.getBoundingClientRect();
  const point = event.touches ? event.touches[0] : event;
  return { x: point.clientX - rect.left, y: point.clientY - rect.top };
}

function startDraw(event) {
  drawing = true;
  const pos = getPos(event);
  lastX = pos.x;
  lastY = pos.y;
}

function draw(event) {
  if (!drawing) return;
  event.preventDefault();
  const pos = getPos(event);
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
  lastX = pos.x;
  lastY = pos.y;
}

function endDraw() {
  drawing = false;
}

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", endDraw);
canvas.addEventListener("mouseleave", endDraw);
canvas.addEventListener("touchstart", startDraw);
canvas.addEventListener("touchmove", draw);
canvas.addEventListener("touchend", endDraw);

document.getElementById("clear").addEventListener("click", () => {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  resultEl.textContent = "숫자를 그려주세요";
});

function canvasToInputArray() {
  const small = document.createElement("canvas");
  small.width = 28;
  small.height = 28;
  const smallCtx = small.getContext("2d");
  smallCtx.drawImage(canvas, 0, 0, 28, 28);
  const imageData = smallCtx.getImageData(0, 0, 28, 28).data;
  const array = new Float32Array(28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    array[i] = imageData[i * 4] / 255; // R 채널 = 흑백 밝기, 0~1 정규화
  }
  return array;
}

let weights = null;

async function loadWeights() {
  const response = await fetch("model/weights.json");
  weights = await response.json();
}

document.getElementById("recognize").addEventListener("click", async () => {
  if (!weights) {
    resultEl.textContent = "모델을 불러오는 중입니다...";
    await loadWeights();
  }

  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  let hasDrawing = false;
  for (let i = 0; i < imageData.length; i += 4) {
    if (imageData[i] > 0) {
      hasDrawing = true;
      break;
    }
  }
  if (!hasDrawing) {
    resultEl.textContent = "그림을 먼저 그려주세요";
    return;
  }

  const input = canvasToInputArray();
  const { predicted, probabilities } = runInference(weights, input);
  const confidence = (probabilities[predicted] * 100).toFixed(1);
  resultEl.textContent = `예측: ${predicted} (${confidence}%)`;
});

loadWeights();
```

- [ ] **Step 3: 브라우저에서 수동 확인**

Run: `cd web_version && python -m http.server 8000` (별도 터미널)
브라우저로 `http://localhost:8000`을 열고: 캔버스에 마우스로 숫자를 그린다 → "인식" 클릭 → 예측 숫자와 확률이 표시되는지 확인한다 → "지우기" 클릭 → 캔버스가 비워지는지 확인한다 → 빈 캔버스에서 "인식"을 누르면 "그림을 먼저 그려주세요"가 뜨는지 확인한다.

- [ ] **Step 4: 커밋**

```bash
git add web_version/index.html web_version/app.js
git commit -m "feat(web): 캔버스 드로잉 UI 추가"
```

---

### Task 11: web_version/CLAUDE.md

**Files:**
- Create: `web_version/CLAUDE.md`

- [ ] **Step 1: 작성**

```markdown
<!-- 2026-09-18 16:56 KST -->
# web_version

손글씨 숫자 인식 웹 버전. 외부 라이브러리, 프레임워크, 빌드 도구를 쓰지 않는 순수 JavaScript로 만들어졌다. GitHub Pages 같은 정적 호스팅에 파일 그대로 올리면 동작한다.

## 구성

- `index.html`: 캔버스와 버튼이 있는 단일 페이지.
- `app.js`: 캔버스 드로잉(마우스/터치), 28x28 다운샘플링, 추론 호출.
- `inference.js`: conv2d/relu/maxPool2d/linear/softmax를 직접 구현한 순수 JS 추론 엔진. 브라우저에서는 전역 함수로, Node 테스트에서는 `require`로 쓰인다.
- `model/weights.json`: `../training/export_web.py`로 만든 가중치. 직접 수정하지 않는다.

## 로컬 실행

```bash
python -m http.server 8000
```

브라우저로 `http://localhost:8000` 접속. `fetch`로 `model/weights.json`을 불러오므로 `file://`로 직접 여는 것보다 로컬 서버를 쓰는 편이 안전하다.

## 테스트

```bash
node --test tests/inference.test.js
```

Node.js 내장 테스트 러너만 쓰며 npm 설치가 필요 없다. `app.js`(캔버스 UI)는 브라우저에서 직접 확인해 주세요.

## 가중치 갱신

모델을 다시 학습했다면 `../training/export_web.py`를 다시 실행해서 `model/weights.json`을 갱신해 주세요. 자동 동기화는 없다.
```

- [ ] **Step 2: 커밋**

```bash
git add web_version/CLAUDE.md
git commit -m "docs(web): CLAUDE.md 추가"
```

---

### Task 12: training/verify_parity.py — PyTorch/JS 교차검증

**Files:**
- Create: `training/verify_parity.py`
- Create: `training/run_web_inference.js`

**Interfaces:**
- Consumes: Task 2의 `training/checkpoints/model.pth`, Task 9의 `web_version/inference.js`(`runInference`), Task 4의 `web_version/model/weights.json`

- [ ] **Step 1: run_web_inference.js 작성**

```js
// training/run_web_inference.js
// 2026-09-18 16:56 KST
const fs = require("node:fs");
const path = require("node:path");
const { runInference } = require("../web_version/inference.js");

const fixture = JSON.parse(
  fs.readFileSync(path.join(__dirname, "fixtures/parity_check.json"), "utf8")
);
const weights = JSON.parse(
  fs.readFileSync(path.join(__dirname, "../web_version/model/weights.json"), "utf8")
);

const predictions = fixture.images.map((flatImage) => {
  const input = Float32Array.from(flatImage);
  const { predicted } = runInference(weights, input);
  return predicted;
});

process.stdout.write(JSON.stringify(predictions));
```

- [ ] **Step 2: verify_parity.py 작성**

```python
# training/verify_parity.py
# 2026-09-18 16:56 KST
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


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 실행해서 확인**

Run: `cd training && python verify_parity.py`
Expected: `일치: 20/20`과 `PASS: ...` 출력. 불일치가 있으면 어느 레이어 구현이 다른지(conv 패딩, weight_shape 순서 등) 확인해서 고친다.

- [ ] **Step 4: 커밋**

```bash
git add training/verify_parity.py training/run_web_inference.js training/fixtures/parity_check.json
git commit -m "test(training): PyTorch/JS 추론 교차검증 스크립트 추가"
```

---

### Task 13: training/CLAUDE.md + 루트 CLAUDE.md

**Files:**
- Create: `training/CLAUDE.md`
- Create: `CLAUDE.md`

- [ ] **Step 1: training/CLAUDE.md 작성**

```markdown
<!-- 2026-09-18 16:56 KST -->
# training

손글씨 숫자 인식 모델을 학습하고, `desktop_version`과 `web_version`이 쓸 가중치를 만드는 곳. 이 폴더의 `model.py`가 모델 구조의 단일 진실 소스다.

## 구성

- `model.py`: CNN 모델 정의(`desktop_version/model.py`와 동일하게 유지해야 함).
- `train.py`: MNIST로 학습, `checkpoints/model.pth` 저장.
- `export_web.py`: `checkpoints/model.pth` → `../web_version/model/weights.json`.
- `verify_parity.py` / `run_web_inference.js`: PyTorch와 `inference.js`의 예측이 일치하는지 교차검증.

## 실행 순서

```bash
pip install -r requirements.txt
python train.py            # checkpoints/model.pth 생성
python export_web.py       # ../web_version/model/weights.json 생성
python verify_parity.py    # 두 구현이 같은 예측을 내는지 확인
```

학습 후에는 `checkpoints/model.pth`를 `../desktop_version/model/weights.pth`로 직접 복사해 주세요(자동화되어 있지 않음).

## 테스트

```bash
pytest tests/ -v
```
```

- [ ] **Step 2: 루트 CLAUDE.md 작성**

```markdown
<!-- 2026-09-18 16:56 KST -->
# 손글씨 숫자 인식

MNIST 손글씨 숫자(0-9)를 인식하는 프로그램. 웹 버전과 데스크톱 버전이 같은 학습된 모델을 공유한다.

설계 배경은 [docs/superpowers/specs/2026-09-18-handwriting-recognition-design.md](docs/superpowers/specs/2026-09-18-handwriting-recognition-design.md)를 참고해 주세요.

## 폴더

- `training/`: 모델 학습과 가중치 export. 자세한 내용은 `training/CLAUDE.md` 참고.
- `desktop_version/`: Python + PyTorch + Tkinter GUI. 자세한 내용은 `desktop_version/CLAUDE.md` 참고.
- `web_version/`: 외부 라이브러리 없는 순수 JS 웹 버전, GitHub Pages 배포 대상. 자세한 내용은 `web_version/CLAUDE.md` 참고.

## 작업 순서

1. `training/`에서 학습하고 가중치를 만든다.
2. `desktop_version/`, `web_version/`은 그 가중치를 각자 포맷대로 가져다 쓴다.
3. 모델 구조를 바꾸면 `training/model.py`와 `desktop_version/model.py`를 함께 수정하고, 다시 학습 → export → `verify_parity.py`로 확인한다.
```

- [ ] **Step 3: 커밋**

```bash
git add training/CLAUDE.md CLAUDE.md
git commit -m "docs: training/루트 CLAUDE.md 추가"
```
