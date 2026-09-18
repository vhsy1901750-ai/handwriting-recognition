<!-- 2026-09-18 16:56 KST -->
# 손글씨 숫자 인식 프로그램 설계 (웹 + 데스크톱)

## 개요

손으로 쓴 숫자(0-9)를 인식하는 프로그램을 웹 버전과 데스크톱 버전으로 나누어 만든다.
모델은 한 번만 학습하고, 두 버전이 같은 학습된 가중치를 공유한다.

- **web_version**: 외부 라이브러리 없이 순수 JavaScript로 추론. GitHub Pages에 정적으로 배포 가능.
- **desktop_version**: Python + PyTorch + Tkinter GUI.
- **training**: 두 버전이 공유하는 모델을 1회 학습하고, 각 버전이 쓸 가중치 포맷으로 export.

## 폴더 구조

```
handwriting-recognition/
├── training/
│   ├── model.py          # 모델 구조 정의 (공유 소스)
│   ├── train.py          # MNIST 학습, checkpoints/model.pth 저장
│   ├── export_web.py     # checkpoints/model.pth → web_version/model/weights.json
│   ├── requirements.txt
│   └── CLAUDE.md
├── desktop_version/
│   ├── model.py          # training/model.py와 동일한 모델 정의 (복사본)
│   ├── app.py             # Tkinter 앱 진입점
│   ├── model/weights.pth  # export된 가중치
│   ├── requirements.txt
│   └── CLAUDE.md
├── web_version/
│   ├── index.html
│   ├── inference.js       # 순수 JS 순전파 구현
│   ├── app.js              # 캔버스 입력, UI 로직
│   ├── model/weights.json  # export된 가중치
│   └── CLAUDE.md
├── docs/superpowers/specs/  # 설계 문서
└── CLAUDE.md                 # 프로젝트 전체 개요
```

`training/model.py`가 모델 구조의 단일 진실 소스(source of truth)다. `desktop_version/model.py`는 추론에만 쓰이는 동일 구조의 복사본이며, 구조를 바꿀 때는 두 파일을 함께 수정해야 한다는 점을 각 CLAUDE.md에 명시한다.

## 모델 구조

작은 CNN, 입력은 28x28 흑백 이미지(0~1 정규화):

```
Conv2d(1, 8, kernel_size=3, padding=1) → ReLU → MaxPool2d(2)   # 28x28 → 14x14, 8채널
Conv2d(8, 16, kernel_size=3, padding=1) → ReLU → MaxPool2d(2)  # 14x14 → 7x7, 16채널
Flatten                                                          # 16*7*7 = 784
Linear(784, 10)
```

(브레인스토밍 단계 초안의 5x5는 padding=1 기준 7x7로 수정 — 아래 자기검토 참고)

출력에 softmax를 적용해 확률로 표시한다.

## 학습 (training/)

- 데이터: `torchvision.datasets.MNIST` (자동 다운로드).
- `train.py`: 표준 학습 루프(Adam, CrossEntropyLoss, ~5 epoch), 학습 종료 후 테스트셋 정확도를 출력하고 **99% 미만이면 경고 출력**(assert는 하지 않음 — 랜덤성으로 인한 실패 방지). 가중치는 `training/checkpoints/model.pth`에 저장.
- `export_web.py`: `checkpoints/model.pth`를 로드해 각 레이어의 weight/bias를 순서가 고정된 JSON 구조로 저장.

### 가중치 JSON 포맷 (web_version/model/weights.json)

```json
{
  "conv1": { "weight": [...], "weight_shape": [8,1,3,3], "bias": [...] },
  "conv2": { "weight": [...], "weight_shape": [16,8,3,3], "bias": [...] },
  "fc":    { "weight": [...], "weight_shape": [10,784], "bias": [...] }
}
```

`weight`는 PyTorch 텐서를 `.flatten().tolist()`한 1차원 배열(row-major, PyTorch 기본 순서와 동일). `inference.js`는 `weight_shape`를 보고 인덱스를 계산해 4차원/2차원처럼 접근한다.

## desktop_version 설계

- `app.py`: Tkinter 창에 `Canvas`(280x280, 확대해서 그리기 편하게), "인식" 버튼, "지우기" 버튼, 결과 레이블(예측 숫자 + 확률 %).
- 그리기: 마우스 드래그 이벤트로 캔버스에 선을 그림(굵기 조절 가능한 흰 선, 검은 배경 — MNIST와 동일한 색 배치).
- 인식 시: 캔버스를 PIL 이미지로 캡처 → 28x28로 리사이즈 → 정규화 → `model.py`의 모델로 forward → argmax + softmax 확률 표시.
- 모델 로드: 시작 시 `model/weights.pth`를 1회 로드.
- 에러 처리: 캔버스가 비어있는 상태(전부 검은색)에서 "인식" 누르면 팝업 없이 "그림을 먼저 그려주세요" 메시지만 결과 레이블에 표시.

## web_version 설계

- `index.html`: `<canvas>` 엘리먼트, "인식"/"지우기" 버튼, 결과 표시 영역. 외부 CDN, 프레임워크, 빌드 도구 없음 — 브라우저에서 파일을 그대로 열거나 GitHub Pages로 서빙하면 동작.
- `app.js`: 마우스/터치 드로잉 이벤트 처리(모바일 지원을 위해 touch 이벤트도 처리), 캔버스 → 28x28 다운샘플링(캔버스 API의 `drawImage`로 축소 후 `getImageData`), 정규화.
- `inference.js`: `weights.json`을 `fetch`로 로드 후 conv2d/relu/maxpool2d/linear/softmax를 `Float32Array` 기반으로 직접 구현. 외부 라이브러리 의존성 없음.
- 정적 배포 구조이므로 상태 저장이나 서버 통신 없음. 페이지 로드 시 `weights.json`을 한 번 fetch하고 이후는 전부 클라이언트 계산.

## 데이터 흐름 요약

```
training/train.py → checkpoints/model.pth
                          │
                          ├─→ (그대로 복사) desktop_version/model/weights.pth
                          └─→ export_web.py → web_version/model/weights.json
```

가중치 갱신은 수동 프로세스(재학습 후 두 파일을 다시 생성/복사)이며, 자동 동기화 파이프라인은 이 프로젝트 범위에 포함하지 않는다.

## 테스트/검증 전략

1. **학습 검증**: `train.py` 실행 후 출력되는 테스트 정확도가 99% 부근인지 육안 확인(자동 assert 없음, 위 참고).
2. **교차검증 스크립트** (`training/verify_parity.py`, 신규):
   - MNIST 테스트셋에서 샘플 20장을 골라 PyTorch 모델의 예측을 기준값으로 저장.
   - 같은 20장에 대해 Node.js로 `web_version/inference.js`를 실행해 예측이 기준값과 일치하는지 비교.
   - 불일치 시 어느 레이어부터 어긋나는지 알 수 있도록 중간 activation도 비교 옵션 제공(디버깅용, 필수 기능은 아님).
3. **desktop_version 수동 확인**: 앱을 실행해 직접 숫자를 그려보고 인식 결과 확인(자동화하지 않음 — GUI 특성상 실행/스모크 테스트로 충분).
4. **web_version 수동 확인**: `web_version/index.html`을 브라우저로 열어(또는 로컬 정적 서버로) 직접 그려보고 확인. GitHub Pages 배포 전 마지막 확인 단계로 포함.

## 범위에서 제외하는 것

- 사용자 인증, 서버 백엔드, 데이터베이스 — 둘 다 순수 클라이언트/로컬 앱.
- 여러 숫자로 구성된 문자열 인식(multi-digit) — 단일 숫자만.
- 모델 재학습 자동화나 CI 파이프라인.
- 데스크톱 앱 패키징(exe/설치파일 생성) — 소스 실행만 지원.
