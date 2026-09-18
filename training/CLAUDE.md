<!-- 2026-09-18 18:35 KST -->
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
