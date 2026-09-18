<!-- 2026-09-18 18:23 KST -->
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
