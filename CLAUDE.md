<!-- 2026-09-18 18:35 KST -->
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
