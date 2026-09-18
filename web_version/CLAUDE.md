<!-- 2026-09-18 18:12 KST -->
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
