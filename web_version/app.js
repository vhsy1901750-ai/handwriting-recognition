// web_version/app.js
// 2026-09-18 18:05 KST
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
