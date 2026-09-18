// 2026-09-18 17:58 KST
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
