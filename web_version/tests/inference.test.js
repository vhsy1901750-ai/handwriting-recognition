const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { conv2d, relu, maxPool2d, linear, softmax, runInference } = require("../inference.js");

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
