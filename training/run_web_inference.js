// training/run_web_inference.js
// 2026-09-18 18:25 KST
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
