const dataset = [
  {
    email: "WIN MONEY NOW!!! Click here to claim your free prize.",
    x: [0.95, 0.85, 0.10, 0.95],
    y: 1
  },
  {
    email: "URGENT: Verify your bank account immediately using this link.",
    x: [0.85, 0.80, 0.15, 1.00],
    y: 1
  },
  {
    email: "You have been selected for an exclusive cash reward.",
    x: [0.90, 0.55, 0.20, 0.75],
    y: 1
  },
  {
    email: "Meeting moved to 14:30 tomorrow. Same room as planned.",
    x: [0.05, 0.10, 0.95, 0.10],
    y: 0
  },
  {
    email: "Hi, attached is the invoice from yesterday.",
    x: [0.05, 0.15, 0.90, 0.10],
    y: 0
  },
  {
    email: "Can you review the pull request when you have time?",
    x: [0.00, 0.05, 0.98, 0.05],
    y: 0
  }
];

const featureNames = [
  "Suspicious words",
  "Links",
  "Sender trust",
  "Urgency"
];

const initialParams = {
  W1: [
    [0.55, -0.25, 0.30, 0.20],
    [0.15, 0.60, -0.45, 0.35],
    [-0.35, 0.20, 0.65, -0.10]
  ],
  b1: [0.05, -0.10, 0.15],
  W2: [0.60, -0.40, 0.55],
  b2: -0.05
};

let params = structuredClone(initialParams);
let grads = null;
let cache = null;
let currentSample = 0;
let currentStep = 0;
let autoplayTimer = null;
let beforeTraining = null;
let afterTraining = null;

const $ = (sel) => document.querySelector(sel);

const els = {
  sampleSelect: $("#sampleSelect"),
  emailText: $("#emailText"),
  featureRow: $("#featureRow"),
  targetLabel: $("#targetLabel"),
  stepCounter: $("#stepCounter"),
  stepTitle: $("#stepTitle"),
  stepDescription: $("#stepDescription"),
  formulaBox: $("#formulaBox"),
  whyText: $("#whyText"),
  predictionValue: $("#predictionValue"),
  targetValue: $("#targetValue"),
  lossValue: $("#lossValue"),
  learningRateValue: $("#learningRateValue"),
  networkSvg: $("#networkSvg"),
  learningRate: $("#learningRate"),
  learningRateOutput: $("#learningRateOutput"),
  beforeSummary: $("#beforeSummary"),
  afterSummary: $("#afterSummary"),
  datasetTableBody: $("#datasetTableBody"),
  prevStep: $("#prevStep"),
  nextStep: $("#nextStep"),
  autoPlay: $("#autoPlay")
};

function relu(z) {
  return Math.max(0, z);
}

function reluPrime(z) {
  return z > 0 ? 1 : 0;
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function clampProbability(p) {
  return Math.min(1 - 1e-7, Math.max(1e-7, p));
}

function dot(a, b) {
  return a.reduce((sum, v, i) => sum + v * b[i], 0);
}

function fmt(n, digits = 3) {
  return Number.isFinite(n) ? n.toFixed(digits) : "—";
}

function classLabel(y) {
  return y === 1 ? "Spam" : "Not spam";
}

function forward(sample = dataset[currentSample]) {
  const z1 = params.W1.map((row, i) => dot(row, sample.x) + params.b1[i]);
  const h = z1.map(relu);
  const z2 = dot(params.W2, h) + params.b2;
  const yHat = sigmoid(z2);
  const p = clampProbability(yHat);
  const loss = -(sample.y * Math.log(p) + (1 - sample.y) * Math.log(1 - p));

  cache = {
    x: [...sample.x],
    y: sample.y,
    z1,
    h,
    z2,
    yHat,
    loss
  };

  return cache;
}

function backward() {
  const c = cache || forward();

  // BCE + sigmoid gives dL/dz2 = yHat - y
  const dz2 = c.yHat - c.y;
  const dW2 = c.h.map((h) => dz2 * h);
  const db2 = dz2;

  const dh = params.W2.map((w) => dz2 * w);
  const dz1 = dh.map((v, i) => v * reluPrime(c.z1[i]));

  const dW1 = dz1.map((g) => c.x.map((x) => g * x));
  const db1 = [...dz1];

  grads = { dW1, db1, dW2, db2, dz2, dz1, dh };
  return grads;
}

function applyGradients() {
  const learningRate = Number(els.learningRate.value);
  if (!grads) backward();

  for (let i = 0; i < params.W1.length; i++) {
    for (let j = 0; j < params.W1[i].length; j++) {
      params.W1[i][j] -= learningRate * grads.dW1[i][j];
    }
    params.b1[i] -= learningRate * grads.db1[i];
  }

  for (let i = 0; i < params.W2.length; i++) {
    params.W2[i] -= learningRate * grads.dW2[i];
  }
  params.b2 -= learningRate * grads.db2;

  grads = null;
  return forward();
}

function trainOnAllSamples(epochs = 1) {
  for (let epoch = 0; epoch < epochs; epoch++) {
    for (let i = 0; i < dataset.length; i++) {
      const previous = currentSample;
      currentSample = i;
      forward(dataset[i]);
      backward();
      applyGradients();
      currentSample = previous;
    }
  }
  return forward(dataset[currentSample]);
}

const steps = [
  {
    title: "Inputs",
    description: "The network does not read words directly in this first lesson. Each email is represented by four normalized numeric features.",
    why: "Neural networks operate on numbers. Features are the numeric representation of the real-world input.",
    formula: () => {
      const s = dataset[currentSample];
      return [
        `x = [${s.x.map(v => fmt(v, 2)).join(", ")}]`,
        `target y = ${s.y}  (${classLabel(s.y)})`
      ];
    },
    focus: "inputs"
  },
  {
    title: "Weighted sums",
    description: "Each hidden neuron receives all four input features. Every connection has its own weight, and each neuron also has a bias.",
    why: "Weights control how strongly each feature influences a neuron. The bias lets the neuron shift its response independently of the inputs.",
    formula: () => {
      const c = cache || forward();
      return [
        `z₁ = Σ(w₁ⱼ · xⱼ) + b₁ = ${fmt(c.z1[0])}`,
        `z₂ = Σ(w₂ⱼ · xⱼ) + b₂ = ${fmt(c.z1[1])}`,
        `z₃ = Σ(w₃ⱼ · xⱼ) + b₃ = ${fmt(c.z1[2])}`
      ];
    },
    focus: "input-hidden"
  },
  {
    title: "Activation function",
    description: "The hidden layer applies ReLU to each weighted sum. Positive values pass through; negative values become zero.",
    why: "Without a nonlinear activation, stacking multiple layers would still behave like one linear transformation.",
    formula: () => {
      const c = cache || forward();
      return c.z1.map((z, i) => `h${i + 1} = ReLU(${fmt(z)}) = ${fmt(c.h[i])}`);
    },
    focus: "hidden"
  },
  {
    title: "Output logit",
    description: "The output neuron combines the hidden activations into one final score called a logit.",
    why: "The logit is an unrestricted real number. We still need to convert it into a probability.",
    formula: () => {
      const c = cache || forward();
      return [
        `z_out = v₁h₁ + v₂h₂ + v₃h₃ + b_out`,
        `z_out = ${fmt(c.z2)}`
      ];
    },
    focus: "hidden-output"
  },
  {
    title: "Sigmoid prediction",
    description: "Sigmoid converts the output logit into a number between 0 and 1, which we interpret as the probability of spam.",
    why: "Binary classification needs one probability. Values near 1 mean spam; values near 0 mean not spam.",
    formula: () => {
      const c = cache || forward();
      return [
        `ŷ = σ(z_out) = 1 / (1 + e^(-z_out))`,
        `ŷ = ${fmt(c.yHat)}  →  ${(c.yHat * 100).toFixed(1)}% spam`
      ];
    },
    focus: "output"
  },
  {
    title: "Loss",
    description: "Binary cross-entropy compares the predicted probability with the true target and gives us one number for how wrong the prediction is.",
    why: "Training needs an objective to minimize. Lower loss means the prediction better matches the target.",
    formula: () => {
      const c = cache || forward();
      return [
        `L = -[y·ln(ŷ) + (1-y)·ln(1-ŷ)]`,
        `L = ${fmt(c.loss)}`
      ];
    },
    focus: "loss"
  },
  {
    title: "Backpropagation",
    description: "We compute how much each parameter contributed to the loss. These derivatives are the gradients.",
    why: "The gradient tells us which direction would increase the loss. To reduce the loss, gradient descent moves the parameters in the opposite direction.",
    formula: () => {
      if (!grads) backward();
      return [
        `∂L/∂z_out = ŷ - y = ${fmt(grads.dz2)}`,
        `∂L/∂v₁ = ${fmt(grads.dW2[0])}`,
        `∂L/∂v₂ = ${fmt(grads.dW2[1])}`,
        `∂L/∂v₃ = ${fmt(grads.dW2[2])}`,
        `gradients continue backward through ReLU`
      ];
    },
    focus: "backprop"
  },
  {
    title: "Gradient descent update",
    description: "Finally, we update every weight and bias by subtracting the learning rate multiplied by its gradient.",
    why: "Repeating forward pass → loss → backpropagation → update is how the neural network learns from data.",
    formula: () => {
      if (!grads) backward();
      const lr = Number(els.learningRate.value);
      return [
        `w_new = w_old - η · ∂L/∂w`,
        `η = ${lr.toFixed(2)}`,
        `example: v₁ = ${fmt(params.W2[0])} - ${lr.toFixed(2)} · (${fmt(grads.dW2[0])})`
      ];
    },
    focus: "update"
  }
];

function buildSelect() {
  dataset.forEach((sample, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `${i + 1}. ${sample.email.slice(0, 54)}${sample.email.length > 54 ? "…" : ""}`;
    els.sampleSelect.appendChild(opt);
  });
}

function buildDatasetTable() {
  els.datasetTableBody.innerHTML = dataset.map((sample) => `
    <tr>
      <td>${escapeHtml(sample.email)}</td>
      ${sample.x.map(v => `<td>${v.toFixed(2)}</td>`).join("")}
      <td>${classLabel(sample.y)}</td>
    </tr>
  `).join("");
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderTask() {
  const s = dataset[currentSample];
  els.emailText.textContent = s.email;
  els.targetLabel.textContent = `${s.y} · ${classLabel(s.y)}`;
  els.targetLabel.className = s.y === 1 ? "badge-spam" : "badge-ham";
  els.featureRow.innerHTML = s.x.map((v, i) => `
    <div class="feature-chip">
      <span>${featureNames[i]}</span>
      <strong>x${i + 1} = ${v.toFixed(2)}</strong>
    </div>
  `).join("");
}

function renderStatus() {
  const c = cache || forward();
  els.predictionValue.textContent = `${(c.yHat * 100).toFixed(1)}% spam`;
  els.targetValue.textContent = `${c.y} · ${classLabel(c.y)}`;
  els.lossValue.textContent = fmt(c.loss);
  els.learningRateValue.textContent = Number(els.learningRate.value).toFixed(2);
}

function formulaHtml(lines) {
  return lines.map(line => `<p class="formula-line">${escapeHtml(line)}</p>`).join("");
}

function renderStep() {
  forward();
  const step = steps[currentStep];
  els.stepCounter.textContent = `Step ${currentStep + 1} / ${steps.length}`;
  els.stepTitle.textContent = step.title;
  els.stepDescription.textContent = step.description;
  els.whyText.textContent = step.why;
  els.formulaBox.innerHTML = formulaHtml(step.formula());
  els.prevStep.disabled = currentStep === 0;
  els.nextStep.textContent = currentStep === steps.length - 1 ? "Finish ✓" : "Next →";
  renderStatus();
  renderNetwork(step.focus);
}

function nodeGroup(id, x, y, title, value, active = false, extraClass = "") {
  return `
    <g class="node ${active ? "active" : ""} ${extraClass}" data-node="${id}">
      <circle class="node-circle" cx="${x}" cy="${y}" r="38"></circle>
      <text class="node-label" x="${x}" y="${y - 3}" text-anchor="middle">${title}</text>
      <text class="node-value" x="${x}" y="${y + 18}" text-anchor="middle">${value}</text>
    </g>
  `;
}

function edgeLine(x1, y1, x2, y2, label, active = false, extra = "") {
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  return `
    <g>
      <line class="edge ${active ? "active" : ""} ${extra}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"></line>
      <text class="edge-label ${active ? "active" : ""}" x="${mx}" y="${my - 7}" text-anchor="middle">${label}</text>
    </g>
  `;
}

function renderNetwork(focus = "inputs") {
  const c = cache || forward();
  if (focus === "backprop" && !grads) backward();

  const inputX = 90;
  const hiddenX = 410;
  const outputX = 720;
  const lossX = 900;

  const inputY = [105, 205, 305, 405];
  const hiddenY = [145, 260, 375];
  const outputY = 260;

  const activeInputs = ["inputs", "input-hidden", "backprop", "update"].includes(focus);
  const activeHidden = ["hidden", "hidden-output", "backprop", "update"].includes(focus);
  const activeOutput = ["output", "loss", "backprop", "update"].includes(focus);

  let edges = "";

  // input -> hidden
  for (let i = 0; i < 4; i++) {
    for (let h = 0; h < 3; h++) {
      const isActive = focus === "input-hidden" || focus === "backprop" || focus === "update";
      const extra = focus === "backprop" ? "backprop" : focus === "update" ? "updated" : "";
      edges += edgeLine(
        inputX + 38,
        inputY[i],
        hiddenX - 38,
        hiddenY[h],
        `w${h + 1}${i + 1}=${fmt(params.W1[h][i], 2)}`,
        isActive,
        extra
      );
    }
  }

  // hidden -> output
  for (let h = 0; h < 3; h++) {
    const isActive = ["hidden-output", "backprop", "update"].includes(focus);
    const extra = focus === "backprop" ? "backprop" : focus === "update" ? "updated" : "";
    edges += edgeLine(
      hiddenX + 38,
      hiddenY[h],
      outputX - 38,
      outputY,
      `v${h + 1}=${fmt(params.W2[h], 2)}`,
      isActive,
      extra
    );
  }

  const lossActive = focus === "loss" || focus === "backprop" || focus === "update";

  const svg = `
    <text class="layer-label" x="${inputX}" y="35" text-anchor="middle">Inputs</text>
    <text class="layer-label" x="${hiddenX}" y="35" text-anchor="middle">Hidden layer + ReLU</text>
    <text class="layer-label" x="${outputX}" y="35" text-anchor="middle">Output + sigmoid</text>
    <text class="layer-label" x="${lossX}" y="35" text-anchor="middle">Loss</text>

    ${edges}

    ${nodeGroup("x1", inputX, inputY[0], "x₁", fmt(dataset[currentSample].x[0], 2), activeInputs)}
    ${nodeGroup("x2", inputX, inputY[1], "x₂", fmt(dataset[currentSample].x[1], 2), activeInputs)}
    ${nodeGroup("x3", inputX, inputY[2], "x₃", fmt(dataset[currentSample].x[2], 2), activeInputs)}
    ${nodeGroup("x4", inputX, inputY[3], "x₄", fmt(dataset[currentSample].x[3], 2), activeInputs)}

    ${nodeGroup("h1", hiddenX, hiddenY[0], "h₁", fmt(c.h[0]), activeHidden)}
    ${nodeGroup("h2", hiddenX, hiddenY[1], "h₂", fmt(c.h[1]), activeHidden)}
    ${nodeGroup("h3", hiddenX, hiddenY[2], "h₃", fmt(c.h[2]), activeHidden)}

    ${nodeGroup("out", outputX, outputY, "ŷ", fmt(c.yHat), activeOutput, "output")}

    <g>
      <line class="edge ${lossActive ? "active" : ""} ${focus === "backprop" ? "backprop" : ""}" x1="${outputX + 38}" y1="${outputY}" x2="${lossX - 62}" y2="${outputY}"></line>
      <rect class="loss-box ${lossActive ? "active" : ""}" x="${lossX - 62}" y="${outputY - 44}" width="124" height="88" rx="14"></rect>
      <text class="loss-text" x="${lossX}" y="${outputY - 5}" text-anchor="middle">L = ${fmt(c.loss)}</text>
      <text class="loss-subtext" x="${lossX}" y="${outputY + 19}" text-anchor="middle">target y=${c.y}</text>
    </g>

    <text class="loss-subtext" x="${hiddenX}" y="485" text-anchor="middle">
      biases: b = [${params.b1.map(v => fmt(v, 2)).join(", ")}]
    </text>
    <text class="loss-subtext" x="${outputX}" y="485" text-anchor="middle">
      b_out = ${fmt(params.b2, 2)}
    </text>
  `;

  els.networkSvg.innerHTML = svg;
}

function updateHistory(before, after) {
  els.beforeSummary.textContent = before
    ? `${(before.yHat * 100).toFixed(1)}% · L ${fmt(before.loss)}`
    : "—";
  els.afterSummary.textContent = after
    ? `${(after.yHat * 100).toFixed(1)}% · L ${fmt(after.loss)}`
    : "—";
}

function stopAutoplay() {
  if (autoplayTimer) {
    clearInterval(autoplayTimer);
    autoplayTimer = null;
  }
  els.autoPlay.textContent = "▶ Auto";
}

els.sampleSelect.addEventListener("change", (e) => {
  currentSample = Number(e.target.value);
  grads = null;
  cache = null;
  currentStep = 0;
  beforeTraining = null;
  afterTraining = null;
  stopAutoplay();
  renderTask();
  renderStep();
  updateHistory(null, null);
});

els.prevStep.addEventListener("click", () => {
  stopAutoplay();
  currentStep = Math.max(0, currentStep - 1);
  grads = null;
  renderStep();
});

els.nextStep.addEventListener("click", () => {
  stopAutoplay();
  if (currentStep < steps.length - 1) {
    currentStep++;
    if (currentStep >= 6) backward();
    renderStep();
  }
});

els.autoPlay.addEventListener("click", () => {
  if (autoplayTimer) {
    stopAutoplay();
    return;
  }

  els.autoPlay.textContent = "⏸ Pause";
  autoplayTimer = setInterval(() => {
    if (currentStep >= steps.length - 1) {
      stopAutoplay();
      return;
    }
    currentStep++;
    if (currentStep >= 6) backward();
    renderStep();
  }, 1200);
});

$("#restartLesson").addEventListener("click", () => {
  stopAutoplay();
  currentStep = 0;
  grads = null;
  renderStep();
});

$("#runForward").addEventListener("click", () => {
  stopAutoplay();
  grads = null;
  currentStep = 4;
  forward();
  renderStep();
});

$("#runBackprop").addEventListener("click", () => {
  stopAutoplay();
  currentStep = 6;
  forward();
  backward();
  renderStep();
});

$("#updateWeights").addEventListener("click", () => {
  stopAutoplay();
  beforeTraining = forward();
  backward();
  applyGradients();
  afterTraining = forward();
  currentStep = 7;
  updateHistory(beforeTraining, afterTraining);
  renderStep();
});

$("#trainTen").addEventListener("click", () => {
  stopAutoplay();
  beforeTraining = forward();
  trainOnAllSamples(10);
  afterTraining = forward();
  currentStep = 7;
  grads = null;
  updateHistory(beforeTraining, afterTraining);
  renderStep();
});

$("#resetWeights").addEventListener("click", () => {
  stopAutoplay();
  params = structuredClone(initialParams);
  grads = null;
  cache = null;
  beforeTraining = null;
  afterTraining = null;
  currentStep = 0;
  updateHistory(null, null);
  renderStep();
});

els.learningRate.addEventListener("input", () => {
  els.learningRateOutput.textContent = Number(els.learningRate.value).toFixed(2);
  els.learningRateValue.textContent = Number(els.learningRate.value).toFixed(2);
  if (currentStep === 7) {
    renderStep();
  }
});

document.querySelectorAll(".mode-btn").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".mode-btn").forEach((b) => {
      b.classList.remove("active");
      b.setAttribute("aria-selected", "false");
    });
    button.classList.add("active");
    button.setAttribute("aria-selected", "true");
    document.body.dataset.mode = button.dataset.mode;
  });
});

function init() {
  buildSelect();
  buildDatasetTable();
  els.sampleSelect.value = "0";
  els.learningRateOutput.textContent = Number(els.learningRate.value).toFixed(2);
  renderTask();
  renderStep();
  updateHistory(null, null);
}

init();
