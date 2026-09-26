const dataset = [
  { email: "WIN MONEY NOW!!! Click here to claim your free prize.", x: [0.95, 0.85, 0.10, 0.95], y: 1 },
  { email: "URGENT: Verify your bank account immediately using this link.", x: [0.85, 0.80, 0.15, 1.00], y: 1 },
  { email: "You have been selected for an exclusive cash reward.", x: [0.90, 0.55, 0.20, 0.75], y: 1 },
  { email: "Meeting moved to 14:30 tomorrow. Same room as planned.", x: [0.05, 0.10, 0.95, 0.10], y: 0 },
  { email: "Hi, attached is the invoice from yesterday.", x: [0.05, 0.15, 0.90, 0.10], y: 0 },
  { email: "Can you review the pull request when you have time?", x: [0.00, 0.05, 0.98, 0.05], y: 0 }
];

const featureNames = ["Suspicious words", "Links", "Sender trust", "Urgency"];

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

const $ = (selector) => document.querySelector(selector);

const els = {
  sampleSelect: $("#sampleSelect"),
  emailText: $("#emailText"),
  targetLabel: $("#targetLabel"),
  featureRow: $("#featureRow"),
  stepRail: $("#stepRail"),
  stepCounter: $("#stepCounter"),
  stepTitle: $("#stepTitle"),
  stepKind: $("#stepKind"),
  stepDescription: $("#stepDescription"),
  formulaBox: $("#formulaBox"),
  whyText: $("#whyText"),
  parameterRows: $("#parameterRows"),
  inspectorHint: $("#inspectorHint"),
  networkSvg: $("#networkSvg"),
  phaseLabel: $("#phaseLabel"),
  phaseDot: $("#phaseDot"),
  predictionValue: $("#predictionValue"),
  targetValue: $("#targetValue"),
  lossValue: $("#lossValue"),
  learningRateValue: $("#learningRateValue"),
  learningRate: $("#learningRate"),
  learningRateOutput: $("#learningRateOutput"),
  beforeSummary: $("#beforeSummary"),
  afterSummary: $("#afterSummary"),
  datasetTableBody: $("#datasetTableBody"),
  prevStep: $("#prevStep"),
  nextStep: $("#nextStep"),
  autoPlay: $("#autoPlay"),
  runForward: $("#runForward"),
  runBackprop: $("#runBackprop"),
  updateWeights: $("#updateWeights")
};

let params = clone(initialParams);
let cache = null;
let grads = null;
let currentSample = 0;
let currentStep = 0;
let lessonTimer = null;
let phaseTimer = null;
let animation = { phase: "idle", stage: -1 };
let beforeTraining = null;
let afterTraining = null;
let lastUpdate = null;

function clone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

function relu(z) {
  return Math.max(0, z);
}

function reluPrime(z) {
  return z > 0 ? 1 : 0;
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function dot(a, b) {
  return a.reduce((sum, value, i) => sum + value * b[i], 0);
}

function clampProbability(p) {
  return Math.min(1 - 1e-7, Math.max(1e-7, p));
}

function fmt(value, digits = 3) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function signed(value, digits = 3) {
  if (!Number.isFinite(value)) return "—";
  const n = value.toFixed(digits);
  return value > 0 ? `+${n}` : n;
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

  // For sigmoid + binary cross entropy:
  // dL/dz_out = y_hat - y
  const dz2 = c.yHat - c.y;
  const dW2 = c.h.map((h) => dz2 * h);
  const db2 = dz2;

  const dh = params.W2.map((w) => dz2 * w);
  const dz1 = dh.map((value, i) => value * reluPrime(c.z1[i]));

  const dW1 = dz1.map((g) => c.x.map((x) => g * x));
  const db1 = [...dz1];

  grads = { dz2, dW2, db2, dh, dz1, dW1, db1 };
  return grads;
}

function captureUpdateSnapshot() {
  const learningRate = Number(els.learningRate.value);
  const oldParams = clone(params);
  const g = grads || backward();
  const newParams = clone(params);

  for (let h = 0; h < 3; h++) {
    for (let i = 0; i < 4; i++) {
      newParams.W1[h][i] = oldParams.W1[h][i] - learningRate * g.dW1[h][i];
    }
    newParams.b1[h] = oldParams.b1[h] - learningRate * g.db1[h];
    newParams.W2[h] = oldParams.W2[h] - learningRate * g.dW2[h];
  }
  newParams.b2 = oldParams.b2 - learningRate * g.db2;

  return { oldParams, newParams, grads: clone(g), learningRate };
}

function applyGradients() {
  if (!grads) backward();
  lastUpdate = captureUpdateSnapshot();
  params = clone(lastUpdate.newParams);
  grads = null;
  return forward();
}

function trainOnAllSamples(epochs = 1) {
  for (let epoch = 0; epoch < epochs; epoch++) {
    for (const sample of dataset) {
      forward(sample);
      backward();
      applyGradients();
    }
  }
  return forward(dataset[currentSample]);
}

const steps = [
  {
    short: "Inputs",
    kind: "Representation",
    title: "Inputs become numbers",
    description: "The network does not read raw words in this first lesson. Each email is represented by four normalized numeric features.",
    why: "Neural networks operate on numbers. A feature vector is the bridge between a real-world example and the mathematical model.",
    focus: "inputs",
    formula: () => {
      const sample = dataset[currentSample];
      return [
        `x = [${sample.x.map(v => fmt(v, 2)).join(", ")}]`,
        `target y = ${sample.y}  (${classLabel(sample.y)})`
      ];
    }
  },
  {
    short: "Weights",
    kind: "Linear layer",
    title: "Weights decide what matters",
    description: "Every hidden neuron receives all four features. Each connection multiplies its input by a trainable weight, then the neuron adds a bias.",
    why: "A larger positive weight increases a feature's influence. A negative weight pushes in the opposite direction. Training will change these numbers.",
    focus: "input-hidden",
    formula: () => {
      const c = cache || forward();
      return [
        `z₁ = Σ(w₁ⱼxⱼ) + b₁ = ${fmt(c.z1[0])}`,
        `z₂ = Σ(w₂ⱼxⱼ) + b₂ = ${fmt(c.z1[1])}`,
        `z₃ = Σ(w₃ⱼxⱼ) + b₃ = ${fmt(c.z1[2])}`
      ];
    }
  },
  {
    short: "ReLU",
    kind: "Activation",
    title: "ReLU adds nonlinearity",
    description: "The hidden layer applies ReLU to each weighted sum: positive values pass through and negative values become zero.",
    why: "Without nonlinear activations, stacking layers would collapse into one linear transformation and could not learn genuinely nonlinear relationships.",
    focus: "hidden",
    formula: () => {
      const c = cache || forward();
      return c.z1.map((z, i) => `h${i + 1} = ReLU(${fmt(z)}) = ${fmt(c.h[i])}`);
    }
  },
  {
    short: "Logit",
    kind: "Linear layer",
    title: "Hidden features produce one score",
    description: "The output neuron combines the three hidden activations into a final unrestricted score called the logit.",
    why: "This last linear layer compresses the learned hidden representation into the one quantity needed by a binary classifier.",
    focus: "hidden-output",
    formula: () => {
      const c = cache || forward();
      return [
        `z_out = v₁h₁ + v₂h₂ + v₃h₃ + b_out`,
        `z_out = ${fmt(c.z2)}`
      ];
    }
  },
  {
    short: "Sigmoid",
    kind: "Prediction",
    title: "Sigmoid turns the score into a probability",
    description: "Sigmoid maps the output logit to a number between 0 and 1. We interpret that number as P(spam).",
    why: "For binary classification, one probability is enough: near 1 means spam; near 0 means not spam.",
    focus: "output",
    formula: () => {
      const c = cache || forward();
      return [
        `ŷ = σ(z_out) = 1 / (1 + e^(-z_out))`,
        `ŷ = ${fmt(c.yHat)}  →  ${(c.yHat * 100).toFixed(1)}% spam`
      ];
    }
  },
  {
    short: "Loss",
    kind: "Objective",
    title: "Loss measures how wrong we are",
    description: "Binary cross-entropy compares the predicted probability with the ground-truth label and returns one scalar error.",
    why: "The optimizer needs a single objective to reduce. Training is ultimately the process of changing parameters so this number becomes smaller.",
    focus: "loss",
    formula: () => {
      const c = cache || forward();
      return [
        `L = -[y·ln(ŷ) + (1-y)·ln(1-ŷ)]`,
        `L = ${fmt(c.loss)}`
      ];
    }
  },
  {
    short: "Backprop",
    kind: "Gradients",
    title: "Backprop assigns blame to every weight",
    description: "Starting from the loss, the chain rule computes how much each weight and bias affected that loss. Those derivatives are the gradients.",
    why: "A gradient is not a new weight. It is a sensitivity: how much the loss would change if we nudged that parameter.",
    focus: "backprop",
    formula: () => {
      if (!grads) backward();
      return [
        `∂L/∂z_out = ŷ - y = ${fmt(grads.dz2)}`,
        `∂L/∂v₁ = ${fmt(grads.dW2[0])}`,
        `∂L/∂v₂ = ${fmt(grads.dW2[1])}`,
        `∂L/∂v₃ = ${fmt(grads.dW2[2])}`,
        `then gradients continue backward through ReLU`
      ];
    }
  },
  {
    short: "Update",
    kind: "Optimizer",
    title: "Gradient descent updates the parameters",
    description: "Each parameter moves opposite its gradient. The learning rate controls the size of that move.",
    why: "Repeat forward → loss → backprop → update over many examples and the network gradually finds parameters that reduce prediction error.",
    focus: "update",
    formula: () => {
      if (!grads) backward();
      const lr = Number(els.learningRate.value);
      return [
        `w_new = w_old - η · ∂L/∂w`,
        `η = ${lr.toFixed(2)}`,
        `example: v₁ = ${fmt(params.W2[0])} - ${lr.toFixed(2)} × (${fmt(grads.dW2[0])})`
      ];
    }
  }
];

function buildSampleSelect() {
  dataset.forEach((sample, i) => {
    const option = document.createElement("option");
    option.value = i;
    option.textContent = `${i + 1}. ${sample.email.slice(0, 50)}${sample.email.length > 50 ? "…" : ""}`;
    els.sampleSelect.appendChild(option);
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

function buildStepRail() {
  els.stepRail.innerHTML = steps.map((step, i) => `
    <button class="step-tab ${i === currentStep ? "active" : ""}" data-step="${i}">
      ${i + 1}. ${step.short}
    </button>
  `).join("");

  els.stepRail.querySelectorAll(".step-tab").forEach(button => {
    button.addEventListener("click", () => {
      stopAllAnimations();
      currentStep = Number(button.dataset.step);
      if (currentStep >= 6) {
        forward();
        backward();
      } else {
        grads = null;
      }
      renderAll();
    });
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderTask() {
  const sample = dataset[currentSample];
  els.emailText.textContent = sample.email;
  els.targetLabel.textContent = `${sample.y} · ${classLabel(sample.y)}`;
  els.targetLabel.className = `target-value ${sample.y === 1 ? "badge-spam" : "badge-ham"}`;

  els.featureRow.innerHTML = sample.x.map((value, i) => `
    <div class="feature-chip">
      <span>${featureNames[i]}</span>
      <strong>x${i + 1} = ${value.toFixed(2)}</strong>
    </div>
  `).join("");
}

function formulaHtml(lines) {
  return lines.map(line => `<p class="formula-line">${escapeHtml(line)}</p>`).join("");
}

function renderStepText() {
  const step = steps[currentStep];
  els.stepCounter.textContent = `Step ${currentStep + 1} of ${steps.length}`;
  els.stepTitle.textContent = step.title;
  els.stepKind.textContent = step.kind;
  els.stepDescription.textContent = step.description;
  els.whyText.textContent = step.why;
  els.formulaBox.innerHTML = formulaHtml(step.formula());

  els.prevStep.disabled = currentStep === 0;
  els.nextStep.textContent = currentStep === steps.length - 1 ? "Lesson complete ✓" : "Next concept →";

  els.stepRail.querySelectorAll(".step-tab").forEach((tab, i) => {
    tab.classList.toggle("active", i === currentStep);
    tab.classList.toggle("done", i < currentStep);
  });
}

function renderStatus() {
  const c = cache || forward();
  els.predictionValue.textContent = `${(c.yHat * 100).toFixed(1)}% spam`;
  els.targetValue.textContent = `${c.y} · ${classLabel(c.y)}`;
  els.lossValue.textContent = fmt(c.loss);
  els.learningRateValue.textContent = Number(els.learningRate.value).toFixed(2);
  els.learningRateOutput.textContent = Number(els.learningRate.value).toFixed(2);
}

function updateHistory(before, after) {
  els.beforeSummary.textContent = before
    ? `${(before.yHat * 100).toFixed(1)}% · loss ${fmt(before.loss)}`
    : "—";
  els.afterSummary.textContent = after
    ? `${(after.yHat * 100).toFixed(1)}% · loss ${fmt(after.loss)}`
    : "—";
}

function renderInspector() {
  const c = cache || forward();
  const g = grads;
  const lr = Number(els.learningRate.value);

  let rows = [];

  if (currentStep < 6 && animation.phase !== "backprop" && animation.phase !== "update") {
    els.inspectorHint.textContent = "At backprop, every trainable parameter gets a gradient.";
    rows = [
      ["v₁", params.W2[0], null, null],
      ["v₂", params.W2[1], null, null],
      ["v₃", params.W2[2], null, null],
      ["b_out", params.b2, null, null]
    ];
  } else {
    if (!grads) backward();
    els.inspectorHint.textContent = "g = ∂L/∂parameter.  Δ = −η·g is the actual gradient-descent move.";
    rows = [
      ["v₁", params.W2[0], grads.dW2[0], -lr * grads.dW2[0]],
      ["v₂", params.W2[1], grads.dW2[1], -lr * grads.dW2[1]],
      ["v₃", params.W2[2], grads.dW2[2], -lr * grads.dW2[2]],
      ["b_out", params.b2, grads.db2, -lr * grads.db2]
    ];
  }

  els.parameterRows.innerHTML = `
    <div class="param-row param-header">
      <span></span><span>value</span><span>gradient g</span><span>update Δ</span>
    </div>
    ${rows.map(([name, value, grad, delta]) => `
      <div class="param-row">
        <span class="param-name">${name}</span>
        <span class="param-cell">${fmt(value)}</span>
        <span class="param-cell gradient">${grad === null ? "—" : signed(grad)}</span>
        <span class="param-cell delta">${delta === null ? "—" : signed(delta)}</span>
      </div>
    `).join("")}
  `;
}

function setPhaseIndicator(phase, text) {
  const container = els.phaseDot.parentElement;
  container.classList.remove("forward", "backprop", "update");
  if (phase !== "idle") container.classList.add(phase);
  els.phaseLabel.textContent = text;

  [els.runForward, els.runBackprop, els.updateWeights].forEach(btn => btn.classList.remove("active"));
  if (phase === "forward") els.runForward.classList.add("active");
  if (phase === "backprop") els.runBackprop.classList.add("active");
  if (phase === "update") els.updateWeights.classList.add("active");
}

function nodeGroup(id, x, y, title, value, state = "") {
  return `
    <g class="node ${state}" data-node="${id}">
      <circle class="node-circle" cx="${x}" cy="${y}" r="38"></circle>
      <text class="node-label" x="${x}" y="${y - 3}" text-anchor="middle">${title}</text>
      <text class="node-value" x="${x}" y="${y + 18}" text-anchor="middle">${value}</text>
    </g>
  `;
}

function tagBox(x, y, weightLabel, gradLabel = null, deltaLabel = null, state = "") {
  const lines = 1 + (gradLabel !== null ? 1 : 0) + (deltaLabel !== null ? 1 : 0);
  const height = lines === 1 ? 22 : lines === 2 ? 34 : 46;
  const width = 82;
  const top = y - height / 2;
  let text = `<text class="weight-text" x="${x}" y="${top + 14}" text-anchor="middle">${weightLabel}</text>`;
  if (gradLabel !== null) {
    text += `<text class="grad-text" x="${x}" y="${top + 26}" text-anchor="middle">g=${gradLabel}</text>`;
  }
  if (deltaLabel !== null) {
    text += `<text class="delta-text" x="${x}" y="${top + 38}" text-anchor="middle">Δ=${deltaLabel}</text>`;
  }

  return `
    <g class="edge-tag ${state}">
      <rect x="${x - width / 2}" y="${top}" width="${width}" height="${height}" rx="7"></rect>
      ${text}
    </g>
  `;
}

function edgeLine({
  x1, y1, x2, y2, weight, grad = null, delta = null,
  edgeState = "", tagState = "", tagT = 0.5, tagYOffset = 0
}) {
  const tx = x1 + (x2 - x1) * tagT;
  const ty = y1 + (y2 - y1) * tagT + tagYOffset;

  return `
    <g>
      <line class="edge ${edgeState}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"></line>
      ${tagBox(
        tx,
        ty,
        weight,
        grad === null ? null : grad,
        delta === null ? null : delta,
        tagState
      )}
    </g>
  `;
}

function edgeVisualState(group) {
  const phase = animation.phase;
  const stage = animation.stage;

  if (phase === "forward") {
    if (group === "ih" && stage >= 1) return ["active-forward", "active"];
    if (group === "ho" && stage >= 3) return ["active-forward", "active"];
    if (group === "ol" && stage >= 5) return ["active-forward", "active"];
  }

  if (phase === "backprop") {
    if (group === "ol" && stage >= 1) return ["active-backprop", "backprop"];
    if (group === "ho" && stage >= 2) return ["active-backprop", "backprop"];
    if (group === "ih" && stage >= 3) return ["active-backprop", "backprop"];
  }

  if (phase === "update") return ["updated", "update"];

  const focus = steps[currentStep].focus;
  if (focus === "input-hidden" && group === "ih") return ["active-forward", "active"];
  if (focus === "hidden-output" && group === "ho") return ["active-forward", "active"];
  if (focus === "backprop") return ["active-backprop", "backprop"];
  if (focus === "update") return ["updated", "update"];
  if (focus === "loss" && group === "ol") return ["active-forward", "active"];

  return ["", ""];
}

function nodeVisualState(group) {
  const phase = animation.phase;
  const stage = animation.stage;

  if (phase === "forward") {
    if (group === "input" && stage >= 0) return "active";
    if (group === "hidden" && stage >= 2) return "active";
    if (group === "output" && stage >= 4) return "active";
  }

  if (phase === "backprop") {
    if (group === "output" && stage >= 1) return "backprop-active";
    if (group === "hidden" && stage >= 2) return "backprop-active";
    if (group === "input" && stage >= 3) return "backprop-active";
  }

  if (phase === "update") return "updated";

  const focus = steps[currentStep].focus;
  if (focus === "inputs" && group === "input") return "active";
  if (focus === "hidden" && group === "hidden") return "active";
  if (focus === "output" && group === "output") return "active";
  if (focus === "backprop") return "backprop-active";
  if (focus === "update") return "updated";

  return "";
}

function gradientForW1(h, i) {
  return grads ? grads.dW1[h][i] : null;
}

function gradientForW2(h) {
  return grads ? grads.dW2[h] : null;
}

function deltaForGradient(g) {
  return g === null ? null : -Number(els.learningRate.value) * g;
}

function renderNetwork() {
  const c = cache || forward();
  const focus = steps[currentStep].focus;

  if ((focus === "backprop" || focus === "update" || animation.phase === "backprop" || animation.phase === "update") && !grads) {
    backward();
  }

  const inputX = 95;
  const hiddenX = 455;
  const outputX = 790;
  const lossX = 1000;

  const inputY = [95, 205, 315, 425];
  const hiddenY = [145, 260, 375];
  const outputY = 260;

  const showGradients = focus === "backprop" || focus === "update" || animation.phase === "backprop" || animation.phase === "update";
  const showDelta = focus === "update" || animation.phase === "update";

  let edges = "";

  for (let i = 0; i < 4; i++) {
    for (let h = 0; h < 3; h++) {
      const [edgeState, tagState] = edgeVisualState("ih");
      const grad = showGradients ? gradientForW1(h, i) : null;
      const delta = showDelta ? deltaForGradient(grad) : null;

      // Stagger labels so crossing edges remain readable.
      const offsets = [-17, 0, 17];
      edges += edgeLine({
        x1: inputX + 38,
        y1: inputY[i],
        x2: hiddenX - 38,
        y2: hiddenY[h],
        weight: `w${h + 1}${i + 1}=${fmt(params.W1[h][i], 2)}`,
        grad: grad === null ? null : signed(grad, 3),
        delta: delta === null ? null : signed(delta, 3),
        edgeState,
        tagState,
        tagT: 0.49,
        tagYOffset: offsets[h]
      });
    }
  }

  for (let h = 0; h < 3; h++) {
    const [edgeState, tagState] = edgeVisualState("ho");
    const grad = showGradients ? gradientForW2(h) : null;
    const delta = showDelta ? deltaForGradient(grad) : null;
    const offsets = [-14, 0, 14];

    edges += edgeLine({
      x1: hiddenX + 38,
      y1: hiddenY[h],
      x2: outputX - 38,
      y2: outputY,
      weight: `v${h + 1}=${fmt(params.W2[h], 2)}`,
      grad: grad === null ? null : signed(grad, 3),
      delta: delta === null ? null : signed(delta, 3),
      edgeState,
      tagState,
      tagT: 0.55,
      tagYOffset: offsets[h]
    });
  }

  const [lossEdgeState] = edgeVisualState("ol");
  const lossState =
    animation.phase === "backprop" || focus === "backprop"
      ? "backprop"
      : (focus === "loss" || animation.phase === "forward")
        ? "active"
        : "";

  const biasBackprop = showGradients;

  els.networkSvg.innerHTML = `
    <text class="layer-label" x="${inputX}" y="32" text-anchor="middle">INPUT FEATURES</text>
    <text class="layer-label" x="${hiddenX}" y="32" text-anchor="middle">HIDDEN + ReLU</text>
    <text class="layer-label" x="${outputX}" y="32" text-anchor="middle">SIGMOID OUTPUT</text>
    <text class="layer-label" x="${lossX}" y="32" text-anchor="middle">BCE LOSS</text>

    ${edges}

    ${nodeGroup("x1", inputX, inputY[0], "x₁", fmt(c.x[0], 2), nodeVisualState("input"))}
    ${nodeGroup("x2", inputX, inputY[1], "x₂", fmt(c.x[1], 2), nodeVisualState("input"))}
    ${nodeGroup("x3", inputX, inputY[2], "x₃", fmt(c.x[2], 2), nodeVisualState("input"))}
    ${nodeGroup("x4", inputX, inputY[3], "x₄", fmt(c.x[3], 2), nodeVisualState("input"))}

    ${nodeGroup("h1", hiddenX, hiddenY[0], "h₁", fmt(c.h[0]), nodeVisualState("hidden"))}
    ${nodeGroup("h2", hiddenX, hiddenY[1], "h₂", fmt(c.h[1]), nodeVisualState("hidden"))}
    ${nodeGroup("h3", hiddenX, hiddenY[2], "h₃", fmt(c.h[2]), nodeVisualState("hidden"))}

    ${nodeGroup("out", outputX, outputY, "ŷ", fmt(c.yHat), nodeVisualState("output"))}

    <g>
      <line class="edge ${lossEdgeState}" x1="${outputX + 38}" y1="${outputY}" x2="${lossX - 65}" y2="${outputY}"></line>
      <rect class="loss-card ${lossState}" x="${lossX - 65}" y="${outputY - 47}" width="130" height="94" rx="15"></rect>
      <text class="loss-title" x="${lossX}" y="${outputY - 8}" text-anchor="middle">L = ${fmt(c.loss)}</text>
      <text class="loss-subtext" x="${lossX}" y="${outputY + 14}" text-anchor="middle">target y = ${c.y}</text>
      <text class="loss-subtext" x="${lossX}" y="${outputY + 31}" text-anchor="middle">${classLabel(c.y)}</text>
    </g>

    <g class="bias-tag ${biasBackprop ? "backprop" : ""}">
      <rect x="${hiddenX - 98}" y="461" width="196" height="${biasBackprop ? 40 : 25}" rx="7"></rect>
      <text x="${hiddenX}" y="477" text-anchor="middle">b = [${params.b1.map(v => fmt(v, 2)).join(", ")}]</text>
      ${biasBackprop ? `<text class="bias-grad" x="${hiddenX}" y="493" text-anchor="middle">g = [${grads.db1.map(v => signed(v, 3)).join(", ")}]</text>` : ""}
    </g>

    <g class="bias-tag ${biasBackprop ? "backprop" : ""}">
      <rect x="${outputX - 75}" y="461" width="150" height="${biasBackprop ? 40 : 25}" rx="7"></rect>
      <text x="${outputX}" y="477" text-anchor="middle">b_out = ${fmt(params.b2, 2)}</text>
      ${biasBackprop ? `<text class="bias-grad" x="${outputX}" y="493" text-anchor="middle">g = ${signed(grads.db2, 3)}</text>` : ""}
    </g>
  `;
}

function renderAll() {
  forward();
  renderTask();
  renderStepText();
  renderStatus();
  renderInspector();
  renderNetwork();
}

function stopLessonAnimation() {
  if (lessonTimer) {
    clearInterval(lessonTimer);
    lessonTimer = null;
  }
  els.autoPlay.textContent = "▶ Play lesson";
}

function stopPhaseAnimation() {
  if (phaseTimer) {
    clearTimeout(phaseTimer);
    phaseTimer = null;
  }
  animation = { phase: "idle", stage: -1 };
  setPhaseIndicator("idle", "Ready");
}

function stopAllAnimations() {
  stopLessonAnimation();
  stopPhaseAnimation();
}

function runStagedAnimation(phase, stages, interval, onFinish = null) {
  stopPhaseAnimation();
  animation = { phase, stage: 0 };
  setPhaseIndicator(
    phase,
    phase === "forward" ? "Forward pass" : phase === "backprop" ? "Backpropagation" : "Updating weights"
  );
  renderInspector();
  renderNetwork();

  let stage = 0;

  const tick = () => {
    stage++;
    animation.stage = stage;
    renderInspector();
    renderNetwork();

    if (stage >= stages - 1) {
      phaseTimer = setTimeout(() => {
        if (onFinish) onFinish();
        animation = { phase: "idle", stage: -1 };
        setPhaseIndicator("idle", "Ready");
        renderAll();
      }, Math.max(300, interval * 0.7));
      return;
    }

    phaseTimer = setTimeout(tick, interval);
  };

  phaseTimer = setTimeout(tick, interval);
}

function runForwardAnimation() {
  stopLessonAnimation();
  grads = null;
  forward();
  currentStep = 4;
  runStagedAnimation("forward", 6, 360);
  renderStepText();
  renderStatus();
}

function runBackpropAnimation() {
  stopLessonAnimation();
  forward();
  backward();
  currentStep = 6;
  runStagedAnimation("backprop", 4, 480);
  renderStepText();
  renderStatus();
}

function runUpdateAnimation() {
  stopLessonAnimation();
  beforeTraining = forward();
  backward();
  lastUpdate = captureUpdateSnapshot();
  currentStep = 7;

  runStagedAnimation("update", 3, 520, () => {
    params = clone(lastUpdate.newParams);
    grads = null;
    afterTraining = forward();
    updateHistory(beforeTraining, afterTraining);
  });

  renderStepText();
  renderStatus();
}

els.sampleSelect.addEventListener("change", (event) => {
  stopAllAnimations();
  currentSample = Number(event.target.value);
  currentStep = 0;
  cache = null;
  grads = null;
  lastUpdate = null;
  beforeTraining = null;
  afterTraining = null;
  updateHistory(null, null);
  renderAll();
});

els.prevStep.addEventListener("click", () => {
  stopAllAnimations();
  currentStep = Math.max(0, currentStep - 1);
  grads = currentStep >= 6 ? backward() : null;
  renderAll();
});

els.nextStep.addEventListener("click", () => {
  stopAllAnimations();
  if (currentStep < steps.length - 1) {
    currentStep++;
    grads = currentStep >= 6 ? backward() : null;
    renderAll();
  }
});

els.autoPlay.addEventListener("click", () => {
  if (lessonTimer) {
    stopLessonAnimation();
    return;
  }

  stopPhaseAnimation();
  els.autoPlay.textContent = "⏸ Pause lesson";

  lessonTimer = setInterval(() => {
    if (currentStep >= steps.length - 1) {
      stopLessonAnimation();
      return;
    }
    currentStep++;
    grads = currentStep >= 6 ? backward() : null;
    renderAll();
  }, 1500);
});

$("#restartLesson").addEventListener("click", () => {
  stopAllAnimations();
  currentStep = 0;
  grads = null;
  renderAll();
});

els.runForward.addEventListener("click", runForwardAnimation);
els.runBackprop.addEventListener("click", runBackpropAnimation);
els.updateWeights.addEventListener("click", runUpdateAnimation);

$("#trainTen").addEventListener("click", () => {
  stopAllAnimations();
  beforeTraining = forward();
  trainOnAllSamples(10);
  afterTraining = forward();
  currentStep = 7;
  grads = null;
  lastUpdate = null;
  updateHistory(beforeTraining, afterTraining);
  setPhaseIndicator("update", "10 epochs complete");
  renderAll();
  setTimeout(() => setPhaseIndicator("idle", "Ready"), 900);
});

$("#resetWeights").addEventListener("click", () => {
  stopAllAnimations();
  params = clone(initialParams);
  cache = null;
  grads = null;
  lastUpdate = null;
  beforeTraining = null;
  afterTraining = null;
  currentStep = 0;
  updateHistory(null, null);
  renderAll();
});

els.learningRate.addEventListener("input", () => {
  els.learningRateOutput.textContent = Number(els.learningRate.value).toFixed(2);
  els.learningRateValue.textContent = Number(els.learningRate.value).toFixed(2);
  renderInspector();
  if (steps[currentStep].focus === "update" || animation.phase === "update") renderNetwork();
});

function init() {
  buildSampleSelect();
  buildDatasetTable();
  buildStepRail();
  els.sampleSelect.value = "0";
  setPhaseIndicator("idle", "Ready");
  renderAll();
  updateHistory(null, null);
}

init();
