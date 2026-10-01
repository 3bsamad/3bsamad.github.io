export const TEACHING_SAMPLE = Object.freeze({
  email: "URGENT: Verify your bank account immediately using this link.",
  x: Object.freeze([0.85, 0.80, 0.15, 1.00]),
  y: 1
});

export const FEATURES = Object.freeze([
  Object.freeze({ key: "suspicious", label: "Suspicious words" }),
  Object.freeze({ key: "links", label: "Links" }),
  Object.freeze({ key: "trust", label: "Sender trust" }),
  Object.freeze({ key: "urgency", label: "Urgency" })
]);

export const INITIAL_PARAMS = Object.freeze({
  W1: Object.freeze([
    Object.freeze([0.55, -0.25, 0.30, 0.20]),
    Object.freeze([0.15, 0.60, -0.45, 0.35]),
    Object.freeze([-0.35, 0.20, 0.65, -0.10])
  ]),
  b1: Object.freeze([0.05, -0.10, 0.15]),
  W2: Object.freeze([0.60, -0.40, 0.55]),
  b2: -0.05
});

const TRAINING_SAMPLES = Object.freeze([
  TEACHING_SAMPLE,
  Object.freeze({ email: "WIN MONEY NOW!!! Click here to claim your free prize.", x: Object.freeze([0.95, 0.85, 0.10, 0.95]), y: 1 }),
  Object.freeze({ email: "You have been selected for an exclusive cash reward.", x: Object.freeze([0.90, 0.55, 0.20, 0.75]), y: 1 }),
  Object.freeze({ email: "Meeting moved to 14:30 tomorrow. Same room as planned.", x: Object.freeze([0.05, 0.10, 0.95, 0.10]), y: 0 }),
  Object.freeze({ email: "Hi, attached is the invoice from yesterday.", x: Object.freeze([0.05, 0.15, 0.90, 0.10]), y: 0 }),
  Object.freeze({ email: "Can you review the pull request when you have time?", x: Object.freeze([0.00, 0.05, 0.98, 0.05]), y: 0 })
]);

const cloneParams = (params) => ({
  W1: params.W1.map((row) => [...row]),
  b1: [...params.b1],
  W2: [...params.W2],
  b2: params.b2
});

const relu = (value) => Math.max(0, value);
const reluPrime = (value) => value > 0 ? 1 : 0;
const sigmoid = (value) => 1 / (1 + Math.exp(-value));
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);

export function forward(params, sample = TEACHING_SAMPLE) {
  const z1 = params.W1.map((row, i) => dot(row, sample.x) + params.b1[i]);
  const h = z1.map(relu);
  const z2 = dot(params.W2, h) + params.b2;
  const yHat = sigmoid(z2);
  const p = Math.min(1 - 1e-7, Math.max(1e-7, yHat));
  const loss = -(sample.y * Math.log(p) + (1 - sample.y) * Math.log(1 - p));

  return {
    x: [...sample.x],
    y: sample.y,
    z1,
    h,
    z2,
    yHat,
    loss
  };
}

export function backward(params, snapshot) {
  const dz2 = snapshot.yHat - snapshot.y;
  const dW2 = snapshot.h.map((value) => dz2 * value);
  const db2 = dz2;
  const dh = params.W2.map((weight) => dz2 * weight);
  const dz1 = dh.map((value, i) => value * reluPrime(snapshot.z1[i]));
  const dW1 = dz1.map((gradient) => snapshot.x.map((x) => gradient * x));
  const db1 = [...dz1];

  return { dz2, dW2, db2, dh, dz1, dW1, db1 };
}

export function updatedParams(params, grads, learningRate = 0.10) {
  const next = cloneParams(params);

  for (let h = 0; h < 3; h++) {
    for (let i = 0; i < 4; i++) {
      next.W1[h][i] = params.W1[h][i] - learningRate * grads.dW1[h][i];
    }
    next.b1[h] = params.b1[h] - learningRate * grads.db1[h];
    next.W2[h] = params.W2[h] - learningRate * grads.dW2[h];
  }

  next.b2 = params.b2 - learningRate * grads.db2;
  return next;
}

export function trainingPreview(
  params = INITIAL_PARAMS,
  samples = TRAINING_SAMPLES,
  learningRate = 0.10,
  epochs = 12
) {
  let current = cloneParams(params);
  const history = [];

  for (let epoch = 0; epoch <= epochs; epoch++) {
    const teaching = forward(current, TEACHING_SAMPLE);
    history.push({
      epoch,
      loss: teaching.loss,
      yHat: teaching.yHat
    });

    if (epoch === epochs) break;

    for (const sample of samples) {
      const snapshot = forward(current, sample);
      const grads = backward(current, snapshot);
      current = updatedParams(current, grads, learningRate);
    }
  }

  return { params: current, history };
}
