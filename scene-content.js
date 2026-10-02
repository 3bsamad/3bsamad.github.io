export const SCENES = Object.freeze([
  { id:"scene-intro", title:"Watch a neural network think.", intuition:"One sample becomes one complete learning story.", rendererState:"dormant" },
  { id:"scene-inputs", title:"The words become signals.", intuition:"The email is represented by four normalized features.", rendererState:"inputs" },
  { id:"scene-weights", title:"Every connection has an opinion.", intuition:"Weights control how strongly each feature influences each hidden neuron.", rendererState:"input-hidden" },
  { id:"scene-relu", title:"Neurons decide what gets through.", intuition:"ReLU preserves positive evidence and clamps negative evidence to zero.", rendererState:"hidden" },
  { id:"scene-output", title:"Three hidden signals become one belief.", intuition:"A logit becomes a probability through sigmoid.", rendererState:"output" },
  { id:"scene-loss", title:"Now the network learns how wrong it was.", intuition:"Binary cross-entropy turns disagreement into a scalar loss.", rendererState:"loss" },
  { id:"scene-backprop", title:"The error flows backward.", intuition:"Gradients measure how sensitive the loss is to each parameter.", rendererState:"backprop" },
  { id:"scene-update", title:"Knowing the gradient changes the network.", intuition:"Gradient descent moves parameters opposite their gradients.", rendererState:"update" },
  { id:"scene-training", title:"Then it happens again. And again.", intuition:"Repeated forward, backward and update cycles turn mechanics into learning.", rendererState:"training" }
]);

const f = (value, digits=3) => Number(value).toFixed(digits);
const signed = (value, digits=3) => `${value >= 0 ? "+" : ""}${Number(value).toFixed(digits)}`;
const paren = (value, digits=2) => value < 0 ? `(${f(value, digits)})` : f(value, digits);
const product = (a, b, digits=3) => f(a * b, digits);
const sumParts = (values, digits=3) => values.map((value, i) => {
  const magnitude = f(Math.abs(value), digits);
  if (i === 0) return value < 0 ? `-${magnitude}` : magnitude;
  return `${value < 0 ? "-" : "+"} ${magnitude}`;
}).join(" ");

export function getSceneContent(sceneId, context) {
  const { snapshot, grads, params, nextParams, learningRate } = context;
  const clampIndex = snapshot.z1.findIndex((v) => v < 0);
  const hi = clampIndex >= 0 ? clampIndex : 0;

  const z1Terms = params.W1[0].map((weight, i) => ({
    weight,
    x: snapshot.x[i],
    contribution: weight * snapshot.x[i]
  }));
  const z1Numeric = z1Terms
    .map(({ weight, x }) => `${paren(weight)} × ${f(x, 2)}`)
    .join(" + ");
  const z1ContributionSum = sumParts(
    [...z1Terms.map(({ contribution }) => contribution), params.b1[0]]
  );
  const outputTerms = params.W2.map((weight, i) => ({
    weight,
    h: snapshot.h[i],
    contribution: weight * snapshot.h[i]
  }));
  const outputNumeric = outputTerms
    .map(({ weight, h }) => `${paren(weight)} × ${f(h)}`)
    .join(" + ");

  const content = {
    "scene-intro": {
      latex: [`x \\rightarrow h \\rightarrow \\hat y \\rightarrow \\mathcal{L}`],
      plainMath: ["x → h → ŷ → loss"],
      detail: "Scroll to follow a single email through the complete learning cycle."
    },
    "scene-inputs": {
      latex: [`x = [${snapshot.x.map(v => v.toFixed(2)).join(",\\; ")}]`],
      plainMath: [`x = [${snapshot.x.map(v => v.toFixed(2)).join(", ")}]`],
      detail: "Suspicious words · links · sender trust · urgency"
    },
    "scene-weights": {
      latex: [
        `z_1 = (0.55)(0.85) + (-0.25)(0.80) + (0.30)(0.15) + (0.20)(1.00) + 0.05`,
        `z_1 = 0.468 - 0.200 + 0.045 + 0.200 + 0.050 = ${f(snapshot.z1[0])}`
      ],
      plainMath: [
        `z1 = ${z1Numeric} + ${f(params.b1[0], 2)}`,
        `z1 = ${z1ContributionSum} = ${f(snapshot.z1[0])}`
      ],
      detail: `All hidden sums: z = [${snapshot.z1.map(v => f(v)).join(", ")}]`
    },
    "scene-relu": {
      latex: [
        `h_{${hi+1}} = \\max(0,${f(snapshot.z1[hi])}) = ${f(snapshot.h[hi])}`,
        `h = [${snapshot.h.map(v => f(v)).join(",\\; ")}]`
      ],
      plainMath: [
        `h${hi+1} = max(0, ${f(snapshot.z1[hi])}) = ${f(snapshot.h[hi])}`,
        `h = [${snapshot.h.map(v => f(v)).join(", ")}]`
      ],
      detail: "h3 was computed normally, then ReLU clamped its negative pre-activation to zero."
    },
    "scene-output": {
      latex: [
        `z_{out} = (0.60)(${f(snapshot.h[0])}) + (-0.40)(${f(snapshot.h[1])}) + (0.55)(${f(snapshot.h[2])}) - 0.050 = ${f(snapshot.z2)}`,
        `\\hat y = \\frac{1}{1+e^{${f(-snapshot.z2)}}} = ${f(snapshot.yHat)}`
      ],
      plainMath: [
        `z_out = ${outputNumeric} ${params.b2 < 0 ? "-" : "+"} ${f(Math.abs(params.b2))} = ${f(snapshot.z2)}`,
        `ŷ = 1 / (1 + e^${f(-snapshot.z2)}) = ${f(snapshot.yHat)}`
      ],
      detail: `The network assigns ${(snapshot.yHat*100).toFixed(1)}% probability to spam.`
    },
    "scene-loss": {
      latex: [
        `\\mathcal{L} = -[1\\ln(${f(snapshot.yHat)}) + 0\\ln(1-${f(snapshot.yHat)})]`,
        `\\mathcal{L} = -\\ln(${f(snapshot.yHat)}) = ${f(snapshot.loss)}`
      ],
      plainMath: [
        `BCE = -[1·ln(${f(snapshot.yHat)}) + 0·ln(1-${f(snapshot.yHat)})]`,
        `loss = -ln(${f(snapshot.yHat)}) = ${f(snapshot.loss)}`
      ],
      detail: "Because this email is spam, y = 1 and BCE simplifies to -ln(ŷ)."
    },
    "scene-backprop": {
      latex: [
        `\\frac{\\partial \\mathcal{L}}{\\partial z_{out}} = ${f(snapshot.yHat)} - 1 = ${signed(grads.dz2)}`,
        `\\frac{\\partial \\mathcal{L}}{\\partial v_1} = (${signed(grads.dz2)})(${f(snapshot.h[0])}) = ${signed(grads.dW2[0])}`
      ],
      plainMath: [
        `∂L/∂z_out = ${f(snapshot.yHat)} - 1 = ${signed(grads.dz2)}`,
        `∂L/∂v1 = ${signed(grads.dz2)} × ${f(snapshot.h[0])} = ${signed(grads.dW2[0])}`
      ],
      detail: `For h3, ReLU'(-0.090)=0, so its incoming weight gradients are zero in this step.`
    },
    "scene-update": {
      latex: [
        `v_1' = 0.600 - 0.10(${signed(grads.dW2[0])}) = ${f(nextParams.W2[0])}`,
        `\\Delta v_1 = ${signed(nextParams.W2[0]-params.W2[0])}`
      ],
      plainMath: [
        `v1 new = ${f(params.W2[0])} - ${learningRate.toFixed(2)} × (${signed(grads.dW2[0])}) = ${f(nextParams.W2[0])}`,
        `Δv1 = ${signed(nextParams.W2[0]-params.W2[0])}`
      ],
      detail: `The negative gradient makes v1 increase by ${f(nextParams.W2[0]-params.W2[0])}.`
    },
    "scene-training": {
      latex: [`\\text{forward} \\rightarrow \\mathcal{L} \\rightarrow \\text{backprop} \\rightarrow \\text{update} \\rightarrow \\cdots`],
      plainMath: ["forward → loss → backprop → update → repeat"],
      detail: "Repeated cycles gradually reduce error over the training examples."
    }
  };
  return content[sceneId] || content["scene-intro"];
}

export function renderMath(panel, expressions, { katex=globalThis.katex } = {}) {
  panel.replaceChildren();
  expressions.forEach((expr) => {
    const row = document.createElement("div");
    row.className = "math-line";
    row.textContent = expr.plain;
    if (katex) {
      try { katex.render(expr.latex, row, { throwOnError:false, displayMode:false }); }
      catch { row.textContent = expr.plain; }
    }
    panel.appendChild(row);
  });
}
