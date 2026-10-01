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

export function getSceneContent(sceneId, context) {
  const { snapshot, grads, params, nextParams, learningRate } = context;
  const clampIndex = snapshot.z1.findIndex((v) => v < 0);
  const hi = clampIndex >= 0 ? clampIndex : 0;

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
        `z_1 = \\sum_j w_{1j}x_j + b_1 = ${f(snapshot.z1[0])}`,
        `w_{11}x_1 = ${f(params.W1[0][0],2)} \\times ${f(snapshot.x[0],2)}`
      ],
      plainMath: [
        `z1 = Σ(w1j xj) + b1 = ${f(snapshot.z1[0])}`,
        `w11 x1 = ${f(params.W1[0][0],2)} × ${f(snapshot.x[0],2)}`
      ],
      detail: "Each hidden neuron computes its own weighted sum."
    },
    "scene-relu": {
      latex: [
        `h_{${hi+1}} = \\operatorname{ReLU}(${f(snapshot.z1[hi])}) = ${f(snapshot.h[hi])}`,
        `\\operatorname{ReLU}(z)=\\max(0,z)`
      ],
      plainMath: [
        `h${hi+1} = ReLU(${f(snapshot.z1[hi])}) = ${f(snapshot.h[hi])}`,
        "ReLU(z) = max(0, z)"
      ],
      detail: snapshot.z1[hi] < 0 ? "Negative evidence is visibly clamped to zero." : "Positive evidence passes through."
    },
    "scene-output": {
      latex: [
        `z_{out} = ${f(snapshot.z2)}`,
        `\\hat y = \\sigma(z_{out}) = ${f(snapshot.yHat)}`
      ],
      plainMath: [`z_out = ${f(snapshot.z2)}`, `ŷ = sigmoid(z_out) = ${f(snapshot.yHat)}`],
      detail: `The network currently assigns ${(snapshot.yHat*100).toFixed(1)}% probability to spam.`
    },
    "scene-loss": {
      latex: [
        `\\mathrm{BCE} = -[y\\ln\\hat y +(1-y)\\ln(1-\\hat y)]`,
        `\\mathcal{L} = ${f(snapshot.loss)}`
      ],
      plainMath: ["BCE loss = -[y ln(ŷ) + (1-y) ln(1-ŷ)]", `loss = ${f(snapshot.loss)}`],
      detail: "The target is y = 1 (spam)."
    },
    "scene-backprop": {
      latex: [
        `\\frac{\\partial \\mathcal{L}}{\\partial z_{out}} = \\hat y-y = ${signed(grads.dz2)}`,
        `\\frac{\\partial \\mathcal{L}}{\\partial v_1} = ${signed(grads.dW2[0])}`
      ],
      plainMath: [
        `∂L/∂z_out = ŷ - y = ${signed(grads.dz2)}`,
        `gradient ∂L/∂v1 = ${signed(grads.dW2[0])}`
      ],
      detail: "The direction reverses: gradients travel from the loss toward earlier parameters."
    },
    "scene-update": {
      latex: [
        `v_1' = v_1 - \\eta \\frac{\\partial \\mathcal{L}}{\\partial v_1}`,
        `${f(params.W2[0])} - ${learningRate.toFixed(2)}(${signed(grads.dW2[0])}) = ${f(nextParams.W2[0])}`
      ],
      plainMath: [
        "v1 new = v1 old - eta × gradient",
        `${f(params.W2[0])} - ${learningRate.toFixed(2)} × (${signed(grads.dW2[0])}) = ${f(nextParams.W2[0])}`,
        `Δ = ${signed(nextParams.W2[0]-params.W2[0])}`
      ],
      detail: `η = ${learningRate.toFixed(2)}. The parameter moves opposite its gradient.`
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
