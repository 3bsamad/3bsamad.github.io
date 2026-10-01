import {
  TEACHING_SAMPLE,
  FEATURES,
  INITIAL_PARAMS,
  forward,
  backward,
  updatedParams,
  trainingPreview
} from "./model.js";
import { createNetworkRenderer } from "./network-renderer.js";
import { SCENES, getSceneContent, renderMath } from "./scene-content.js";
import { createScrollStory } from "./scroll-scenes.js";

document.documentElement.classList.add("js-ready");

const learningRate = 0.10;
const snapshot = forward(INITIAL_PARAMS, TEACHING_SAMPLE);
const grads = backward(INITIAL_PARAMS, snapshot);
const nextParams = updatedParams(INITIAL_PARAMS, grads, learningRate);
const afterUpdate = forward(nextParams, TEACHING_SAMPLE);
const training = trainingPreview(INITIAL_PARAMS, undefined, learningRate, 12);

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const compactQuery = window.matchMedia("(max-width: 820px)");

const els = {
  svg: document.querySelector("#networkSvg"),
  emailText: document.querySelector("#emailText"),
  featureStrip: document.querySelector("#featureStrip"),
  mathExpression: document.querySelector("#mathExpression"),
  mathDetail: document.querySelector("#mathDetail"),
  prediction: document.querySelector("#predictionMetric"),
  loss: document.querySelector("#lossMetric"),
  status: document.querySelector("#stageStatus"),
  finale: document.querySelector("#trainingFinale"),
  summary: document.querySelector("#trainingSummary"),
  lossChart: document.querySelector("#lossChart"),
  progress: document.querySelector("#scrollProgressBar")
};

const sceneStatus = {
  "scene-intro": "Dormant network",
  "scene-inputs": "Encoding input features",
  "scene-weights": "Signals moving forward",
  "scene-relu": "Hidden neurons activating",
  "scene-output": "Forming spam probability",
  "scene-loss": "Measuring prediction error",
  "scene-backprop": "Gradients moving backward",
  "scene-update": "Updating parameters",
  "scene-training": "Repeating the learning loop"
};

function renderEmail() {
  const escaped = TEACHING_SAMPLE.email
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");

  els.emailText.innerHTML = escaped
    .replace(/URGENT/gi, '<span class="signal-word">$&</span>')
    .replace(/bank account/gi, '<span class="signal-word">$&</span>')
    .replace(/immediately/gi, '<span class="signal-word">$&</span>')
    .replace(/link/gi, '<span class="signal-word">$&</span>');

  els.featureStrip.innerHTML = FEATURES.map((feature, i) => `
    <div class="feature-token">
      <span>${feature.label}</span>
      <strong>x${i + 1} · ${TEACHING_SAMPLE.x[i].toFixed(2)}</strong>
    </div>
  `).join("");
}

function renderLossChart() {
  const points = training.history;
  const width = 320;
  const height = 86;
  const padX = 12;
  const padY = 12;
  const minLoss = Math.min(...points.map((p) => p.loss));
  const maxLoss = Math.max(...points.map((p) => p.loss));
  const span = Math.max(1e-6, maxLoss - minLoss);

  const coords = points.map((point, i) => {
    const x = padX + (i / (points.length - 1)) * (width - padX * 2);
    const y = padY + ((maxLoss - point.loss) / span) * (height - padY * 2);
    return { x, y };
  });

  const d = coords.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(" ");
  const first = coords[0];
  const last = coords.at(-1);

  els.lossChart.innerHTML = `
    <title>Loss over 12 short training epochs</title>
    <line class="loss-chart-axis" x1="12" y1="74" x2="308" y2="74"></line>
    <path class="loss-chart-line" d="${d}"></path>
    <circle class="loss-chart-dot" cx="${first.x}" cy="${first.y}" r="3"></circle>
    <circle class="loss-chart-dot" cx="${last.x}" cy="${last.y}" r="4"></circle>
    <text class="loss-chart-label" x="12" y="84">0</text>
    <text class="loss-chart-label" x="308" y="84" text-anchor="end">12 epochs</text>
  `;

  els.summary.textContent =
    `Loss ${points[0].loss.toFixed(3)} → ${points.at(-1).loss.toFixed(3)} · spam probability ${(points[0].yHat*100).toFixed(1)}% → ${(points.at(-1).yHat*100).toFixed(1)}%`;
}

renderEmail();
renderLossChart();

const renderer = createNetworkRenderer(els.svg, {
  reducedMotion,
  gsap: window.gsap
});

renderer.setSnapshot(snapshot, grads, INITIAL_PARAMS);
renderer.setLayout(compactQuery.matches ? "compact" : "desktop");

compactQuery.addEventListener("change", (event) => {
  renderer.setLayout(event.matches ? "compact" : "desktop");
});

function sceneContext(sceneId) {
  return {
    snapshot,
    grads,
    params: INITIAL_PARAMS,
    nextParams,
    learningRate,
    afterUpdate,
    training,
    sceneId
  };
}

function renderSceneUI(sceneId) {
  document.body.dataset.scene = sceneId;
  els.status.textContent = sceneStatus[sceneId] || "Neural network";

  const content = getSceneContent(sceneId, sceneContext(sceneId));
  const expressions = content.latex.map((latex, i) => ({
    latex,
    plain: content.plainMath[i] || content.plainMath[0]
  }));

  renderMath(els.mathExpression, expressions, { katex: window.katex });
  els.mathDetail.textContent = content.detail;

  const showPrediction = [
    "scene-output","scene-loss","scene-backprop","scene-update","scene-training"
  ].includes(sceneId);
  const showLoss = [
    "scene-loss","scene-backprop","scene-update","scene-training"
  ].includes(sceneId);

  els.prediction.textContent = showPrediction
    ? `${(snapshot.yHat * 100).toFixed(1)}% spam`
    : "—";
  els.loss.textContent = showLoss ? snapshot.loss.toFixed(3) : "—";

  if (sceneId === "scene-update") {
    els.prediction.textContent =
      `${(snapshot.yHat*100).toFixed(1)}% → ${(afterUpdate.yHat*100).toFixed(1)}%`;
    els.loss.textContent =
      `${snapshot.loss.toFixed(3)} → ${afterUpdate.loss.toFixed(3)}`;
  }

  els.finale.hidden = sceneId !== "scene-training";

  const badge = document.querySelector(".spam-badge");
  if (badge) {
    if (showPrediction) {
      badge.textContent = "spam · target 1";
      badge.classList.add("is-spam");
    } else {
      badge.textContent = "unknown";
      badge.classList.remove("is-spam");
    }
  }
}

const story = createScrollStory({
  renderer,
  reducedMotion,
  gsap: window.gsap,
  ScrollTrigger: window.ScrollTrigger,
  onSceneChange: renderSceneUI
});

renderSceneUI("scene-intro");

if (window.ScrollTrigger && window.gsap && !reducedMotion) {
  window.gsap.set(els.progress, { scaleX: 0, transformOrigin: "left center" });
  window.ScrollTrigger.create({
    start: 0,
    end: "max",
    onUpdate: (self) => window.gsap.set(els.progress, { scaleX: self.progress })
  });
}

document.querySelector(".restart-link")?.addEventListener("click", (event) => {
  event.preventDefault();
  story.scrollToScene("scene-intro");
});
