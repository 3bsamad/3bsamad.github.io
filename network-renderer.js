const SVG_NS = "http://www.w3.org/2000/svg";

export function getNetworkGeometry(mode = "desktop") {
  if (mode === "compact") {
    return {
      width: 540, height: 700,
      inputs: [{x:70,y:110},{x:205,y:110},{x:335,y:110},{x:470,y:110}],
      hidden: [{x:135,y:315},{x:270,y:315},{x:405,y:315}],
      output: {x:270,y:505},
      loss: {x:270,y:635}
    };
  }
  return {
    width: 1080, height: 620,
    inputs: [{x:95,y:145},{x:95,y:255},{x:95,y:365},{x:95,y:475}],
    hidden: [{x:450,y:195},{x:450,y:310},{x:450,y:425}],
    output: {x:760,y:310},
    loss: {x:985,y:310}
  };
}

export function getSceneVisualState(sceneId) {
  const states = {
    "scene-intro": { focus: "dormant", direction: "none" },
    "scene-inputs": { focus: "inputs", direction: "forward" },
    "scene-weights": { focus: "input-hidden", direction: "forward" },
    "scene-relu": { focus: "hidden", direction: "forward" },
    "scene-output": { focus: "output", direction: "forward" },
    "scene-loss": { focus: "loss", direction: "forward" },
    "scene-backprop": { focus: "all", direction: "backward" },
    "scene-update": { focus: "all", direction: "update" },
    "scene-training": { focus: "all", direction: "cycle" }
  };
  return states[sceneId] || { focus: "dormant", direction: "none" };
}

const create = (tag, attrs = {}) => {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
  return el;
};

const textNode = (x, y, cls, value, anchor = "middle") => {
  const t = create("text", { x, y, class: cls, "text-anchor": anchor });
  t.textContent = value;
  return t;
};

function curve(a, b, mode) {
  if (mode === "compact") {
    const midY = (a.y + b.y) / 2;
    return \`M ${a.x} ${a.y} C ${a.x} ${midY}, ${b.x} ${midY}, ${b.x} ${b.y}\`;
  }
  const midX = (a.x + b.x) / 2;
  return \`M ${a.x} ${a.y} C ${midX} ${a.y}, ${midX} ${b.y}, ${b.x} ${b.y}\`;
}

function makeNeuron(id, label) {
  const g = create("g", { class: "neuron", "data-neuron": id });
  const halo = create("circle", { r: 49, class: "neuron-halo" });
  const shell = create("circle", { r: 36, class: "neuron-shell" });
  const ring = create("circle", { r: 29, class: "neuron-ring" });
  const core = create("circle", { r: 20, class: "neuron-core" });
  const pulse = create("circle", { r: 7, class: "neuron-pulse" });
  const labelEl = textNode(0, -5, "neuron-label", label);
  const valueEl = textNode(0, 16, "neuron-value", "—");
  const preEl = textNode(0, 59, "neuron-pre", "");
  g.append(halo, shell, ring, core, pulse, labelEl, valueEl, preEl);
  return { g, valueEl, preEl };
}

function makeLoss() {
  const g = create("g", { class: "loss-node", "data-neuron": "loss" });
  const halo = create("rect", { x:-78, y:-48, width:156, height:96, rx:18, class:"loss-halo" });
  const body = create("rect", { x:-68, y:-40, width:136, height:80, rx:14, class:"loss-body" });
  const title = textNode(0, -8, "loss-node-title", "BCE loss");
  const value = textNode(0, 18, "loss-node-value", "—");
  g.append(halo, body, title, value);
  return { g, value };
}

function key(group, a, b) {
  return \`${group}-${a}-${b}\`;
}

export function createNetworkRenderer(svg, {
  reducedMotion = false,
  gsap = globalThis.gsap
} = {}) {
  let layout = "desktop";
  let geometry = getNetworkGeometry(layout);
  let snapshot = null;
  let grads = null;
  let params = null;
  let currentScene = "scene-intro";

  const defs = create("defs");
  defs.innerHTML = \`
    <filter id="softGlow" x="-80%" y="-80%" width="260%" height="260%">
      <feGaussianBlur stdDeviation="4.5" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  \`;
  svg.appendChild(defs);

  const guideLayer = create("g", { class: "network-guides" });
  const connectionLayer = create("g", { class: "connection-layer" });
  const signalLayer = create("g", { class: "signal-layer" });
  const neuronLayer = create("g", { class: "neuron-layer" });
  svg.append(guideLayer, connectionLayer, signalLayer, neuronLayer);

  const neurons = {
    inputs: ["x1","x2","x3","x4"].map((id, i) => makeNeuron(id, \`x${i + 1}\`)),
    hidden: ["h1","h2","h3"].map((id, i) => makeNeuron(id, \`h${i + 1}\`)),
    output: makeNeuron("output", "ŷ"),
    loss: makeLoss()
  };

  [...neurons.inputs, ...neurons.hidden, neurons.output].forEach((n) => neuronLayer.appendChild(n.g));
  neuronLayer.appendChild(neurons.loss.g);

  const connections = new Map();

  function buildConnections() {
    connectionLayer.replaceChildren();
    connections.clear();

    for (let i = 0; i < 4; i++) {
      for (let h = 0; h < 3; h++) {
        const id = key("ih", i, h);
        const group = create("g", { class:"connection", "data-connection":id, "data-group":"ih" });
        const base = create("path", { class:"connection-base" });
        const active = create("path", { class:"connection-active" });
        const label = create("g", { class:"connection-label" });
        const box = create("rect", { x:-32, y:-11, width:64, height:22, rx:6 });
        const text = textNode(0, 3, "connection-label-text", "");
        label.append(box, text);
        group.append(base, active, label);
        connectionLayer.appendChild(group);
        connections.set(id, { group, base, active, label, text, from:i, to:h, kind:"ih" });
      }
    }

    for (let h = 0; h < 3; h++) {
      const id = key("ho", h, 0);
      const group = create("g", { class:"connection", "data-connection":id, "data-group":"ho" });
      const base = create("path", { class:"connection-base" });
      const active = create("path", { class:"connection-active" });
      const label = create("g", { class:"connection-label" });
      const box = create("rect", { x:-30, y:-11, width:60, height:22, rx:6 });
      const text = textNode(0, 3, "connection-label-text", "");
      label.append(box, text);
      group.append(base, active, label);
      connectionLayer.appendChild(group);
      connections.set(id, { group, base, active, label, text, from:h, to:0, kind:"ho" });
    }

    const id = "ol-0-0";
    const group = create("g", { class:"connection", "data-connection":id, "data-group":"ol" });
    const base = create("path", { class:"connection-base" });
    const active = create("path", { class:"connection-active" });
    group.append(base, active);
    connectionLayer.appendChild(group);
    connections.set(id, { group, base, active, from:0, to:0, kind:"ol" });
  }

  function connectionPoints(conn) {
    if (conn.kind === "ih") return [geometry.inputs[conn.from], geometry.hidden[conn.to]];
    if (conn.kind === "ho") return [geometry.hidden[conn.from], geometry.output];
    return [geometry.output, geometry.loss];
  }

  function positionConnection(conn) {
    const [a, b] = connectionPoints(conn);
    const d = curve(a, b, layout);
    conn.base.setAttribute("d", d);
    conn.active.setAttribute("d", d);
    if (conn.label) {
      const t = conn.kind === "ih" ? 0.50 : 0.56;
      conn.label.setAttribute("transform", \`translate(${a.x + (b.x-a.x)*t} ${a.y + (b.y-a.y)*t})\`);
    }
  }

  function positionNeuron(neuron, point) {
    neuron.g.setAttribute("transform", \`translate(${point.x} ${point.y})\`);
  }

  function drawGuides() {
    guideLayer.replaceChildren();
    const labels = layout === "compact"
      ? [["INPUTS",24,44],["HIDDEN + ReLU",24,244],["OUTPUT",24,445],["LOSS",24,585]]
      : [["INPUTS",95,62],["HIDDEN + ReLU",450,62],["OUTPUT",760,62],["LOSS",985,62]];

    labels.forEach(([label, x, y]) => {
      guideLayer.appendChild(textNode(x, y, "layer-title", label, layout === "compact" ? "start" : "middle"));
    });
  }

  function positionAll() {
    svg.setAttribute("viewBox", \`0 0 ${geometry.width} ${geometry.height}\`);
    neurons.inputs.forEach((n, i) => positionNeuron(n, geometry.inputs[i]));
    neurons.hidden.forEach((n, i) => positionNeuron(n, geometry.hidden[i]));
    positionNeuron(neurons.output, geometry.output);
    positionNeuron(neurons.loss, geometry.loss);
    connections.forEach(positionConnection);
    drawGuides();
  }

  function setLayout(next) {
    if (next === layout) return;
    layout = next;
    geometry = getNetworkGeometry(layout);
    buildConnections();
    positionAll();
    if (snapshot) setSnapshot(snapshot, grads, params);
    applyScene(currentScene, { immediate:true });
  }

  function setSnapshot(nextSnapshot, nextGrads, nextParams) {
    snapshot = nextSnapshot;
    grads = nextGrads;
    params = nextParams;

    neurons.inputs.forEach((n, i) => {
      n.valueEl.textContent = snapshot.x[i].toFixed(2);
      n.preEl.textContent = "";
    });
    neurons.hidden.forEach((n, i) => {
      n.valueEl.textContent = snapshot.h[i].toFixed(3);
      n.preEl.textContent = \`z=${snapshot.z1[i].toFixed(3)}\`;
      n.g.dataset.zeroed = snapshot.h[i] === 0 ? "true" : "false";
    });
    neurons.output.valueEl.textContent = \`${(snapshot.yHat*100).toFixed(1)}%\`;
    neurons.output.preEl.textContent = \`z=${snapshot.z2.toFixed(3)}\`;
    neurons.loss.value.textContent = snapshot.loss.toFixed(3);

    connections.forEach((conn) => {
      if (conn.kind === "ih") {
        const weight = params.W1[conn.to][conn.from];
        conn.base.style.setProperty("--weight-width", (1.1 + Math.abs(weight)*2.2).toFixed(2));
        conn.text.textContent = \`w=${weight.toFixed(2)}\`;
      } else if (conn.kind === "ho") {
        const weight = params.W2[conn.from];
        conn.base.style.setProperty("--weight-width", (1.1 + Math.abs(weight)*2.2).toFixed(2));
        conn.text.textContent = \`v=${weight.toFixed(2)}\`;
      }
    });
  }

  function clearStates() {
    [...neurons.inputs,...neurons.hidden,neurons.output].forEach((n) => {
      n.g.classList.remove("is-active","is-receiving","is-gradient","is-updated","is-zeroed");
    });
    neurons.loss.g.classList.remove("is-active","is-gradient","is-updated");
    connections.forEach((conn) => {
      conn.group.classList.remove("is-forward","is-gradient","is-updated","is-emphasized");
      conn.label?.classList.remove("is-visible");
      if (params && conn.text) {
        if (conn.kind === "ih") conn.text.textContent = `w=${params.W1[conn.to][conn.from].toFixed(2)}`;
        if (conn.kind === "ho") conn.text.textContent = `v=${params.W2[conn.from].toFixed(2)}`;
      }
    });
    signalLayer.replaceChildren();
  }

  function applyScene(sceneId, { immediate=false } = {}) {
    currentScene = sceneId;
    clearStates();

    if (sceneId === "scene-intro") return;

    if (sceneId === "scene-inputs") {
      neurons.inputs.forEach((n) => n.g.classList.add("is-active"));
      return;
    }

    if (sceneId === "scene-weights") {
      neurons.inputs.forEach((n) => n.g.classList.add("is-active"));
      neurons.hidden.forEach((n) => n.g.classList.add("is-receiving"));
      connections.forEach((conn) => {
        if (conn.kind === "ih") {
          conn.group.classList.add("is-forward","is-emphasized");
          if (layout === "desktop") conn.label?.classList.add("is-visible");
        }
      });
      if (!immediate) pulseForward("ih");
      return;
    }

    if (sceneId === "scene-relu") {
      neurons.hidden.forEach((n) => n.g.classList.add(n.g.dataset.zeroed === "true" ? "is-zeroed" : "is-active"));
      connections.forEach((conn) => { if (conn.kind === "ih") conn.group.classList.add("is-forward"); });
      return;
    }

    if (sceneId === "scene-output") {
      neurons.hidden.forEach((n) => n.g.classList.add(n.g.dataset.zeroed === "true" ? "is-zeroed" : "is-active"));
      neurons.output.g.classList.add("is-receiving");
      connections.forEach((conn) => {
        if (conn.kind === "ho") {
          conn.group.classList.add("is-forward","is-emphasized");
          if (layout === "desktop") conn.label?.classList.add("is-visible");
        }
      });
      if (!immediate) pulseForward("ho");
      return;
    }

    if (sceneId === "scene-loss") {
      neurons.output.g.classList.add("is-active");
      neurons.loss.g.classList.add("is-active");
      connections.get("ol-0-0")?.group.classList.add("is-forward","is-emphasized");
      if (!immediate) pulseForward("ol");
      return;
    }

    if (sceneId === "scene-backprop") {
      neurons.loss.g.classList.add("is-gradient");
      [...neurons.inputs,...neurons.hidden,neurons.output].forEach((n) => n.g.classList.add("is-gradient"));
      connections.forEach((conn) => {
        conn.group.classList.add("is-gradient","is-emphasized");
        if (layout === "desktop" && conn.label && grads) {
          if (conn.kind === "ih") conn.text.textContent = \`g=${grads.dW1[conn.to][conn.from].toFixed(3)}\`;
          if (conn.kind === "ho") conn.text.textContent = \`g=${grads.dW2[conn.from].toFixed(3)}\`;
          conn.label.classList.add("is-visible");
        }
      });
      if (!immediate) pulseBackward("all");
      return;
    }

    if (sceneId === "scene-update") {
      [...neurons.inputs,...neurons.hidden,neurons.output].forEach((n) => n.g.classList.add("is-updated"));
      neurons.loss.g.classList.add("is-updated");
      connections.forEach((conn) => conn.group.classList.add("is-updated"));
      if (!immediate) flashUpdate();
      return;
    }

    if (sceneId === "scene-training") {
      [...neurons.inputs,...neurons.hidden,neurons.output].forEach((n) => n.g.classList.add("is-active"));
      connections.forEach((conn) => conn.group.classList.add("is-forward"));
    }
  }

  function pulsePath(path, colorClass, reverse=false, delay=0) {
    if (reducedMotion) return;
    const pulse = create("circle", { r:5, class:\`signal-pulse ${colorClass}\` });
    signalLayer.appendChild(pulse);
    const length = path.getTotalLength?.() || 1;
    const update = (progress) => {
      const point = path.getPointAtLength(reverse ? length*(1-progress) : length*progress);
      pulse.setAttribute("cx", point.x);
      pulse.setAttribute("cy", point.y);
    };

    if (gsap) {
      const tracker = { p:0 };
      gsap.to(tracker, {
        p:1, delay, duration:1.05, ease:"power1.inOut",
        onUpdate:() => update(tracker.p),
        onComplete:() => pulse.remove()
      });
    } else {
      update(1);
      setTimeout(() => pulse.remove(), 220);
    }
  }

  function pulseForward(group="all") {
    let i = 0;
    connections.forEach((conn) => {
      if (group === "all" || conn.kind === group) pulsePath(conn.active, "is-forward", false, Math.min(i++*0.035,0.28));
    });
  }

  function pulseBackward(group="all") {
    let i = 0;
    [...connections.values()].reverse().forEach((conn) => {
      if (group === "all" || conn.kind === group) pulsePath(conn.active, "is-gradient", true, Math.min(i++*0.03,0.28));
    });
  }

  function flashUpdate() {
    if (reducedMotion || !gsap) return;
    gsap.fromTo(
      [...connections.values()].map((c) => c.active),
      { opacity:0.25 },
      { opacity:1, duration:0.35, yoyo:true, repeat:1, stagger:0.015 }
    );
  }

  buildConnections();
  positionAll();

  return {
    setLayout, setSnapshot, applyScene, pulseForward, pulseBackward, flashUpdate,
    get layout() { return layout; }
  };
}
