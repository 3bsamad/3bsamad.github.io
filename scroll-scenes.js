import { SCENES } from "./scene-content.js";

export function createScrollStory({
  renderer,
  reducedMotion = false,
  onSceneChange = () => {},
  gsap = globalThis.gsap,
  ScrollTrigger = globalThis.ScrollTrigger
}) {
  const sections = SCENES.map((scene) => document.getElementById(scene.id)).filter(Boolean);
  let activeScene = "scene-intro";
  const disposables = [];

  function activate(sceneId, { immediate = reducedMotion } = {}) {
    activeScene = sceneId;
    renderer.applyScene(sceneId, { immediate });

    document.querySelectorAll(".story-scene").forEach((section) => {
      section.classList.toggle("is-active", section.id === sceneId);
    });

    document.querySelectorAll("#sceneNav [data-scene]").forEach((button) => {
      if (button.dataset.scene === sceneId) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    });

    onSceneChange(sceneId);
  }

  if (gsap && ScrollTrigger && !reducedMotion) {
    gsap.registerPlugin(ScrollTrigger);
    sections.forEach((section, index) => {
      const previous = sections[Math.max(0, index - 1)];
      disposables.push(ScrollTrigger.create({
        trigger: section,
        start: "top 58%",
        end: "bottom 42%",
        onEnter: () => activate(section.id),
        onLeaveBack: () => activate(previous.id, { immediate: true })
      }));
    });
  } else if ("IntersectionObserver" in globalThis) {
    const observer = new IntersectionObserver((entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (visible) activate(visible.target.id, { immediate:true });
    }, { threshold:[0.25,0.5,0.7] });
    sections.forEach((section) => observer.observe(section));
    disposables.push({ kill:() => observer.disconnect() });
  }

  function scrollToScene(sceneId) {
    const target = document.getElementById(sceneId);
    if (!target) return;
    target.scrollIntoView({ behavior:reducedMotion ? "auto" : "smooth", block:"center" });
    if (reducedMotion) activate(sceneId, { immediate:true });
  }

  document.querySelectorAll("#sceneNav [data-scene]").forEach((button) => {
    button.addEventListener("click", () => scrollToScene(button.dataset.scene));
  });

  activate(activeScene, { immediate:true });

  return {
    activate,
    scrollToScene,
    destroy() { disposables.forEach((item) => item.kill?.()); },
    get activeScene() { return activeScene; }
  };
}
