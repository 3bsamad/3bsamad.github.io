import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
const css = readFileSync(new URL("../styles.css", import.meta.url), "utf8");

test("page defines the ordered scrollytelling scenes", () => {
  const ids = [
    "scene-intro","scene-inputs","scene-weights","scene-relu",
    "scene-output","scene-loss","scene-backprop","scene-update","scene-training"
  ];
  let last = -1;
  for (const id of ids) {
    const matches = [...html.matchAll(new RegExp(`id=["']${id}["']`, "g"))];
    assert.equal(matches.length, 1, `${id} should exist exactly once`);
    const index = html.indexOf(`id="${id}"`);
    assert.ok(index > last, `${id} should follow the previous scene`);
    last = index;
  }
});

test("page exposes the V2 visualization contract", () => {
  for (const id of ["network-stage","networkSvg","sceneNav","mathPanel","emailCard","lossChart"]) {
    assert.match(html, new RegExp(`id=["']${id}["']`));
  }
  assert.doesNotMatch(html, /id=["']sampleSelect["']/);
  assert.match(html, /<script[^>]+type=["']module["'][^>]+src=["']app\.js["']/);
  assert.match(html, /gsap/i);
  assert.match(html, /ScrollTrigger/i);
  assert.match(html, /katex/i);
});

test("CSS provides a sticky network whose containing column spans the full story", () => {
  assert.match(css, /\.visual-column\s*\{[^}]*align-self:\s*stretch/s);
  assert.match(css, /position:\s*sticky/);
  assert.match(css, /@media \(max-width:\s*820px\)[\s\S]*?\.story-shell\s*\{[^}]*display:\s*grid[^}]*grid-template-columns:\s*1fr/s);
  assert.match(css, /@media \(max-width:\s*820px\)[\s\S]*?\.visual-column\s*\{[^}]*grid-row:\s*1/s);
  assert.match(css, /@media \(max-width:\s*820px\)[\s\S]*?\.narrative-column\s*\{[^}]*grid-row:\s*1[^}]*padding-top:/s);
  assert.match(css, /prefers-reduced-motion:\s*reduce/);
  assert.match(css, /overflow-x:\s*clip|overflow-x:\s*hidden/);
  assert.doesNotMatch(css, /#networkSvg[^}]*min-width:\s*[1-9]/s);
});


test("desktop stage reserves the main height for the network instead of the email card", () => {
  assert.match(css, /\.network-stage\s*\{[^}]*grid-template-rows:\s*auto\s+minmax\(0,\s*1fr\)\s+auto\s+auto/s);
  assert.match(css, /\.email-card\s*\{[^}]*position:\s*absolute/s);
  assert.match(css, /body\[data-scene="scene-weights"\][\s\S]*?\.email-card[^}]*opacity:\s*0/s);
  assert.match(css, /#networkSvg\s*\{[^}]*display:\s*block/s);
});


test("email staging area does not cover the network and zeroed ReLU neurons are visually distinct", () => {
  assert.match(css, /body\[data-scene="scene-intro"\][\s\S]*?\.network-frame[^}]*padding-top:/s);
  assert.match(css, /body\[data-scene="scene-inputs"\][\s\S]*?\.network-frame[^}]*padding-top:/s);
  assert.match(css, /\.neuron\.is-zeroed\s*\{[^}]*opacity:\s*1/s);
  assert.match(css, /\.neuron\.is-zeroed\s+\.neuron-halo[^}]*fill:\s*var\(--amber-soft\)/s);
});
