/* Omar OS: progressive enhancement for the supplied portfolio windows. */
(function () {
  'use strict';
  var wins = Array.from(document.querySelectorAll('.win'));
  var mobile = window.matchMedia('(max-width:760px)');
  var launchers = new Map();
  var order = [];
  function byId(name) { return document.getElementById('w-' + name); }

  function fit(w) {
    if (mobile.matches) return;
    var minX = 144;
    var maxX = Math.max(minX, innerWidth - w.offsetWidth - 12);
    var maxY = Math.max(36, innerHeight - w.offsetHeight - 96);
    w.style.left = Math.max(minX, Math.min(maxX, parseFloat(w.style.left) || minX)) + 'px';
    w.style.top = Math.max(36, Math.min(maxY, parseFloat(w.style.top) || 70)) + 'px';
  }
  function front(w) {
    order = order.filter(function (x) { return x !== w; });
    order.push(w);
    wins.forEach(function (x) { x.classList.toggle('front', x === w); });
    order.forEach(function (x, i) { x.style.zIndex = 20 + i; });
  }
  function syncButtons() {
    document.querySelectorAll('[data-open]').forEach(function (el) {
      var w = byId(el.dataset.open);
      el.setAttribute('aria-controls', w.id);
      el.setAttribute('aria-expanded', String(mobile.matches || w.classList.contains('open')));
      if (el.classList.contains('icon')) el.classList.toggle('sel', w.classList.contains('front'));
    });
  }
  function open(name, launcher) {
    var w = byId(name);
    if (!w) return;
    if (launcher) launchers.set(w, launcher);
    w.classList.add('open');
    front(w);
    fit(w);
    syncButtons();
    if (launcher) {
      if (mobile.matches) w.scrollIntoView({block:'start', behavior:'instant'});
      w.focus({preventScroll:true});
    }
  }
  function close(w) {
    if (mobile.matches) return;
    w.classList.remove('open', 'front');
    order = order.filter(function (x) { return x !== w; });
    var next = order[order.length - 1];
    if (next) front(next);
    syncButtons();
    var launcher = launchers.get(w) || document.querySelector('.icon[data-open="' + w.id.slice(2) + '"]');
    if (launcher) launcher.focus({preventScroll:true});
  }

  document.querySelectorAll('[data-open]').forEach(function (el) {
    el.addEventListener('click', function () { open(el.dataset.open, el); });
  });
  document.querySelector('.skip-link').addEventListener('click', function (e) {
    e.preventDefault();
    open('about', e.currentTarget);
  });
  wins.forEach(function (w) {
    w.addEventListener('pointerdown', function () { front(w); syncButtons(); });
    w.addEventListener('focusin', function () { front(w); syncButtons(); });
    var closeButton = w.querySelector('.close');
    closeButton.setAttribute('aria-label', 'Close ' + w.getAttribute('aria-label'));
    closeButton.addEventListener('click', function () { close(w); });
    var bar = w.querySelector('.titlebar');
    var drag = null;
    bar.addEventListener('pointerdown', function (e) {
      if (mobile.matches || e.button !== 0 || e.target.closest('button')) return;
      drag = {x:e.clientX - w.offsetLeft, y:e.clientY - w.offsetTop};
      bar.setPointerCapture(e.pointerId);
      bar.style.cursor = 'grabbing';
    });
    bar.addEventListener('pointermove', function (e) {
      if (!drag) return;
      w.style.left = (e.clientX - drag.x) + 'px';
      w.style.top = (e.clientY - drag.y) + 'px';
      fit(w);
    });
    function stopDrag() { drag = null; bar.style.cursor = ''; }
    bar.addEventListener('pointerup', stopDrag);
    bar.addEventListener('pointercancel', stopDrag);
    bar.addEventListener('lostpointercapture', stopDrag);
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && !mobile.matches && order.length) {
      close(order[order.length - 1]);
    }
  });
  function reflow() {
    wins.filter(function (w) { return w.classList.contains('open'); }).forEach(fit);
    syncButtons();
  }
  window.addEventListener('resize', reflow);
  mobile.addEventListener('change', reflow);
  // Only hide the fallback panels after the window controls are ready.
  document.documentElement.classList.add('enhanced');
  open('about');
  function tick() {
    document.getElementById('clock').textContent = new Intl.DateTimeFormat('en-GB', {
      hour:'2-digit', minute:'2-digit'
    }).format(new Date());
  }
  tick();
  setInterval(tick, 15000);
})();

