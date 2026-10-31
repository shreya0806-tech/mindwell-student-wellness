/* MindWell interactions (Bootstrap 5) */
(function () {
  // reveal on scroll
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
    });
  }, { threshold: 0.1 });
  document.querySelectorAll('.reveal').forEach(function (el) { io.observe(el); });

  // breathing exercise tied to the Bootstrap modal
  var modal = document.getElementById('breathModal');
  var breathTimer = null;
  function startBreath() {
    var circle = document.getElementById('breathCircle');
    var label = document.getElementById('breathLabel');
    var count = document.getElementById('breathCount');
    if (!circle) return;
    stopBreath();
    var phases = [
      { t: 'Breathe in…', s: 4, scale: 1.35 },
      { t: 'Hold', s: 4, scale: 1.35 },
      { t: 'Breathe out…', s: 6, scale: 1.0 },
      { t: 'Rest', s: 2, scale: 1.0 }
    ];
    var round = 1, pi = 0, left = phases[0].s;
    function tick() {
      var ph = phases[pi];
      if (label) label.textContent = ph.t + '  ·  round ' + round + ' of 4';
      if (count) count.textContent = left;
      circle.style.transform = 'scale(' + ph.scale + ')';
      left -= 1;
      if (left < 0) {
        pi = (pi + 1) % phases.length;
        if (pi === 0) {
          round += 1;
          if (round > 4) {
            stopBreath();
            if (label) label.textContent = 'Well done — notice how you feel. 💜';
            if (count) count.textContent = '✓';
            circle.style.transform = 'scale(1)';
            return;
          }
        }
        left = phases[pi].s;
      }
    }
    tick();
    breathTimer = setInterval(tick, 1000);
  }
  function stopBreath() { if (breathTimer) { clearInterval(breathTimer); breathTimer = null; } }
  if (modal) {
    modal.addEventListener('shown.bs.modal', startBreath);
    modal.addEventListener('hidden.bs.modal', function () {
      stopBreath();
      var circle = document.getElementById('breathCircle');
      if (circle) circle.style.transform = 'scale(1)';
    });
  }
  // if URL hash points inside a tab pane (e.g. #helplines), activate that tab
  if (window.location.hash) {
    var pane = document.querySelector(window.location.hash);
    if (pane) {
      var tabPane = pane.closest('.tab-pane');
      if (tabPane && tabPane.id) {
        var trigger = document.querySelector('[data-bs-target="#' + tabPane.id + '"]');
        if (trigger && window.bootstrap) {
          new bootstrap.Tab(trigger).show();
        }
      }
      setTimeout(function () { pane.scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 350);
    }
  }
})();

// CSRF helper for fetch POSTs
function getCookie(name) {
  var v = document.cookie.match('(^|;)\\s*' + name + '\\s*=\\s*([^;]+)');
  return v ? v.pop() : '';
}
