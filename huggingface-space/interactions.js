(() => {
  const root = document.querySelector('.action-app');
  if (!root || root.dataset.interactionsReady === '1') return;
  root.dataset.interactionsReady = '1';

  const status = root.querySelector('.status-message');
  const engine = root.querySelector('.engine-chip');
  const progress = root.querySelector('.progress-control input');
  const start = root.querySelector('.start-action');
  const reset = root.querySelector('.stop-action');
  const clear = [...root.querySelectorAll('button')].find(b => /Clear video/i.test(b.innerText));
  const video = root.querySelector('#video-stage video');
  const setEngine = (label, mode) => {
    if (!engine) return;
    engine.innerHTML = '<span class="pulse-dot"></span><span>' + label + '</span>';
    engine.style.borderColor = mode === 'busy' ? '#8d6acb' : mode === 'done' ? '#4d806a' : '#4d705f';
    engine.style.background = mode === 'busy' ? '#302449' : mode === 'done' ? '#1d352e' : '#1d352e';
    engine.style.color = mode === 'busy' ? '#d8c5ff' : '#b8f4d7';
  };
  if (start) start.addEventListener('click', () => setEngine('ANALYSIS RUNNING', 'busy'));
  if (reset) reset.addEventListener('click', () => setEngine('ENGINE STANDBY', 'idle'));
  if (clear) clear.addEventListener('click', () => setEngine('ENGINE STANDBY', 'idle'));
  if (status && engine) {
    const observer = new MutationObserver(() => {
      const message = status.innerText || '';
      if (/completed|loaded:/i.test(message)) setEngine('RESULTS READY', 'done');
      else if (/error|could not|choose a video/i.test(message)) setEngine('CHECK INPUT', 'idle');
      else if (/reset|cleared|ready for input/i.test(message)) setEngine('ENGINE STANDBY', 'idle');
    });
    observer.observe(status, {subtree:true,childList:true,characterData:true});
  }
  const bindVideo = () => {
    const player = root.querySelector('#video-stage video');
    if (!player || player.dataset.bound === '1') return;
    player.dataset.bound = '1';
    player.addEventListener('loadedmetadata', () => {
      const chip = root.querySelector('.state-pill');
      if (chip) chip.textContent = '● VIDEO LOADED';
    });
  };
  bindVideo();
  new MutationObserver(bindVideo).observe(root, {subtree:true,childList:true});
  root.querySelectorAll('button').forEach(button => {
    button.addEventListener('pointerdown', event => {
      const ripple = document.createElement('span');
      ripple.className = 'button-ripple';
      const rect = button.getBoundingClientRect();
      ripple.style.left = (event.clientX - rect.left) + 'px';
      ripple.style.top = (event.clientY - rect.top) + 'px';
      button.appendChild(ripple);
      setTimeout(() => ripple.remove(), 550);
    });
  });
})();