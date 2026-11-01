CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

:root {
  --bg: #170d29; --surface: #24123b; --panel: #2b1747; --panel2: #351b55;
  --line: #7544a1; --text: #fffaff; --muted: #e0cbed;
  --violet: #bd4cff; --violet-bright: #e0a0ff; --pink: #ff76dc;
  --cyan: #76efff; --green: #70f4c4; --red: #ff82ad; --amber: #ffd18b;
}
* { box-sizing: border-box; }
html, body { background: #170d29 !important; color: var(--text) !important; }
body { overflow-x: hidden !important; }
.gradio-container {
  width: 100% !important; max-width: 1600px !important; min-width: 0 !important;
  margin: 0 auto !important; padding: 22px clamp(12px, 2.6vw, 40px) 30px !important;
  background:
    radial-gradient(ellipse at 10% 0%, #8738c966 0%, transparent 38%),
    radial-gradient(ellipse at 92% 15%, #d342bb38 0%, transparent 32%),
    radial-gradient(ellipse at 55% 100%, #5933a944 0%, transparent 40%),
    linear-gradient(135deg, #211039 0%, #170d29 52%, #29113d 100%) !important;
  color: var(--text) !important; font-family: 'DM Sans', Arial, sans-serif !important;
}
footer { display:none !important; }
.action-app, .action-app * { min-width: 0; }
.hero-header {
 display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:20px;
 padding:23px 28px; border:1px solid #b66be0; border-radius:20px;
 background:linear-gradient(110deg,#55277d 0%,#3c205d 55%,#6c286e 100%);
 box-shadow:0 12px 32px #0b061866,0 0 24px #c35aff38,inset 0 1px #ffffff35;
}
.hero-brand {display:flex;align-items:center;gap:16px}
.brand-symbol {width:58px;height:58px;display:grid;place-items:center;position:relative;flex-shrink:0;border:1px solid #f0c4ff;border-radius:16px;background:linear-gradient(145deg,#e28aff,#982be0);box-shadow:0 0 22px #d35bff99,inset 0 0 14px #fff5}
.brand-symbol span {font:700 17px 'Space Mono',monospace;color:white;text-shadow:0 0 10px #fff}
.brand-symbol i {position:absolute;width:8px;height:8px;right:8px;top:8px;border-radius:50%;background:var(--cyan);box-shadow:0 0 10px var(--cyan)}
.eyebrow {font:700 10px 'Space Mono',monospace;letter-spacing:1.3px;color:#efbdff}
.brand-title {font-size:clamp(22px,3vw,33px);font-weight:800;letter-spacing:.4px;color:#fff;line-height:1.2}
.brand-title span {color:#f0b1ff;text-shadow:0 0 18px #e68affaa}
.brand-subtitle {font-size:13px;color:#f0def9;margin-top:6px}
.header-meta {display:flex;flex-direction:column;align-items:flex-end;gap:10px}
.engine-chip {display:flex;align-items:center;gap:9px;border:1px solid #d18bea;background:#51266d;border-radius:30px;padding:8px 13px;color:#fff;font:700 10px 'Space Mono',monospace}
.pulse-dot {width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green)}
.version-label {font:10px 'Space Mono',monospace;letter-spacing:.8px;color:#e8d5f4}
.version-label b {color:white}
.welcome-row {display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:16px;padding:27px 4px 19px}
.welcome-row h1 {font-size:clamp(23px,3vw,31px);font-weight:800;margin:7px 0 4px;color:#fff}
.welcome-row p {margin:0;color:#e0cbed;font-size:14px}
.workspace-tag {font:10px 'Space Mono',monospace;letter-spacing:.7px;color:#f8eaff;border:1px solid #9862ba;background:#3a2053;padding:10px 13px;border-radius:9px;white-space:nowrap}
.tiny-square {display:inline-block;width:7px;height:7px;background:var(--cyan);margin-right:8px;border-radius:2px;box-shadow:0 0 9px var(--cyan)}
.app-tabs {gap:14px !important}
.app-tabs [role=tablist] {display:flex !important;flex-wrap:wrap !important;gap:8px !important;margin-bottom:19px !important;padding:7px !important;border:1px solid #714395 !important;border-radius:14px !important;background:#2a1742 !important}
.app-tabs button[role=tab] {color:#ead8f5 !important;border:1px solid transparent !important;border-radius:9px !important;background:transparent !important;padding:11px 17px !important;font-size:13px !important;font-weight:700 !important}
.app-tabs button[role=tab]:hover {border-color:#c27ce9 !important;color:#fff !important;background:#4a2865 !important}
.app-tabs button[role=tab][aria-selected=true] {color:#fff !important;background:linear-gradient(110deg,#a83ee0,#7732b4) !important;border-color:#e2a5ff !important;box-shadow:0 0 17px #c14aff55,inset 0 1px #fff5 !important}
.panel {width:100% !important;min-width:0 !important;background:linear-gradient(145deg,#3a2255,#2b1944) !important;border:1px solid #8051a5 !important;border-radius:17px !important;padding:21px !important;box-shadow:0 10px 25px #0c061b55,inset 0 1px #ffffff12 !important}
.stage-grid,.lower-grid,.analytics-grid {align-items:stretch !important;gap:18px !important;min-width:0 !important}
.stage-grid > *,.lower-grid > *,.analytics-grid > * {min-width:0 !important}
.panel-topline,.section-heading {display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:17px}
.section-title {font-size:19px;font-weight:800;letter-spacing:-.3px;color:#fff;margin-top:5px}
.state-pill {font:700 10px 'Space Mono',monospace;color:#caffea;border:1px solid #55b99a;background:#205644;padding:7px 10px;border-radius:7px}
.metric-caption {font:10px 'Space Mono',monospace;letter-spacing:.8px;color:#e0c4f0}
.video-panel {border-color:#c17be5 !important;background:linear-gradient(145deg,#40245f,#2b1944) !important;box-shadow:0 12px 35px #0c061b66,0 0 25px #c14aff22 !important}
.video-input {width:100% !important;border:1px solid #a86bc8 !important;border-radius:13px !important;overflow:hidden !important;background:#160d25 !important}
.video-input video {width:100% !important;max-height:490px !important;object-fit:contain !important;background:#160d25 !important}
.status-message {font-size:13px !important;color:#fff !important;background:#492962 !important;border:1px solid #9560b6 !important;border-left:4px solid #e29aff !important;border-radius:9px !important;padding:12px 14px !important;margin:14px 0 !important}
.status-message p {margin:0 !important}
.video-actions {display:flex !important;flex-wrap:wrap !important;gap:11px !important;margin:12px 0 7px !important}
.video-actions > * {flex:1 1 170px !important}
button {min-height:44px !important;border-radius:10px !important;font-family:'DM Sans',sans-serif !important;font-size:13px !important;font-weight:700 !important;transition:filter .15s ease,transform .15s ease,box-shadow .15s ease !important}
button:hover {filter:brightness(1.12) !important;transform:translateY(-1px) !important}
button.primary,.start-action {background:linear-gradient(110deg,#d05aff,#8e32d0) !important;border:1px solid #f0bdff !important;color:white !important;box-shadow:0 0 17px #bd4cff66,inset 0 1px #fff5 !important}
button.secondary,.stop-action {background:#56336f !important;border:1px solid #a875c6 !important;color:#fff !important}
.progress-control {margin-top:10px !important}
input[type=range] {accent-color:#e09aff !important}
.progress-control label,.action-panel label,.control-rail label {font:700 10px 'Space Mono',monospace !important;letter-spacing:.7px !important;color:#efdafa !important}
.control-rail {background:linear-gradient(155deg,#48265f,#321d4c) !important}
.control-rail input,.control-rail textarea,.control-rail select,.control-rail [role=combobox] {background:#512d6b !important;color:#fff !important;border:1px solid #a36cc0 !important;border-radius:9px !important}
.control-rail .wrap {margin-bottom:17px !important}
.rail-divider {height:1px;background:#8d5eaa;margin:20px 0 17px}
.rail-actions {gap:9px !important;margin-top:10px !important}
.rail-actions button {width:100% !important}
.download-action {width:100% !important;background:linear-gradient(110deg,#7c39a5,#592d7d) !important;border:1px solid #c08ae0 !important;color:white !important}
.rail-tip {display:flex;gap:11px;align-items:flex-start;margin-top:20px;padding:13px;border:1px solid #9362ad;border-radius:11px;background:#3d2455}
.tip-mark {font-size:20px;color:#f1b0ff;text-shadow:0 0 12px #dc78ff}
.rail-tip b {font-size:12px;color:#fff}
.rail-tip p {font-size:11px;line-height:1.55;color:#e4d0f0;margin:4px 0 0}
.metrics-section {margin-top:18px;padding:18px 20px 20px;border:1px solid #8051a5;border-radius:17px;background:linear-gradient(120deg,#392153,#2c1944);box-shadow:0 9px 24px #0c061b44}
.section-heading {margin-bottom:13px}
.metric-strip {display:flex !important;gap:13px !important;align-items:stretch !important}
.metric-strip > * {flex:1 1 0 !important;min-width:0 !important}
.metric-card {background:linear-gradient(145deg,#563276,#40245f) !important;border:1px solid #9366b0 !important;border-radius:12px !important;padding:14px !important}
.metric-card label {font:700 9px 'Space Mono',monospace !important;letter-spacing:.5px !important;color:#f0dcfa !important}
.metric-card input {font-size:clamp(23px,2.3vw,31px) !important;font-weight:800 !important;color:#fff !important;padding:3px 0 !important;background:transparent !important;border:0 !important}
.metric-cyan {border-top:3px solid var(--cyan) !important}.metric-violet {border-top:3px solid #e0a0ff !important}.metric-rose {border-top:3px solid var(--pink) !important}.metric-amber {border-top:3px solid var(--amber) !important}
.metric-cyan input {color:#a5f5ff !important}.metric-violet input {color:#edc1ff !important}.metric-rose input {color:#ffb0dc !important}.metric-amber input {color:#ffe0ad !important}
.lower-grid {margin-top:18px !important}
.action-panel input[type=range] {accent-color:#e09aff !important}
.action-panel .wrap {margin-bottom:15px !important}
.session-info-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.session-info {padding:15px;border:1px solid #80549d;border-radius:11px;background:#40265a}
.session-info span,.session-info b,.session-info small {display:block}
.session-info span {font:10px 'Space Mono',monospace;letter-spacing:.7px;color:#d9b9eb}
.session-info b {font-size:13px;color:#fff;margin-top:7px}
.session-info small {font-size:10px;color:#d8c2e7;margin-top:4px}
.analytics-grid > * {flex:1 1 380px !important}
.gradio-plot {background:transparent !important;min-width:0 !important;width:100% !important;border:0 !important}
.gradio-plot .plot-container {max-width:100% !important;border-radius:10px !important;overflow:hidden !important}
.dataframe {max-width:100% !important;overflow-x:auto !important;border:1px solid #80549d !important;border-radius:10px !important}
.dataframe table {min-width:420px !important;background:#39234f !important;color:#fff !important}
.dataframe th {background:#593477 !important;color:#fff !important;font:700 10px 'Space Mono',monospace !important}
.dataframe td {border-color:#70478c !important;font-size:12px !important}
.empty-events {color:#e2cfee !important;font-size:13px !important}
.settings-panel {max-width:950px}
.settings-panel .prose {color:#e6d3f0 !important}
.settings-panel strong {color:#fff}
.app-footer {display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px;padding:22px 4px 5px;color:#d2b9e1;font:10px 'Space Mono',monospace;letter-spacing:.6px}
.app-footer b,.app-footer i {color:#e3a1ff;font-style:normal;text-shadow:0 0 10px #c35aff88}
@media (max-width:1050px) {
 .stage-grid,.lower-grid,.analytics-grid {flex-direction:column !important}
 .stage-grid > *,.lower-grid > *,.analytics-grid > * {width:100% !important;max-width:100% !important;flex:1 1 auto !important}
 .video-input video {max-height:390px !important}
}
@media (max-width:700px) {
 .gradio-container {padding:12px 10px 18px !important}
 .hero-header {padding:15px;border-radius:14px;align-items:flex-start}
 .hero-brand {gap:11px}.brand-symbol {width:45px;height:45px;border-radius:12px}
 .brand-title {font-size:20px}.brand-subtitle {font-size:10px;max-width:220px}
 .eyebrow {font-size:9px}.header-meta {align-items:flex-start;gap:7px}
 .engine-chip {font-size:8px;padding:7px 9px}.version-label {font-size:8px}
 .welcome-row {padding:21px 3px 15px;align-items:flex-start}
 .welcome-row h1 {font-size:23px}.welcome-row p {font-size:12px;max-width:260px}
 .workspace-tag {font-size:8px;padding:8px}
 .app-tabs [role=tablist] {flex-wrap:nowrap !important;overflow-x:auto !important}
 .app-tabs button[role=tab] {flex:0 0 auto !important;font-size:10px !important;padding:10px 9px !important;white-space:nowrap !important}
 .panel {padding:14px !important;border-radius:13px !important}
 .section-title {font-size:16px}.video-input video {max-height:280px !important}
 .metric-strip {display:grid !important;grid-template-columns:repeat(2,minmax(0,1fr)) !important}
 .metric-card {padding:10px !important}.metric-card input {font-size:24px !important}
 .metric-card label {font-size:8px !important}.metrics-section {padding:14px !important}
 .session-info-grid {grid-template-columns:1fr 1fr}
 .app-footer {font-size:8px;flex-direction:column}
}
@media (max-width:390px) {
 .brand-subtitle,.version-label {display:none}.welcome-row {flex-direction:column}
 .workspace-tag {align-self:flex-start}.app-tabs button[role=tab] {padding:9px 7px !important;font-size:9px !important}
 .session-info-grid {grid-template-columns:1fr}
}

/* Native Gradio video: replace the default white upload/player surfaces */
#video-stage {
  width:100% !important; min-height:460px !important; padding:0 !important;
  border:1px solid #bd79e8 !important; border-radius:15px !important; overflow:hidden !important;
  background:radial-gradient(ellipse at 50% 0%,#743c9c55 0%,transparent 65%),linear-gradient(145deg,#241337,#160d25) !important;
  box-shadow:inset 0 0 35px #a33fe51c,0 0 18px #bd4cff20 !important;
}
#video-stage > div,#video-stage .wrap,#video-stage .video-container,
#video-stage .upload-container,#video-stage .empty,#video-stage [data-testid="video"],
#video-stage [data-testid="video-upload"],#video-stage .drop-zone {
  background-color:transparent !important; color:#f8edff !important;
  border-color:#80519d !important; box-shadow:none !important;
}
#video-stage .upload-container,#video-stage .drop-zone,#video-stage [data-testid="video-upload"] {
  min-height:420px !important; border:1px dashed #bd79e8 !important; border-radius:13px !important;
  background:radial-gradient(ellipse at center,#743c9c30 0%,transparent 68%),linear-gradient(145deg,#28163d,#1b102b) !important;
  transition:background .2s ease,border-color .2s ease,box-shadow .2s ease !important;
}
#video-stage .upload-container:hover,#video-stage .drop-zone:hover,#video-stage [data-testid="video-upload"]:hover {
  border-color:#edb4ff !important;
  background:radial-gradient(ellipse at center,#a64ce84a 0%,transparent 72%),#28163d !important;
  box-shadow:inset 0 0 30px #c14aff22,0 0 18px #bd4cff2b !important;
}
#video-stage button,#video-stage [role="button"] {
  color:#fff !important; background:linear-gradient(120deg,#71349a,#4c286d) !important;
  border:1px solid #ae75ce !important; border-radius:9px !important; box-shadow:0 0 12px #a94ce52b !important;
}
#video-stage button:hover,#video-stage [role="button"]:hover {
  background:linear-gradient(120deg,#9846c6,#65328b) !important; border-color:#e5b0ff !important;
}
#video-stage svg {color:#e8b5ff !important}
#video-stage p,#video-stage span,#video-stage label,#video-stage small {color:#ead9f5 !important}
#video-stage video {
  display:block !important; width:100% !important; max-height:460px !important;
  object-fit:contain !important; background:#10091b !important; border-radius:12px !important;
}
#video-stage input,#video-stage [type=file] {color:#f8edff !important}
#video-stage [data-testid="file-upload"],#video-stage .file-preview,#video-stage .thumbnail-item {
  background:#29163e !important; border:1px solid #80519d !important; border-radius:10px !important;
}
@media(max-width:700px) {
  #video-stage {min-height:280px !important}
  #video-stage .upload-container,#video-stage .drop-zone,#video-stage [data-testid="video-upload"] {min-height:260px !important}
  #video-stage video {max-height:280px !important}
}

"""
