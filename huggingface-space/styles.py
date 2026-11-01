CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

:root {
  --page: #101b2d;
  --surface: #192943;
  --card: #20334f;
  --card-soft: #263b59;
  --line: #3b5270;
  --text: #f5f8ff;
  --muted: #b4c2d5;
  --cyan: #54dcf2;
  --blue: #5596ff;
  --violet: #b5a0ff;
  --rose: #ff829d;
  --amber: #ffd18a;
}
* { box-sizing: border-box; }
body { background: var(--page) !important; color: var(--text) !important; }
.gradio-container {
  max-width: 1480px !important;
  padding: 26px clamp(14px, 3vw, 42px) 24px !important;
  background: linear-gradient(145deg, #14233a 0%, #101b2d 58%, #172943 100%) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', Arial, sans-serif !important;
}
footer { display:none !important; }
.action-app { min-height:100vh; }
.hero-header {
  display:flex; align-items:center; justify-content:space-between; gap:24px;
  padding:24px 28px; border:1px solid #49627f; border-radius:18px;
  background:linear-gradient(110deg,#294566,#203650 60%,#1b304b);
  box-shadow:0 12px 32px #07132666;
}
.hero-brand { display:flex; align-items:center; gap:18px; min-width:0; }
.brand-symbol {
  width:58px;height:58px;display:grid;place-items:center;position:relative;flex-shrink:0;
  border:1px solid #72dff2;border-radius:15px;background:#1a7893;
  box-shadow:0 5px 18px #06182a55;
}
.brand-symbol span {font:700 17px 'Space Mono',monospace;color:white}
.brand-symbol i {position:absolute;width:8px;height:8px;right:8px;top:8px;border-radius:50%;background:#a4ffdb}
.eyebrow {font:700 10px 'Space Mono',monospace;letter-spacing:1.2px;color:#a9c7e7}
.brand-title {font-size:clamp(22px,3vw,31px);font-weight:800;letter-spacing:.4px;color:#fff;line-height:1.3}
.brand-title span {color:#79e9f5}
.brand-subtitle {font-size:13px;color:#d1deed;margin-top:5px}
.header-meta {display:flex;flex-direction:column;align-items:flex-end;gap:10px}
.engine-chip {display:flex;align-items:center;gap:9px;border:1px solid #4c9b8d;background:#1b625c;border-radius:30px;padding:8px 13px;color:#d2fff1;font:700 10px 'Space Mono',monospace;letter-spacing:.5px}
.pulse-dot {width:8px;height:8px;border-radius:50%;background:#8fffd0;box-shadow:0 0 8px #8fffd0}
.version-label {font:10px 'Space Mono',monospace;letter-spacing:.8px;color:#c0d0e3}
.version-label b {color:#fff}
.welcome-row {display:flex;align-items:flex-end;justify-content:space-between;gap:18px;padding:30px 4px 20px}
.welcome-row h1 {font-size:29px;font-weight:800;letter-spacing:-.6px;margin:7px 0 4px;color:#fff}
.welcome-row p {margin:0;color:#bdcadd;font-size:14px}
.workspace-tag {font:10px 'Space Mono',monospace;letter-spacing:.7px;color:#d3e5f8;border:1px solid #49617d;background:#203651;padding:10px 13px;border-radius:8px;white-space:nowrap}
.tiny-square {display:inline-block;width:7px;height:7px;background:var(--cyan);margin-right:8px;border-radius:2px}
.app-tabs {gap:12px !important}
.app-tabs [role=tablist] {border-bottom:1px solid #49617c !important;gap:8px !important;margin-bottom:20px !important;padding-bottom:7px !important}
.app-tabs button[role=tab] {color:#c3d0e1 !important;border:1px solid #405773 !important;border-radius:9px !important;background:#1b2d46 !important;padding:12px 17px !important;font-size:13px !important;font-weight:700 !important}
.app-tabs button[role=tab][aria-selected=true] {color:#fff !important;background:#236b88 !important;border-color:#65d9ed !important;box-shadow:0 4px 14px #061b2b55 !important}
.panel {
  background:#20334d !important;border:1px solid #435a76 !important;border-radius:15px !important;
  padding:22px !important;min-width:0 !important;box-shadow:0 8px 22px #08172a55 !important;
}
.top-grid,.lower-grid,.analytics-grid {align-items:stretch !important;gap:18px !important}
.panel-topline {display:flex;align-items:center;justify-content:space-between;gap:14px;margin-bottom:18px}
.section-title {font-size:19px;font-weight:800;letter-spacing:-.3px;color:#fff;margin-top:5px}
.state-pill {font:700 10px 'Space Mono',monospace;color:#c8ffeb;border:1px solid #398e7b;background:#205d56;padding:7px 10px;border-radius:6px;white-space:nowrap}
.metric-caption {font:10px 'Space Mono',monospace;letter-spacing:.8px;color:#b2c5dc}
.video-panel {border-color:#4d7895 !important}
.video-input {border:1px solid #607a98 !important;border-radius:11px !important;overflow:hidden !important;background:#111f32 !important}
.video-input video {max-height:420px !important;object-fit:contain !important;background:#111f32 !important}
.status-message {font-size:13px !important;color:#e1eaf5 !important;background:#2b4562 !important;border:1px solid #55718f !important;border-left:4px solid var(--cyan) !important;border-radius:8px !important;padding:12px 14px !important;margin:14px 0 !important}
.status-message p {margin:0 !important}
.video-actions {gap:12px !important;margin:12px 0 7px !important}
button {min-height:44px !important;border-radius:9px !important;font-family:'DM Sans',sans-serif !important;font-size:13px !important;font-weight:700 !important;transition:filter .15s ease,transform .15s ease !important}
button:hover {filter:brightness(1.12) !important;transform:translateY(-1px) !important}
button.primary,.start-action {background:#167fa5 !important;border:1px solid #76e6f4 !important;color:white !important;box-shadow:0 4px 12px #071b2b55 !important}
button.secondary,.stop-action {background:#344c69 !important;border:1px solid #66809d !important;color:#f2f7ff !important}
.download-action {width:100% !important;margin-top:10px !important;background:#285d79 !important;border:1px solid #62b9d0 !important;color:white !important}
.progress-control {margin-top:10px !important}
.progress-control input[type=range],input[type=range] {accent-color:var(--cyan) !important}
.progress-control label,.action-panel label,.controls-panel label {font:700 10px 'Space Mono',monospace !important;letter-spacing:.7px !important;color:#c2d1e3 !important}
.insights-panel {background:#243a57 !important}
.metric-row {gap:12px !important;margin-bottom:12px !important}
.metric-card {background:#2b4260 !important;border:1px solid #526c89 !important;border-radius:11px !important;padding:15px !important;min-width:0 !important}
.metric-card label {font:700 9px 'Space Mono',monospace !important;letter-spacing:.5px !important;color:#c2d2e5 !important}
.metric-card input {font-size:31px !important;font-weight:800 !important;color:#fff !important;padding:3px 0 !important;background:transparent !important;border:0 !important}
.metric-cyan {border-top:3px solid var(--cyan) !important}
.metric-violet {border-top:3px solid var(--violet) !important}
.metric-rose {border-top:3px solid var(--rose) !important}
.metric-amber {border-top:3px solid var(--amber) !important}
.metric-cyan input {color:#8af0ff !important}.metric-violet input {color:#d0c1ff !important}
.metric-rose input {color:#ffabbc !important}.metric-amber input {color:#ffdda8 !important}
.insight-note {display:flex;align-items:center;gap:13px;padding:14px;border:1px solid #526b87;border-radius:10px;background:#2b4260;margin-top:10px}
.note-icon {width:34px;height:34px;display:grid;place-items:center;border-radius:9px;background:#26708a;color:#fff;font-size:18px;flex-shrink:0}
.insight-note b,.insight-note span {display:block}.insight-note b {font-size:12px;color:#fff}.insight-note span {font-size:11px;color:#d0dced;margin-top:4px;line-height:1.5}
.mini-divider {height:1px;background:#526984;margin:17px 0 13px}
.quick-facts {display:grid;grid-template-columns:1fr 1fr;gap:14px}
.quick-facts span,.quick-facts b {display:block}
.quick-facts span {font:10px 'Space Mono',monospace;letter-spacing:.7px;color:#b0c3d9}
.quick-facts b {font-size:12px;color:#fff;margin-top:5px}
.lower-grid {margin-top:18px !important}
.action-panel input[type=range] {accent-color:var(--cyan) !important}
.action-panel .wrap,.controls-panel .wrap {margin-bottom:15px !important}
.select-row {gap:12px !important}
.controls-panel input,.controls-panel textarea,.controls-panel select,.controls-panel [role=combobox] {background:#2a405d !important;color:#fff !important;border:1px solid #58718e !important;border-radius:8px !important}
.controls-panel .utility-actions {gap:10px !important;margin-top:10px !important}
.utility-actions button {flex:1}
.gradio-plot {background:transparent !important;min-width:0 !important;border:0 !important}
.gradio-plot .plot-container {border-radius:9px !important}
.analytics-grid {gap:18px !important}
.dataframe {overflow-x:auto !important;border:1px solid #536b87 !important;border-radius:9px !important}
.dataframe table {background:#243a56 !important;color:#f0f5fc !important}
.dataframe th {background:#315171 !important;color:#d9f8ff !important;font:700 10px 'Space Mono',monospace !important}
.dataframe td {border-color:#49617c !important;font-size:12px !important}
.empty-events {color:#c1d0e2 !important;font-size:13px !important}
.settings-panel {max-width:950px}
.settings-panel .prose {color:#d5e0ed !important}
.settings-panel strong {color:#fff}
.app-footer {display:flex;justify-content:space-between;gap:14px;padding:22px 4px 5px;color:#aebfd4;font:10px 'Space Mono',monospace;letter-spacing:.6px}
.app-footer b,.app-footer i {color:#75e7f5;font-style:normal}
@media(max-width:900px) {
  .top-grid,.lower-grid,.analytics-grid {flex-direction:column !important}
  .top-grid>div,.lower-grid>div,.analytics-grid>div {width:100% !important;flex:none !important}
  .video-input video {max-height:350px !important}
}
@media(max-width:600px) {
  .gradio-container {padding:12px 10px 16px !important}
  .hero-header {padding:15px;border-radius:13px;align-items:flex-start}
  .hero-brand {gap:12px}.brand-symbol {width:45px;height:45px;border-radius:11px}
  .brand-title {font-size:20px}.brand-subtitle {font-size:10px;max-width:220px}
  .eyebrow {font-size:9px}.header-meta {gap:7px}.engine-chip {font-size:8px;padding:7px 9px}
  .version-label {font-size:8px}.welcome-row {padding:22px 3px 16px;align-items:flex-start}
  .welcome-row h1 {font-size:24px}.welcome-row p {font-size:12px;max-width:260px}
  .workspace-tag {font-size:8px;padding:8px}
  .app-tabs [role=tablist] {gap:5px !important;overflow-x:auto !important}
  .app-tabs button[role=tab] {font-size:10px !important;padding:10px 9px !important;white-space:nowrap}
  .panel {padding:15px !important;border-radius:12px !important}
  .section-title {font-size:16px}.video-input video {max-height:260px !important}
  .metric-card {padding:10px !important}.metric-card input {font-size:25px !important}
  .metric-card label {font-size:8px !important}.select-row {flex-wrap:wrap !important}
  .select-row>div {min-width:100% !important}.app-footer {font-size:8px;flex-direction:column}
  .video-actions button {font-size:11px !important}
}
@media(max-width:380px) {
  .brand-subtitle,.version-label {display:none}
  .header-meta {align-self:flex-start}.welcome-row {flex-direction:column}
  .workspace-tag {align-self:flex-start}
  .app-tabs button[role=tab] {padding:9px 7px !important;font-size:9px !important}
}
"""