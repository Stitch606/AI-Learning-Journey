CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

:root {
  --bg: #090511;
  --bg-2: #10091c;
  --surface: #160d24;
  --panel: #1c102d;
  --panel-2: #241238;
  --panel-3: #2c1743;
  --line: #482466;
  --line-soft: #38204e;
  --text: #fbf7ff;
  --muted: #c0add3;
  --violet: #b52cff;
  --violet-bright: #d45cff;
  --violet-soft: #8b35e8;
  --pink: #ff4fd8;
  --cyan: #54eaff;
  --green: #54f0bd;
  --red: #ff5d91;
  --amber: #ffc56e;
}
* { box-sizing: border-box; }
html, body { background: var(--bg) !important; color: var(--text) !important; }
body { overflow-x: hidden !important; }
.gradio-container {
  width: 100% !important;
  max-width: 1540px !important;
  min-width: 0 !important;
  margin: 0 auto !important;
  padding: 24px clamp(12px, 2.5vw, 38px) 28px !important;
  background:
    radial-gradient(ellipse at 12% 0%, #42146a55 0%, transparent 36%),
    radial-gradient(ellipse at 92% 14%, #8b1a8d22 0%, transparent 32%),
    linear-gradient(145deg, #10091b 0%, #090511 58%, #12091e 100%) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', Arial, sans-serif !important;
}
footer { display: none !important; }
.action-app { min-height: 100vh; min-width: 0 !important; }
.action-app * { min-width: 0; }

.hero-header {
  display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 20px;
  padding: 24px 28px; border: 1px solid #8735b8; border-radius: 20px;
  background: linear-gradient(112deg, #27113d 0%, #1c102d 54%, #32123d 100%);
  box-shadow: 0 0 0 1px #b52cff18, 0 12px 38px #0008, 0 0 28px #a72aff20;
}
.hero-brand { display: flex; align-items: center; gap: 17px; min-width: 0; }
.brand-symbol {
  width: 58px; height: 58px; display: grid; place-items: center; position: relative; flex-shrink: 0;
  border: 1px solid #e18aff; border-radius: 16px;
  background: linear-gradient(145deg, #bd35ff, #7118bc);
  box-shadow: 0 0 12px #c43dff99, 0 0 30px #a52aff55, inset 0 0 14px #fff2;
}
.brand-symbol span { font: 700 17px 'Space Mono', monospace; color: white; text-shadow: 0 0 10px #fff9; }
.brand-symbol i { position: absolute; width: 8px; height: 8px; right: 8px; top: 8px; border-radius: 50%; background: var(--cyan); box-shadow: 0 0 10px var(--cyan); }
.eyebrow { font: 700 10px 'Space Mono', monospace; letter-spacing: 1.35px; color: #d58aff; }
.brand-title { font-size: clamp(22px, 3vw, 32px); font-weight: 800; letter-spacing: .5px; color: #fff; line-height: 1.25; }
.brand-title span { color: var(--violet-bright); text-shadow: 0 0 18px #bd42ff88; }
.brand-subtitle { font-size: 13px; color: #d5c1e7; margin-top: 6px; }
.header-meta { display: flex; flex-direction: column; align-items: flex-end; gap: 10px; }
.engine-chip { display: flex; align-items: center; gap: 9px; border: 1px solid #7e3ba1; background: #28143a; border-radius: 30px; padding: 8px 13px; color: #f1d8ff; font: 700 10px 'Space Mono', monospace; letter-spacing: .5px; }
.pulse-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); box-shadow: 0 0 7px var(--green), 0 0 15px #54f0bd99; }
.version-label { font: 10px 'Space Mono', monospace; letter-spacing: .8px; color: #c8b1db; }
.version-label b { color: #fff; }
.welcome-row { display: flex; align-items: flex-end; justify-content: space-between; flex-wrap: wrap; gap: 16px; padding: 30px 4px 20px; }
.welcome-row h1 { font-size: clamp(23px, 3vw, 30px); font-weight: 800; letter-spacing: -.6px; margin: 7px 0 4px; color: #fff; }
.welcome-row p { margin: 0; color: #c6b2d8; font-size: 14px; }
.workspace-tag { font: 10px 'Space Mono', monospace; letter-spacing: .7px; color: #ead6fa; border: 1px solid #69368a; background: #211331; padding: 10px 13px; border-radius: 9px; white-space: nowrap; }
.tiny-square { display: inline-block; width: 7px; height: 7px; background: var(--cyan); margin-right: 8px; border-radius: 2px; box-shadow: 0 0 9px var(--cyan); }

.app-tabs { gap: 12px !important; min-width: 0 !important; }
.app-tabs [role=tablist] { display: flex !important; flex-wrap: wrap !important; border-bottom: 1px solid #4d2b68 !important; gap: 8px !important; margin-bottom: 20px !important; padding-bottom: 9px !important; }
.app-tabs button[role=tab] { color: #d1bedf !important; border: 1px solid #503067 !important; border-radius: 10px !important; background: #1a1028 !important; padding: 11px 16px !important; font-size: 13px !important; font-weight: 700 !important; transition: .18s ease !important; }
.app-tabs button[role=tab]:hover { border-color: var(--violet-bright) !important; color: #fff !important; box-shadow: 0 0 14px #b52cff33 !important; }
.app-tabs button[role=tab][aria-selected=true] { color: #fff !important; background: linear-gradient(110deg, #8424bd, #5e1c91) !important; border-color: #db8aff !important; box-shadow: 0 0 0 1px #c64aff33, 0 0 18px #a72aff55 !important; }

.panel {
  width: 100% !important; min-width: 0 !important;
  background: linear-gradient(150deg, #211331, #180e25) !important;
  border: 1px solid #54316d !important; border-radius: 16px !important;
  padding: 21px !important; box-shadow: 0 10px 28px #0006, inset 0 1px #ffffff08 !important;
}
.top-grid, .lower-grid, .analytics-grid { align-items: stretch !important; gap: 18px !important; min-width: 0 !important; }
.top-grid > *, .lower-grid > *, .analytics-grid > * { min-width: 0 !important; }
.panel-topline { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-bottom: 17px; }
.section-title { font-size: 19px; font-weight: 800; letter-spacing: -.3px; color: #fff; margin-top: 5px; }
.state-pill { font: 700 10px 'Space Mono', monospace; color: #c8ffeb; border: 1px solid #368b77; background: #12382f; padding: 7px 10px; border-radius: 7px; white-space: nowrap; }
.metric-caption { font: 10px 'Space Mono', monospace; letter-spacing: .8px; color: #c5a9db; }

.video-panel { border-color: #8a42b1 !important; box-shadow: 0 10px 32px #0007, 0 0 24px #a52aff18 !important; }
.video-input { width: 100% !important; min-width: 0 !important; border: 1px solid #754295 !important; border-radius: 12px !important; overflow: hidden !important; background: #0c0713 !important; }
.video-input video { width: 100% !important; max-height: 420px !important; object-fit: contain !important; background: #0c0713 !important; }
.status-message { font-size: 13px !important; color: #eadcf5 !important; background: #241432 !important; border: 1px solid #633783 !important; border-left: 4px solid var(--violet-bright) !important; border-radius: 9px !important; padding: 12px 14px !important; margin: 14px 0 !important; }
.status-message p { margin: 0 !important; }
.video-actions { display: flex !important; flex-wrap: wrap !important; gap: 11px !important; margin: 12px 0 7px !important; }
.video-actions > * { flex: 1 1 150px !important; min-width: 0 !important; }
button { min-height: 44px !important; border-radius: 10px !important; font-family: 'DM Sans', sans-serif !important; font-size: 13px !important; font-weight: 700 !important; transition: filter .15s ease, transform .15s ease, box-shadow .15s ease !important; }
button:hover { filter: brightness(1.13) !important; transform: translateY(-1px) !important; }
button.primary, .start-action { background: linear-gradient(110deg, #b62dff, #7c22c9) !important; border: 1px solid #e2a0ff !important; color: white !important; box-shadow: 0 0 14px #ad31ff66, inset 0 1px #fff3 !important; }
button.secondary, .stop-action { background: #2a1a3b !important; border: 1px solid #70458b !important; color: #f6eaff !important; }
.download-action { width: 100% !important; margin-top: 10px !important; background: linear-gradient(110deg, #542078, #38204f) !important; border: 1px solid #9d59c3 !important; color: white !important; }
.progress-control { margin-top: 10px !important; }
.progress-control input[type=range], input[type=range] { accent-color: var(--violet-bright) !important; }
.progress-control label, .action-panel label, .controls-panel label { font: 700 10px 'Space Mono', monospace !important; letter-spacing: .7px !important; color: #d2bce2 !important; }

.insights-panel { background: linear-gradient(150deg, #251337, #1b1029) !important; }
.metric-row { display: flex !important; flex-wrap: wrap !important; gap: 12px !important; margin-bottom: 12px !important; }
.metric-row > * { flex: 1 1 120px !important; min-width: 0 !important; }
.metric-card { background: linear-gradient(145deg, #2a1740, #211331) !important; border: 1px solid #583371 !important; border-radius: 12px !important; padding: 14px !important; min-width: 0 !important; }
.metric-card label { font: 700 9px 'Space Mono', monospace !important; letter-spacing: .5px !important; color: #d2bce2 !important; }
.metric-card input { font-size: clamp(23px, 2.3vw, 31px) !important; font-weight: 800 !important; color: #fff !important; padding: 3px 0 !important; background: transparent !important; border: 0 !important; }
.metric-cyan { border-top: 3px solid var(--cyan) !important; }
.metric-violet { border-top: 3px solid var(--violet-bright) !important; }
.metric-rose { border-top: 3px solid var(--pink) !important; }
.metric-amber { border-top: 3px solid var(--amber) !important; }
.metric-cyan input { color: #8cf2ff !important; }
.metric-violet input { color: #e0a6ff !important; }
.metric-rose input { color: #ff9cde !important; }
.metric-amber input { color: #ffdda8 !important; }
.insight-note { display: flex; align-items: center; gap: 13px; padding: 14px; border: 1px solid #563270; border-radius: 11px; background: #241432; margin-top: 10px; }
.note-icon { width: 36px; height: 36px; display: grid; place-items: center; border-radius: 10px; background: #542078; color: #fff; font-size: 18px; flex-shrink: 0; box-shadow: 0 0 14px #b52cff55; }
.insight-note b, .insight-note span { display: block; }
.insight-note b { font-size: 12px; color: #fff; }
.insight-note span { font-size: 11px; color: #d4c1e3; margin-top: 4px; line-height: 1.5; }
.mini-divider { height: 1px; background: #503064; margin: 17px 0 13px; }
.quick-facts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.quick-facts span, .quick-facts b { display: block; }
.quick-facts span { font: 10px 'Space Mono', monospace; letter-spacing: .7px; color: #c2a9d5; }
.quick-facts b { font-size: 12px; color: #fff; margin-top: 5px; }

.lower-grid { margin-top: 18px !important; }
.action-panel input[type=range] { accent-color: var(--violet-bright) !important; }
.action-panel .wrap, .controls-panel .wrap { margin-bottom: 15px !important; }
.select-row { display: flex !important; flex-wrap: wrap !important; gap: 12px !important; }
.select-row > * { flex: 1 1 190px !important; min-width: 0 !important; }
.controls-panel input, .controls-panel textarea, .controls-panel select, .controls-panel [role=combobox] { background: #281738 !important; color: #fff !important; border: 1px solid #684184 !important; border-radius: 9px !important; }
.controls-panel .utility-actions { display: flex !important; flex-wrap: wrap !important; gap: 10px !important; margin-top: 10px !important; }
.utility-actions > * { flex: 1 1 140px !important; min-width: 0 !important; }
.gradio-plot { background: transparent !important; min-width: 0 !important; width: 100% !important; border: 0 !important; }
.gradio-plot .plot-container { max-width: 100% !important; border-radius: 10px !important; overflow: hidden !important; }
.analytics-grid { gap: 18px !important; }
.analytics-grid > * { flex: 1 1 360px !important; }
.dataframe { max-width: 100% !important; overflow-x: auto !important; border: 1px solid #563270 !important; border-radius: 10px !important; }
.dataframe table { min-width: 420px !important; background: #211331 !important; color: #f7efff !important; }
.dataframe th { background: #38204f !important; color: #f0d9ff !important; font: 700 10px 'Space Mono', monospace !important; }
.dataframe td { border-color: #4b2d62 !important; font-size: 12px !important; }
.empty-events { color: #cdb9dc !important; font-size: 13px !important; }
.settings-panel { max-width: 950px; }
.settings-panel .prose { color: #d8c7e5 !important; }
.settings-panel strong { color: #fff; }
.app-footer { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 12px; padding: 22px 4px 5px; color: #bda8cf; font: 10px 'Space Mono', monospace; letter-spacing: .6px; }
.app-footer b, .app-footer i { color: var(--violet-bright); font-style: normal; text-shadow: 0 0 10px #b52cff77; }

@media (max-width: 900px) {
  .top-grid, .lower-grid, .analytics-grid { flex-direction: column !important; }
  .top-grid > *, .lower-grid > *, .analytics-grid > * { width: 100% !important; max-width: 100% !important; flex: 1 1 auto !important; }
  .video-input video { max-height: 350px !important; }
}
@media (max-width: 600px) {
  .gradio-container { padding: 12px 10px 18px !important; }
  .hero-header { padding: 15px; border-radius: 14px; align-items: flex-start; }
  .hero-brand { gap: 11px; }
  .brand-symbol { width: 45px; height: 45px; border-radius: 12px; }
  .brand-title { font-size: 20px; }
  .brand-subtitle { font-size: 10px; max-width: 220px; }
  .eyebrow { font-size: 9px; }
  .header-meta { align-items: flex-start; gap: 7px; }
  .engine-chip { font-size: 8px; padding: 7px 9px; }
  .version-label { font-size: 8px; }
  .welcome-row { padding: 22px 3px 16px; align-items: flex-start; }
  .welcome-row h1 { font-size: 23px; }
  .welcome-row p { font-size: 12px; max-width: 260px; }
  .workspace-tag { font-size: 8px; padding: 8px; }
  .app-tabs [role=tablist] { flex-wrap: nowrap !important; overflow-x: auto !important; }
  .app-tabs button[role=tab] { flex: 0 0 auto !important; font-size: 10px !important; padding: 10px 9px !important; white-space: nowrap !important; }
  .panel { padding: 14px !important; border-radius: 13px !important; }
  .section-title { font-size: 16px; }
  .video-input video { max-height: 260px !important; }
  .metric-card { padding: 10px !important; }
  .metric-card input { font-size: 24px !important; }
  .metric-card label { font-size: 8px !important; }
  .select-row > * { flex-basis: 100% !important; }
  .app-footer { font-size: 8px; flex-direction: column; }
  .video-actions button { font-size: 11px !important; }
}
@media (max-width: 380px) {
  .brand-subtitle, .version-label { display: none; }
  .header-meta { align-self: flex-start; }
  .welcome-row { flex-direction: column; }
  .workspace-tag { align-self: flex-start; }
  .app-tabs button[role=tab] { padding: 9px 7px !important; font-size: 9px !important; }
  .quick-facts { grid-template-columns: 1fr; }
}
"""
