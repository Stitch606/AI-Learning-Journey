CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');
:root{--bg:#100d1c;--surface:#191528;--panel:#201b32;--panel-hi:#28213d;--line:#39304f;--text:#f8f5ff;--muted:#aaa2bd;--purple:#b77cff;--purple2:#8d5cf6;--cyan:#68e5ee;--pink:#ff83c8;--green:#66e0b0;--amber:#f5c879}
*{box-sizing:border-box}
html,body{background:var(--bg)!important;color:var(--text)!important}
body{overflow-x:hidden!important}
.gradio-container{max-width:1540px!important;width:100%!important;margin:auto!important;padding:22px clamp(12px,2.5vw,36px) 30px!important;background:radial-gradient(ellipse at 5% 0%,#6840a52b,transparent 36%),radial-gradient(ellipse at 95% 8%,#2c9fa51b,transparent 30%),linear-gradient(150deg,#151123,#100d1c 65%,#171225)!important;color:var(--text)!important;font-family:'DM Sans',sans-serif!important}
footer{display:none!important}
.action-app,.action-app *{min-width:0}
.hero-header{display:flex;align-items:center;justify-content:space-between;gap:20px;flex-wrap:wrap;padding:22px 26px;border:1px solid #55466f;border-radius:18px;background:linear-gradient(110deg,#28213c,#211a35 60%,#1d2839);box-shadow:0 14px 36px #08061180,inset 0 1px #ffffff0c}
.hero-brand{display:flex;align-items:center;gap:15px}
.brand-symbol{width:54px;height:54px;display:grid;place-items:center;position:relative;flex-shrink:0;border:1px solid #b58aff;border-radius:15px;background:linear-gradient(145deg,#a77aff,#6240b4);box-shadow:0 0 24px #9d71ff55}
.brand-symbol span{font:700 16px 'Space Mono';color:white}.brand-symbol i{position:absolute;width:7px;height:7px;right:8px;top:8px;border-radius:50%;background:var(--cyan);box-shadow:0 0 10px var(--cyan)}
.eyebrow{font:700 9px 'Space Mono',monospace;letter-spacing:1.4px;color:#bca9df}
.brand-title{font-size:clamp(22px,3vw,32px);font-weight:800;letter-spacing:.4px;line-height:1.2;color:#fff}.brand-title span{color:#c9a9ff}
.brand-subtitle{font-size:12px;color:#bcb5cc;margin-top:6px}
.header-meta{display:flex;flex-direction:column;align-items:flex-end;gap:9px}
.engine-chip{display:flex;align-items:center;gap:8px;border:1px solid #4d705f;background:#1d352e;border-radius:30px;padding:8px 12px;color:#b8f4d7;font:700 9px 'Space Mono'}
.pulse-dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 9px var(--green)}
.version-label{font:9px 'Space Mono';letter-spacing:.8px;color:#9e96b0}.version-label b{color:#ddd3f0}
.welcome-row{display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:15px;padding:27px 4px 18px}
.welcome-row h1{font-size:clamp(23px,3vw,30px);font-weight:800;margin:7px 0 4px;color:#fff}.welcome-row p{margin:0;color:var(--muted);font-size:13px}
.workspace-tag{font:9px 'Space Mono';letter-spacing:.7px;color:#d7d0e5;border:1px solid #443b59;background:#211b32;padding:9px 12px;border-radius:8px;white-space:nowrap}
.tiny-square{display:inline-block;width:7px;height:7px;background:var(--cyan);margin-right:8px;border-radius:2px;box-shadow:0 0 8px #68e5ee88}
.app-tabs{gap:14px!important}
.app-tabs [role=tablist]{display:flex!important;flex-wrap:wrap!important;gap:7px!important;margin-bottom:17px!important;padding:6px!important;border:1px solid #39324b!important;border-radius:12px!important;background:#191528!important}
.app-tabs button[role=tab]{color:#aaa2bd!important;border:1px solid transparent!important;border-radius:8px!important;background:transparent!important;padding:10px 15px!important;font-size:12px!important;font-weight:700!important}
.app-tabs button[role=tab]:hover{background:#2c2540!important;color:white!important}
.app-tabs button[role=tab][aria-selected=true]{color:#fff!important;background:linear-gradient(110deg,#8054d7,#6746b2)!important;border-color:#a88ae9!important;box-shadow:0 5px 18px #714dc944!important}
.panel{width:100%!important;min-width:0!important;background:linear-gradient(145deg,#211c33,#1b172b)!important;border:1px solid #3b334f!important;border-radius:15px!important;padding:19px!important;box-shadow:0 9px 25px #08061155,inset 0 1px #ffffff08!important}
.stage-grid,.lower-grid,.analytics-grid{align-items:stretch!important;gap:16px!important;min-width:0!important}
.stage-grid>*,.lower-grid>*,.analytics-grid>*{min-width:0!important}
.panel-topline,.section-heading{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:15px}
.section-title{font-size:18px;font-weight:800;color:#f8f5ff;margin-top:5px;letter-spacing:-.3px}
.state-pill{font:700 9px 'Space Mono';color:#a9f0d1;border:1px solid #3e745d;background:#1b352b;padding:6px 9px;border-radius:6px}
.metric-caption{font:9px 'Space Mono';letter-spacing:.8px;color:#938aa7}
.video-panel{border-color:#51446c!important;background:linear-gradient(145deg,#242039,#1b172b)!important}
.video-input{width:100%!important;border:1px solid #51466b!important;border-radius:12px!important;overflow:hidden!important;background:#100d1c!important}
.video-input video{width:100%!important;max-height:490px!important;object-fit:contain!important;background:#100d1c!important}
.status-message{font-size:12px!important;color:#e8e1f3!important;background:#28223b!important;border:1px solid #49405e!important;border-left:3px solid var(--purple)!important;border-radius:8px!important;padding:11px 13px!important;margin:13px 0!important}
.status-message p{margin:0!important}
.video-actions{display:flex!important;flex-wrap:wrap!important;gap:10px!important;margin:11px 0 6px!important}.video-actions>*{flex:1 1 160px!important}
button{min-height:42px!important;border-radius:9px!important;font-family:'DM Sans',sans-serif!important;font-size:12px!important;font-weight:700!important;transition:filter .16s ease,transform .16s ease,box-shadow .16s ease!important}
button:hover{filter:brightness(1.12)!important;transform:translateY(-1px)!important}
button.primary,.start-action{background:linear-gradient(110deg,#9468e9,#704ac1)!important;border:1px solid #bba0f4!important;color:white!important;box-shadow:0 5px 16px #7653c533!important}
button.secondary,.stop-action{background:#302942!important;border:1px solid #514663!important;color:#e8e1f3!important}
.control-rail{background:linear-gradient(155deg,#242039,#1d192e)!important}
.control-rail input,.control-rail textarea,.control-rail select,.control-rail [role=combobox]{background:#28223b!important;color:#f8f5ff!important;border:1px solid #514663!important;border-radius:8px!important}
.control-rail .wrap{margin-bottom:15px!important}
.control-rail label,.progress-control label,.action-panel label{font:700 9px 'Space Mono'!important;letter-spacing:.6px!important;color:#b9b0ca!important}
.rail-divider{height:1px;background:#393149;margin:18px 0 15px}
.rail-actions{gap:8px!important;margin-top:9px!important}.rail-actions button{width:100%!important}
.download-action{width:100%!important;background:#302942!important;border:1px solid #514663!important;color:#e8e1f3!important}
.rail-tip{display:flex;gap:10px;align-items:flex-start;margin-top:17px;padding:12px;border:1px solid #48405a;border-radius:9px;background:#252039}
.tip-mark{font-size:18px;color:#c6a5ff}.rail-tip b{font-size:11px;color:#f8f5ff}.rail-tip p{font-size:10px;line-height:1.55;color:#aaa2bd;margin:4px 0 0}
.metrics-section{margin-top:16px;padding:17px 18px 18px;border:1px solid #39324b;border-radius:15px;background:#1b172b;box-shadow:0 8px 22px #08061140}
.section-heading{margin-bottom:12px}.metric-strip{display:flex!important;gap:11px!important;align-items:stretch!important}.metric-strip>*{flex:1 1 0!important;min-width:0!important}
.metric-card{background:linear-gradient(145deg,#28223b,#211c32)!important;border:1px solid #443b58!important;border-radius:11px!important;padding:13px!important}
.metric-card label{font:700 8px 'Space Mono'!important;letter-spacing:.5px!important;color:#a9a1bb!important}
.metric-card input{font-size:clamp(23px,2.2vw,29px)!important;font-weight:800!important;color:#fff!important;padding:3px 0!important;background:transparent!important;border:0!important}
.metric-cyan{border-top:2px solid var(--cyan)!important}.metric-violet{border-top:2px solid var(--purple)!important}.metric-rose{border-top:2px solid var(--pink)!important}.metric-amber{border-top:2px solid var(--amber)!important}
.metric-cyan input{color:#8debf0!important}.metric-violet input{color:#c5a6ff!important}.metric-rose input{color:#ff9ed2!important}.metric-amber input{color:#f6d18c!important}
.lower-grid{margin-top:16px!important}.action-panel .wrap{margin-bottom:13px!important}
.session-info-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
.session-info{padding:13px;border:1px solid #403850;border-radius:9px;background:#252039}
.session-info span,.session-info b,.session-info small{display:block}.session-info span{font:9px 'Space Mono';letter-spacing:.7px;color:#938aa7}.session-info b{font-size:12px;color:#eee8f7;margin-top:6px}.session-info small{font-size:10px;color:#9f97b1;margin-top:4px}
.analytics-grid>*{flex:1 1 360px!important}.gradio-plot{background:transparent!important;min-width:0!important;width:100%!important;border:0!important}
.gradio-plot .plot-container{max-width:100%!important;border-radius:9px!important;overflow:hidden!important}
.dataframe{max-width:100%!important;overflow-x:auto!important;border:1px solid #403850!important;border-radius:9px!important}
.dataframe table{min-width:420px!important;background:#211c32!important;color:#eee8f7!important}
.dataframe th{background:#302942!important;color:#eee8f7!important;font:700 9px 'Space Mono'!important}
.dataframe td{border-color:#393149!important;font-size:11px!important}.empty-events{color:#aaa2bd!important;font-size:12px!important}
.settings-panel{max-width:950px}.settings-panel .prose{color:#c4bdcf!important}.settings-panel strong{color:#fff}
.app-footer{display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;padding:21px 4px 5px;color:#817991;font:9px 'Space Mono';letter-spacing:.6px}
.app-footer b,.app-footer i{color:#b99bfa;font-style:normal}
input[type=range]{accent-color:#aa83f4!important}
@media(max-width:1050px){.stage-grid,.lower-grid,.analytics-grid{flex-direction:column!important}.stage-grid>*,.lower-grid>*,.analytics-grid>*{width:100%!important;max-width:100%!important;flex:1 1 auto!important}.video-input video{max-height:380px!important}}
@media(max-width:700px){
.gradio-container{padding:11px 9px 17px!important}.hero-header{padding:14px;border-radius:13px;align-items:flex-start}.hero-brand{gap:10px}.brand-symbol{width:43px;height:43px;border-radius:11px}.brand-title{font-size:19px}.brand-subtitle{font-size:10px;max-width:215px}.eyebrow{font-size:8px}.header-meta{align-items:flex-start;gap:6px}.engine-chip{font-size:8px;padding:6px 8px}.version-label{font-size:8px}
.welcome-row{padding:19px 2px 14px;align-items:flex-start}.welcome-row h1{font-size:22px}.welcome-row p{font-size:11px;max-width:260px}.workspace-tag{font-size:8px;padding:8px}
.app-tabs [role=tablist]{flex-wrap:nowrap!important;overflow-x:auto!important}.app-tabs button[role=tab]{flex:0 0 auto!important;font-size:10px!important;padding:9px 10px!important;white-space:nowrap!important}
.panel{padding:13px!important;border-radius:12px!important}.section-title{font-size:16px}.video-input video{max-height:270px!important}
.metric-strip{display:grid!important;grid-template-columns:repeat(2,minmax(0,1fr))!important}.metric-card{padding:10px!important}.metric-card input{font-size:23px!important}.metric-card label{font-size:8px!important}.metrics-section{padding:13px!important}
.session-info-grid{grid-template-columns:1fr 1fr}.app-footer{font-size:8px;flex-direction:column}
}
@media(max-width:390px){.brand-subtitle,.version-label{display:none}.welcome-row{flex-direction:column}.workspace-tag{align-self:flex-start}.app-tabs button[role=tab]{padding:8px 7px!important;font-size:9px!important}.session-info-grid{grid-template-columns:1fr}}
/* Video widget surface */
#video-stage{width:100%!important;min-height:360px!important;padding:0!important;border:1px solid #51466b!important;border-radius:12px!important;overflow:hidden!important;background:radial-gradient(ellipse at 50% 0%,#65449a26,transparent 65%),#100d1c!important;box-shadow:inset 0 0 28px #9d71ff0d!important}
#video-stage>div,#video-stage .wrap,#video-stage .video-container,#video-stage .upload-container,#video-stage .empty,#video-stage [data-testid="video"],#video-stage [data-testid="video-upload"],#video-stage .drop-zone{background-color:transparent!important;color:#eee8f7!important;border-color:#51466b!important;box-shadow:none!important}
#video-stage .upload-container,#video-stage .drop-zone,#video-stage [data-testid="video-upload"]{min-height:330px!important;border:1px dashed #695a86!important;border-radius:10px!important;background:radial-gradient(ellipse at center,#7954b01c,transparent 70%),#171326!important}
#video-stage button,#video-stage [role=button]{color:#f8f5ff!important;background:#302942!important;border:1px solid #665580!important;border-radius:8px!important}
#video-stage svg{color:#c6a5ff!important}#video-stage p,#video-stage span,#video-stage label,#video-stage small{color:#c8bfd8!important}
#video-stage video{display:block!important;width:100%!important;max-height:460px!important;object-fit:contain!important;background:#100d1c!important;border-radius:10px!important}
@media(max-width:700px){#video-stage{min-height:260px!important}#video-stage .upload-container,#video-stage .drop-zone,#video-stage [data-testid="video-upload"]{min-height:245px!important}#video-stage video{max-height:270px!important}}
"""