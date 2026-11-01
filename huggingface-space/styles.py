
CSS = r"""
/* ==========================================
   ACTION DETECTOR - DESIGN SYSTEM
   ========================================== */

:root {
    --bg: #020817;
    --bg-secondary: #07132d;
    --panel: #07142b;
    --panel-hover: #0b1c3c;

    --blue: #087bff;
    --cyan: #00d9ff;
    --purple: #a000ff;
    --green: #00e6a0;
    --red: #ff2870;
    --orange: #ffaa00;

    --text: #e8f2ff;
    --muted: #91a9ce;

    --border: rgba(0, 119, 255, 0.65);
    --glow: 0 0 9px rgba(0, 119, 255, 0.25);

    --radius: 13px;
}

/* ==========================================
   GLOBAL
   ========================================== */

.gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: Inter, Arial, sans-serif !important;
    max-width: 100% !important;
    padding: 12px !important;
}

body {
    background: var(--bg) !important;
    color: var(--text) !important;
}

* {
    box-sizing: border-box;
}

footer {
    display: none !important;
}

/* ==========================================
   HEADER
   ========================================== */

.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 15px;
    padding: 14px 20px;
    margin-bottom: 15px;

    background: linear-gradient(
        100deg,
        #07132d,
        #020817
    );

    border-bottom: 1px solid var(--border);
    border-radius: 10px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 13px;
}

.brand-icon {
    font-size: 36px;
    color: var(--purple);
}

.brand-title {
    font-size: 27px;
    font-weight: 800;
    letter-spacing: 0.5px;
    color: white;
}

.brand-title span {
    color: var(--cyan);
}

.brand-subtitle {
    color: #9db7df;
    font-size: 13px;
    margin-top: 4px;
}

.system-status {
    color: var(--green);
    font-size: 13px;
    font-weight: 600;
}

/* ==========================================
   MAIN LAYOUT
   ========================================== */

.main-layout {
    display: grid;
    grid-template-columns: 175px minmax(0, 1fr);
    gap: 16px;
    align-items: start;
}

.sidebar {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 10px 0;
}

.sidebar-item {
    padding: 13px 14px;
    border-radius: 9px;
    color: var(--muted);
    background: transparent;
    border: 1px solid transparent;
    font-size: 14px;
}

.sidebar-item.active {
    color: white;
    background: linear-gradient(
        100deg,
        #064acb,
        #09236b
    );
    border: 1px solid var(--cyan);
    box-shadow: var(--glow);
}

.dashboard-content {
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 15px;
}

/* ==========================================
   PANELS
   ========================================== */

.panel {
    background: linear-gradient(
        145deg,
        #07152e,
        #030b1d
    ) !important;

    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;

    padding: 15px !important;
    box-shadow: var(--glow);
    min-width: 0;
}

.panel-title {
    color: #d9eaff !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    margin-bottom: 12px !important;
}

/* ==========================================
   VIDEO PANEL
   ========================================== */

.video-panel {
    border: 1px solid var(--blue) !important;
    border-radius: 13px !important;
    padding: 12px !important;
    background: #030b1c !important;
    box-shadow: 0 0 12px rgba(0, 100, 255, 0.2);
}

.video-frame {
    width: 100%;
    aspect-ratio: 16 / 9;
    background: #000;
    overflow: hidden;
    border-radius: 9px;
}

.video-frame video,
.video-frame img {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

/* ==========================================
   STATISTICS CARDS
   ========================================== */

.stats-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
}

.stat-card {
    min-height: 85px;
    display: flex;
    align-items: center;
    gap: 13px;

    padding: 13px;
    border-radius: 13px;

    background: #071632;
    border: 1px solid var(--blue);
}

.stat-card .stat-icon {
    font-size: 27px;
}

.stat-card .stat-value {
    font-size: 25px;
    font-weight: 800;
}

.stat-card .stat-label {
    font-size: 12px;
    color: var(--muted);
}

.stat-blue {
    border-color: var(--blue);
    color: var(--cyan);
}

.stat-purple {
    border-color: var(--purple);
    color: #c18aff;
}

.stat-red {
    border-color: var(--red);
    color: #ff5b91;
}

.stat-green {
    border-color: var(--green);
    color: var(--green);
}

/* ==========================================
   ACTION PROGRESS
   ========================================== */

.action-list {
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.action-item {
    display: grid;
    grid-template-columns: 30px minmax(0, 1fr);
    gap: 10px;
    align-items: center;
}

.action-icon {
    font-size: 22px;
}

.action-info {
    min-width: 0;
}

.action-info-header {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    font-size: 13px;
    margin-bottom: 7px;
}

.progress-track {
    height: 8px;
    background: #102650;
    border-radius: 20px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 20px;
}

.progress-blue {
    background: var(--cyan);
}

.progress-green {
    background: var(--green);
}

.progress-orange {
    background: var(--orange);
}

.progress-red {
    background: var(--red);
}

/* ==========================================
   BUTTONS
   ========================================== */

button.primary,
.primary-btn {
    background: linear-gradient(
        100deg,
        #064cdb,
        #087bff
    ) !important;

    border: 1px solid var(--cyan) !important;
    color: white !important;
    border-radius: 9px !important;
}

button.secondary,
.secondary-btn {
    background: #091a3b !important;
    border: 1px solid #174c9c !important;
    color: #dceaff !important;
    border-radius: 9px !important;
}

button.danger,
.danger-btn {
    background: #310b29 !important;
    border: 1px solid var(--red) !important;
    color: #ff6095 !important;
    border-radius: 9px !important;
}

button {
    transition: background-color 0.15s ease,
                border-color 0.15s ease !important;
}

button:hover {
    border-color: var(--cyan) !important;
}

/* ==========================================
   INPUTS
   ========================================== */

input,
textarea,
select,
.gradio-container input {
    background: #091a3b !important;
    color: white !important;
    border: 1px solid #164b91 !important;
    border-radius: 8px !important;
}

label {
    color: #c8dcfa !important;
}

/* ==========================================
   VIDEO CONTROLS
   ========================================== */

.controls-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 8px;
}

.control-button {
    min-height: 70px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    background: #091a3b;
    border: 1px solid #164b91;
    border-radius: 9px;
    color: white;
}

/* ==========================================
   CHARTS
   ========================================== */

.chart-panel {
    min-height: 260px;
    overflow: hidden;
}

/* ==========================================
   EVENTS
   ========================================== */

.event-list {
    display: flex;
    flex-direction: column;
    gap: 0;
}

.event-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    padding: 12px 5px;
    border-bottom: 1px solid #142b50;
}

.event-name {
    font-size: 14px;
    color: white;
}

.event-time {
    color: var(--muted);
    font-size: 12px;
}

/* ==========================================
   ACTION BUTTONS
   ========================================== */

.action-buttons {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
}

/* ==========================================
   RESPONSIVE - TABLETS
   ========================================== */

@media (max-width: 900px) {

    .main-layout {
        grid-template-columns: 1fr;
    }

    .sidebar {
        flex-direction: row;
        overflow-x: auto;
        padding-bottom: 5px;
    }

    .sidebar-item {
        white-space: nowrap;
        flex-shrink: 0;
    }

    .dashboard-content {
        width: 100%;
    }
}

/* ==========================================
   RESPONSIVE - PHONES
   ========================================== */

@media (max-width: 600px) {

    .gradio-container {
        padding: 7px !important;
    }

    .app-header {
        padding: 10px;
        gap: 8px;
    }

    .brand-icon {
        font-size: 25px;
    }

    .brand-title {
        font-size: 18px;
    }

    .brand-subtitle {
        font-size: 10px;
    }

    .system-status {
        font-size: 10px;
    }

    .sidebar {
        gap: 6px;
    }

    .sidebar-item {
        padding: 10px;
        font-size: 12px;
    }

    .panel {
        padding: 10px !important;
    }

    .stats-grid {
        gap: 7px;
    }

    .stat-card {
        min-height: 72px;
        padding: 9px;
        gap: 7px;
    }

    .stat-card .stat-icon {
        font-size: 21px;
    }

    .stat-card .stat-value {
        font-size: 20px;
    }

    .stat-card .stat-label {
        font-size: 10px;
    }

    .controls-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .action-buttons {
        grid-template-columns: 1fr;
    }

    .chart-panel {
        min-height: 220px;
    }

    .panel-title {
        font-size: 14px !important;
    }

}

/* ==========================================
   SMALL PHONES
   ========================================== */

@media (max-width: 360px) {

    .brand-title {
        font-size: 15px;
    }

    .brand-subtitle {
        display: none;
    }

    .stat-card .stat-value {
        font-size: 17px;
    }

    .sidebar-item {
        padding: 8px;
    }

}
"""
