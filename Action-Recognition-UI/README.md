# Human Action Detection UI

Standalone HTML/CSS/JavaScript dashboard. Gradio is intentionally not connected yet.

- HTML: structure
- CSS: dark/light themes and responsive layout
- JavaScript: page state, browser camera/video controls, and the future bridge contract

The future Gradio integration should call `window.ActionUI.setState(payload)`. The UI exposes `window.ActionUI.getState()` and `setTheme()`.

## Local preview
`python -m http.server 4173`, then open http://localhost:4173/Action-Recognition-UI/

Camera access requires HTTPS or localhost.

## Visual QA
The GitHub Actions workflow opens Chromium at 1280x1170 and 390x844, captures screenshots, tests the theme toggle, state bridge and horizontal overflow, then uploads screenshots as an artifact.
