"""Optional Chromium interaction checks against local, explicitly synthetic UI fixtures.

Set RIFT_TEST_BROWSER to a Chromium/headless-shell executable to run these checks.
No Riot requests, training, Node tooling or browser downloads are involved.
"""

import html
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from offline_rift.ui.server import DashboardServer
from rift_common.pipeline.actions import ACTION_NAMES

BROWSER = os.environ.get("RIFT_TEST_BROWSER")

CHECKS = r"""
const browserErrors = [];
window.addEventListener('error', event => browserErrors.push(event.message));
window.addEventListener('unhandledrejection', event => browserErrors.push(String(event.reason)));
(async () => {
  const check = (condition, message) => { if (!condition) throw Error(message); };
  const wait = async (condition) => {
    for (let i = 0; i < 100; i++) {
      if (condition()) return;
      await new Promise(resolve => setTimeout(resolve, 20));
    }
    throw Error('Timed out waiting for UI state');
  };
  const originalFetch = window.fetch;
  const result = document.createElement('pre');
  result.id = 'browser-test-result';
  result.hidden = true;
  try {
    await wait(() => document.querySelector('.decision.active'));
    check(document.getElementById('decision-title').textContent === 'Group mid', 'Blue perspective');
    check(document.querySelectorAll('.alternative-row').length === 3, 'Ranked alternatives');
    check(document.querySelectorAll('.action-row').length === 12, 'All score rows');
    check(document.getElementById('model-kind').textContent === __MODEL_LABEL__, 'Loaded model identity');
    check(document.documentElement.scrollWidth <= window.innerWidth, 'Horizontal overflow');
    const red = document.querySelector('input[value="200"]');
    red.checked = true;
    red.dispatchEvent(new Event('change'));
    check(document.getElementById('decision-title').textContent === 'Own-half defensive formation', 'Red perspective');
    document.getElementById('focus').click();
    check(getComputedStyle(document.querySelector('.secondary')).display === 'none', 'Focus hides secondary panels');
    check(document.getElementById('focus').getAttribute('aria-pressed') === 'true', 'Focus accessible state');
    document.getElementById('focus').click();
    check(getComputedStyle(document.querySelector('.secondary')).display !== 'none', 'Full view restored');
    document.getElementById('pause').click();
    check(document.getElementById('decision-tag').textContent === 'PAUSED', 'Paused state');
    check(document.querySelectorAll('.action-row.selected').length === 0, 'Pause clears scores');
    check(document.querySelectorAll('.alternative-row').length === 0, 'Pause clears alternatives');
    document.getElementById('pause').click();
    await wait(() => document.querySelector('.decision.active'));
    check(document.getElementById('decision-title').textContent === 'Own-half defensive formation', 'Explicit team survives refresh');
    window.fetch = async (...args) => {
      const response = await originalFetch(...args);
      const data = await response.json();
      data.valid_for_ms = 100;
      return { ok: true, json: async () => data };
    };
    document.getElementById('pause').click();
    document.getElementById('pause').click();
    await wait(() => document.querySelector('.decision.active'));
    await wait(() => document.getElementById('decision-tag').textContent === 'UPDATING');
    check(document.getElementById('game-clock').textContent === '--:--', 'Expired clock cleared');
    check(document.querySelectorAll('.action-row.selected').length === 0, 'Expired scores cleared');
    check(document.querySelectorAll('.alternative-row').length === 0, 'Expired alternatives cleared');
    window.fetch = async () => { throw Error('Test disconnection'); };
    document.getElementById('pause').click();
    document.getElementById('pause').click();
    await wait(() => document.getElementById('decision-tag').textContent === 'OFFLINE');
    check(document.querySelectorAll('.action-row.selected').length === 0, 'Failure clears scores');
    window.fetch = originalFetch;
    document.getElementById('pause').click();
    document.getElementById('pause').click();
    await wait(() => document.querySelector('.decision.active'));
    document.querySelector('input[value="100"]').click();
    document.getElementById('focus').dispatchEvent(new KeyboardEvent('keydown', { key: 'f', bubbles: true }));
    check(document.body.classList.contains('focus'), 'Focus keyboard shortcut');
    document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'f', bubbles: true }));
    check(!document.body.classList.contains('focus'), 'Full view keyboard shortcut');
    check(browserErrors.length === 0, browserErrors.join('; '));
    result.textContent = JSON.stringify({ passed: true, width: innerWidth });
  } catch (error) {
    result.textContent = JSON.stringify({ passed: false, message: String(error), errors: browserErrors });
  }
  document.body.append(result);
})();
"""


@pytest.mark.skipif(
    not BROWSER, reason="Set RIFT_TEST_BROWSER for optional browser interaction checks"
)
@pytest.mark.parametrize("width,height", [(1365, 1050), (390, 844)])
@pytest.mark.parametrize(
    "mode,label",
    [("balanced_bc", "Neural classifier"), ("hybrid", "CQL + neural model"), ("cql", "CQL model")],
)
def test_dashboard_browser_interactions(tmp_path, width, height, mode, label):
    def state():
        return {
            "status": "live",
            "checked_at": time.time(),
            "valid_for_ms": 30000,
            "interval_seconds": 30,
            "default_team": 100,
            "transport": "local",
            "actions": ACTION_NAMES,
            "model": {
                "prediction_method": mode,
                "action_counts": [10] * 12,
                "prediction_counts": [10] * 12,
                "action_coverage": 12,
                "collapsed": False,
            },
            "recommendations": {
                str(team): {
                    "action_id": action,
                    "action": ACTION_NAMES[action],
                    "game_time": 1285,
                    "macro_scores": [5.0 if i == action else -i / 10 for i in range(12)]
                    if mode != "cql"
                    else None,
                    "q_values": [5.0 if i == action else -i / 10 for i in range(12)]
                    if mode != "balanced_bc"
                    else None,
                }
                for team, action in ((100, 5), (200, 8))
            },
        }

    with DashboardServer(SimpleNamespace(state=state), 0) as server:
        source, mime = server.assets["/app.js"]
        server.assets["/app.js"] = (
            source + CHECKS.replace("__MODEL_LABEL__", json.dumps(label)).encode(),
            mime,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = subprocess.run(
                [
                    BROWSER,
                    "--headless",
                    "--no-sandbox",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--no-proxy-server",
                    "--hide-scrollbars",
                    f"--user-data-dir={tmp_path / 'browser-profile'}",
                    f"--window-size={width},{height}",
                    "--virtual-time-budget=6000",
                    f"--screenshot={tmp_path / 'dashboard.png'}",
                    "--dump-dom",
                    f"http://127.0.0.1:{server.server_port}",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, result.stderr
            match = re.search(r'<pre id="browser-test-result"[^>]*>(.*?)</pre>', result.stdout)
            assert match, result.stdout[-2500:] + result.stderr[-1500:]
            report = json.loads(html.unescape(match[1]))
            assert report["passed"], report
            assert Path(tmp_path / "dashboard.png").is_file()
        finally:
            server.shutdown()
            thread.join(timeout=5)
