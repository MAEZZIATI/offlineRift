"""Optional compiled HUD checks against a synthetic local producer, never live data."""

import json
import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from offline_rift.ui.server import HudServer

BROWSER = os.environ.get("RIFT_TEST_BROWSER")
NODE = os.environ.get("RIFT_TEST_NODE", "node")
BRIDGE = r"""
let receive;
window.fixtureReceive = (frame) => receive?.(frame);
window.riftDesktop = {
  onFrame(callback) {
    receive = callback;
    fetch('/api/hud').then(r => r.json()).then(frame => receive?.(frame));
    return () => {receive = null;};
  },
  onToggleDetails: () => () => {},
  onInteraction: () => () => {},
  configure: async (context) => {
    const response = await fetch('/api/context', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(context)});
    if (!response.ok) throw Error('Selection rejected');
    return response.json();
  },
  openSettings: () => location.assign('/?settings=1'),
};
"""
CHECKS = r"""
await page.goto(url);
await page.locator('.hud-action-title').waitFor();
assert.equal(await page.locator('.hud-action-title').textContent(), 'Rotate toward mid');
assert.equal((await page.locator('body').textContent()).includes('+14.2%'), false);
await page.getByText('Choose your role in HUD settings').waitFor();
await page.getByLabel('Open coach settings').click();
await page.locator('.runtime-settings select').first().waitFor();
assert.equal((await page.locator('.runtime-model').textContent()).includes('6 POLICIES'), true);
assert.equal(await page.locator('.runtime-policies > div').count(), 6);
await page.getByLabel('Your role', {exact:true}).selectOption('JUNGLE');
await page.waitForFunction(() => !document.querySelectorAll('select')[1].disabled);
await page.goto(url);
await page.getByText('Clear jungle camps', {exact:true}).waitFor();
await page.locator('.hud-expand').click();
await page.getByText('Uncalibrated', {exact:true}).waitFor();
assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
await page.evaluate(() => window.fixtureReceive({schema:'macro-hud-frame-v1',session_id:'fixture',context_version:2,status:'waiting_for_game',message:'Waiting for League of Legends',detail:'No game',hud:null,context:{team:100,actor:'active',role:null},players:[],model:{display_name:'fixture-coach',fingerprint:'abc',policy_count:6,policies:{}}}));
await page.getByRole('heading', {name:'Waiting for League of Legends'}).waitFor();
assert.equal(await page.locator('.hud-action-title').count(), 0);
"""


@pytest.mark.skipif(not BROWSER, reason="Set RIFT_TEST_BROWSER to a Chromium executable")
@pytest.mark.parametrize("width,height", [(1365, 1050), (390, 844)])
def test_compiled_hud_model_role_and_withdrawal(tmp_path, width, height):
    role = [None]
    version = [0]

    def frame():
        return {
            "schema": "macro-hud-frame-v1",
            "session_id": "fixture",
            "context_version": version[0],
            "status": "live",
            "message": "",
            "detail": "",
            "players": [],
            "context": {"team": 100, "actor": "active", "role": role[0]},
            "model": {
                "display_name": "fixture-coach",
                "fingerprint": "abc",
                "policy_count": 6,
                "policies": {
                    scope: {"algorithm": "bc", "candidate": "fixture"}
                    for scope in ("TEAM", "TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
                },
            },
            "hud": {
                "timestamp": "18:42",
                "urgency": "normal",
                "valid_for_ms": 6000,
                "macro": {"title": "Rotate toward mid", "subtext": "Check the map."},
                "role": {
                    "player_role": role[0],
                    "action": "Clear jungle camps",
                    "details": "Check available information.",
                }
                if role[0]
                else None,
                "notice": None if role[0] else "Choose your role in HUD settings",
                "telemetry": {"ai_confidence": None, "state_hash": "abc"},
                "coordination": "independent",
            },
        }

    def configure(context):
        role[0] = context["role"]
        version[0] += 1
        return frame()

    with HudServer(SimpleNamespace(frame=frame, configure=configure), 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            playwright = (
                Path(__file__).resolve().parents[1]
                / "production/frontend/macro-hud/node_modules/playwright-core/index.mjs"
            )
            # Use the real browser clock. Chromium's --virtual-time-budget can run
            # timer-based assertions before the store's animation frame is painted.
            script = f"""
import {{ chromium }} from {json.dumps(playwright.as_uri())};
import assert from "node:assert/strict";
const [executablePath, url, screenshot, width, height] = process.argv.slice(2);
const browser = await chromium.launch({{
  executablePath, headless: true,
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--no-proxy-server'],
}});
try {{
  const page = await browser.newPage({{viewport: {{width: +width, height: +height}}}});
  page.setDefaultTimeout(5000);
  await page.addInitScript({json.dumps(BRIDGE)});
  {CHECKS}
  console.log(JSON.stringify({{passed: true}}));
  await page.screenshot({{path: screenshot}});
}} finally {{ await browser.close(); }}
"""
            result = subprocess.run(
                [
                    NODE,
                    "--input-type=module",
                    "-",
                    BROWSER,
                    f"http://127.0.0.1:{server.server_port}",
                    str(tmp_path / "hud.png"),
                    str(width),
                    str(height),
                ],
                input=script,
                capture_output=True,
                text=True,
                timeout=40,
                check=False,
            )
            assert result.returncode == 0, result.stderr
            report = json.loads(result.stdout)
            assert report["passed"], report
            assert (tmp_path / "hud.png").exists()
        finally:
            server.shutdown()
            thread.join(timeout=5)
