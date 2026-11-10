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
CHECKS = r"""
(async () => {
 const report = document.createElement('pre'); report.id='browser-result'; report.hidden=true;
 const wait=async(fn)=>{for(let i=0;i<200;i++){if(fn())return;await new Promise(r=>setTimeout(r,25));}throw Error('HUD timeout: '+document.body.textContent);};
 const check=(ok,message)=>{if(!ok)throw Error(message);};
 try {
  await wait(()=>document.querySelector('.hud-action-title'));
  check(document.querySelector('.hud-action-title').textContent==='Rotate toward mid','Actual macro payload');
  check(!document.body.textContent.includes('+14.2%'),'No fabricated impact');
  check(document.body.textContent.includes('Choose your role'),'Missing role is explicit');
  document.querySelector('[aria-label="Open coach settings"]').click();
  await wait(()=>document.querySelector('.runtime-settings select'));
  check(document.querySelector('.runtime-model').textContent.includes('6 POLICIES'),'Selected six-policy model');
  check(document.querySelectorAll('.runtime-policies > div').length===6,'All six policies');
  const select=document.querySelectorAll('select')[1];
  select.value='JUNGLE';select.dispatchEvent(new Event('change',{bubbles:true}));
  await wait(()=>document.querySelectorAll('select')[1].value==='JUNGLE');
  document.querySelector('[aria-label="Close settings"]').click();
  await wait(()=>document.querySelector('.hud-role-action')?.textContent==='Clear jungle camps');
  document.querySelector('.hud-expand').click();
  await wait(()=>document.querySelector('.hud-confidence'));
  check(document.querySelector('.hud-confidence').textContent.includes('Uncalibrated'),'No fake confidence');
  check(document.documentElement.scrollWidth<=innerWidth,'No horizontal overflow');
  window.fetch=async()=>new Response(JSON.stringify({schema:'macro-hud-frame-v1',session_id:'fixture',context_version:2,status:'waiting_for_game',message:'Waiting for League of Legends',detail:'No game',hud:null,context:{team:100,actor:'active',role:null},players:[],model:{display_name:'fixture-coach',fingerprint:'abc',policy_count:6,policies:{}}}),{status:200,headers:{'Content-Type':'application/json'}});
  await wait(()=>document.querySelector('.hud-empty h2')?.textContent==='Waiting for League of Legends');
  check(!document.querySelector('.hud-action-title'),'Advice withdrawn when game unavailable');
  report.textContent=JSON.stringify({passed:true});
 }catch(error){report.textContent=JSON.stringify({passed:false,error:String(error)});}
 document.body.append(report);
})();
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
const [executablePath, url, screenshot, width, height] = process.argv.slice(2);
const browser = await chromium.launch({{
  executablePath, headless: true,
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--no-proxy-server'],
}});
try {{
  const page = await browser.newPage({{viewport: {{width: +width, height: +height}}}});
  await page.goto(url);
  await page.evaluate({json.dumps(CHECKS)});
  console.log(await page.locator('#browser-result').textContent());
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
