"""Optional real-browser checks, using synthetic local next-action UI fixtures only."""

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
from rift_common.actions import ACTION_NAMES, action_catalog

BROWSER = os.environ.get("RIFT_TEST_BROWSER")
CHECKS = r"""
(async () => {
 const report = document.createElement('pre'); report.id='browser-result'; report.hidden=true;
 const errors=[]; window.addEventListener('error',e=>errors.push(e.message));
 const check=(ok,message)=>{if(!ok)throw Error(message);};
 const wait=async(fn)=>{for(let i=0;i<100;i++){if(fn())return;await new Promise(r=>setTimeout(r,20));}throw Error('UI state timeout');};
 const choose=(id,value)=>{const el=document.getElementById(id);el.value=value;el.dispatchEvent(new Event('change'));};
 const title=()=>document.getElementById('decision-title').textContent;
 const originalFetch=window.fetch;
 try {
  await wait(()=>title()==='Choose your role.');
  choose('role','TOP');await wait(()=>title()==='Farm top lane');
  check(document.getElementById('model-kind').textContent.includes(window.testAlgorithm==='iql'?'IQL':window.testAlgorithm==='cql'?'CQL':'imitation'),'Algorithm is identified');
  check(document.getElementById('score-label').textContent==={cql:'Q-VALUE',iql:'POLICY SCORE',bc:'BEHAVIOR SCORE'}[window.testAlgorithm],'Score meaning matches the algorithm');
  check(!document.getElementById('decision-check').hidden,'Action checks visible');
  check(document.getElementById('decision-horizon').textContent.includes('60'),'Action horizon visible');
  check(document.querySelectorAll('.action-row').length===15,'Full action catalog');
  check(document.querySelectorAll('.alternative-row').length===3,'Supported alternatives');
  check(document.documentElement.scrollWidth<=innerWidth,'Horizontal overflow');
  choose('actor','team');await wait(()=>title()==='Rotate toward mid');
  check(document.getElementById('role').disabled,'Team does not request a player role');
  check(!document.getElementById('model-warning').hidden,'Low team coverage is exposed');
  choose('actor','p2');await wait(()=>title()==='Clear jungle camps');
  check(document.getElementById('perspective').textContent.includes('Jungle'),'Role context rendered');
  check(document.getElementById('model-warning-text').textContent.includes('90%'),'Role concentration is exposed');
  check(document.querySelector('input[value="200"]').checked,'Player team selected');
  choose('actor','active');await wait(()=>title()==='Choose your role.');
  choose('role','TOP');await wait(()=>title()==='Farm top lane');
  document.getElementById('focus').click();
  check(getComputedStyle(document.querySelector('.secondary')).display==='none','Focus view');
  document.getElementById('focus').click();
  document.getElementById('pause').click();
  check(document.getElementById('decision-tag').textContent==='PAUSED','Pause state');
  check(document.querySelectorAll('.action-row.selected').length===0,'No paused scores');
  document.getElementById('pause').click();await wait(()=>title()==='Farm top lane');
  window.fetch=async(...args)=>{const response=await originalFetch(...args);const data=await response.json();data.valid_for_ms=100;return {ok:true,json:async()=>data};};
  document.getElementById('pause').click();document.getElementById('pause').click();
  await wait(()=>document.querySelector('.decision.active'));
  await wait(()=>document.getElementById('decision-tag').textContent==='UPDATING');
  check(document.querySelectorAll('.action-row.selected').length===0,'Expired scores cleared');
  check(document.getElementById('decision-check').hidden,'Expired checks cleared');
  window.fetch=async()=>{throw Error('Test disconnection');};
  document.getElementById('pause').click();document.getElementById('pause').click();
  await wait(()=>document.getElementById('decision-tag').textContent==='OFFLINE');
  check(document.querySelectorAll('.alternative-row').length===0,'No stale alternatives');
  window.fetch=originalFetch;
  document.getElementById('pause').click();document.getElementById('pause').click();
  await wait(()=>title()==='Farm top lane');
  check(errors.length===0,errors.join(';'));
  report.textContent=JSON.stringify({passed:true});
 } catch(error) {report.textContent=JSON.stringify({passed:false,error:String(error),errors});}
 document.body.append(report);
})();
"""


@pytest.mark.skipif(not BROWSER, reason="Set RIFT_TEST_BROWSER to an installed Chromium executable")
@pytest.mark.parametrize("width,height", [(1365, 1050), (390, 844)])
@pytest.mark.parametrize("algorithm", ["cql", "bc", "iql"])
def test_next_action_ui_subject_role_and_stale_states(tmp_path, width, height, algorithm):
    def state(*, team=None, actor=None, role=None):
        team = team or 100
        actor = actor or "active"
        if actor == "active":
            actor = "p1"
            team = 100
        if actor == "p2":
            team = 200
            role = role or "JUNGLE"
        if actor == "team":
            role = "TEAM"
        action = 6 if actor == "team" else 4 if actor == "p2" else 1
        result = {
            "team": team,
            "actor": actor,
            "role": role,
            "subject": "Team" if actor == "team" else "Test champion",
            "game_time": 1285,
        }
        if not role:
            result.update(status="needs_role", message="Select your role.")
        else:
            detail = action_catalog()[action]
            result.update(
                status="recommendation",
                action_id=action,
                action=ACTION_NAMES[action],
                instruction=detail["instruction"],
                check=detail["check"],
                horizon_seconds=60,
                method=algorithm,
                score_meaning="return estimate; not a probability",
                scores=[5 if i == action else -i / 10 for i in range(15)],
            )
        return {
            "status": "live",
            "checked_at": time.time(),
            "valid_for_ms": 30000,
            "interval_seconds": 30,
            "transport": "local",
            "default_team": 100,
            "actions": ACTION_NAMES,
            "catalog": action_catalog(),
            "players": [
                {"id": "p1", "team": 100, "champion": "Test top", "role": ""},
                {"id": "p2", "team": 200, "champion": "Test jungle", "role": "JUNGLE"},
            ],
            "active_player": "p1",
            "recommendation": result,
            "model": {
                "prediction_method": algorithm,
                "action_counts": [10] * 15,
                "prediction_counts": [10] * 15,
                "action_coverage": 15,
                "collapsed": False,
                "per_group": {
                    "TEAM": {
                        "prediction_counts": [6] + [0] * 11 + [4, 0, 0],
                        "prediction_coverage": 2,
                    },
                    "JUNGLE": {
                        "prediction_counts": [0] * 4 + [1] + [0] * 7 + [18, 0, 1],
                        "prediction_coverage": 3,
                    },
                },
            },
        }

    with DashboardServer(SimpleNamespace(state=state), 0) as server:
        source, mime = server.assets["/app.js"]
        script = f"window.testAlgorithm={json.dumps(algorithm)};\n" + CHECKS
        server.assets["/app.js"] = (source + script.encode(), mime)
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
                    f"--user-data-dir={tmp_path / 'profile'}",
                    f"--window-size={width},{height}",
                    "--virtual-time-budget=7000",
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
            match = re.search(r'<pre id="browser-result"[^>]*>(.*?)</pre>', result.stdout)
            assert match, result.stdout[-2000:] + result.stderr[-1000:]
            report = json.loads(html.unescape(match[1]))
            assert report["passed"], report
            assert Path(tmp_path / "dashboard.png").exists()
        finally:
            server.shutdown()
            thread.join(timeout=5)
