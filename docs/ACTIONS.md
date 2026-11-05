# Action catalog

Every recommendation identifies the **subject and role**, names a **next behavior**, gives
an instruction and prerequisite checks, and uses a nominal **60-second horizon**. Reassess
when the situation changes. This is a coarse option, not a command held for exactly a minute.
The UI shows other supported choices; scores are never presented as win probabilities.

| Option | Meaning and reconstruction evidence |
|---|---|
| Continue current activity | No large resolved area transition; for a team, no observed majority transition. Not a universal missing-data fallback. |
| Farm top lane | Lane CS increases while both endpoints remain in the top lane band. Check waves/threats yourself. |
| Farm mid lane | The equivalent observed mid-lane income behavior. |
| Farm bottom lane | The equivalent observed bottom-lane income behavior. |
| Clear jungle camps | Jungle CS increases while remaining in the jungle area. Current camp availability is unobserved live. |
| Rotate toward top | Sufficient observed movement ending in the top lane band. May reinforce, roam or pressure; not a certified gank. |
| Rotate toward mid | The equivalent movement toward mid. |
| Rotate toward bottom | The equivalent movement toward bottom. |
| Prepare dragon area | Approach or maintain area presence over the following interval during the modeled dragon-pit window; includes the pit's Elder phase. |
| Prepare Baron area | The corresponding Baron approach/presence behavior during its modeled window. |
| Prepare Herald area | The corresponding Herald approach/presence behavior before its conservative expiry uncertainty. |
| Return to base and regroup | Base arrival from outside base without an observed death explaining it. Does not distinguish recall, teleport or walking. |
| Participate in a nearby fight | Recorded kill, assist **or death** in the interval. Fight initiation, safety and agency remain unverified. |
| Wait to respawn | Observed dead state at the start. This is forced waiting, not a strategic recommendation to seek death. |
| Rotate through jungle | A resolved move from another coarse area into jungle. Does not assert invade safety, camp availability or an exact route. |

The player version is conditioned on TOP, JUNGLE, MIDDLE, BOTTOM or UTILITY. Team actions
require at least three members with the corresponding behavior; individual farming and
respawn options are unavailable for the team subject. A player's potential choices are
restricted by observed role support, not by assuming each role must always perform one
scripted behavior.

## Explicitly unsupported fine actions

These belong in the new research roadmap but are not fabricated as learned options:

- **Freeze, slow push, crash a wave, deny a specific minion:** reliable wave/minion state
  is absent from current live observations and minute-level endpoints do not establish it.
- **Specific camp routes, invades or counter-jungle choices:** camp-level availability and
  visible/hidden opponent positioning are not established.
- **Gank/counter-gank commitments:** rotations are observable approximations; intent,
  failed attempts without visible events and local opportunities need better evidence.
- **Ward placement, clearing vision and safe paths:** ward/vision state is not consistently
  available in the live contract. No unexposed vision is invented.
- **Precise engage/disengage, target selection, combos, dodges, spell timing and last hits:**
  require much finer combat/control observations than the cached timelines provide.
- **A distinct Void Grubs option and detailed siege/defense options:** require a validated,
  patch-scoped availability/evidence contract and appropriate execution labels. Ordinary
  rotations can currently cover movement, but they do not claim a learned Grubs or siege plan.

Completeness means explicit supported behaviors and explicit evidence gaps. New classes
must be backed by execution evidence, input masks, action feasibility, terminal alignment
and held-out support checks before entering the production action contract.
