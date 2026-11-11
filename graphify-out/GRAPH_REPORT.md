# Graph Report - offlineRift  (2026-11-11)

## Corpus Check
- 114 files · ~67,912 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6 file(s) not represented in the graph (top: .css 3, (none) 2, .ipynb 1)

## Summary
- 1303 nodes · 3971 edges · 73 communities (63 shown, 10 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 475 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2f072053`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- offline_rift/workflow.py
- offline_rift/workflow.py
- s
- artifacts.py
- sd
- atomic_json
- ud
- production/frontend/macro-hud/package.json
- NextActionPolicy
- b0
- D
- index-wRf3c-mw.js
- bh
- st
- types.ts
- test_workflow.py
- test_six_real_models_are_scope_trained_reload_route_and_detect_corruption
- numpy
- main.cjs
- Fl
- LiveDataSource
- read_match
- runtime.tsx
- test_hud_desktop.py
- vite.config.ts
- compilerOptions
- Sh
- Lo
- MacroHUD.tsx
- Eo
- LiveClientInference
- t1
- Action
- test_hud_service.py
- server.py
- artifacts.py
- fixture.tsx
- Hl
- live_client.py
- notebook_section
- L
- zh
- LiveHudService
- README.md
- G
- Decision problem and robustness study
- jd
- e1
- os
- HudHandler
- scripts
- zr
- SYNTHESIS.md
- Offline Rift repository instructions
- Six-policy architecture
- Modeling contract
- desktop/package.json
- .recommend
- as
- Th
- Offline Rift desktop HUD
- Offline Rift
- European development and game review — 2026-11-10
- test_architecture.py
- Later ranked acceptance: show, record and review advice
- vitest
- .prettierrc.json
- l
- fixtures/macro-hud/package.json
- tests/frontend/macro-hud/package.json
- offline-rift
- PROTOCOL.md

## God Nodes (most connected - your core abstractions)
1. `bh()` - 394 edges
2. `s()` - 67 edges
3. `sd()` - 56 edges
4. `ud()` - 44 edges
5. `atomic_json()` - 43 edges
6. `read_json()` - 39 edges
7. `Action` - 37 edges
8. `Vl()` - 35 edges
9. `Jv()` - 31 edges
10. `NextActionPolicy` - 29 edges

## Surprising Connections (you probably didn't know these)
- `test_subject_changes_share_snapshot_but_not_cached_advice()` --calls--> `LiveClientUnavailable`  [INFERRED]
  tests/test_hud_service.py → production/src/offline_rift/inference/transport.py
- `test_local_desktop_api_validates_context_and_keeps_assets_local()` --uses--> `HudServer`  [INFERRED]
  tests/test_hud_service.py → production/src/offline_rift/ui/server.py
- `test_ctrl_c_notifies_native_host_even_when_windows_proxy_has_exited()` --uses--> `MacroCoach`  [INFERRED]
  tests/test_hud_desktop.py → production/src/offline_rift/inference/coach.py
- `Maintained tools` --references--> `synthesize()`  [INFERRED]
  research/README.md → research/src/rift_research/reporting.py
- `dashboard()` --uses--> `LiveStateAdapter`  [INFERRED]
  tests/test_hud_service.py → production/src/offline_rift/inference/live_client.py

## Import Cycles
- None detected.

## Communities (73 total, 10 thin omitted)

### Community 0 - "offline_rift/workflow.py"
Cohesion: 0.07
Nodes (14): bellman_target(), train_coach(), expectile_loss(), IQLUpdate, critic_targets(), main(), PolicyTrainer, TrainConfig (+6 more)

### Community 1 - "offline_rift/workflow.py"
Cohesion: 0.09
Nodes (10): run_id(), Collector, load_api_key(), RiotAPIError, RiotClient, main(), Workflow, test_collection_pagination_dedup_resume_and_interruption() (+2 more)

### Community 2 - "s"
Cohesion: 0.12
Nodes (56): a1(), b1(), cn(), de(), ef(), id(), Ii(), Il() (+48 more)

### Community 3 - "artifacts.py"
Cohesion: 0.07
Nodes (21): group_index(), ActivityEvidence, Behavior, distance(), player_activity(), player_behavior(), team_activity(), team_behavior() (+13 more)

### Community 4 - "sd"
Cohesion: 0.10
Nodes (50): a0(), Ac(), ad(), ai(), Be(), Bu(), Cc(), Ce() (+42 more)

### Community 5 - "atomic_json"
Cohesion: 0.11
Nodes (26): atomic_json(), file_digest(), read_json(), verify_bundle_manifest(), write_bundle_manifest(), CatalogCache, patch_of(), review_adherence() (+18 more)

### Community 6 - "ud"
Cohesion: 0.11
Nodes (42): bf(), Bi(), cd(), dd(), gd(), ge(), gi(), gr() (+34 more)

### Community 7 - "production/frontend/macro-hud/package.json"
Cohesion: 0.07
Nodes (34): { contextBridge, ipcRenderer }, devDependencies, electron@45.2.3, @electron/packager@20.3.0, jsdom@^26.1.0, @playwright/test@1.51.1, prettier@3.6.2, tailwindcss@^4.1.0 (+26 more)

### Community 8 - "NextActionPolicy"
Cohesion: 0.13
Nodes (12): alignment(), withhold_conflicting_role(), load_policy(), MacroCoach, NextActionPolicy, assemble_coach(), reselect_coach(), select_candidate() (+4 more)

### Community 9 - "b0"
Cohesion: 0.09
Nodes (33): _0(), al(), b0(), Bc(), C(), c0(), Cl(), d0() (+25 more)

### Community 10 - "D"
Cohesion: 0.13
Nodes (33): af(), bd(), co(), D(), d1(), Df(), ed(), er() (+25 more)

### Community 11 - "index-wRf3c-mw.js"
Cohesion: 0.08
Nodes (31): Ah, Bh, Bl(), Ch, Dh, Ee, Fh, Gh (+23 more)

### Community 12 - "bh"
Cohesion: 0.09
Nodes (20): bh(), _e(), fs(), G1(), Gu(), H1(), Hv(), jo() (+12 more)

### Community 13 - "st"
Cohesion: 0.14
Nodes (29): Bt(), Dm(), _f(), g0(), h0(), Ic(), ju(), Ke() (+21 more)

### Community 14 - "types.ts"
Cohesion: 0.14
Nodes (20): createHudStore(), flush(), update(), HudSnapshot, shared(), StoreOptions, clockSeconds(), Coordination (+12 more)

### Community 15 - "test_workflow.py"
Cohesion: 0.39
Nodes (8): prepare(), test_activation_retains_previous_reference_and_verifies_first(), test_failed_training_does_not_publish_latest_model(), test_immutable_build_cache_reuse_and_changed_input(), test_legacy_workflow_is_rejected_without_modification(), test_ranked_roster_registers_verified_unique_players_and_preserves_primary(), test_reload_parity_failure_preserves_active_model_and_failed_report(), test_reused_model_is_reloaded_and_rejects_corruption()

### Community 16 - "test_six_real_models_are_scope_trained_reload_route_and_detect_corruption"
Cohesion: 0.16
Nodes (12): DatasetBuilder, load_dataset(), ItemCatalog, StateEncoder, test_six_real_models_are_scope_trained_reload_route_and_detect_corruption(), test_explicit_duration_terminal_rewards_all_roles_and_integrity(), test_ambiguous_rows_keep_exact_episodes_and_are_not_training_support(), test_development_checkpoint_is_restored_and_phase_support_reloads() (+4 more)

### Community 17 - "numpy"
Cohesion: 0.09
Nodes (25): feasible_actions(), episode_returns(), behavior_metrics(), evidence_patterns(), ObservationAugmenter, scope_diagnostics(), history_consistency(), checkpoint_diagnostics() (+17 more)

### Community 18 - "main.cjs"
Cohesion: 0.12
Nodes (18): {
  app,
  BrowserWindow,
  globalShortcut,
  ipcMain,
  screen,
  Tray,
  Menu,
  nativeImage,
  session,
}, backend, broadcast(), fs, interaction(), lifetime, openSettings(), path (+10 more)

### Community 19 - "Fl"
Cohesion: 0.23
Nodes (24): $1(), Aa(), At(), ba(), br(), ci(), eh(), En() (+16 more)

### Community 20 - "LiveDataSource"
Cohesion: 0.19
Nodes (10): LiveClientUnavailable, LiveDataSource, test_explicit_local_and_injected_sessions_do_not_use_windows(), test_missing_windows_interop_and_native_auto(), test_transport_cannot_target_remote_urls(), test_windows_certificate_path_is_translated(), test_windows_errors_are_recoverable_offline_states(), test_windows_rejects_redirects_and_invalid_responses() (+2 more)

### Community 21 - "read_match"
Cohesion: 0.35
Nodes (4): ObjectiveRules, ObjectiveTracker, test_herald_single_spawn_and_conservative_uncertainty(), test_soul_counts_same_team_and_unknown_subtype_masks_availability()

### Community 22 - "runtime.tsx"
Cohesion: 0.14
Nodes (9): dependencies, lucide-react@^0.468.0, DesktopBridge, Frame, Runtime(), settingsWindow, store, PLAYER_ROLES (+1 more)

### Community 24 - "vite.config.ts"
Cohesion: 0.13
Nodes (6): require, directory, fixtures, repository, directory, frontend

### Community 25 - "compilerOptions"
Cohesion: 0.12
Nodes (16): compilerOptions, allowImportingTsExtensions, baseUrl, jsx, lib, module, moduleResolution, noEmit (+8 more)

### Community 26 - "Sh"
Cohesion: 0.20
Nodes (13): Sh(), ae(), al(), be(), dt(), Ht(), I(), K() (+5 more)

### Community 27 - "Lo"
Cohesion: 0.20
Nodes (16): Di(), Ds(), fr(), gs(), ir(), Iu(), ki(), la() (+8 more)

### Community 28 - "MacroHUD.tsx"
Cohesion: 0.28
Nodes (13): Countdown, Details, EmptyState, Header, MacroCard, MacroHUD, MacroHUDProps, ROLE_ICON (+5 more)

### Community 29 - "Eo"
Cohesion: 0.15
Nodes (15): ar(), bo(), ch(), dh(), Eo(), fh(), Fv(), Kr() (+7 more)

### Community 30 - "LiveClientInference"
Cohesion: 0.16
Nodes (12): LiveClientInference, LiveStateAdapter, SessionCapture, live_data(), test_active_player_is_not_silently_replaced_with_team(), test_capture_is_disabled_by_default_and_interrupt_retains_progress(), test_future_history_marker_cannot_enable_objectives_and_game_end_clears_advice(), test_history_rollbacks_duplicate_clock_and_incomplete_events() (+4 more)

### Community 31 - "t1"
Cohesion: 0.27
Nodes (14): Au(), Bs(), eu(), fd(), hu(), Jc(), l1(), ni() (+6 more)

### Community 32 - "Action"
Cohesion: 0.15
Nodes (12): Action, action_support(), OptionNetwork, rank_options(), phase_counts(), phase_indices(), phase_support(), coordination_metrics() (+4 more)

### Community 33 - "test_hud_service.py"
Cohesion: 0.14
Nodes (7): dashboard(), test_hud_pairs_actual_macro_and_selected_role_without_calibrated_metrics(), test_local_desktop_api_validates_context_and_keeps_assets_local(), test_missing_role_uses_team_policy_without_guessing(), test_restart_changes_session_and_offline_withdraws_hud(), test_scoped_diagnostics_use_the_selected_role(), test_subject_changes_share_snapshot_but_not_cached_advice()

### Community 34 - "server.py"
Cohesion: 0.10
Nodes (8): launch_desktop(), _windows_local_executable(), add_ui_arguments(), main(), run_hud(), test_compiled_hud_model_role_and_withdrawal(), configure(), frame()

### Community 35 - "artifacts.py"
Cohesion: 0.09
Nodes (3): main(), numbers(), canonical_champion()

### Community 36 - "fixture.tsx"
Cohesion: 0.18
Nodes (9): react@^19.2.0, react-dom@^19.2.0, peerDependencies, react@^19.2.0, react-dom@^19.2.0, react, react-dom, durations (+1 more)

### Community 37 - "Hl"
Cohesion: 0.18
Nodes (12): bn(), Da(), dr(), Ga(), Hl(), ie(), _n(), nr() (+4 more)

### Community 38 - "live_client.py"
Cohesion: 0.43
Nodes (3): AdviceController, test_retained_role_advice_uses_fresh_macro_direction_and_rechecks_alignment(), test_advice_persistence_requires_support_and_clears_on_death()

### Community 40 - "L"
Cohesion: 0.42
Nodes (11): L(), gh(), yh(), M(), Nt(), O(), Ot(), ql() (+3 more)

### Community 41 - "zh"
Cohesion: 0.20
Nodes (10): ee(), hh(), Qi(), sn(), Uo(), vh(), al(), H() (+2 more)

### Community 42 - "LiveHudService"
Cohesion: 0.16
Nodes (5): action_catalog(), HudServer, LiveHudService, test_windows_native_hud_starts_and_closes_after_ctrl_c(), test_lifetime_connection_delivers_shutdown_even_after_the_process_proxy_exits()

### Community 43 - "README.md"
Cohesion: 0.22
Nodes (5): Action catalog, Explicitly unsupported fine actions, Historical studies and cleanup, Maintained tools, Offline policy research

### Community 44 - "G"
Cohesion: 0.20
Nodes (9): c1(), G(), j1(), or(), sr(), u1(), x1(), y1() (+1 more)

### Community 45 - "Decision problem and robustness study"
Cohesion: 0.22
Nodes (8): Availability-aware consistency, Baselines and conservative regression, Coherence and abstention, Decision problem and robustness study, European representation and checkpoint experiment, Selection and evidence boundaries, Shared omission regularizer, What is optimized

### Community 46 - "jd"
Cohesion: 0.42
Nodes (9): A(), Cu(), hf(), jd(), jf(), Kd(), Rt(), wu() (+1 more)

### Community 47 - "e1"
Cohesion: 0.33
Nodes (9): cs(), e1(), ec(), es(), ht(), is(), lc(), Ma() (+1 more)

### Community 48 - "os"
Cohesion: 0.22
Nodes (9): Dn(), Ev(), gv(), ls(), os(), Po(), Su(), Sv() (+1 more)

### Community 50 - "scripts"
Cohesion: 0.25
Nodes (8): scripts, build, dev, format:check, package:linux, package:win, test, test:browser

### Community 51 - "zr"
Cohesion: 0.25
Nodes (7): ah(), An(), ji(), nh(), tt(), uh(), zr()

### Community 52 - "SYNTHESIS.md"
Cohesion: 0.25
Nodes (7): Data, targets and exclusions, Engineering, cleanup and next tests, History omission on reserved matches, Mathematical design and robustness, Offline Rift: six-policy macro coach, Reserved-match results, Training and development comparison

### Community 53 - "Offline Rift repository instructions"
Cohesion: 0.29
Nodes (6): APIs, data and artifacts, graphify, Modeling invariants, Objective and structure, Offline Rift repository instructions, Validation and handoff

### Community 54 - "Six-policy architecture"
Cohesion: 0.29
Nodes (6): Bundle and publication contract, Coordination contract, Product and boundaries, Scalability and current limits, Six-policy architecture, Source map

### Community 55 - "Modeling contract"
Cohesion: 0.29
Nodes (6): Composition and evaluation, Modeling contract, Objective and scope, Observations and actions, Rewards and explicit critic targets, Versioned evidence and complete episodes

### Community 56 - "desktop/package.json"
Cohesion: 0.29
Nodes (6): author, description, license, main, name, version

### Community 59 - "as"
Cohesion: 0.29
Nodes (7): as(), cr(), If(), jm(), ll(), Pi(), Zs()

### Community 60 - "Th"
Cohesion: 0.29
Nodes (3): Eh(), Ro(), Th()

### Community 61 - "Offline Rift desktop HUD"
Cohesion: 0.33
Nodes (5): Build and run, Controls, Offline Rift desktop HUD, Runtime and performance, Verification

### Community 62 - "Offline Rift"
Cohesion: 0.33
Nodes (6): Architecture, Data and offline training, Install, Offline Rift, Product behavior, Validation and research

### Community 63 - "European development and game review — 2026-11-10"
Cohesion: 0.33
Nodes (6): Adaptive regional selection, Data and modeling changes, European development and game review — 2026-11-10, First reserved European comparison, Fresh evaluation after the adaptive revision, One-game notebook

### Community 66 - "Later ranked acceptance: show, record and review advice"
Cohesion: 0.40
Nodes (4): After the game, Before loading the map, Later ranked acceptance: show, record and review advice, Stronger review using the completed Match-V5 game

### Community 67 - "vitest"
Cohesion: 0.40
Nodes (4): @testing-library/react@^16.3.0, vitest@4.0.18, @testing-library/react, vitest

### Community 69 - ".prettierrc.json"
Cohesion: 0.50
Nodes (3): printWidth, singleQuote, trailingComma

### Community 70 - "l"
Cohesion: 0.67
Nodes (4): tm(), e(), l(), t()

## Knowledge Gaps
- **2 isolated node(s):** `offline-rift`, `offline-rift-research`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 335 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `bh()` connect `bh` to `s`, `sd`, `ud`, `b0`, `D`, `index-wRf3c-mw.js`, `st`, `Fl`, `Lo`, `Eo`, `t1`, `Hl`, `L`, `zh`, `G`, `jd`, `e1`, `os`, `zr`, `as`, `Th`, `l`?**
  _High betweenness centrality (0.148) - this node is a cross-community bridge._
- **Are the 40 inferred relationships involving `bh()` (e.g. with `_0()` and `$1()`) actually correct?**
  _`bh()` has 40 INFERRED edges - model-reasoned connections that need verification._
- **What connects `offline-rift`, `offline-rift-research` to the rest of the system?**
  _2 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `offline_rift/workflow.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07293868921775898 - nodes in this community are weakly interconnected._
- **Why does `synthesize()` connect `atomic_json` to `artifacts.py`, `README.md`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `s()` (e.g. with `os()` and `a()`) actually correct?**
  _`s()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Should `offline_rift/workflow.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08758503401360544 - nodes in this community are weakly interconnected._