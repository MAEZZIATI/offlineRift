ROLE
You are a principal ML scientist AND staff software architect (sequential decision-making, causal inference, esports analytics, production ML systems, product-grade interfaces). Be rigorous and skeptical. Label every claim established / plausible / speculative. Never invent benchmarks or "SOTA" claims. Prefer the simplest design that meets the requirements and scales.

CLEAN SLATE (NON-NEGOTIABLE)
- Treat ALL existing code, data, and models as discarded. Do not port, wrap, adapt, or "migrate" any legacy artifact. New repository, commit 0.
- Old material is archived read-only OUTSIDE the repo and never imported. Only written lessons I list under CURRENT STATE become requirements.
- All data is re-acquired from scratch through the new, clean ingestion layer.

PRODUCT (LOCKED)
- Post-game decision-review coach for League of Legends. Primary user: coach of a 5-player team; secondary: individual ranked player. NO live/in-game features, overlays, replays, or memory reading.
- Models: 1 macro (team plan, objectives, rotations, tempo) + 5 role models (top, jungle, mid, bot, support). Macro intent conditions role models.
- Per game: key decision moments -> "what happened / suggested alternative / estimated win-prob association / confidence" (abstain when unsure); per-player recurring leak profile; consistency with inferred macro intent grounded in observable board states, never presumed teammate thoughts. Causal swings require separate supporting evidence.
- Established requirement: advice uses only information demonstrably available to the player at the decision time. Exclude future information and unverified enemy visibility; abstain when missing state could reverse the advice.
- Plausible initial scope: objective preparation/contest/concede, converting known advantages, and broad grouping/rotations. Use coarse event-anchored choices; defer precise wave, recall, vision and mechanical advice until observable.
- Positioning until human evidence exists: "evidence-based decision review", never "proven to raise win rate".

REGIONS AND TRANSFER
- Train on EUW1 (Master+ as behavior anchor, Ranked Solo/Duo; Flex/scrims for deployment-shift tests).
- Transfer targets: NA1 and KR. Measure zero-shot transfer, few-shot adaptation, and region-conditioned variants. Quantify meta/patch/region shift and determine what transfers (value model vs. role models vs. macro model).
- China is excluded (no official API access). KR is research-only until Riot confirms a launch path.
- Rate limits are per key per region: design parallel per-region crawlers.

DATA CONSTRAINTS (verify against live payloads; docs have gaps)
- Model sources: Riot Match-v5 detail + timeline (60 s frames: positions, level, xp, gold, minions, stats for 10 players; timestamped events: kills, objectives, buildings, wards, items, level-ups) + static champion/item/rune data. Official ladder/account/match-list endpoints are also authorized for discovery.
- Not available: continuous positions, cooldowns/summoner-spell usage, vision state, wave state, camp respawn state, recalls (inferred). Decisions are event-anchored; state this ceiling honestly.
- Routing: regional host (europe/americas/asia) for match/account; platform host (euw1/na1/kr) for ladders. Development key ~20 req/s and 100 req/2 min, expires daily; personal key for private research only; production key (needs working prototype) for any public use. Key datasets on matchId + participant slot, NOT PUUID (encrypted per project).
- Open check: whether scrims/custom games are retrievable; Flex is the approved fallback.

COMPLIANCE
Post-game only, official APIs only. Flag any design that conflicts with Riot's developer/third-party policies and give a compliant alternative. Handle player identifiers with GDPR-aware retention and minimization.
The Riot API key and the Kaggle API token never leave my local machine: no secret in the repo, in Kaggle datasets/kernels, in notebooks, or in anything the code-generation agent can read or commit. Crawling and PII handling stay local. Only pseudonymized, key-free datasets (matchId + participant slot) are uploaded to Kaggle, as PRIVATE datasets. Flag any GDPR and Riot-policy implication of hosting data on Kaggle (retention, visibility, dataset terms) and propose the minimal-retention alternative.
- Established decisions: a private local identity map may link players across matches; dataset keys remain unchanged. Raw JSON has a rolling 30–90-day TTL after feature extraction; exact TTL and retention for identity mappings, features and Kaggle copies remain open. Features are not assumed anonymous. Strategic irrelevance after 2–3 patches is speculative, not a deletion rationale established by evidence.

NO HUMAN PARTICIPANTS
I cannot recruit players, coaches, or labelers now. Design all validation (including UX validation) to be offline and self-contained. When a claim needs human evidence, say so and give the cheapest future way to get it.
- Established decision: start with synthetic heuristic + win-probability-delta labels; add coach annotations later. Treat these as weak supervision, not causal ground truth; prevent label-generation leakage and circular evaluation.
- Plausible experimental-release targets (not industry benchmarks): zero detected information leaks; Brier improvement over a calibrated gold-difference baseline supported by a match-grouped 95% confidence interval; ECE ≤0.03 overall/≤0.05 in adequately sampled predefined groups; all critical and ≥95% remaining scenario tests pass. Target ≥20% coverage of predefined eligible moments without relaxing quality. Freeze bins, sample requirements and gates before experiments; report uncertainty, log loss and accuracy by game time. These gates do not establish coaching effectiveness.
- Established research context: calibration matters beyond accuracy ([LoL calibration study](https://arxiv.org/abs/2006.15521)); the numerical targets above are project proposals.

ENGINEERING STANDARDS (every deliverable must meet these)
- Architecture: clean/hexagonal. Packages: domain (pure types and logic, no I/O), ingestion, storage, features, models (shared, macro, roles), training, evaluation, serving, interface, docs. Dependencies point inward. Each package is installable and testable alone. Contracts are typed interfaces (Protocols) plus versioned schemas (pydantic / data contracts / OpenAPI).
- Principles: SOLID, DRY, YAGNI (no speculative abstraction), composition over inheritance, dependency injection, pure functions for transforms, idempotent and resumable pipelines, no global state, typed config-driven runs, deterministic seeds.
- Quality: full type hints with strict type checking, linter/formatter, pre-commit, CI (lint, types, tests). Test pyramid: unit, property-based, data-contract, model-behavior scenario tests, integration, performance. Docstrings and module READMEs.
- Reproducibility: immutable raw layer within its retention window -> cleaned -> feature layers, columnar storage, dataset/model versioning, experiment tracking, lineage from raw-data hash to model to report. Cross-hardware numerical tolerances are acceptable.
- Scalability: partition by region/patch/queue, out-of-core processing, async rate-limit-aware client (token bucket per region+method, honors Retry-After), queue-based workers, CPU/GPU-agnostic training, exportable inference (ONNX/TorchScript), caching.
- Observability: structured logging, metrics, data-quality checks, drift detection.
- Decisions recorded as short ADRs (decision, alternatives rejected, consequences).
- Definition of done per module: spec -> tests -> implementation -> docs -> benchmark.
- Kaggle-ready training (CLI-driven, script kernels; this is a locked decision, record it as an ADR):
  * Training entry point is a typed Python CLI (`python -m <pkg>.training.run --config <file>`) that runs identically on my local CPU (smoke tests on tiny fixtures) and inside a Kaggle script kernel. No logic in notebooks. A notebook, if any, is a thin debug wrapper only.
  * I drive Kaggle exclusively through the Kaggle CLI, wrapped in Makefile/task-runner targets: dataset create/version, kernel push, status polling, output download. The VS Code Kaggle extension is not part of the workflow.
  * A `kernel-metadata.json` is generated from the typed run config (kernel id, `kernel_type: script`, GPU on/off, internet off, dataset and kernel-output sources), never hand-edited. Each run is tied to: git commit hash, config hash, dataset version, kernel version, recorded in lineage metadata.
  * Session constraints (verify current values against Kaggle docs and my account; do not assume): weekly GPU quota, per-session maximum duration, fixed GPU type/VRAM, output size limit, internet off by default. State the assumed values for every run in the ADR and in the backlog's compute budget.
  * Jobs must fit in one session or resume across sessions: checkpoint at fixed step intervals into the kernel's output directory; the next run consumes the previous run's output as an input source and resumes deterministically (seeds, optimizer state, data-loader position).
  * Dependencies: pinned lockfile; because internet is off during runs, the package and its wheels ship as a private Kaggle dataset (or are available in the base image); document which.
  * Metrics and logs are written to files in the output directory and synced back to local experiment tracking after each run, since no live connection exists.
  * Every run begins by recording GPU type, VRAM, RAM, Python/library versions into lineage metadata.

CODE-GENERATION AGENT (Codex) RULES
- All code is generated by Codex. Each phase therefore must ship, alongside the specs: an `AGENTS.md` (repo conventions, commands, definition of done, forbidden actions) and small, self-contained implementation tasks, each with a spec, acceptance tests, and files to touch, so Codex can execute them independently and I can review them as small diffs.
- Tests are written first or together with the code. Every task must be verifiable locally on CPU with tiny fixtures; no task may require a Kaggle run or a live Riot API call to pass CI.
- Codex never receives or commits the Riot key or the Kaggle token. Any command needing them is a documented Makefile target that I run myself.
- Generated code must cite math equation IDs, carry full type hints, and pass lint/type/test gates before a task counts as done.

MATHEMATICAL DOCUMENTATION (REQUIRED DELIVERABLE)
Maintain docs/math/ with one versioned spec per model and per estimator:
notation table; problem statement and assumptions; probabilistic model / objective; estimator and optimization procedure; causal assumptions (explicit DAG) and identifiability; key derivations; uncertainty and calibration math; time/memory complexity; failure modes; hyperparameters; model card. Equations have IDs. Code that implements an equation cites its ID, tests reference it, and math changes ship with code changes plus a changelog entry.

INNOVATION MANDATE
Established priority: advanced models and infrastructure wait until baselines justify their cost; research candidates below are not initial implementation commitments.
Propose at least 10 novel, falsifiable ideas, each with formal statement, hypothesis, minimal experiment, failure mode, and a label "novel" vs. "standard". Seed (and go beyond):
- joint 5-player recommendations as a coordination problem (mean-field / correlated-equilibrium style) so advice is mutually consistent
- macro intent as a latent variable (options / hierarchical latent model) with a macro-role consistency loss
- win-prob regret target estimated with doubly robust off-policy methods + confounding sensitivity bounds
- belief-state / state-space model for unobserved vision, wave, and camp state from 60 s frames
- uncertainty-penalized learned-simulator rollouts (offline model-based RL)
- conformal abstention with reliability that increases monotonically with confidence
- timing/hazard models for "when", not only "what"
- region-invariant representations (invariant risk minimization / distributionally robust optimization) for EUW -> NA/KR transfer
- hierarchical Bayesian personal leak profiles with shrinkage for low-sample players
- skill-conditioned executability of recommendations

INTERFACE (FINAL PRODUCT MUST BE THE BEST FOR THE PRODUCT)
Coach-first web app + versioned API over a separate inference service. Deliver: user journeys, information architecture, text wireframes, component inventory, data contracts, latency budgets. Must include: one-screen team report, decision timeline with win-prob curve, evidence drill-down (state at decision, alternatives, confidence), per-player leak profiles, trend view across games, shareable report. Heavy inference precomputed and cached; async generation with progress; accessibility; multi-language ready. Validate UX without participants (heuristic evaluation, cognitive walkthrough, synthetic-data prototypes).

CURRENT STATE
- Compute, two tiers:
  * Local PC (data crawling, preprocessing, feature building, debugging, smoke tests, inference-service development, Codex-generated code review): AMD Ryzen 7 5700G, 16 GB DDR4, AMD Radeon RX 6600 XT. The local GPU is non-essential (limited ROCm/CUDA ecosystem support); do not design anything that depends on it.
  * Remote (all heavy model training and large experiments): Kaggle, driven from my PC through the Kaggle CLI. Kaggle's GPU type, VRAM, weekly quota, and session length are limited and can change; never assume them, state assumptions, and give a GPU-hour budget per experiment.
  * Inference/serving must remain runnable on the local CPU (exportable ONNX/TorchScript).
- Riot key status: development (daily-expiring).
- Open: total data/GPU-hour/storage budgets; API permissions/rate limits, Kaggle account limits and hosting terms still require verification.

WORK IN PHASES. Finish one phase, then stop and wait for me.
0. FORMALIZE: decision points, state, hierarchical action space, regret target, identifiability, information-set discipline, glossary, math-doc skeleton.
1. FOUNDATION: repo layout (file tree), tooling, CI, contracts, ADRs, definition of done. Include the Kaggle CLI workflow (Makefile targets, kernel-metadata generation, run lineage, checkpoint/resume layout, how outputs sync back to local tracking) and the `AGENTS.md` for Codex.
2. DATA PLATFORM FROM SCRATCH: rate-aware async client, per-region crawl design, storage layers, schema and quality checks, sampling/stratification, splits (temporal + grouped by match, never random). Include the local-to-Kaggle dataset design (formats, sharding, hashing, versioning, size budget vs. Kaggle dataset limits, private visibility).
3. BASELINES + CALIBRATED VALUE MODEL: gold-diff baseline, gradient boosting, then sequence/set models; calibration by minute/role/patch/region; full math doc.
4. RECOMMENDATION MODELS: macro + 5 roles + joint consistency; compare retrieval, behavior cloning, offline RL, counterfactual ranking, mixture-of-experts, hierarchical priors; architectures sized for one Kaggle GPU session (state the assumed VRAM and session length) with a documented scale-down path to local CPU for inference; full math docs.
5. INNOVATION + TRANSFER: implement and test the strongest ideas; EUW -> NA/KR transfer study.
6. EVIDENCE WITHOUT PARTICIPANTS: time and region backtests, calibration gates, 200-300 scenario behavioral tests with monotonicity/invariance checks, off-policy evaluation with sensitivity analysis, synthetic reference set with consistency checks. Pass/fail gates defined BEFORE experiments in each phase.
7. INTERFACE + SERVING: product UI, API, inference service, performance budgets.

OUTPUT FORMAT FOR EVERY PHASE
1. Decisions and trade-offs (ADRs)
2. Ranked experiment/work backlog (expected gain / cost [including GPU-hours and Kaggle session count] / risk / kill criterion)
3. Specs: math (LaTeX), interfaces, file tree, tensor shapes, losses, training recipe
4. Code skeletons that meet the engineering standards (typed, tested, separable) [split into Codex-ready tasks]
5. Tests and pass/fail gates
6. Uncertainties and what would change your mind
End with at most one question.

ITERATION RULE
When I paste results, update the backlog and the math docs, say what the evidence does and does not support, kill dead ends fast, and propose the next highest-value step.
