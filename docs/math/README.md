# Mathematical documentation boundary

Established: Phase 1 defines workflow/contracts only. There is no implemented
model, loss, estimator, tensor training recipe, or scientific equation to cite.
No Phase 0 formalization or future model design is supplied implicitly.

Established convention for later phases: add `<model-or-estimator>/v1.md` from
[TEMPLATE.md](TEMPLATE.md); equation IDs are stable and namespaced
`<NAME>-v1-<NN>`. Code and tests implementing an equation cite its ID. Mathematical
changes ship with code/tests and a changelog entry. Do not invent empirical
performance or causal identifiability claims.
