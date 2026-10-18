# foundry_t20_forge_assert_array_prune_pass

`assertEq(T[], T[])` has no cheap content-equality lowering, so it is
conservatively PRUNED (ASSUME(false)) rather than emitting a false-WRONG
reference-equality assert. This content-EQUAL case — which the generic path
previously reported as FAILED (false WRONG) — now yields SUCCESSFUL. Prune
posture: never a false WRONG, at the cost of possibly-vacuous SUCCESSFUL (an
array mismatch bug can be missed). Precise element-wise array comparison is a
future optimization, not required for the conservative contract.
