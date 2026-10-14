# Regression for esbmc/esbmc#4509: deterministic transitive-import case.
#
# main directly imports `direct_value` from `leaf`, so `leaf` enters round 1
# of the parser's import-discovery fixpoint with specific_names = {direct_value}.
# `mid` is also in round 1; parsing it adds `inner` to module_imports. `inner`
# is processed in round 2 and only then adds `deep_value` to leaf's
# specific_names. A single-phase emitter (the pre-fix bug) would have already
# written leaf.json with only direct_value in round 1, and never re-emit it —
# so `inner`'s call to `deep_value()` would fail to resolve regardless of set
# iteration order. The fix's discovery/emission split makes this case pass.
from leaf import direct_value
from mid import via_mid

assert direct_value() == 10
assert via_mid() == 20
