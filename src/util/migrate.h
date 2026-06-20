#ifndef _ESBMC_UTIL_MIGRATE_H_
#define _ESBMC_UTIL_MIGRATE_H_

// "Migration" refers to the process of moving an expression from the prior
// string-based internal representation to the newer typed representation.
// There's a full mapping in both directions.

#include <irep2/irep2.h>
#include <util/std_expr.h>
#include <util/std_types.h>

// Don't ask
class namespacet;
class symbolt;
// thread_local so parallel symex threads can each set their own
// namespace pointer without racing.
extern thread_local const namespacet *migrate_namespace_lookup;

type2tc migrate_type(const typet &type);
void migrate_expr(const exprt &expr, expr2tc &new_expr);

// IREP2 form of a symbol's type. Named chokepoint for symbol type reads in
// the migration layer (esbmc/esbmc#4715, B2): returns `sym.get_type2()` which
// is the IREP2 source of truth after the B2 storage flip. In debug builds it
// also asserts the IREP2 form is stable under a round-trip through legacy
// irept -- the property (proven for synthetic types in
// unit/util/migrate.test.cpp) that makes the lazy legacy cache on `symbolt`
// lossless, now checked on every real symbol type the pipeline reads.
type2tc migrate_symbol_type(const symbolt &sym);

// IREP2 form of a symbol's value. Value-side counterpart of
// migrate_symbol_type. The debug cross-check is *guarded*: function bodies
// (sym.get_type().is_code()) are skipped (no caller goes through this path on
// them today, and the assertion would force a potentially-large body
// round-trip for no signal); nil values are skipped too.
void migrate_symbol_value(const symbolt &sym, expr2tc &dest);

// Set a symbol's type from an IREP2 form (esbmc/esbmc#4715, B2). The
// chokepoint for symbol-type writes: routes through symbolt's IREP2-side
// setter, which stores the form natively and invalidates the lazy legacy
// cache. Replaces the previous round-trip pattern
// `sym.set_type(migrate_type_back(t))` at the call sites.
void set_symbol_type(symbolt &sym, const type2tc &t);

typet migrate_type_back(const type2tc &ref);
exprt migrate_expr_back(const expr2tc &ref);

// --- Phase 4.2 construction helpers (Part IV §6: "build once, shared") -------
// IREP2 replacements for the two legacy expression constructors the frontends
// lean on most, so frontend code can build IREP2 nodes directly instead of
// `symbol_expr(symbolt)` / `side_effect_expr_function_callt`. These are pure
// construction helpers with round-trip unit tests; no call site is rewired
// here -- the wiring is Phase 4.3/4.4 work, shipped separately so this
// infrastructure carries zero coverage-axis risk (the V-track lesson).

// IREP2 form of `symbol_expr(const symbolt&)`: a level-0 `symbol2t` carrying the
// symbol's IREP2 type (read via migrate_symbol_type, the B2 source of truth)
// and its identifier. The legacy node also stores a cosmetic display name;
// IREP2 symbols carry only the identifier, so it is neither represented nor
// needed (`migrate_expr` drops it on the same path).
expr2tc symbol_expr2tc(const symbolt &sym);

// IREP2 form of `side_effect_expr_function_callt`: an expression-context call
// `function(arguments...)` evaluating to `return_type`, i.e. a `sideeffect2t`
// with allockind::function_call. Mirrors migrate.cpp's function_call lowering
// (operand = callee, arguments = args, no size/alloctype).
expr2tc side_effect_function_call2tc(
  const type2tc &return_type,
  const expr2tc &function,
  const std::vector<expr2tc> &arguments);

#endif /* _ESBMC_UTIL_MIGRATE_H_ */
