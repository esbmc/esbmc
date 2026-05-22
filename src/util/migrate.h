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
extern const namespacet *migrate_namespace_lookup;

type2tc migrate_type(const typet &type);
void migrate_expr(const exprt &expr, expr2tc &new_expr);

// IREP2 form of a symbol's type. Single chokepoint for the symbol-table
// migration (esbmc/esbmc#4715, B2 S1): equivalent to
// migrate_type(sym.get_type()) today, but centralised so the storage can later
// become IREP2-native without touching callers again. In debug builds it also
// asserts the IREP2 form is stable under a round-trip through legacy irept --
// the property (proven for synthetic types in unit/util/migrate.test.cpp) that
// makes deriving the legacy field from IREP2 lossless, now checked on every
// real symbol type the pipeline reads.
type2tc migrate_symbol_type(const symbolt &sym);

// IREP2 form of a symbol's value. Value-side counterpart of
// migrate_symbol_type (esbmc/esbmc#4715, B2 S2). The debug cross-check is
// *guarded*: function bodies (sym.get_type().is_code()) are skipped because
// migrate_expr_back cannot reconstruct a code_block -- a body is migrated
// forward only (see unit/util/migrate.test.cpp); nil values are skipped too.
void migrate_symbol_value(const symbolt &sym, expr2tc &dest);

typet migrate_type_back(const type2tc &ref);
exprt migrate_expr_back(const expr2tc &ref);

#endif /* _ESBMC_UTIL_MIGRATE_H_ */
