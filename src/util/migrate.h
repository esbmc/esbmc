#ifndef _ESBMC_UTIL_MIGRATE_H_
#define _ESBMC_UTIL_MIGRATE_H_

// "Migration" refers to the process of moving an expression from the prior
// string-based internal representation to the newer typed representation.
// There's a full mapping in both directions.

#include <util/irep2.h>
#include <util/std_expr.h>
#include <util/std_types.h>

// Don't ask
class namespacet;
extern namespacet *migrate_namespace_lookup;

void real_migrate_type(const typet &type, type2tc &new_type,
                  const namespacet *ns = NULL, bool cache = true);
void migrate_type(const typet &type, type2tc &new_type,
                  const namespacet *ns = NULL, bool cache = true);
void migrate_expr(const exprt &expr, expr2tc &new_expr);
typet migrate_type_back(const type2tc &ref);
exprt migrate_expr_back(const expr2tc &ref);

#endif /* _ESBMC_UTIL_MIGRATE_H_ */
