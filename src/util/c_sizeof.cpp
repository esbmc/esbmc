#include <util/arith_tools.h>
#include <util/c_sizeof.h>
#include <util/c_types.h>
#include <irep2/irep2_utils.h>
#include <util/migrate.h>
#include <util/simplify_expr.h>
#include <util/type_byte_size.h>

exprt c_sizeof(const typet &src, const namespacet &ns)
{
  type2tc t = migrate_type(ns.follow(src));

  // Array size simplification and so forth will have already occurred in
  // migration, but we might still run into a nondeterministically sized
  // array.
  expr2tc size = type_byte_size_expr(t);

  return migrate_expr_back(size);
}
