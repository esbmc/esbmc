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

  // If it is an array of infinite size, we just return
  // "infinity" expression and multiply it recursively
  // by the size of the array subtype
  if(is_array_type(t) && to_array_type(t).size_is_infinite)
    return mult_exprt(
      c_sizeof(migrate_type_back(to_array_type(t).subtype), ns),
      exprt("infinity"));

  // Array size simplification and so forth will have already occurred in
  // migration, but we might still run into a nondeterministically sized
  // array.
  expr2tc size = type_byte_size_expr(t);

  return migrate_expr_back(size);
}
