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
  BigInt size;

  size = type_byte_size(t); // let the exception propagate
  constant_int2tc theval(get_uint32_type(), size);
  return migrate_expr_back(theval);
}
