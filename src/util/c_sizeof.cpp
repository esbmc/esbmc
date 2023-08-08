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
  return migrate_expr_back(c_sizeof(t, ns));
}

expr2tc c_sizeof(const type2tc &src, const namespacet &ns)
{
  type2tc t = ns.follow(src);
  // A trick to deal with infinitely sized arrays. This only
  // comes useful when we need to "bound" infinite sizes
  // (e.g., in GOTO to C translator).
  if(is_array_type(t) && to_array_type(t).size_is_infinite)
  {
    // Right now we just introduce a symbol __ESBMC_INF_SIZE to represent
    // the "infinity" expression in irep1. (As of now migrating
    // the latter into irep2 results in an error.)
    expr2tc inf_size = symbol2tc(get_uint64_type(), "__ESBMC_INF_SIZE");
    expr2tc subtype_size = constant_int2tc(
      get_uint64_type(), type_byte_size(to_array_type(t).subtype));
    return mul2tc(get_uint64_type(), inf_size, subtype_size);
  }
  // Array size simplification and so forth will have already occurred in
  // migration, but we might still run into a nondeterministically sized
  // array.
  BigInt size = type_byte_size(t);
  return constant_int2tc(get_uint64_type(), size);
}
