#include "char_utils.h"

#include <irep2/irep2_utils.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/migrate.h>
#include <util/std_types.h>
#include <util/std_expr.h>

namespace python_char_utils
{
namespace
{
// V.3: IREP2 expression-construction helpers (exact round-trip of the legacy
// constructors; behaviour-preserving). Back-migrated for the legacy seam.
exprt build_index(const exprt &arr, const exprt &idx, const typet &t)
{
  expr2tc arr2, idx2;
  migrate_expr(arr, arr2);
  migrate_expr(idx, idx2);
  return migrate_expr_back(index2tc(migrate_type(t), arr2, idx2));
}

exprt build_typecast(const exprt &from, const typet &t)
{
  expr2tc from2;
  migrate_expr(from, from2);
  return migrate_expr_back(typecast2tc(migrate_type(t), from2));
}

exprt build_dereference(const exprt &ptr, const typet &t)
{
  expr2tc ptr2;
  migrate_expr(ptr, ptr2);
  return migrate_expr_back(dereference2tc(migrate_type(t), ptr2));
}

bool is_char_array_of_length(const typet &type, std::size_t expected_length)
{
  if (!type.is_array())
    return false;

  const auto &array_type = to_array_type(type);
  if (array_type.subtype() != char_type())
    return false;

  if (!array_type.size().is_constant())
    return false;

  BigInt size_int;
  if (to_integer(to_constant_expr(array_type.size()), size_int))
    return false;

  return size_int == BigInt(expected_length);
}
} // namespace

bool try_get_single_char_expr(
  const exprt &expr,
  exprt &char_expr,
  bool allow_void_pointer)
{
  if (expr.type() == char_type())
  {
    char_expr = expr;
    return true;
  }

  if (is_char_array_of_length(expr.type(), 2))
  {
    // Source is a 2-char array (checked above) -> index2t source ok.
    exprt zero_index = from_integer(0, index_type());
    char_expr = build_index(expr, zero_index, char_type());
    return true;
  }

  if (
    allow_void_pointer && expr.type().is_pointer() &&
    (expr.type().subtype().is_nil() || expr.type().subtype().id() == "empty"))
  {
    exprt char_ptr = build_typecast(expr, pointer_typet(char_type()));
    char_expr = build_dereference(char_ptr, char_type());
    return true;
  }

  return false;
}

exprt get_char_value_as_int(const exprt &expr, bool allow_void_pointer)
{
  exprt char_expr;
  if (!try_get_single_char_expr(expr, char_expr, allow_void_pointer))
    return nil_exprt();

  return build_typecast(char_expr, signedbv_typet(8));
}

} // namespace python_char_utils
