#include "char_utils.h"

#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/std_types.h>
#include <util/std_expr.h>

namespace python_char_utils
{
namespace
{
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
    exprt zero_index = from_integer(0, index_type());
    char_expr = index_exprt(expr, zero_index, char_type());
    return true;
  }

  if (
    allow_void_pointer && expr.type().is_pointer() &&
    (expr.type().subtype().is_nil() || expr.type().subtype().id() == "empty"))
  {
    exprt char_ptr = typecast_exprt(expr, pointer_typet(char_type()));
    char_expr = dereference_exprt(char_ptr, char_type());
    return true;
  }

  return false;
}

exprt get_char_value_as_int(const exprt &expr, bool allow_void_pointer)
{
  exprt char_expr;
  if (!try_get_single_char_expr(expr, char_expr, allow_void_pointer))
    return nil_exprt();

  return typecast_exprt(char_expr, signedbv_typet(8));
}

} // namespace python_char_utils
