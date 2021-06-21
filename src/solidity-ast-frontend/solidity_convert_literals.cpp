#include <solidity-ast-frontend/solidity_convert.h>
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/ieee_float.h>
#include <util/string_constant.h>

bool solidity_convertert::convert_integer_literal(
  const IntegerLiteralTracker* integer_literal,
  exprt &dest)
{
  typet type;
  if(get_type(integer_literal->get_qualtype_tracker(), type))
    return true;

  assert(type.is_unsignedbv() || type.is_signedbv()); // for "_x=100", false || true

  exprt the_val;
  if(type.is_unsignedbv())
  {
    assert(!"Unimplemented - type.is_unsignedbv()");
  }
  else // "_x=100" uses this. "100" is considered as signed by default
  {
    the_val = constant_exprt(
      integer2binary(integer_literal->get_sgn_ext_value(), bv_width(type)), // val.getSExtValue()=100, bv_width(type)=32
      integer2string(integer_literal->get_sgn_ext_value()),
      type);
  }

  dest.swap(the_val);
  return false;
}
