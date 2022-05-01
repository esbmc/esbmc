#include <solidity-frontend/solidity_convert.h>
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/ieee_float.h>
#include <util/string_constant.h>
#include <util/std_expr.h>

// Integer literal
bool solidity_convertert::convert_integer_literal(
  const nlohmann::json &integer_literal,
  std::string the_value,
  exprt &dest)
{
  typet type;
  if(get_type_description(integer_literal, type))
    return true;

  assert(
    type.is_unsignedbv() || type.is_signedbv()); // for "_x=100", false || true

  exprt the_val;
  if(type.is_unsignedbv())
  {
    // extract the value: unsigned
    unsigned z_ext_value = std::stoul(the_value, nullptr);
    the_val = constant_exprt(
      integer2binary(z_ext_value, bv_width(type)),
      integer2string(z_ext_value),
      type);
  }
  else
  {
    assert(!"Unimplemented - Got signed type. Add sigend bv conversion");
  }

  dest.swap(the_val);
  return false;
}

bool solidity_convertert::convert_bool_literal(
  const nlohmann::json &bool_literal,
  std::string the_value,
  exprt &dest)
{
  typet type;
  if(get_type_description(bool_literal, type))
    return true;

  assert(type.is_bool());

  if(the_value == "true")
  {
    dest = true_exprt();
    return false;
  }

  if(the_value == "false")
  {
    dest = false_exprt();
    return false;
  }

  return true;
}

// TODO: Character literal
// TODO: String literal
// TODO: Float literal.
//    - Note: Currently Solidity does NOT support floating point data types or fp arithmetic.
//      Everything is done in fixed-point arithmetic as of Solidity compiler v0.8.6.
