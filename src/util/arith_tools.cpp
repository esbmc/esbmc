#include <cassert>
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <irep2/irep2_utils.h>

bool to_integer(const exprt &expr, BigInt &int_value)
{
  if (!expr.is_constant())
    return true;

  const std::string &value = expr.value().as_string();
  const irep_idt &type_id = expr.type().id();

  if (type_id == "pointer")
  {
    if (value == "NULL")
    {
      int_value = 0;
      return false;
    }
  }
  else if (type_id == "c_enum" || type_id == "symbol")
  {
    int_value = string2integer(value);
    return false;
  }
  else if (type_id == "unsignedbv")
  {
    int_value = binary2integer(value, false);
    return false;
  }
  else if (type_id == "signedbv")
  {
    int_value = binary2integer(value, true);
    return false;
  }

  return true;
}

exprt from_integer(const BigInt &int_value, const typet &type)
{
  exprt expr;

  expr.clear();
  expr.type() = type;
  expr.id("constant");

  const irep_idt &type_id = type.id();

  if (type_id == "unsignedbv" || type_id == "signedbv")
  {
    expr.value(integer2binary(int_value, bv_width(type)));
    return expr;
  }
  if (type_id == "bool")
  {
    if (int_value == 0)
    {
      expr.make_false();
      return expr;
    }
    if (int_value == 1)
    {
      expr.make_true();
      return expr;
    }
  }

  expr.make_nil();
  return expr;
}

expr2tc from_integer(const BigInt &int_value, const type2tc &type)
{
  switch (type->type_id)
  {
  case type2t::bool_id:
    return !int_value.is_zero() ? gen_true_expr() : gen_false_expr();

  case type2t::unsignedbv_id:
  case type2t::signedbv_id:
    return constant_int2tc(type, int_value);

  case type2t::fixedbv_id:
  {
    expr2tc f = constant_fixedbv2tc(fixedbvt(fixedbv_spect(
      to_fixedbv_type(type).width, to_fixedbv_type(type).integer_bits)));
    to_constant_fixedbv2t(f).value.from_integer(int_value);
    return f;
  }

  case type2t::floatbv_id:
  {
    expr2tc f = constant_floatbv2tc(ieee_floatt(ieee_float_spect(
      to_floatbv_type(type).fraction, to_floatbv_type(type).exponent)));
    to_constant_floatbv2t(f).value.from_integer(int_value);
    return f;
  }
  default:
    abort();
  }
}

exprt from_double(double val, const typet &type)
{
  ieee_floatt f;
  f.rounding_mode = ieee_floatt::ROUND_TO_EVEN;

  if (type.is_floatbv())
    f.change_spec(ieee_float_spect(to_floatbv_type(type)));
  else
    throw "from_double: unsupported type";

  f.from_double(val);
  return f.to_expr();
}

BigInt power(const BigInt &base, const BigInt &exponent)
{
  assert(exponent >= 0);

  if (exponent == 0)
    return 1;

  BigInt result(base);
  BigInt count(exponent - 1);

  while (count != 0)
  {
    result *= base;
    --count;
  }

  return result;
}

/// ceil(log2(size))
BigInt address_bits(const BigInt &size)
{
  BigInt result, x = 2;
  for (result = 1; x < size; result += 1, x *= 2)
    ;

  return result;
}
