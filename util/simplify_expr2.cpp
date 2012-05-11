#include "irep2.h"

expr2tc
expr2t::do_simplify(void) const
{

  return expr2tc();
}

static void
to_fixedbv(const expr2tc &op, fixedbvt &bv)
{

  switch (op->expr_id) {
  case expr2t::constant_int_id:
    bv.spec = fixedbv_spect(64, 64); // XXX
    bv.from_integer(to_constant_int2t(op).constant_value);
    break;
  case expr2t::constant_bool_id:
    bv.spec = fixedbv_spect(1, 1); // XXX
    bv.from_integer((to_constant_bool2t(op).constant_value)
                     ? BigInt(1) : BigInt(0));
    break;
  case expr2t::constant_fixedbv_id:
    bv = to_constant_fixedbv2t(op).value;
  default:
    assert(0 && "Unexpectedly typed argument to to_fixedbv");
  }
}

static expr2tc
from_fixedbv(const fixedbvt &bv, const type2tc &type)
{

  switch (type->type_id) {
  case type2t::bool_id:
    {
    bool theval = bv.is_zero() ? false : true;
    return expr2tc(new constant_bool2t(theval));
    }
  case type2t::unsignedbv_id:
  case type2t::signedbv_id:
    return expr2tc(new constant_int2t(type, bv.to_integer()));
  case type2t::fixedbv_id:
    return expr2tc(new constant_fixedbv2t(type, bv));
  default:
    assert(0 && "Unexpected typed argument to from_fixedbv");
  }
}

expr2tc
add2t::do_simplify(void) const
{

  if (!is_constant_expr(side_1) || !is_constant_expr(side_2))
    return expr2tc();

  assert((is_constant_int2t(side_1) || is_constant_bool2t(side_1) ||
          is_constant_fixedbv2t(side_1)) &&
         (is_constant_int2t(side_2) || is_constant_bool2t(side_2) ||
          is_constant_fixedbv2t(side_2)) &&
          "Operands to simplified add must be int, bool or fixedbv");

  // The plan: convert everything to a fixedbv, operate, and convert back to
  // whatever form we need. Fixedbv appears to be a wrapper around BigInt.
  fixedbvt operand1, operand2;
  to_fixedbv(side_1, operand1);
  to_fixedbv(side_2, operand2);

  operand1 += operand2;

  return from_fixedbv(operand1, type);
}

expr2tc
sub2t::do_simplify(void) const
{

  if (!is_constant_expr(side_1) || !is_constant_expr(side_2))
    return expr2tc();

  assert((is_constant_int2t(side_1) || is_constant_bool2t(side_1) ||
          is_constant_fixedbv2t(side_1)) &&
         (is_constant_int2t(side_2) || is_constant_bool2t(side_2) ||
          is_constant_fixedbv2t(side_2)) &&
          "Operands to simplified sub must be int, bool or fixedbv");

  fixedbvt operand1, operand2;
  to_fixedbv(side_1, operand1);
  to_fixedbv(side_2, operand2);

  operand1 -= operand2;

  return from_fixedbv(operand1, type);
}

expr2tc
mul2t::do_simplify(void) const
{

  if (!is_constant_expr(side_1) || !is_constant_expr(side_2))
    return expr2tc();

  assert((is_constant_int2t(side_1) || is_constant_bool2t(side_1) ||
          is_constant_fixedbv2t(side_1)) &&
         (is_constant_int2t(side_2) || is_constant_bool2t(side_2) ||
          is_constant_fixedbv2t(side_2)) &&
          "Operands to simplified mul must be int, bool or fixedbv");

  fixedbvt operand1, operand2;
  to_fixedbv(side_1, operand1);
  to_fixedbv(side_2, operand2);

  operand1 *= operand2;

  return from_fixedbv(operand1, type);
}

expr2tc
div2t::do_simplify(void) const
{

  if (!is_constant_expr(side_1) || !is_constant_expr(side_2))
    return expr2tc();

  assert((is_constant_int2t(side_1) || is_constant_bool2t(side_1) ||
          is_constant_fixedbv2t(side_1)) &&
         (is_constant_int2t(side_2) || is_constant_bool2t(side_2) ||
          is_constant_fixedbv2t(side_2)) &&
          "Operands to simplified div must be int, bool or fixedbv");

  fixedbvt operand1, operand2;
  to_fixedbv(side_1, operand1);
  to_fixedbv(side_2, operand2);

  operand1 /= operand2;

  return from_fixedbv(operand1, type);
}

expr2tc
modulus2t::do_simplify(void) const
{

  if (!is_constant_expr(side_1) || !is_constant_expr(side_2))
    return expr2tc();

  assert((is_constant_int2t(side_1) || is_constant_bool2t(side_1) ||
          is_constant_fixedbv2t(side_1)) &&
         (is_constant_int2t(side_2) || is_constant_bool2t(side_2) ||
          is_constant_fixedbv2t(side_2)) &&
          "Operands to simplified div must be int, bool or fixedbv");

  fixedbvt operand1, operand2;
  to_fixedbv(side_1, operand1);
  to_fixedbv(side_2, operand2);

  fixedbvt quotient = operand1;
  quotient /= operand2; // calculate quotient.
  quotient *= operand2; // to subtract.
  operand1 -= quotient; // And finally, the remainder.

  return from_fixedbv(operand1, type);
}

expr2tc
with2t::do_simplify(void) const
{

  if (is_constant_struct2t(source_value)) {
    const constant_struct2t &c_struct = to_constant_struct2t(source_value);
    const constant_string2t &memb = to_constant_string2t(update_field);
    unsigned no = get_component_number(type, memb.value);
    assert(no < c_struct.datatype_members.size());

    // Clone constant struct, update its field according to this "with".
    constant_struct2tc s = expr2tc(c_struct.clone());
    s.get()->datatype_members[no] = update_value;
    return expr2tc(s);
  } else if (is_constant_array2t(source_value) &&
             is_constant_int2t(update_field)) {
    const constant_array2t &array = to_constant_array2t(source_value);
    const constant_int2t &index = to_constant_int2t(update_field);

    // Index may be out of bounds. That's an error in the program, but not in
    // the model we're generating, so permit it. Can't simplify it though.
    if (index.as_ulong() >= array.datatype_members.size())
      return expr2tc();

    constant_array2tc arr = expr2tc(array.clone());
    arr.get()->datatype_members[index.as_ulong()] = update_value;
    return expr2tc(arr);
  } else {
    return expr2tc();
  }
}

expr2tc
member2t::do_simplify(void) const
{

  if (is_constant_struct2t(source_value) || is_constant_union2t(source_value)) {
    unsigned no = get_component_number(source_value->type, member);

    // Clone constant struct, update its field according to this "with".
    expr2tc s;
    if (is_constant_struct2t(source_value)) {
      s = to_constant_struct2t(source_value).datatype_members[no];
    } else {
      s = to_constant_union2t(source_value).datatype_members[no];
    }

    assert(type == s->type);
    return s;
  } else {
    return expr2tc();
  }
}
