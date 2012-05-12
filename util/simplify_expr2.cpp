#include "irep2.h"

#include <string.h>

#include <boost/static_assert.hpp>

#include <ansi-c/c_types.h>

expr2tc
expr2t::do_simplify(void) const
{

  return expr2tc();
}

static const type2tc &
decide_on_expr_type(const expr2tc &side1, const expr2tc &side2)
{

  // For some arithmetic expr, decide on the result of operating on them.

  // Fixedbv's take precedence.
  if (is_fixedbv_type(side1->type))
    return side1->type;
  if (is_fixedbv_type(side2->type))
    return side2->type;

  // If one operand is bool, return the other, as that's either bool or will
  // have a higher rank.
  if (is_bool_type(side1->type))
    return side2->type;
  else if (is_bool_type(side2->type))
    return side1->type;

  assert(is_bv_type(side1->type) && is_bv_type(side2->type));

  unsigned int side1_width = side1->type->get_width();
  unsigned int side2_width = side2->type->get_width();

  if (side1->type == side2->type) {
    if (side1_width > side2_width)
      return side1->type;
    else
      return side2->type;
  }

  // Differing between signed/unsigned bv type. Take unsigned if greatest.
  if (is_unsignedbv_type(side1->type) && side1_width >= side2_width)
    return side1->type;

  if (is_unsignedbv_type(side2->type) && side2_width >= side1_width)
    return side2->type;

  // Otherwise return the signed one;
  if (is_signedbv_type(side1->type))
    return side1->type;
  else
    return side2->type;
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
    break;
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

static void
fetch_ops_from_this_type(std::list<expr2tc> &ops, expr2t::expr_ids id,
                         const expr2tc &expr)
{

  if (expr->expr_id == id) {
    std::vector<const expr2tc*> operands;
    expr->list_operands(operands);
    for (std::vector<const expr2tc*>::const_iterator it = operands.begin();
         it != operands.end(); it++)
      fetch_ops_from_this_type(ops, id, **it);
  } else {
    ops.push_back(expr);
  }
}

static bool
rebalance_associative_tree(const expr2tc &expr, std::list<expr2tc> &ops,
        expr2tc (*create_obj_wrapper)(const expr2tc &arg1, const expr2tc &arg2))
{

  // So the purpose of this is to take a tree of all-the-same-operation and
  // re-arrange it so that there are some operations that we can simplify.
  // In old irep things like addition or subtraction or whatever could take
  // a whole set of operands (however many you shoved in the vector) and those
  // could all be simplified with each other. However, now that we've moved to
  // binary-only ireps, this isn't possible (and it's causing high
  // inefficiencies).
  // So instead, reconstruct a tree of all-the-same ireps into a vector and
  // try to simplify all of their contents, then try to reconfigure into another
  // set of operations.
  // There's great scope for making this /much/ more efficient via passing modes
  // and vectors downwards, but lets not prematurely optimise. All this is
  // faster than stringly stuff.

  std::list<expr2tc> operands;
  fetch_ops_from_this_type(operands, expr->expr_id, expr);

  // Are there enough constant values in there?
  unsigned int const_values = 0;
  unsigned int orig_size = operands.size();
  for (std::list<expr2tc>::const_iterator it = operands.begin();
       it != operands.end(); it++)
    if (is_constant_expr(*it))
      const_values++;

  // Nothing for us to simplify.
  if (const_values <= 1)
    return false;

  // Otherwise, we can go through simplifying operands.
  expr2tc accuml;
  for (std::list<expr2tc>::iterator it = operands.begin();
       it != operands.end(); it++) {
    if (!is_constant_expr(*it))
      continue;

    // We have a constant; do we have another constant to simplify with?
    if (is_nil_expr(accuml)) {
      // Juggle iterators, our iterator becomes invalid when we erase it.
      std::list<expr2tc>::iterator back = it;
      back--;
      accuml = *it;
      operands.erase(it);
      it = back;
      continue;
    }

    // Now attempt to simplify that. Create a new associative object and
    // give it a shot.
    expr2tc tmp = create_obj_wrapper(accuml, *it);
    if (is_nil_expr(tmp))
      continue; // Creating wrapper rejected it.

    tmp = tmp->simplify();
    if (is_nil_expr(tmp))
      // For whatever reason we're unable to simplify these two constants.
      continue;

    // It's good; remove that object from the list.
    accuml = tmp;
    std::list<expr2tc>::iterator back = it;
    back--;
    operands.erase(it);
    it = back;
  }

  // So, we've attempted to remove some things. There are three cases.
  // First, nothing was pulled out of the list. Shouldn't happen, but just
  // in case...
  if (operands.size() == orig_size)
    return false;

  // If only one constant value was removed from the list, then we attempted to
  // simplify two constants and it failed. No simplification.
  if (operands.size() == orig_size - 1)
    return false;

  // Finally; we've succeeded and simplified something. Push the simplified
  // constant back at the end of the list.
  operands.push_back(accuml);
  return true;
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

expr2tc
pointer_offs_simplify_2(const expr2tc &offs)
{

  if (is_symbol2t(offs) || is_constant_string2t(offs)) {
    return expr2tc(new constant_int2t(int_type2(), BigInt(0)));
  } else {
    // XXX - index simplification exists in old irep, however it looks quite
    // broken.
    return expr2tc();
  }
}

expr2tc
pointer_offset2t::do_simplify(void) const
{

  if (is_address_of2t(ptr_obj)) {
    const address_of2t &addrof = to_address_of2t(ptr_obj);
    return pointer_offs_simplify_2(addrof.ptr_obj);
  } else if (is_typecast2t(ptr_obj)) {
    const typecast2t &cast = to_typecast2t(ptr_obj);
    expr2tc new_ptr_offs = expr2tc(new pointer_offset2t(type, cast.from));
    expr2tc reduced = new_ptr_offs->simplify();

    // No good simplification -> return nothing
    if (is_nil_expr(reduced))
      return reduced;

    // If it simplified to zero, that's fine, return that.
    if (is_constant_int2t(reduced) &&
        to_constant_int2t(reduced).constant_value.is_zero())
      return reduced;

    // If it didn't reduce to zero, give up. Not sure why this is the case,
    // but it's what the old irep code does.
    return expr2tc();
  } else if (is_add2t(ptr_obj)) {
    const add2t &add = to_add2t(ptr_obj);

    // So, one of these should be a ptr type, or there isn't any point in this
    // being a pointer_offset irep.
    if (!is_pointer_type(add.side_1->type) &&
        !is_pointer_type(add.side_2->type))
      return expr2tc();

    // Can't have pointer-on-pointer arith.
    assert(!(is_pointer_type(add.side_1->type) &&
             is_pointer_type(add.side_2->type)));

    const expr2tc ptr_op = (is_pointer_type(add.side_1->type))
                           ? add.side_1 : add.side_2;
    const expr2tc non_ptr_op = (is_pointer_type(add.side_1->type))
                               ? add.side_2 : add.side_1;

    // Turn the pointer one into pointer_offset.
    expr2tc new_ptr_op = expr2tc(new pointer_offset2t(type, ptr_op));
    expr2tc new_add = expr2tc(new add2t(type, new_ptr_op, non_ptr_op));

    // XXX XXX XXX
    // XXX XXX XXX  This may be the source of pointer arith fail. Or lack of
    // XXX XXX XXX  consideration of pointer arithmetic.
    // XXX XXX XXX

    // So, this add is a valid simplification. We may be able to simplify
    // further though.
    expr2tc tmp = new_add->simplify();
    if (is_nil_expr(tmp))
      return new_add;
    else
      return tmp;
  } else {
    return expr2tc();
  }
}

expr2tc
index2t::do_simplify(void) const
{

  if (is_with2t(source_value)) {
    if (index == to_with2t(source_value).update_field) {
      // Index is the same as an update to the thing we're indexing; we can
      // just take the update value from the "with" below.
      return to_with2t(source_value).update_value;
    }

    // XXX jmorse old irep has an additional simplification of indexes with
    // a with below it; I haven't implemented it here out of partial lazyness,
    // but so that it can be studied in the future to see if it makes a
    // difference.
    return expr2tc();
  } else if (is_constant_array2t(source_value) && is_constant_int2t(index)) {
    const constant_array2t &arr = to_constant_array2t(source_value);
    const constant_int2t &idx = to_constant_int2t(index);

    // Index might be greater than the constant array size. This means we can't
    // simplify it, and the user might be eaten by an assertion failure in the
    // model. We don't have to think about this now though.
    unsigned long the_idx = idx.as_ulong();
    if (the_idx >= arr.datatype_members.size())
      return expr2tc();

    return arr.datatype_members[the_idx];
  } else if (is_constant_string2t(source_value) && is_constant_int2t(index)) {
    const constant_string2t &str = to_constant_string2t(source_value);
    const constant_int2t &idx = to_constant_int2t(index);

    // Same index situation
    unsigned long the_idx = idx.as_ulong();
    if (the_idx > str.value.as_string().size()) // allow reading null term.
      return expr2tc();

    // String constants had better be some kind of integer type
    assert(is_bv_type(type));
    unsigned long val = str.value.as_string().c_str()[the_idx];
    return expr2tc(new constant_int2t(type, BigInt(val)));
  } else if (is_constant_array_of2t(source_value)) {
    // XXX jmorse - here's hoping that something else is doing the bounds
    // checking on arrays here.
    return to_constant_array_of2t(source_value).initializer;
  } else {
    return expr2tc();
  }
}

expr2tc
not2t::do_simplify(void) const
{

  if (!is_constant_bool2t(value))
    return expr2tc();

  const constant_bool2t &val = to_constant_bool2t(value);
  return expr2tc(new constant_bool2t(!val.constant_value));
}

expr2tc
and2t::do_simplify(void) const
{

  if (!is_constant_bool2t(side_1) || !is_constant_bool2t(side_2))
    return expr2tc();

  const constant_bool2t &val1 = to_constant_bool2t(side_1);
  const constant_bool2t &val2 = to_constant_bool2t(side_2);
  return expr2tc(new constant_bool2t(val1.constant_value &&
                                     val2.constant_value));
}

expr2tc
or2t::do_simplify(void) const
{

  if (is_constant_bool2t(side_1) && to_constant_bool2t(side_1).constant_value)
    return expr2tc(new constant_bool2t(true));

  if (is_constant_bool2t(side_2) && to_constant_bool2t(side_2).constant_value)
    return expr2tc(new constant_bool2t(true));

  return expr2tc();
}

expr2tc
xor2t::do_simplify(void) const
{

  if (!is_constant_bool2t(side_1) || !is_constant_bool2t(side_2))
    return expr2tc();

  const constant_bool2t &val1 = to_constant_bool2t(side_1);
  const constant_bool2t &val2 = to_constant_bool2t(side_2);
  return expr2tc(new constant_bool2t(val1.constant_value ^
                                     val2.constant_value));
}

expr2tc
implies2t::do_simplify(void) const
{

  // False => * evaluate to true, always
  if (is_constant_bool2t(side_1) && !to_constant_bool2t(side_1).constant_value)
    return expr2tc(new constant_bool2t(true));

  // Otherwise, the only other thing that will make this expr always true is
  // if side 2 is true.
  if (is_constant_bool2t(side_2) && to_constant_bool2t(side_2).constant_value)
    return expr2tc(new constant_bool2t(true));

  return expr2tc();
}

static expr2tc
do_bit_munge_operation(void (*opfunc)(uint8_t *, uint8_t *, size_t),
                       const type2tc &type, const expr2tc &side_1,
                       const expr2tc &side_2)
{
  uint8_t buffer1[256], buffer2[256];
  BOOST_STATIC_ASSERT(sizeof(buffer1) == sizeof(buffer2));

  // Only support integer and's. If you're a float, pointer, or whatever, you're
  // on your own.
  if (!is_constant_int2t(side_1) || !is_constant_int2t(side_2))
    return expr2tc();

  // So - we can't make BigInt by itself do an and operation. But we can dump
  // it to a binary representation, and then and that.
  const constant_int2t &int1 = to_constant_int2t(side_1);
  const constant_int2t &int2 = to_constant_int2t(side_2);

  assert(int1.constant_value.get_len() < sizeof(buffer1) &&
         int2.constant_value.get_len() < sizeof(buffer2) &&
         "You've successfully generated an integer that's bigger than a massive"
         " stack buffer -- well done, this abort is for you!");

  unsigned int maxsize = std::max(int1.constant_value.get_len(),
                                  int2.constant_value.get_len());

  // Dump will zero-prefix and right align the output number.
  int1.constant_value.dump(buffer1, maxsize);
  int2.constant_value.dump(buffer2, maxsize);

  opfunc(buffer1, buffer2, maxsize);

  // And now, restore.
  constant_int2t *theint = new constant_int2t(type, BigInt(0));
  theint->constant_value.load(buffer1, maxsize);
  return expr2tc(theint);
}

static void
do_bitand_op(uint8_t *op1, uint8_t *op2, size_t n)
{
  for (size_t i = 0; i < n; i++)
    op1[i] &= op2[i];
}

expr2tc
bitand2t::do_simplify(void) const
{
  return do_bit_munge_operation(do_bitand_op, type, side_1, side_2);
}

static void
do_bitor_op(uint8_t *op1, uint8_t *op2, size_t n)
{
  for (size_t i = 0; i < n; i++)
    op1[i] |= op2[i];
}

expr2tc
bitor2t::do_simplify(void) const
{
  return do_bit_munge_operation(do_bitor_op, type, side_1, side_2);
}

static void
do_bitxor_op(uint8_t *op1, uint8_t *op2, size_t n)
{
  for (size_t i = 0; i < n; i++)
    op1[i] ^= op2[i];
}

expr2tc
bitxor2t::do_simplify(void) const
{
  return do_bit_munge_operation(do_bitxor_op, type, side_1, side_2);
}

static void
do_bitnand_op(uint8_t *op1, uint8_t *op2, size_t n)
{
  for (size_t i = 0; i < n; i++) {
    op1[i] &= op2[i];
    op1[i] = ~op1[i];
  }
}

expr2tc
bitnand2t::do_simplify(void) const
{
  return do_bit_munge_operation(do_bitnand_op, type, side_1, side_2);
}

static void
do_bitnor_op(uint8_t *op1, uint8_t *op2, size_t n)
{
  for (size_t i = 0; i < n; i++) {
    op1[i] |= op2[i];
    op1[i] = ~op1[i];
  }
}

expr2tc
bitnor2t::do_simplify(void) const
{
  return do_bit_munge_operation(do_bitnor_op, type, side_1, side_2);
}

static void
do_bitnxor_op(uint8_t *op1, uint8_t *op2, size_t n)
{
  for (size_t i = 0; i < n; i++) {
    op1[i] ^= op2[i];
    op1[i] = ~op1[i];
  }
}

expr2tc
bitnxor2t::do_simplify(void) const
{
  return do_bit_munge_operation(do_bitnxor_op, type, side_1, side_2);
}

static void
do_bitnot_op(uint8_t *op1, uint8_t *op2 __attribute__((unused)), size_t n)
{
  for (size_t i = 0; i < n; i++)
    op1[i] = ~op1[i];
}

expr2tc
bitnot2t::do_simplify(void) const
{
  return do_bit_munge_operation(do_bitnot_op, type, value, value);
}
