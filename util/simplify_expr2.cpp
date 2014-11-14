#include "irep2.h"

#include <string.h>

#include <boost/static_assert.hpp>

#include <ansi-c/c_types.h>
#include <base_type.h>
#include <type_byte_size.h>

expr2tc
expr2t::do_simplify(bool second __attribute__((unused))) const
{

  return expr2tc();
}

static const type2tc &
decide_on_expr_type(const expr2tc &side1, const expr2tc &side2)
{

  // For some arithmetic expr, decide on the result of operating on them.
  if (is_pointer_type(side1))
    return side1->type;
  if (is_pointer_type(side2))
    return side2->type;

  // Fixedbv's take precedence.
  if (is_fixedbv_type(side1))
    return side1->type;
  if (is_fixedbv_type(side2))
    return side2->type;

  // If one operand is bool, return the other, as that's either bool or will
  // have a higher rank.
  if (is_bool_type(side1))
    return side2->type;
  else if (is_bool_type(side2))
    return side1->type;

  assert(is_bv_type(side1) && is_bv_type(side2));

  unsigned int side1_width = side1->type->get_width();
  unsigned int side2_width = side2->type->get_width();

  if (side1->type == side2->type) {
    if (side1_width > side2_width)
      return side1->type;
    else
      return side2->type;
  }

  // Differing between signed/unsigned bv type. Take unsigned if greatest.
  if (is_unsignedbv_type(side1) && side1_width >= side2_width)
    return side1->type;

  if (is_unsignedbv_type(side2) && side2_width >= side1_width)
    return side2->type;

  // Otherwise return the signed one;
  if (is_signedbv_type(side1))
    return side1->type;
  else
    return side2->type;
}

static void
to_fixedbv(const expr2tc &op, fixedbvt &bv)
{

  // XXX XXX XXX -- this would appear to be broken in a couple of cases.
  // Take a look at the typecast cvt code -- where we're taking the target
  // type, fetching the fixedbv spec from that, and constructing from there.
  // Which turns out not to break test cases like 01_cbmc_Fixedbv8

  switch (op->expr_id) {
  case expr2t::constant_int_id:
    bv.spec = fixedbv_spect(128, 64); // XXX
    bv.from_integer(to_constant_int2t(op).constant_value);
    break;
  case expr2t::constant_bool_id:
    bv.spec = fixedbv_spect(32, 16); // XXX
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
    {
    // To integer truncates non-integer bits, it turns out.
    BigInt tmp = bv.to_integer();

    // Round away upper bits, just in case we're decreasing accuracy here.
    unsigned int bits = type->get_width();
    fixedbvt tmp_bv;
    tmp_bv.spec = bv.spec;
    tmp_bv.from_integer(tmp);
    tmp_bv.round(fixedbv_spect(bits*2, bits));

    // If we're converting to a signedbv, the top bit being set means negative.
    if (is_signedbv_type(type) && !tmp.is_negative()) {
      assert(type->get_width() <= 64);
      uint64_t top_bit = 1ULL << (type->get_width()-1);
      uint64_t cur_val = tmp_bv.to_integer().to_uint64();
      if (cur_val >= top_bit) {
        // Construct some bit mask gumpf as a sign extension
        int64_t large_int = -1;
        large_int <<= (type->get_width() - 1);
        large_int |= cur_val;
        tmp_bv.from_integer(large_int);
      }
    } else if (is_signedbv_type(type)) {
      int64_t theval = tmp.to_int64();
      tmp_bv.from_integer(theval);
    } else if (is_unsignedbv_type(type) && tmp_bv.to_integer().is_negative()) {
      // Need to switch this number to being an unsigned representation of the
      // same bit vector.
      int64_t the_num = tmp_bv.to_integer().to_int64();

      unsigned int width = type->get_width();
      uint64_t mask = (1ULL << width) - 1ULL;
      if (width == 64)
        mask = 0xFFFFFFFFFFFFFFFF;

      uint64_t output = the_num & mask;
      tmp_bv.from_integer(BigInt(output));
    }

    // And done.
    return expr2tc(new constant_int2t(type, tmp_bv.to_integer()));
    }
  case type2t::fixedbv_id:
    return expr2tc(new constant_fixedbv2t(type, bv));
  default:
    assert(0 && "Unexpected typed argument to from_fixedbv");
  }
}

void
make_fixedbv_types_match(fixedbvt &bv1, fixedbvt &bv2)
{

  // First, simple case,
  if (bv1.spec.width == bv2.spec.width &&
      bv1.spec.integer_bits == bv2.spec.integer_bits)
    return;

  // Otherwise, pick the large one, assuming we're always keeping the int/frac
  // division at the middle,
  if (bv1.spec.width > bv2.spec.width)
    bv2.round(bv1.spec);
  else
    bv1.round(bv2.spec);

  return;
}

static void
fetch_ops_from_this_type(std::list<expr2tc> &ops, expr2t::expr_ids id,
                         const expr2tc &expr)
{

  if (expr->expr_id == id) {
    forall_operands2(it, idx, expr)
      fetch_ops_from_this_type(ops, id, *it);
  } else {
    ops.push_back(expr);
  }
}

static bool
rebalance_associative_tree(const expr2t &expr, std::list<expr2tc> &ops,
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

  // Extract immediate operands
  std::list<const expr2tc*> immediate_operands;
  expr.list_operands(immediate_operands);
  for (std::list<const expr2tc*>::const_iterator
       it = immediate_operands.begin(); it != immediate_operands.end(); it++)
      fetch_ops_from_this_type(ops, expr.expr_id, **it);

  // Are there enough constant values in there?
  unsigned int const_values = 0;
  unsigned int orig_size = ops.size();
  for (std::list<expr2tc>::const_iterator it = ops.begin();
       it != ops.end(); it++)
    if (is_constant_expr(*it))
      const_values++;

  // Nothing for us to simplify.
  if (const_values <= 1)
    return false;

  // Otherwise, we can go through simplifying operands.
  expr2tc accuml;
  for (std::list<expr2tc>::iterator it = ops.begin();
       it != ops.end(); it++) {
    if (!is_constant_expr(*it))
      continue;

    // We have a constant; do we have another constant to simplify with?
    if (is_nil_expr(accuml)) {
      // Juggle iterators, our iterator becomes invalid when we erase it.
      std::list<expr2tc>::iterator back = it;
      back--;
      accuml = *it;
      ops.erase(it);
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
    ops.erase(it);
    it = back;
  }

  // So, we've attempted to remove some things. There are three cases.
  // First, nothing was pulled out of the list. Shouldn't happen, but just
  // in case...
  if (ops.size() == orig_size)
    return false;

  // If only one constant value was removed from the list, then we attempted to
  // simplify two constants and it failed. No simplification.
  if (ops.size() == orig_size - 1)
    return false;

  // Finally; we've succeeded and simplified something. Push the simplified
  // constant back at the end of the list.
  ops.push_back(accuml);
  return true;
}

expr2tc
attempt_associative_simplify(const expr2t &expr,
        expr2tc (*create_obj_wrapper)(const expr2tc &arg1, const expr2tc &arg2))
{

  std::list<expr2tc> operands;
  if (rebalance_associative_tree(expr, operands, create_obj_wrapper)) {
    // Horray, we simplified. Recreate.
    assert(operands.size() >= 2);
    std::list<expr2tc>::const_iterator it = operands.begin();
    expr2tc accuml = *it;
    it++;
    for ( ; it != operands.end(); it++) {
      expr2tc tmp;
      accuml = create_obj_wrapper(accuml, *it);
      if (is_nil_expr(accuml))
        return expr2tc(); // wrapper rejected new obj :O
    }

    return accuml;
  } else {
    return expr2tc();
  }
}

static expr2tc
create_add_wrapper(const expr2tc &arg1, const expr2tc &arg2)
{

  const type2tc &type = decide_on_expr_type(arg1, arg2);
  return expr2tc(new add2t(type, arg1, arg2));
}

expr2tc
add2t::do_simplify(bool second) const
{

  if (!is_constant_expr(side_1) || !is_constant_expr(side_2)) {
    if (!second)
      // Wait until operands are simplified
      return expr2tc();

    // If one is zero, return the other.
    if (is_constant_expr(side_1)) {
      fixedbvt op;
      to_fixedbv(side_1, op);
      if (op.is_zero())
        return side_2;
    }

    if (is_constant_expr(side_2)) {
      fixedbvt op;
      to_fixedbv(side_2, op);
      if (op.is_zero())
        return side_1;
    }

    // Attempt to simplify associative tree.
    return attempt_associative_simplify(*this, create_add_wrapper);
  }

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

  make_fixedbv_types_match(operand1, operand2);
  operand1 += operand2;

  return from_fixedbv(operand1, type);
}

expr2tc
sub2t::do_simplify(bool second __attribute__((unused))) const
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

  make_fixedbv_types_match(operand1, operand2);
  operand1 -= operand2;

  return from_fixedbv(operand1, type);
}

static expr2tc
mul_check_for_zero_or_one(const expr2tc &checking, const expr2tc &other_op,
                          const type2tc &ourtype)
{

  fixedbvt operand;
  to_fixedbv(checking, operand);

  if (operand.is_zero())
    return from_fixedbv(operand, ourtype); // Mul by zero -> return zero

  fixedbvt one = operand;
  one.from_integer(BigInt(1));
  if (operand == one) { // Mul by one -> return other operand.
    if (other_op->type == ourtype)
      return other_op;
    else
      return expr2tc(new typecast2t(ourtype, other_op));
  }

  // Return nothing, nothing to simplify.
  return expr2tc();
}

expr2tc
mul2t::do_simplify(bool second __attribute__((unused))) const
{

  // If we don't have two constant operands, check for one being zero.
  if (!is_constant_expr(side_1) || !is_constant_expr(side_2)) {
    if (is_constant_expr(side_1)) {
      expr2tc tmp = mul_check_for_zero_or_one(side_1, side_2, type);
      if (!is_nil_expr(tmp))
        return tmp;
    }

    if (is_constant_expr(side_2)) {
      expr2tc tmp = mul_check_for_zero_or_one(side_2, side_1, type);
      if (!is_nil_expr(tmp))
        return tmp;
    }

    return expr2tc();
  }

  assert((is_constant_int2t(side_1) || is_constant_bool2t(side_1) ||
          is_constant_fixedbv2t(side_1)) &&
         (is_constant_int2t(side_2) || is_constant_bool2t(side_2) ||
          is_constant_fixedbv2t(side_2)) &&
          "Operands to simplified mul must be int, bool or fixedbv");

  fixedbvt operand1, operand2;
  to_fixedbv(side_1, operand1);
  to_fixedbv(side_2, operand2);

  // Multiplication by any zero operand -> zero
  if (operand1.is_zero())
    return from_fixedbv(operand1, type);
  if (operand2.is_zero())
    return from_fixedbv(operand2, type);

  make_fixedbv_types_match(operand1, operand2);
  operand1 *= operand2;

  return from_fixedbv(operand1, type);
}

expr2tc
div2t::do_simplify(bool second __attribute__((unused))) const
{

  if (!is_constant_expr(side_1) || !is_constant_expr(side_2)) {
    // If side_1 is zero, result is zero.
    if (is_constant_expr(side_1)) {
      fixedbvt operand1;
      to_fixedbv(side_1, operand1);
      if (operand1.is_zero())
        return from_fixedbv(operand1, type);
      else
        return expr2tc();
    } else {
      return expr2tc();
    }
  }

  assert((is_constant_int2t(side_1) || is_constant_bool2t(side_1) ||
          is_constant_fixedbv2t(side_1)) &&
         (is_constant_int2t(side_2) || is_constant_bool2t(side_2) ||
          is_constant_fixedbv2t(side_2)) &&
          "Operands to simplified div must be int, bool or fixedbv");

  fixedbvt operand1, operand2;
  to_fixedbv(side_1, operand1);
  to_fixedbv(side_2, operand2);

  if (operand1.is_zero())
    return from_fixedbv(operand1, type);

  // Div by zero -> not allowed. XXX - this should never reach this point, but
  // if it does, perhaps the caller has some nondet guard that guarentees it's
  // never evaluated. Either way, don't explode, just refuse to simplify.
  if (operand2.is_zero())
    return expr2tc();

  make_fixedbv_types_match(operand1, operand2);
  operand1 /= operand2;

  return from_fixedbv(operand1, type);
}

expr2tc
modulus2t::do_simplify(bool second __attribute__((unused))) const
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

  make_fixedbv_types_match(operand1, operand2);
  fixedbvt quotient = operand1;
  quotient /= operand2; // calculate quotient.
  // Truncate fraction bits.
  quotient.from_integer(quotient.to_integer());
  quotient *= operand2; // to subtract.
  operand1 -= quotient; // And finally, the remainder.

  return from_fixedbv(operand1, type);
}

expr2tc
neg2t::do_simplify(bool second __attribute__((unused))) const
{

  if (!is_constant_expr(value))
    return expr2tc();

  assert((is_constant_int2t(value) || is_constant_bool2t(value) ||
          is_constant_fixedbv2t(value)) &&
          "Operands to simplified neg2t must be int, bool or fixedbv");

  // The plan: convert everything to a fixedbv, operate, and convert back to
  // whatever form we need. Fixedbv appears to be a wrapper around BigInt.
  fixedbvt operand;
  to_fixedbv(value, operand);

  operand.negate();

  return from_fixedbv(operand, type);
}

expr2tc
with2t::do_simplify(bool second __attribute__((unused))) const
{

  if (is_constant_struct2t(source_value)) {
    const constant_struct2t &c_struct = to_constant_struct2t(source_value);
    const constant_string2t &memb = to_constant_string2t(update_field);
    unsigned no = static_cast<const struct_union_data&>(*type.get())
                  .get_component_number(memb.value);
    assert(no < c_struct.datatype_members.size());

    // Clone constant struct, update its field according to this "with".
    constant_struct2tc s = expr2tc(c_struct.clone());
    s.get()->datatype_members[no] = update_value;
    return expr2tc(s);
  } else if (is_constant_union2t(source_value)) {
    const constant_union2t &c_union = to_constant_union2t(source_value);
    const union_type2t &thetype = to_union_type(c_union.type);
    const constant_string2t &memb = to_constant_string2t(update_field);
    unsigned no = static_cast<const struct_union_data&>(*c_union.type.get())
                  .get_component_number(memb.value);
    assert(no < thetype.member_names.size());

    // If the update value type matches the current lump of data's type, we can
    // just replace it with the new value. As far as I can tell, constant unions
    // only ever contain one member, and it's the member most recently written.
    if (thetype.members[no] != update_value->type)
      return expr2tc();

    std::vector<expr2tc> newmembers;
    newmembers.push_back(update_value);
    return expr2tc(new constant_union2t(type, newmembers));
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
  } else if (is_constant_array_of2t(source_value)) {
    const constant_array_of2t &array = to_constant_array_of2t(source_value);

    // We don't simplify away these withs if // the array_of is infinitely
    // sized. This is because infinitely sized arrays are no longer converted
    // correctly in the solver backend (they're simply not supported by SMT).
    // Thus it becomes important to be able to assign a value to a field in an
    // aray_of and not have it const propagatated away.
    const constant_array_of2t &thearray = to_constant_array_of2t(source_value);
    const array_type2t &arr_type = to_array_type(thearray.type);
    if (arr_type.size_is_infinite)
      return expr2tc();

    // We can eliminate this operation if the operand to this with is the same
    // as the initializer.
    if (update_value == array.initializer)
      return source_value;
    else
      return expr2tc();
  } else {
    return expr2tc();
  }
}

expr2tc
member2t::do_simplify(bool second __attribute__((unused))) const
{

  if (is_constant_struct2t(source_value) || is_constant_union2t(source_value)) {
    unsigned no =
      static_cast<const struct_union_data&>(*source_value->type.get())
      .get_component_number(member);

    // Clone constant struct, update its field according to this "with".
    expr2tc s;
    if (is_constant_struct2t(source_value)) {
      s = to_constant_struct2t(source_value).datatype_members[no];

      assert(is_pointer_type(type) ||
             base_type_eq(type, s->type, namespacet(contextt())));
    } else {
      // XXX jmorse HHHNNGGGGGG, it would appear that the constant arrays spat
      // out by the parser are somewhat undefined; to the extent that there are
      // an undefined number of operands in the datatype_members vector. So
      // bounds check first that we can actually perform this member operation.
      const constant_union2t &uni = to_constant_union2t(source_value);
      if (uni.datatype_members.size() <= no)
        return expr2tc();

      s = uni.datatype_members[no];

      // If the type we just selected isn't compatible, it means that whatever
      // field is in the constant union /isn't/ the field we're selecting from
      // it. So don't simplify it, because we can't.
      if (!is_pointer_type(type) &&
          !base_type_eq(type, s->type, namespacet(contextt())))
        return expr2tc();
    }


    return s;
  } else {
    return expr2tc();
  }
}

expr2tc
pointer_offs_simplify_2(const expr2tc &offs, const type2tc &type)
{

  if (is_symbol2t(offs) || is_constant_string2t(offs)) {
    return expr2tc(new constant_int2t(type, BigInt(0)));
  } else if (is_index2t(offs)) {
    const index2t &index = to_index2t(offs);

    if (is_symbol2t(index.source_value) && is_constant_int2t(index.index)) {
      // We can reduce to that index offset.
      const array_type2t &arr = to_array_type(index.source_value->type);
      unsigned int widthbits = arr.subtype->get_width();
      unsigned int widthbytes = widthbits / 8;
      BigInt val = to_constant_int2t(index.index).constant_value;
      val *= widthbytes;
      return expr2tc(new constant_int2t(type, val));
    } else if (is_constant_string2t(index.source_value) &&
               is_constant_int2t(index.index)) {
      // This can also be simplified to an array offset. Just return the index,
      // as the string elements are all 8 bit bytes.
      return index.index;
    } else {
      return expr2tc();
    }
  } else {
    return expr2tc();
  }
}

expr2tc
pointer_offset2t::do_simplify(bool second) const
{

  // XXX - this could be better. But the current implementation catches most
  // cases that ESBMC produces internally.

  if (second && is_address_of2t(ptr_obj)) {
    const address_of2t &addrof = to_address_of2t(ptr_obj);
    return pointer_offs_simplify_2(addrof.ptr_obj, type);
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
    if (!is_pointer_type(add.side_1) &&
        !is_pointer_type(add.side_2))
      return expr2tc();

    // Can't have pointer-on-pointer arith.
    assert(!(is_pointer_type(add.side_1) &&
             is_pointer_type(add.side_2)));

    expr2tc ptr_op = (is_pointer_type(add.side_1)) ? add.side_1 : add.side_2;
    expr2tc non_ptr_op =
      (is_pointer_type(add.side_1)) ? add.side_2 : add.side_1;

    // Can't do any kind of simplification if the ptr op has a symbolic type.
    // Let the SMT layer handle this. In the future, can we pass around a
    // namespace?
    if (is_symbol_type(to_pointer_type(ptr_op->type).subtype))
      return expr2tc();

    // Turn the pointer one into pointer_offset.
    expr2tc new_ptr_op = expr2tc(new pointer_offset2t(type, ptr_op));
    // And multiply the non pointer one by the type size.
    type2tc ptr_int_type = get_int_type(config.ansi_c.pointer_width);
    type2tc ptr_subtype = to_pointer_type(ptr_op->type).subtype;
    mp_integer thesize = (is_empty_type(ptr_subtype)) ? 1
                          : type_byte_size(*ptr_subtype.get());
#if 0
    constant_int2tc type_size(ptr_int_type, thesize);
#endif
    constant_int2tc type_size(type, thesize);

#if 0
    if (non_ptr_op->type->get_width() != config.ansi_c.pointer_width)
      non_ptr_op = typecast2tc(ptr_int_type, non_ptr_op);
#endif
    // Herp derp tacas
    if (non_ptr_op->type->get_width() != type->get_width())
      non_ptr_op = typecast2tc(type, non_ptr_op);

#if 0
    mul2tc new_non_ptr_op(ptr_int_type, non_ptr_op, type_size);
#endif
    mul2tc new_non_ptr_op(type, non_ptr_op, type_size);

    expr2tc new_add = expr2tc(new add2t(type, new_ptr_op, new_non_ptr_op));

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
index2t::do_simplify(bool second __attribute__((unused))) const
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
    if (idx.constant_value.is_negative())
      return expr2tc();

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
not2t::do_simplify(bool second __attribute__((unused))) const
{

  if (is_not2t(value))
    // Bam. These negate.
    return to_not2t(value).value;

  if (!is_constant_bool2t(value))
    return expr2tc();

  const constant_bool2t &val = to_constant_bool2t(value);
  return expr2tc(new constant_bool2t(!val.constant_value));
}

expr2tc
and2t::do_simplify(bool second __attribute__((unused))) const
{

  if (is_constant_bool2t(side_1)) {
   if (to_constant_bool2t(side_1).constant_value)
     // constant true; other operand determines truth
     return side_2;
   else
     // constant false; never true.
     return side_1;
  }

  if (is_constant_bool2t(side_2)) {
   if (to_constant_bool2t(side_2).constant_value)
     // constant true; other operand determines truth
     return side_1;
   else
     // constant false; never true.
     return side_2;
  }

  if (!is_constant_bool2t(side_1) || !is_constant_bool2t(side_2))
    return expr2tc();

  const constant_bool2t &val1 = to_constant_bool2t(side_1);
  const constant_bool2t &val2 = to_constant_bool2t(side_2);
  return expr2tc(new constant_bool2t(val1.constant_value &&
                                     val2.constant_value));
}

expr2tc
or2t::do_simplify(bool second __attribute__((unused))) const
{

  // If either operand is true, the expr is true
  if (is_constant_bool2t(side_1) && to_constant_bool2t(side_1).constant_value)
    return true_expr;

  if (is_constant_bool2t(side_2) && to_constant_bool2t(side_2).constant_value)
    return true_expr;

  // If both or operands are false, the expr is false.
  if (is_constant_bool2t(side_1)
      && !to_constant_bool2t(side_1).constant_value
      && is_constant_bool2t(side_2)
      && !to_constant_bool2t(side_2).constant_value)
    return false_expr;

  return expr2tc();
}

expr2tc
xor2t::do_simplify(bool second __attribute__((unused))) const
{

  if (!is_constant_bool2t(side_1) || !is_constant_bool2t(side_2))
    return expr2tc();

  const constant_bool2t &val1 = to_constant_bool2t(side_1);
  const constant_bool2t &val2 = to_constant_bool2t(side_2);
  return expr2tc(new constant_bool2t(val1.constant_value ^
                                     val2.constant_value));
}

expr2tc
implies2t::do_simplify(bool second __attribute__((unused))) const
{

  // False => * evaluate to true, always
  if (is_constant_bool2t(side_1) && !to_constant_bool2t(side_1).constant_value)
    return true_expr;

  // Otherwise, the only other thing that will make this expr always true is
  // if side 2 is true.
  if (is_constant_bool2t(side_2) && to_constant_bool2t(side_2).constant_value)
    return true_expr;

  return expr2tc();
}

static expr2tc
do_bit_munge_operation(int64_t (*opfunc)(int64_t, int64_t),
                       const type2tc &type, const expr2tc &side_1,
                       const expr2tc &side_2)
{
  int64_t val1, val2;

  // Only support integer and's. If you're a float, pointer, or whatever, you're
  // on your own.
  if (!is_constant_int2t(side_1) || !is_constant_int2t(side_2))
    return expr2tc();

  // So - we can't make BigInt by itself do an and operation. But we can dump
  // it to a binary representation, and then and that.
  const constant_int2t &int1 = to_constant_int2t(side_1);
  const constant_int2t &int2 = to_constant_int2t(side_2);

  // Drama: BigInt does *not* do any kind of twos compliment representation.
  // In fact, negative numbers are stored as positive integers, but marked as
  // being negative. To get around this, perform operations in an {u,}int64,
  if (int1.constant_value.get_len() * sizeof(BigInt::onedig_t)
                                         > sizeof(int64_t) ||
      int2.constant_value.get_len() * sizeof(BigInt::onedig_t)
                                           > sizeof(int64_t))
    return expr2tc();

  // Dump will zero-prefix and right align the output number.
  val1 = int1.constant_value.to_int64();
  val2 = int2.constant_value.to_int64();

  if (int1.constant_value.is_negative()) {
    if (val1 & 0x8000000000000000ULL) {
      // Too large to fit, negative, in an int64_t.
      return expr2tc();
    } else {
      val1 = -val1;
    }
  }

  if (int2.constant_value.is_negative()) {
    if (val2 & 0x8000000000000000ULL) {
      // Too large to fit, negative, in an int64_t.
      return expr2tc();
    } else {
      val2 = -val2;
    }
  }

  val1 = opfunc(val1, val2);

  // This has potentially become negative. Check the top bit.
  if (val1 & (1 << (type->get_width() - 1)) && is_signedbv_type(type)) {
    // Sign extend.
    val1 |= -1LL << (type->get_width());
  }

  // And now, restore, paying attention to whether this is supposed to be
  // signed or not.
  constant_int2t *theint;
  if (is_signedbv_type(type))
    theint = new constant_int2t(type, BigInt(val1));
  else
    theint = new constant_int2t(type, BigInt((uint64_t)val1));

  return expr2tc(theint);
}

static int64_t
do_bitand_op(int64_t op1, int64_t op2)
{
  return op1 & op2;
}

expr2tc
bitand2t::do_simplify(bool second __attribute__((unused))) const
{
  return do_bit_munge_operation(do_bitand_op, type, side_1, side_2);
}

static int64_t
do_bitor_op(int64_t op1, int64_t op2)
{
  return op1 | op2;
}

expr2tc
bitor2t::do_simplify(bool second __attribute__((unused))) const
{
  return do_bit_munge_operation(do_bitor_op, type, side_1, side_2);
}

static int64_t
do_bitxor_op(int64_t op1, int64_t op2)
{
  return op1 ^ op2;
}

expr2tc
bitxor2t::do_simplify(bool second __attribute__((unused))) const
{
  return do_bit_munge_operation(do_bitxor_op, type, side_1, side_2);
}

static int64_t
do_bitnand_op(int64_t op1, int64_t op2)
{
  return ~(op1 & op2);
}

expr2tc
bitnand2t::do_simplify(bool second __attribute__((unused))) const
{
  return do_bit_munge_operation(do_bitnand_op, type, side_1, side_2);
}

static int64_t
do_bitnor_op(int64_t op1, int64_t op2)
{
  return ~(op1 | op2);
}

expr2tc
bitnor2t::do_simplify(bool second __attribute__((unused))) const
{
  return do_bit_munge_operation(do_bitnor_op, type, side_1, side_2);
}

static int64_t
do_bitnxor_op(int64_t op1, int64_t op2)
{
  return ~(op1 ^ op2);
}

expr2tc
bitnxor2t::do_simplify(bool second __attribute__((unused))) const
{
  return do_bit_munge_operation(do_bitnxor_op, type, side_1, side_2);
}

static int64_t
do_bitnot_op(int64_t op1, int64_t op2 __attribute__((unused)))
{
  return ~op1;
}

expr2tc
bitnot2t::do_simplify(bool second __attribute__((unused))) const
{
  return do_bit_munge_operation(do_bitnot_op, type, value, value);
}

static int64_t
do_shl_op(int64_t op1, int64_t op2)
{
  return op1 << op2;
}

expr2tc
shl2t::do_simplify(bool second __attribute__((unused))) const
{
  return do_bit_munge_operation(do_shl_op, type, side_1, side_2);
}

static int64_t
do_lshr_op(int64_t op1, int64_t op2)
{
  return ((uint64_t)op1) >> ((uint64_t)op2);
}

expr2tc
lshr2t::do_simplify(bool second __attribute__((unused))) const
{
  return do_bit_munge_operation(do_lshr_op, type, side_1, side_2);
}

static int64_t
do_ashr_op(int64_t op1, int64_t op2)
{
  return op1 >> op2;
}

expr2tc
ashr2t::do_simplify(bool second __attribute__((unused))) const
{
  return do_bit_munge_operation(do_ashr_op, type, side_1, side_2);
}

expr2tc
typecast2t::do_simplify(bool second) const
{

  // Follow approach of old irep, i.e., copy it
  if (type == from->type) {
    // Typecast to same type means this can be eliminated entirely
    return from;
  } else if (is_bool_type(type)) {
    // Bool type -> turn into equality with zero
    expr2tc zero;
    if (is_pointer_type(from)) {
      zero = expr2tc(new symbol2t(from->type, irep_idt("NULL")));
    } else {
      fixedbvt bv;
      bv.from_integer(BigInt(0));
      zero = from_fixedbv(bv, from->type);
    }
    expr2tc eq = expr2tc(new equality2t(from, zero));
    expr2tc noteq = expr2tc(new not2t(eq));
    return noteq;
  } else if (is_symbol2t(from) && to_symbol2t(from).thename == "NULL"
             && is_pointer_type(type)){
    // Casts of null can operate on null directly. So long as we're casting it
    // to a pointer. Code like 32_floppy casts it to an int though; were we to
    // simplify that away, we end up with Z3 type errors.
    // Use of strings here is inefficient XXX jmorse
    return from;
  } else if (is_pointer_type(type) && is_pointer_type(from)) {
    // Casting from one pointer to another is meaningless... except when there's
    // pointer arithmetic about to be applied to it. So, only nurk typecasts
    // that don't change the subtype width.
    const pointer_type2t &ptr_to = to_pointer_type(type);
    const pointer_type2t &ptr_from = to_pointer_type(from->type);

    if (is_symbol_type(ptr_to.subtype) || is_symbol_type(ptr_from.subtype) ||
        is_code_type(ptr_to.subtype) || is_code_type(ptr_from.subtype))
      return expr2tc(); // Not worth thinking about

    if (is_array_type(ptr_to.subtype) &&
        is_symbol_type(get_array_subtype(ptr_to.subtype)))
      return expr2tc(); // Not worth thinking about

    if (is_array_type(ptr_from.subtype) &&
        is_symbol_type(get_array_subtype(ptr_from.subtype)))
      return expr2tc(); // Not worth thinking about

    try {
      unsigned int to_width = (is_empty_type(ptr_to.subtype)) ? 8
                              : ptr_to.subtype->get_width();
      unsigned int from_width = (is_empty_type(ptr_from.subtype)) ? 8
                              : ptr_from.subtype->get_width();

      if (to_width == from_width)
        return from;
      else
        return expr2tc();
    } catch (array_type2t::dyn_sized_array_excp*e) {
      // Something crazy, and probably C++ based, occurred. Don't attempt to
      // simplify.
      return expr2tc();
    }
  } else if (is_constant_expr(from)) {
    // Casts from constant operands can be done here.
    if (is_constant_bool2t(from) && is_bv_type(type)) {
      if (to_constant_bool2t(from).constant_value) {
        return expr2tc(new constant_int2t(type, BigInt(1)));
      } else {
        return expr2tc(new constant_int2t(type, BigInt(0)));
      }
    } else if (is_bool_type(type) && (is_constant_int2t(from) ||
                                      is_constant_fixedbv2t(from))) {
      fixedbvt bv;
      to_fixedbv(from, bv);
      if (bv.get_value().is_zero()) {
        return false_expr;
      } else {
        return true_expr;
      }
    } else if (is_bv_type(from) && is_fixedbv_type(type)) {
      fixedbvt f;
      f.spec = to_fixedbv_type(migrate_type_back(type)); // Dodgy.
      f.from_integer(to_constant_int2t(from).constant_value);
      exprt ref = f.to_expr();
      expr2tc cvt;
      migrate_expr(ref, cvt);
      return cvt;
    } else if (is_fixedbv_type(from) && is_fixedbv_type(type)) {
      fixedbvt f(to_constant_fixedbv2t(from).value);
      f.round(to_fixedbv_type(migrate_type_back(type)));
      exprt ref = f.to_expr();
      expr2tc cvt;
      migrate_expr(ref, cvt);
      return cvt;
    } else if ((is_bv_type(type) || is_fixedbv_type(type)) &&
                (is_bv_type(from) || is_fixedbv_type(from))) {
      fixedbvt bv;
      to_fixedbv(from, bv);
      return from_fixedbv(bv, type);
    } else {
      return expr2tc();
    }
  } else if (is_typecast2t(from) && type == from->type) {
    // Typecast from a typecast can be eliminated. We'll be simplified even
    // further by the caller.
    return expr2tc(new typecast2t(type, to_typecast2t(from).from));
  } else if (second && is_bv_type(type) && is_bv_type(from) &&
             (is_add2t(from) || is_sub2t(from) || is_mul2t(from) ||
              is_neg2t(from)) && from->type->get_width() <= type->get_width()) {
    // So, if this is an integer type, performing an integer arith operation,
    // and the type we're casting to isn't _supposed_ to result in a loss of
    // information, push the cast downwards.
    // XXXjmorse - I'm not convinced that potentially increasing int width is
    // a good plan, but this is what CBMC was doing, so don't change
    // behaviour.
    std::list<expr2tc> set2;
    forall_operands2(it, idx, from) {
      expr2tc cast = expr2tc(new typecast2t(type, *it));
      set2.push_back(cast);
    }

    // Now clone the expression and update its operands.
    expr2tc newobj = expr2tc(from->clone());
    newobj.get()->type = type;

    std::list<expr2tc>::const_iterator it2 = set2.begin();
    Forall_operands2(it3, idx2, newobj) {
      *it3 = *it2;
      it2++;
    }

    // Caller won't simplify us further if it's called us with second=true, so
    // give simplification another shot ourselves.
    expr2tc tmp = newobj->simplify();
    if (is_nil_expr(tmp))
      return newobj;
    else
      return tmp;
  } else {
    return expr2tc();
  }

  assert(0 && "Fell through typecast2t::do_simplify");
}

expr2tc
address_of2t::do_simplify(bool second __attribute__((unused))) const
{

  // NB: address of never has its operands simplified below its feet for sanitys
  // sake.
  // Only attempt to simplify indexes. Whatever we're taking the address of,
  // we can't simplify away the symbol.
  if (is_index2t(ptr_obj)) {
    const index2t &idx = to_index2t(ptr_obj);
    const pointer_type2t &ptr_type = to_pointer_type(type);

    // Don't simplify &a[0]
    if (is_constant_int2t(idx.index) &&
        to_constant_int2t(idx.index).constant_value.is_zero())
      return expr2tc();

    expr2tc new_index = idx.index->simplify();
    if (is_nil_expr(new_index))
      new_index = idx.index;

    expr2tc zero = expr2tc(new constant_int2t(index_type2(), BigInt(0)));
    expr2tc new_idx = expr2tc(new index2t(idx.type, idx.source_value, zero));
    expr2tc sub_addr_of = expr2tc(new address_of2t(ptr_type.subtype, new_idx));

    return expr2tc(new add2t(type, sub_addr_of, new_index));
  } else {
    return expr2tc();
  }
}

static expr2tc
do_rel_simplify(const expr2tc &side1, const expr2tc &side2,
                bool (*do_rel)(const fixedbvt &bv1, const fixedbvt &bv2))
{
  fixedbvt bv1, bv2;

  to_fixedbv(side1, bv1);
  to_fixedbv(side2, bv2);

  make_fixedbv_types_match(bv1, bv2);
  bool res = do_rel(bv1, bv2);

  return expr2tc(new constant_bool2t(res));
}

bool
do_fixedbv_eq(const fixedbvt &bv1, const fixedbvt &bv2)
{
  return bv1 == bv2;
}

expr2tc
equality2t::do_simplify(bool second __attribute__((unused))) const
{

  if (is_constant_expr(side_1) && is_constant_expr(side_2))
    return do_rel_simplify(side_1, side_2, do_fixedbv_eq);
  else
    return expr2tc();
}

bool
do_fixedbv_ineq(const fixedbvt &bv1, const fixedbvt &bv2)
{
  return bv1 != bv2;
}

expr2tc
notequal2t::do_simplify(bool second __attribute__((unused))) const
{

  if (is_constant_expr(side_1) && is_constant_expr(side_2))
    return do_rel_simplify(side_1, side_2, do_fixedbv_ineq);
  else
    return expr2tc();
}

bool
do_fixedbv_lt(const fixedbvt &bv1, const fixedbvt &bv2)
{
  return bv1 < bv2;
}

expr2tc
lessthan2t::do_simplify(bool second __attribute__((unused))) const
{

  if (is_constant_expr(side_1) && is_constant_expr(side_2))
    return do_rel_simplify(side_1, side_2, do_fixedbv_lt);
  else
    return expr2tc();
}

bool
do_fixedbv_gt(const fixedbvt &bv1, const fixedbvt &bv2)
{
  return bv1 > bv2;
}

expr2tc
greaterthan2t::do_simplify(bool second __attribute__((unused))) const
{

  if (is_constant_expr(side_1) && is_constant_expr(side_2))
    return do_rel_simplify(side_1, side_2, do_fixedbv_gt);
  else
    return expr2tc();
}

bool
do_fixedbv_le(const fixedbvt &bv1, const fixedbvt &bv2)
{
  return bv1 <= bv2;
}

expr2tc
lessthanequal2t::do_simplify(bool second __attribute__((unused))) const
{

  if (is_constant_expr(side_1) && is_constant_expr(side_2))
    return do_rel_simplify(side_1, side_2, do_fixedbv_le);
  else
    return expr2tc();
}

bool
do_fixedbv_ge(const fixedbvt &bv1, const fixedbvt &bv2)
{
  return bv1 >= bv2;
}

expr2tc
greaterthanequal2t::do_simplify(bool second __attribute__((unused))) const
{

  if (is_constant_expr(side_1) && is_constant_expr(side_2))
    return do_rel_simplify(side_1, side_2, do_fixedbv_ge);
  else
    return expr2tc();
}

expr2tc
if2t::do_simplify(bool second __attribute__((unused))) const
{
  if (is_constant_expr(cond)) {
    // We can simplify this.
    if (is_constant_bool2t(cond)) {
      if (to_constant_bool2t(cond).constant_value) {
        return true_value;
      } else {
        return false_value;
      }
    } else {
      // Cast towards a bool type.
      expr2tc cast = expr2tc(new typecast2t(type_pool.get_bool(), cond));
      cast = cast->simplify();
      assert(!is_nil_expr(cast) && "We should always be able to cast a "
             "constant value to a constant bool");

      if (to_constant_bool2t(cast).constant_value) {
        return true_value;
      } else {
        return false_value;
      }
    }
  } else {
    return expr2tc();
  }
}

expr2tc
overflow2t::do_simplify(bool second __attribute__((unused))) const
{
  unsigned int num_const = 0;
  bool simplified = false;

  // Non constant expression. We can't just simplify the operand, because it has
  // to remain the operation we expect (i.e., add2t shouldn't distribute itself)
  // so simplify its operands right here.
  if (second)
    return expr2tc();

  expr2tc new_operand = operand->clone();
  Forall_operands2(it, idx, new_operand) {
    expr2tc tmp = (**it).simplify();
    if (!is_nil_expr(tmp)) {
      *it= tmp;
      simplified = true;
    }

    if (is_constant_expr(*it))
      num_const++;
  }

  // If we don't have two constant operands, we can't simplify this expression.
  // We also don't want the underlying addition / whatever to become
  // distributed, so if the sub expressions are simplifiable, return a new
  // overflow with simplified subexprs, but no distribution.
  // The 'simplified' test guards against a continuous chain of simplifying the
  // same overflow expression over and over again.
  if (num_const != 2) {
    if (simplified)
      return expr2tc(new overflow2t(new_operand));
    else
      return expr2tc();
  }

  // Can only simplify ints
  if (!is_bv_type(new_operand))
    return expr2tc(new overflow2t(new_operand));

  // We can simplify that expression, so do it. And how do we detect overflows?
  // Perform the operation twice, once with a small type, one with huge, and
  // see if they differ. Max we can do is 64 bits, so if the expression already
  // has that size, give up.
  if (new_operand->type->get_width() == 64)
    return expr2tc();

  expr2tc simpl_op = new_operand->simplify();
  assert(is_constant_expr(simpl_op));
  expr2tc op_with_big_type = new_operand->clone();
  op_with_big_type.get()->type = (is_signedbv_type(new_operand))
                                 ? type_pool.get_int(64)
                                 : type_pool.get_uint(64);
  op_with_big_type = op_with_big_type->simplify();

  // Now ensure they're the same.
  equality2t eq(simpl_op, op_with_big_type);
  expr2tc tmp = eq.simplify();

  // And the inversion of that is the result of this overflow operation (i.e.
  // if not equal, then overflow).
  tmp = expr2tc(new not2t(tmp));
  tmp = tmp->simplify();
  assert(!is_nil_expr(tmp) && is_constant_bool2t(tmp));
  return tmp;
}

// Heavily inspired by cbmc's simplify_exprt::objects_equal_address_of
static expr2tc
obj_equals_addr_of(const expr2tc &a, const expr2tc &b)
{

  if (is_symbol2t(a) && is_symbol2t(b)) {
    if (a == b)
      return true_expr;
  } else if (is_index2t(a) && is_index2t(b)) {
    return obj_equals_addr_of(to_index2t(a).source_value,
                              to_index2t(b).source_value);
  } else if (is_member2t(a) && is_member2t(b)) {
    return obj_equals_addr_of(to_member2t(a).source_value,
                              to_member2t(b).source_value);
  } else if (is_constant_string2t(a) && is_constant_string2t(b)) {
    bool val = (to_constant_string2t(a).value == to_constant_string2t(b).value);
    if (val)
      return true_expr;
    else
      return false_expr;
  }

  return expr2tc();
}

expr2tc
same_object2t::do_simplify(bool second __attribute__((unused))) const
{

  if (is_address_of2t(side_1) && is_address_of2t(side_2))
    return obj_equals_addr_of(to_address_of2t(side_1).ptr_obj,
                              to_address_of2t(side_2).ptr_obj);

  if (is_symbol2t(side_1) && is_symbol2t(side_2) &&
      to_symbol2t(side_1).get_symbol_name() == "NULL" &&
      to_symbol2t(side_1).get_symbol_name() == "NULL")
    return true_expr;

  return expr2tc();
}

expr2tc
concat2t::do_simplify(bool second __attribute__((unused))) const
{

  if (!is_constant_int2t(side_1) || !is_constant_int2t(side_2))
    return expr2tc();

  const mp_integer &value1 = to_constant_int2t(side_1).constant_value;
  const mp_integer &value2 = to_constant_int2t(side_2).constant_value;

  // k; Take the values, and concatenate. Side 1 has higher end bits.
  mp_integer accuml = value1;
  accuml *= (1ULL << side_2->type->get_width());
  accuml += value2;

  return constant_int2tc(type, accuml);
}
