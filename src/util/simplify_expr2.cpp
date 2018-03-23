#include <climits>
#include <cstring>
#include <util/arith_tools.h>
#include <util/base_type.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/irep2.h>
#include <util/irep2_utils.h>
#include <util/type_byte_size.h>

expr2tc expr2t::do_simplify(bool second __attribute__((unused))) const
{
  return expr2tc();
}

static expr2tc try_simplification(const expr2tc &expr)
{
  expr2tc to_simplify = expr->do_simplify();
  if(is_nil_expr(to_simplify))
    to_simplify = expr2tc(expr->clone());
  return expr2tc(to_simplify->clone());
}

static expr2tc typecast_check_return(const type2tc &type, const expr2tc &expr)
{
  // If the expr is already nil, do nothing
  if(is_nil_expr(expr))
    return expr2tc();

  // Don't type cast from constant to pointer
  // TODO: check if this is right
  if(is_pointer_type(type) && is_number_type(expr))
    return try_simplification(expr);

  // No need to typecast
  if(expr->type == type)
    return expr;

  // Create a typecast of the result
  expr2tc typecast = expr2tc(new typecast2t(type, expr));

  return try_simplification(typecast);
}

static void fetch_ops_from_this_type(
  std::list<expr2tc> &ops,
  expr2t::expr_ids id,
  const expr2tc &expr)
{
  if(expr->expr_id == id)
  {
    expr->foreach_operand(
      [&ops, id](const expr2tc &e) { fetch_ops_from_this_type(ops, id, e); });
  }
  else
  {
    ops.push_back(expr);
  }
}

static bool rebalance_associative_tree(
  const expr2t &expr,
  std::list<expr2tc> &ops,
  const std::function<expr2tc(const expr2tc &arg1, const expr2tc &arg2)>
    &op_wrapper)
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
  expr.foreach_operand([&ops, &expr](const expr2tc &e) {
    fetch_ops_from_this_type(ops, expr.expr_id, e);
  });

  // Are there enough constant values in there?
  unsigned int const_values = 0;
  unsigned int orig_size = ops.size();
  for(std::list<expr2tc>::const_iterator it = ops.begin(); it != ops.end();
      it++)
    if(is_constant_expr(*it))
      const_values++;

  // Nothing for us to simplify.
  if(const_values <= 1)
    return false;

  // Otherwise, we can go through simplifying operands.
  expr2tc accuml;
  for(std::list<expr2tc>::iterator it = ops.begin(); it != ops.end(); it++)
  {
    if(!is_constant_expr(*it))
      continue;

    // We have a constant; do we have another constant to simplify with?
    if(is_nil_expr(accuml))
    {
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
    expr2tc tmp = op_wrapper(accuml, *it);
    if(is_nil_expr(tmp))
      continue; // Creating wrapper rejected it.

    tmp = tmp->simplify();
    if(is_nil_expr(tmp))
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
  if(ops.size() == orig_size)
    return false;

  // If only one constant value was removed from the list, then we attempted to
  // simplify two constants and it failed. No simplification.
  if(ops.size() == orig_size - 1)
    return false;

  // Finally; we've succeeded and simplified something. Push the simplified
  // constant back at the end of the list.
  ops.push_back(accuml);
  return true;
}

expr2tc attempt_associative_simplify(
  const expr2t &expr,
  const std::function<expr2tc(const expr2tc &arg1, const expr2tc &arg2)>
    &op_wrapper)
{
  std::list<expr2tc> operands;
  if(rebalance_associative_tree(expr, operands, op_wrapper))
  {
    // Horray, we simplified. Recreate.
    assert(operands.size() >= 2);
    std::list<expr2tc>::const_iterator it = operands.begin();
    expr2tc accuml = *it;
    it++;
    for(; it != operands.end(); it++)
    {
      expr2tc tmp;
      accuml = op_wrapper(accuml, *it);
      if(is_nil_expr(accuml))
        return expr2tc(); // wrapper rejected new obj :O
    }

    return accuml;
  }

  return expr2tc();
}

template <template <typename> class TFunctor, typename constructor>
static expr2tc simplify_arith_2ops(
  const type2tc &type,
  const expr2tc &side_1,
  const expr2tc &side_2)
{
  if(!is_number_type(type) && !is_pointer_type(type))
    return expr2tc();

  // Try to recursively simplify nested operations both sides, if any
  expr2tc simplied_side_1 = try_simplification(side_1);
  expr2tc simplied_side_2 = try_simplification(side_2);

  if(!is_constant_expr(simplied_side_1) && !is_constant_expr(simplied_side_2))
  {
    // Were we able to simplify the sides?
    if((side_1 != simplied_side_1) || (side_2 != simplied_side_2))
    {
      expr2tc new_op =
        expr2tc(new constructor(type, simplied_side_1, simplied_side_2));

      return typecast_check_return(type, new_op);
    }

    return expr2tc();
  }

  // This should be handled by ieee_*
  assert(!is_floatbv_type(type));

  expr2tc simpl_res;
  if(is_bv_type(simplied_side_1) || is_bv_type(simplied_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_int2t;

    std::function<BigInt &(expr2tc &)> get_value = [](expr2tc &c) -> BigInt & {
      return to_constant_int2t(c).value;
    };

    simpl_res = TFunctor<BigInt>::simplify(
      simplied_side_1, simplied_side_2, is_constant, get_value);

    // Fix rounding when an overflow occurs
    if(!is_nil_expr(simpl_res) && is_constant_int2t(simpl_res))
      migrate_expr(
        from_integer(
          to_constant_int2t(simpl_res).value,
          migrate_type_back(simpl_res->type)),
        simpl_res);
  }
  else if(is_fixedbv_type(simplied_side_1) || is_fixedbv_type(simplied_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_fixedbv2t;

    std::function<fixedbvt &(expr2tc &)> get_value =
      [](expr2tc &c) -> fixedbvt & { return to_constant_fixedbv2t(c).value; };

    simpl_res = TFunctor<fixedbvt>::simplify(
      simplied_side_1, simplied_side_2, is_constant, get_value);
  }
  else if(is_bool_type(simplied_side_1) || is_bool_type(simplied_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_bool2t;

    std::function<bool &(expr2tc &)> get_value = [](expr2tc &c) -> bool & {
      return to_constant_bool2t(c).value;
    };

    simpl_res = TFunctor<bool>::simplify(
      simplied_side_1, simplied_side_2, is_constant, get_value);
  }
  else
    return expr2tc();

  return typecast_check_return(type, simpl_res);
}

template <class constant_type>
struct Addtor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    if(is_constant(op1))
    {
      // Found a zero? Simplify to op2
      expr2tc c1 = op1;
      if(get_value(c1) == 0)
        return op2;
    }

    if(is_constant(op2))
    {
      // Found a zero? Simplify to op1
      expr2tc c2 = op2;
      if(get_value(c2) == 0)
        return op1;
    }

    // Two constants? Simplify to result of the addition
    if(is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      get_value(c1) += get_value(c2);
      return c1;
    }

    return expr2tc();
  }
};

expr2tc add2t::do_simplify(bool __attribute__((unused))) const
{
  expr2tc res = simplify_arith_2ops<Addtor, add2t>(type, side_1, side_2);
  if(!is_nil_expr(res))
    return res;

  // Attempt associative simplification
  std::function<expr2tc(const expr2tc &arg1, const expr2tc &arg2)> add_wrapper =
    [this](const expr2tc &arg1, const expr2tc &arg2) -> expr2tc {
    return add2tc(this->type, arg1, arg2);
  };

  return attempt_associative_simplify(*this, add_wrapper);
}

template <class constant_type>
struct Subtor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    if(is_constant(op1))
    {
      // Found a zero? Simplify to -op2
      expr2tc c1 = op1;
      if(get_value(c1) == 0)
      {
        neg2tc c(op2->type, op2);
        ::simplify(c);
        return c;
      }
    }

    if(is_constant(op2))
    {
      // Found a zero? Simplify to op1
      expr2tc c2 = op2;
      if(get_value(c2) == 0)
        return op1;
    }

    // Two constants? Simplify to result of the subtraction
    if(is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      get_value(c1) -= get_value(c2);
      return c1;
    }

    return expr2tc();
  }
};

expr2tc sub2t::do_simplify(bool second __attribute__((unused))) const
{
  return simplify_arith_2ops<Subtor, sub2t>(type, side_1, side_2);
}

template <class constant_type>
struct Multor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    if(is_constant(op1))
    {
      expr2tc c1 = op1;

      // Found a zero? Simplify to zero
      if(get_value(c1) == 0)
        return op1;

      // Found an one? Simplify to op2
      if(get_value(c1) == 1)
        return op2;
    }

    if(is_constant(op2))
    {
      expr2tc c2 = op2;

      // Found a zero? Simplify to zero
      if(get_value(c2) == 0)
        return op2;

      // Found an one? Simplify to op1
      if(get_value(c2) == 1)
        return op1;
    }

    // Two constants? Simplify to result of the multiplication
    if(is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      get_value(c1) *= get_value(c2);
      return c1;
    }

    return expr2tc();
  }
};

expr2tc mul2t::do_simplify(bool second __attribute__((unused))) const
{
  return simplify_arith_2ops<Multor, mul2t>(type, side_1, side_2);
}

template <class constant_type>
struct Divtor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    if(is_constant(op1))
    {
      // Numerator is zero? Simplify to zero
      expr2tc c1 = op1;
      if(get_value(c1) == 0)
        return op1;
    }

    if(is_constant(op2))
    {
      expr2tc c2 = op2;

      // Denominator is zero? Don't simplify
      if(get_value(c2) == 0)
        return expr2tc();

      // Denominator is one? Simplify to numerator's constant
      if(get_value(c2) == 1)
        return op1;
    }

    // Two constants? Simplify to result of the division
    if(is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      get_value(c1) /= get_value(c2);
      return c1;
    }

    return expr2tc();
  }
};

expr2tc div2t::do_simplify(bool second __attribute__((unused))) const
{
  return simplify_arith_2ops<Divtor, div2t>(type, side_1, side_2);
}

expr2tc modulus2t::do_simplify(bool second __attribute__((unused))) const
{
  if(!is_number_type(type))
    return expr2tc();

  // Try to recursively simplify nested operations both sides, if any
  expr2tc simplied_side_1 = try_simplification(side_1);
  expr2tc simplied_side_2 = try_simplification(side_2);

  if(!is_constant_expr(simplied_side_1) || !is_constant_expr(simplied_side_2))
  {
    // Were we able to simplify the sides?
    if((side_1 != simplied_side_1) || (side_2 != simplied_side_2))
    {
      expr2tc new_mod =
        expr2tc(new modulus2t(type, simplied_side_1, simplied_side_2));

      return typecast_check_return(type, new_mod);
    }

    return expr2tc();
  }

  if(is_bv_type(type))
  {
    if(is_constant_int2t(simplied_side_2))
    {
      // Denominator is one? Simplify to zero
      if(to_constant_int2t(simplied_side_2).value == 1)
        return constant_int2tc(type, BigInt(0));
    }

    if(is_constant_int2t(simplied_side_1) && is_constant_int2t(simplied_side_2))
    {
      const constant_int2t &numerator = to_constant_int2t(simplied_side_1);
      const constant_int2t &denominator = to_constant_int2t(simplied_side_2);

      auto c = numerator.value;
      c %= denominator.value;

      return constant_int2tc(type, c);
    }
  }

  return expr2tc();
}

template <template <typename> class TFunctor, typename constructor>
static expr2tc simplify_arith_1op(const type2tc &type, const expr2tc &value)
{
  if(!is_number_type(type))
    return expr2tc();

  // Try to recursively simplify nested operation, if any
  expr2tc to_simplify = try_simplification(value);
  if(!is_constant_expr(to_simplify))
  {
    // Were we able to simplify anything?
    if(value != to_simplify)
    {
      expr2tc new_neg = expr2tc(new constructor(type, to_simplify));
      return typecast_check_return(type, new_neg);
    }

    return expr2tc();
  }

  expr2tc simpl_res;
  if(is_bv_type(value))
  {
    std::function<constant_int2t &(expr2tc &)> to_constant =
      (constant_int2t & (*)(expr2tc &)) to_constant_int2t;

    simpl_res = TFunctor<constant_int2t>::simplify(to_simplify, to_constant);
  }
  else if(is_fixedbv_type(value))
  {
    std::function<constant_fixedbv2t &(expr2tc &)> to_constant =
      (constant_fixedbv2t & (*)(expr2tc &)) to_constant_fixedbv2t;

    simpl_res =
      TFunctor<constant_fixedbv2t>::simplify(to_simplify, to_constant);
  }
  else if(is_floatbv_type(value))
  {
    std::function<constant_floatbv2t &(expr2tc &)> to_constant =
      (constant_floatbv2t & (*)(expr2tc &)) to_constant_floatbv2t;

    simpl_res =
      TFunctor<constant_floatbv2t>::simplify(to_simplify, to_constant);
  }
  else if(is_bool_type(value))
  {
    std::function<constant_bool2t &(expr2tc &)> to_constant =
      (constant_bool2t & (*)(expr2tc &)) to_constant_bool2t;

    simpl_res = TFunctor<constant_bool2t>::simplify(to_simplify, to_constant);
  }
  else
    return expr2tc();

  return typecast_check_return(type, simpl_res);
}

template <class constant_type>
struct Negator
{
  static expr2tc simplify(
    const expr2tc &number,
    std::function<constant_type &(expr2tc &)> to_constant)
  {
    expr2tc c = number;
    to_constant(c).value = !to_constant(c).value;
    return c;
  }
};

expr2tc neg2t::do_simplify(bool second __attribute__((unused))) const
{
  return simplify_arith_1op<Negator, neg2t>(type, value);
}

template <class constant_type>
struct abstor
{
  static expr2tc simplify(
    const expr2tc &number,
    std::function<constant_type &(expr2tc &)> to_constant)
  {
    expr2tc c = number;

    if(to_constant(c).value > 0)
      return number;

    to_constant(c).value = !to_constant(c).value;
    return c;
  }
};

expr2tc abs2t::do_simplify(bool second __attribute__((unused))) const
{
  return simplify_arith_1op<abstor, abs2t>(type, value);
}

expr2tc with2t::do_simplify(bool second __attribute__((unused))) const
{
  if(is_constant_struct2t(source_value))
  {
    const constant_struct2t &c_struct = to_constant_struct2t(source_value);
    const constant_string2t &memb = to_constant_string2t(update_field);
    unsigned no = static_cast<const struct_union_data &>(*type.get())
                    .get_component_number(memb.value);
    assert(no < c_struct.datatype_members.size());

    // Clone constant struct, update its field according to this "with".
    constant_struct2tc s = c_struct;
    s->datatype_members[no] = update_value;
    return expr2tc(s);
  }
  if(is_constant_union2t(source_value))
  {
    const constant_union2t &c_union = to_constant_union2t(source_value);
    const union_type2t &thetype = to_union_type(c_union.type);
    const constant_string2t &memb = to_constant_string2t(update_field);
    unsigned no = static_cast<const struct_union_data &>(*c_union.type.get())
                    .get_component_number(memb.value);
    assert(no < thetype.member_names.size());

    // If the update value type matches the current lump of data's type, we can
    // just replace it with the new value. As far as I can tell, constant unions
    // only ever contain one member, and it's the member most recently written.
    if(thetype.members[no] != update_value->type)
      return expr2tc();

    std::vector<expr2tc> newmembers;
    newmembers.push_back(update_value);
    return constant_union2tc(type, newmembers);
  }
  else if(is_constant_array2t(source_value) && is_constant_int2t(update_field))
  {
    const constant_array2t &array = to_constant_array2t(source_value);
    const constant_int2t &index = to_constant_int2t(update_field);

    // Index may be out of bounds. That's an error in the program, but not in
    // the model we're generating, so permit it. Can't simplify it though.
    if(index.value.is_negative())
      return expr2tc();

    if(index.as_ulong() >= array.datatype_members.size())
      return expr2tc();

    constant_array2tc arr = array;
    arr->datatype_members[index.as_ulong()] = update_value;
    return arr;
  }
  else if(is_constant_array_of2t(source_value))
  {
    const constant_array_of2t &array = to_constant_array_of2t(source_value);

    // We don't simplify away these withs if // the array_of is infinitely
    // sized. This is because infinitely sized arrays are no longer converted
    // correctly in the solver backend (they're simply not supported by SMT).
    // Thus it becomes important to be able to assign a value to a field in an
    // aray_of and not have it const propagatated away.
    const constant_array_of2t &thearray = to_constant_array_of2t(source_value);
    const array_type2t &arr_type = to_array_type(thearray.type);
    if(arr_type.size_is_infinite)
      return expr2tc();

    // We can eliminate this operation if the operand to this with is the same
    // as the initializer.
    if(update_value == array.initializer)
      return source_value;

    return expr2tc();
  }
  else
  {
    return expr2tc();
  }
}

expr2tc member2t::do_simplify(bool second __attribute__((unused))) const
{
  if(is_constant_struct2t(source_value) || is_constant_union2t(source_value))
  {
    unsigned no =
      static_cast<const struct_union_data &>(*source_value->type.get())
        .get_component_number(member);

    // Clone constant struct, update its field according to this "with".
    expr2tc s;
    if(is_constant_struct2t(source_value))
    {
      s = to_constant_struct2t(source_value).datatype_members[no];

      assert(
        is_pointer_type(type) ||
        base_type_eq(type, s->type, namespacet(contextt())));
    }
    else
    {
      // The constant array has some number of elements, up to the size of the
      // array, but possibly fewer. This is legal C. So bounds check first that
      // we can actually perform this member operation.
      const constant_union2t &uni = to_constant_union2t(source_value);
      if(uni.datatype_members.size() <= no)
        return expr2tc();

      s = uni.datatype_members[no];

      // If the type we just selected isn't compatible, it means that whatever
      // field is in the constant union /isn't/ the field we're selecting from
      // it. So don't simplify it, because we can't.
      if(
        !is_pointer_type(type) &&
        !base_type_eq(type, s->type, namespacet(contextt())))
        return expr2tc();
    }

    return s;
  }

  return expr2tc();
}

expr2tc pointer_offset2t::do_simplify(bool second __attribute__((unused))) const
{
  // XXX - this could be better. But the current implementation catches most
  // cases that ESBMC produces internally.

  if(is_address_of2t(ptr_obj))
  {
    const address_of2t &addrof = to_address_of2t(ptr_obj);
    if(is_symbol2t(addrof.ptr_obj) || is_constant_string2t(addrof.ptr_obj))
      return gen_zero(type);

    if(is_index2t(addrof.ptr_obj))
    {
      const index2t &index = to_index2t(addrof.ptr_obj);
      if(is_constant_int2t(index.index))
      {
        if(is_symbol2t(index.source_value))
        {
          // We can reduce to that index offset.
          const array_type2t &arr = to_array_type(index.source_value->type);
          unsigned int widthbits = arr.subtype->get_width();
          unsigned int widthbytes = widthbits / 8;
          BigInt val = to_constant_int2t(index.index).value;
          val *= widthbytes;
          return constant_int2tc(type, val);
        }

        if(is_constant_string2t(index.source_value))
        {
          // This can also be simplified to an array offset. Just return the index,
          // as the string elements are all 8 bit bytes.
          return index.index;
        }
      }
    }
  }
  else if(is_typecast2t(ptr_obj))
  {
    const typecast2t &cast = to_typecast2t(ptr_obj);
    expr2tc new_ptr_offs = pointer_offset2tc(type, cast.from);
    expr2tc reduced = new_ptr_offs->simplify();

    // No good simplification -> return nothing
    if(is_nil_expr(reduced))
      return reduced;

    // If it simplified to zero, that's fine, return that.
    if(is_constant_int2t(reduced) && to_constant_int2t(reduced).value.is_zero())
      return reduced;

    // If it didn't reduce to zero, give up. Not sure why this is the case,
    // but it's what the old irep code does.
  }
  else if(is_add2t(ptr_obj))
  {
    const add2t &add = to_add2t(ptr_obj);

    // So, one of these should be a ptr type, or there isn't any point in this
    // being a pointer_offset irep.
    if(!is_pointer_type(add.side_1) && !is_pointer_type(add.side_2))
      return expr2tc();

    // Can't have pointer-on-pointer arith.
    assert(!(is_pointer_type(add.side_1) && is_pointer_type(add.side_2)));

    expr2tc ptr_op = (is_pointer_type(add.side_1)) ? add.side_1 : add.side_2;
    expr2tc non_ptr_op =
      (is_pointer_type(add.side_1)) ? add.side_2 : add.side_1;

    // Can't do any kind of simplification if the ptr op has a symbolic type.
    // Let the SMT layer handle this. In the future, can we pass around a
    // namespace?
    if(is_symbol_type(to_pointer_type(ptr_op->type).subtype))
      return expr2tc();

    // Turn the pointer one into pointer_offset.
    expr2tc new_ptr_op = pointer_offset2tc(type, ptr_op);
    // And multiply the non pointer one by the type size.
    type2tc ptr_int_type = get_int_type(config.ansi_c.pointer_width);
    type2tc ptr_subtype = to_pointer_type(ptr_op->type).subtype;
    mp_integer thesize =
      (is_empty_type(ptr_subtype)) ? 1 : type_byte_size(ptr_subtype);
    constant_int2tc type_size(type, thesize);

    // SV-Comp workaround
    if(non_ptr_op->type->get_width() != type->get_width())
      non_ptr_op = typecast2tc(type, non_ptr_op);

    mul2tc new_non_ptr_op(type, non_ptr_op, type_size);

    expr2tc new_add = add2tc(type, new_ptr_op, new_non_ptr_op);

    // So, this add is a valid simplification. We may be able to simplify
    // further though.
    expr2tc tmp = new_add->simplify();
    if(is_nil_expr(tmp))
      return new_add;

    return tmp;
  }

  return expr2tc();
}

expr2tc index2t::do_simplify(bool second __attribute__((unused))) const
{
  if(is_with2t(source_value))
  {
    if(index == to_with2t(source_value).update_field)
    {
      // Index is the same as an update to the thing we're indexing; we can
      // just take the update value from the "with" below.
      return to_with2t(source_value).update_value;
    }

    return expr2tc();
  }
  if(is_constant_array2t(source_value) && is_constant_int2t(index))
  {
    const constant_array2t &arr = to_constant_array2t(source_value);
    const constant_int2t &idx = to_constant_int2t(index);

    // Index might be greater than the constant array size. This means we can't
    // simplify it, and the user might be eaten by an assertion failure in the
    // model. We don't have to think about this now though.
    if(idx.value.is_negative())
      return expr2tc();

    unsigned long the_idx = idx.as_ulong();
    if(the_idx >= arr.datatype_members.size())
      return expr2tc();

    return arr.datatype_members[the_idx];
  }
  else if(is_constant_string2t(source_value) && is_constant_int2t(index))
  {
    const constant_string2t &str = to_constant_string2t(source_value);
    const constant_int2t &idx = to_constant_int2t(index);

    // Same index situation
    if(idx.value.is_negative())
      return expr2tc();

    unsigned long the_idx = idx.as_ulong();
    if(the_idx > str.value.as_string().size()) // allow reading null term.
      return expr2tc();

    // String constants had better be some kind of integer type
    assert(is_bv_type(type));
    unsigned long val = str.value.as_string().c_str()[the_idx];
    return constant_int2tc(type, BigInt(val));
  }
  else if(is_constant_array_of2t(source_value))
  {
    // Only thing this index can evaluate to is the default value of this array
    return to_constant_array_of2t(source_value).initializer;
  }
  else
  {
    return expr2tc();
  }
}

expr2tc not2t::do_simplify(bool second __attribute__((unused))) const
{
  expr2tc simp = try_simplification(value);

  if(is_not2t(simp))
    // These negate.
    return to_not2t(simp).value;

  if(!is_constant_bool2t(simp))
    return expr2tc();

  const constant_bool2t &val = to_constant_bool2t(simp);
  return expr2tc(new constant_bool2t(!val.value));
}

template <template <typename> class TFunctor, typename constructor>
static expr2tc simplify_logic_2ops(
  const type2tc &type,
  const expr2tc &side_1,
  const expr2tc &side_2)
{
  if(!is_number_type(type))
    return expr2tc();

  // Try to recursively simplify nested operations both sides, if any
  expr2tc simplied_side_1 = try_simplification(side_1);
  expr2tc simplied_side_2 = try_simplification(side_2);

  if(!is_constant_expr(simplied_side_1) && !is_constant_expr(simplied_side_2))
  {
    // Were we able to simplify the sides?
    if((side_1 != simplied_side_1) || (side_2 != simplied_side_2))
    {
      expr2tc new_op =
        expr2tc(new constructor(simplied_side_1, simplied_side_2));

      return typecast_check_return(type, new_op);
    }

    return expr2tc();
  }

  expr2tc simpl_res;

  if(is_bv_type(simplied_side_1) || is_bv_type(simplied_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_int2t;

    std::function<BigInt &(expr2tc &)> get_value = [](expr2tc &c) -> BigInt & {
      return to_constant_int2t(c).value;
    };

    simpl_res = TFunctor<BigInt>::simplify(
      simplied_side_1, simplied_side_2, is_constant, get_value);
  }
  else if(is_fixedbv_type(simplied_side_1) || is_fixedbv_type(simplied_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_fixedbv2t;

    std::function<fixedbvt &(expr2tc &)> get_value =
      [](expr2tc &c) -> fixedbvt & { return to_constant_fixedbv2t(c).value; };

    simpl_res = TFunctor<fixedbvt>::simplify(
      simplied_side_1, simplied_side_2, is_constant, get_value);
  }
  else if(is_floatbv_type(simplied_side_1) || is_floatbv_type(simplied_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_floatbv2t;

    std::function<ieee_floatt &(expr2tc &)> get_value = [](
      expr2tc &c) -> ieee_floatt & { return to_constant_floatbv2t(c).value; };

    simpl_res = TFunctor<ieee_floatt>::simplify(
      simplied_side_1, simplied_side_2, is_constant, get_value);
  }
  else if(is_bool_type(simplied_side_1) || is_bool_type(simplied_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_bool2t;

    std::function<bool &(expr2tc &)> get_value = [](expr2tc &c) -> bool & {
      return to_constant_bool2t(c).value;
    };

    simpl_res = TFunctor<bool>::simplify(
      simplied_side_1, simplied_side_2, is_constant, get_value);
  }
  else
    return expr2tc();

  return typecast_check_return(type, simpl_res);
}

template <class constant_type>
struct Andtor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    if(is_constant(op1))
    {
      // False? never true
      expr2tc c1 = op1;
      return (get_value(c1) == 0) ? c1 : op2;
    }

    if(is_constant(op2))
    {
      // False? never true
      expr2tc c2 = op2;
      return (get_value(c2) == 0) ? c2 : op1;
    }

    // Two constants? Simplify to result of the and
    if(is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      return constant_bool2tc(!(get_value(c1) == 0) && !(get_value(c2) == 0));
    }

    return expr2tc();
  }
};

expr2tc and2t::do_simplify(bool second __attribute__((unused))) const
{
  return simplify_logic_2ops<Andtor, and2t>(type, side_1, side_2);
}

template <class constant_type>
struct Ortor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    if(is_constant(op1))
    {
      // True? Simplify to op2
      expr2tc c1 = op1;
      if(!(get_value(c1) == 0))
        return gen_true_expr();
    }

    if(is_constant(op2))
    {
      // True? Simplify to op1
      expr2tc c2 = op2;
      if(!(get_value(c2) == 0))
        return gen_true_expr();
    }

    // Two constants? Simplify to result of the or
    if(is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      return constant_bool2tc(!(get_value(c1) == 0) || !(get_value(c2) == 0));
    }

    return expr2tc();
  }
};

expr2tc or2t::do_simplify(bool second __attribute__((unused))) const
{
  // Special case: if one side is a not of the other, and they're otherwise
  // identical, simplify to true
  if(is_not2t(side_1))
  {
    const not2t &ref = to_not2t(side_1);
    if(ref.value == side_2)
      return gen_true_expr();
  }
  else if(is_not2t(side_2))
  {
    const not2t &ref = to_not2t(side_2);
    if(ref.value == side_1)
      return gen_true_expr();
  }

  // Otherwise, default
  return simplify_logic_2ops<Ortor, or2t>(type, side_1, side_2);
}

template <class constant_type>
struct Xortor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    if(is_constant(op1))
    {
      expr2tc c1 = op1;
      // False? Simplify to op2
      if(get_value(c1) == 0)
        return op2;
    }

    if(is_constant(op2))
    {
      expr2tc c2 = op2;
      // False? Simplify to op1
      if(get_value(c2) == 0)
        return op1;
    }

    // Two constants? Simplify to result of the xor
    if(is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      return constant_bool2tc(!(get_value(c1) == 0) ^ !(get_value(c2) == 0));
    }

    return expr2tc();
  }
};

expr2tc xor2t::do_simplify(bool second __attribute__((unused))) const
{
  return simplify_logic_2ops<Xortor, xor2t>(type, side_1, side_2);
}

template <class constant_type>
struct Impliestor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    // False => * evaluate to true, always
    if(is_constant(op1))
    {
      expr2tc c1 = op1;
      if(get_value(c1) == 0)
        return gen_true_expr();
    }

    // Otherwise, the only other thing that will make this expr always true is
    // if side 2 is true.
    if(is_constant(op2))
    {
      expr2tc c2 = op2;
      if(!(get_value(c2) == 0))
        return gen_true_expr();
    }

    return expr2tc();
  }
};

expr2tc implies2t::do_simplify(bool second __attribute__((unused))) const
{
  return simplify_logic_2ops<Impliestor, implies2t>(type, side_1, side_2);
}

template <typename constructor>
static expr2tc do_bit_munge_operation(
  const std::function<int64_t(int64_t, int64_t)> &opfunc,
  const type2tc &type,
  const expr2tc &side_1,
  const expr2tc &side_2)
{
  // Try to recursively simplify nested operations both sides, if any
  expr2tc simplied_side_1 = try_simplification(side_1);
  expr2tc simplied_side_2 = try_simplification(side_2);

  if(!is_constant_expr(simplied_side_1) && !is_constant_expr(simplied_side_2))
  {
    // Were we able to simplify the sides?
    if((side_1 != simplied_side_1) || (side_2 != simplied_side_2))
    {
      expr2tc new_op =
        expr2tc(new constructor(type, simplied_side_1, simplied_side_2));

      return typecast_check_return(type, new_op);
    }

    return expr2tc();
  }

  // Only support integer and's. If you're a float, pointer, or whatever, you're
  // on your own.
  if(!is_constant_int2t(side_1) || !is_constant_int2t(side_2))
    return expr2tc();

  // So - we can't make BigInt by itself do an and operation. But we can dump
  // it to a binary representation, and then and that.
  const constant_int2t &int1 = to_constant_int2t(side_1);
  const constant_int2t &int2 = to_constant_int2t(side_2);

  // Drama: BigInt does *not* do any kind of twos compliment representation.
  // In fact, negative numbers are stored as positive integers, but marked as
  // being negative. To get around this, perform operations in an {u,}int64,
  if(
    ((int1.value.get_len() * sizeof(BigInt::onedig_t)) > sizeof(int64_t)) ||
    ((int2.value.get_len() * sizeof(BigInt::onedig_t)) > sizeof(int64_t)))
    return expr2tc();

  // Dump will zero-prefix and right align the output number.
  int64_t val1 = int1.value.to_int64();
  int64_t val2 = int2.value.to_int64();

  if(int1.value.is_negative())
  {
    if(val1 & 0x8000000000000000ULL)
    {
      // Too large to fit, negative, in an int64_t.
      return expr2tc();
    }

    val1 = -val1;
  }

  if(int2.value.is_negative())
  {
    if(val2 & 0x8000000000000000ULL)
    {
      // Too large to fit, negative, in an int64_t.
      return expr2tc();
    }

    val2 = -val2;
  }

  val1 = opfunc(val1, val2);

  // This has potentially become negative. Check the top bit.
  if(val1 & (1 << (type->get_width() - 1)) && is_signedbv_type(type))
  {
    // Sign extend.
    val1 |= ULLONG_MAX << (type->get_width());
  }

  // And now, restore, paying attention to whether this is supposed to be
  // signed or not.
  constant_int2t *theint;
  if(is_signedbv_type(type))
    theint = new constant_int2t(type, BigInt(val1));
  else
    theint = new constant_int2t(type, BigInt((uint64_t)val1));

  return expr2tc(theint);
}

expr2tc bitand2t::do_simplify(bool second __attribute__((unused))) const
{
  std::function<int64_t(int64_t, int64_t)> op = [](int64_t op1, int64_t op2) {
    return (op1 & op2);
  };

  return do_bit_munge_operation<bitand2t>(op, type, side_1, side_2);
}

expr2tc bitor2t::do_simplify(bool second __attribute__((unused))) const
{
  std::function<int64_t(int64_t, int64_t)> op = [](int64_t op1, int64_t op2) {
    return (op1 | op2);
  };

  return do_bit_munge_operation<bitor2t>(op, type, side_1, side_2);
}

expr2tc bitxor2t::do_simplify(bool second __attribute__((unused))) const
{
  std::function<int64_t(int64_t, int64_t)> op = [](int64_t op1, int64_t op2) {
    return (op1 ^ op2);
  };

  return do_bit_munge_operation<bitxor2t>(op, type, side_1, side_2);
}

expr2tc bitnand2t::do_simplify(bool second __attribute__((unused))) const
{
  std::function<int64_t(int64_t, int64_t)> op = [](int64_t op1, int64_t op2) {
    return ~(op1 & op2);
  };

  return do_bit_munge_operation<bitnand2t>(op, type, side_1, side_2);
}

expr2tc bitnor2t::do_simplify(bool second __attribute__((unused))) const
{
  std::function<int64_t(int64_t, int64_t)> op = [](int64_t op1, int64_t op2) {
    return ~(op1 | op2);
  };

  return do_bit_munge_operation<bitnor2t>(op, type, side_1, side_2);
}

expr2tc bitnxor2t::do_simplify(bool second __attribute__((unused))) const
{
  std::function<int64_t(int64_t, int64_t)> op = [](int64_t op1, int64_t op2) {
    return ~(op1 ^ op2);
  };

  return do_bit_munge_operation<bitnxor2t>(op, type, side_1, side_2);
}

expr2tc bitnot2t::do_simplify(bool second __attribute__((unused))) const
{
  std::function<int64_t(int64_t, int64_t)> op =
    [](int64_t op1, int64_t op2 __attribute__((unused))) { return ~(op1); };

  return do_bit_munge_operation<bitnot2t>(op, type, value, value);
}

expr2tc shl2t::do_simplify(bool second __attribute__((unused))) const
{
  std::function<int64_t(int64_t, int64_t)> op = [](int64_t op1, int64_t op2) {
    return (op1 << op2);
  };

  return do_bit_munge_operation<shl2t>(op, type, side_1, side_2);
}

expr2tc lshr2t::do_simplify(bool second __attribute__((unused))) const
{
  std::function<int64_t(int64_t, int64_t)> op = [](int64_t op1, int64_t op2) {
    return ((uint64_t)op1) >> ((uint64_t)op2);
  };

  return do_bit_munge_operation<lshr2t>(op, type, side_1, side_2);
}

expr2tc ashr2t::do_simplify(bool second __attribute__((unused))) const
{
  std::function<int64_t(int64_t, int64_t)> op = [](int64_t op1, int64_t op2) {
    return (op1 >> op2);
  };

  return do_bit_munge_operation<ashr2t>(op, type, side_1, side_2);
}

expr2tc typecast2t::do_simplify(bool second __attribute__((unused))) const
{
  // Follow approach of old irep, i.e., copy it
  if(type == from->type)
  {
    // Typecast to same type means this can be eliminated entirely
    return from;
  }

  auto simp = try_simplification(from);

  if(is_constant_expr(simp))
  {
    // Casts from constant operands can be done here.
    if(is_bool_type(simp) && is_number_type(type))
    {
      if(is_bv_type(type))
      {
        // bool to int
        return constant_int2tc(type, BigInt(to_constant_bool2t(simp).value));
      }
      if(is_fixedbv_type(type))
      {
        fixedbvt fbv;
        fbv.spec = to_fixedbv_type(migrate_type_back(type));
        fbv.from_integer(to_constant_bool2t(simp).value);
        return constant_fixedbv2tc(fbv);
      }
      else if(is_floatbv_type(simp))
      {
        if(!is_constant_int2t(rounding_mode))
          return expr2tc();

        ieee_floatt fpbv;

        mp_integer rm_value = to_constant_int2t(rounding_mode).value;
        fpbv.rounding_mode = ieee_floatt::rounding_modet(rm_value.to_int64());

        fpbv.from_expr(to_constant_floatbv2t(simp).value.to_expr());
        fpbv.change_spec(to_floatbv_type(migrate_type_back(type)));

        return constant_floatbv2tc(fpbv);
      }
    }
    else if(is_bv_type(simp) && is_number_type(type))
    {
      // int to int/float/double
      const constant_int2t &theint = to_constant_int2t(simp);

      if(is_bv_type(type))
      {
        // If we are typecasting from integer to a smaller integer,
        // this will return the number with the smaller size
        exprt number = from_integer(theint.value, migrate_type_back(type));

        BigInt new_number;
        if(to_integer(number, new_number))
          return expr2tc();

        return constant_int2tc(type, new_number);
      }
      if(is_fixedbv_type(type))
      {
        fixedbvt fbv;
        fbv.spec = to_fixedbv_type(migrate_type_back(type));
        fbv.from_integer(theint.value);
        return constant_fixedbv2tc(fbv);
      }
      else if(is_bool_type(type))
      {
        const constant_int2t &theint = to_constant_int2t(simp);
        return theint.value.is_zero() ? gen_false_expr() : gen_true_expr();
      }
      else if(is_floatbv_type(type))
      {
        if(!is_constant_int2t(rounding_mode))
          return expr2tc();

        ieee_floatt fpbv;

        mp_integer rm_value = to_constant_int2t(rounding_mode).value;
        fpbv.rounding_mode = ieee_floatt::rounding_modet(rm_value.to_int64());

        fpbv.spec = to_floatbv_type(migrate_type_back(type));
        fpbv.from_integer(to_constant_int2t(simp).value);

        return constant_floatbv2tc(fpbv);
      }
    }
    else if(is_fixedbv_type(simp) && is_number_type(type))
    {
      // float/double to int/float/double
      fixedbvt fbv(to_constant_fixedbv2t(simp).value);

      if(is_bv_type(type))
      {
        return constant_int2tc(type, fbv.to_integer());
      }
      if(is_fixedbv_type(type))
      {
        fbv.round(to_fixedbv_type(migrate_type_back(type)));
        return constant_fixedbv2tc(fbv);
      }
      else if(is_bool_type(type))
      {
        const constant_fixedbv2t &fbv = to_constant_fixedbv2t(simp);
        return fbv.value.is_zero() ? gen_false_expr() : gen_true_expr();
      }
    }
    else if(is_floatbv_type(simp) && is_number_type(type))
    {
      // float/double to int/float/double
      if(!is_constant_int2t(rounding_mode))
        return expr2tc();

      ieee_floatt fpbv(to_constant_floatbv2t(simp).value);

      mp_integer rm_value = to_constant_int2t(rounding_mode).value;
      fpbv.rounding_mode = ieee_floatt::rounding_modet(rm_value.to_int64());

      if(is_bv_type(type))
      {
        return constant_int2tc(type, fpbv.to_integer());
      }
      if(is_floatbv_type(type))
      {
        fpbv.change_spec(to_floatbv_type(migrate_type_back(type)));
        return constant_floatbv2tc(fpbv);
      }
      else if(is_bool_type(type))
      {
        return fpbv.is_zero() ? gen_false_expr() : gen_true_expr();
      }
    }
  }
  else if(is_bool_type(type))
  {
    // Bool type -> turn into equality with zero
    exprt zero = gen_zero(migrate_type_back(simp->type));

    expr2tc zero2;
    migrate_expr(zero, zero2);

    return not2tc(equality2tc(simp, zero2));
  }
  else if(is_pointer_type(type) && is_pointer_type(simp))
  {
    // Casting from one pointer to another is meaningless... except when there's
    // pointer arithmetic about to be applied to it. So, only remove typecasts
    // that don't change the subtype width.
    const pointer_type2t &ptr_to = to_pointer_type(type);
    const pointer_type2t &ptr_from = to_pointer_type(simp->type);

    if(
      is_symbol_type(ptr_to.subtype) || is_symbol_type(ptr_from.subtype) ||
      is_code_type(ptr_to.subtype) || is_code_type(ptr_from.subtype))
      return expr2tc(); // Not worth thinking about

    if(
      is_array_type(ptr_to.subtype) &&
      is_symbol_type(get_array_subtype(ptr_to.subtype)))
      return expr2tc(); // Not worth thinking about

    if(
      is_array_type(ptr_from.subtype) &&
      is_symbol_type(get_array_subtype(ptr_from.subtype)))
      return expr2tc(); // Not worth thinking about

    try
    {
      unsigned int to_width =
        (is_empty_type(ptr_to.subtype)) ? 8 : ptr_to.subtype->get_width();
      unsigned int from_width =
        (is_empty_type(ptr_from.subtype)) ? 8 : ptr_from.subtype->get_width();

      if(to_width == from_width)
        return simp;

      return expr2tc();
    }
    catch(array_type2t::dyn_sized_array_excp *e)
    {
      // Something crazy, and probably C++ based, occurred. Don't attempt to
      // simplify.
      return expr2tc();
    }
  }
  else if(is_typecast2t(simp) && type == simp->type)
  {
    // Typecast from a typecast can be eliminated. We'll be simplified even
    // further by the caller.
    return expr2tc(new typecast2t(type, to_typecast2t(simp).from));
  }

  return expr2tc();
}

expr2tc nearbyint2t::do_simplify(bool second __attribute__((unused))) const
{
  if(!is_number_type(type))
    return expr2tc();

  // Try to recursively simplify nested operation, if any
  expr2tc to_simplify = try_simplification(from);
  if(!is_constant_floatbv2t(to_simplify))
  {
    // Were we able to simplify anything?
    if(from != to_simplify)
      return typecast_check_return(type, nearbyint2tc(type, to_simplify));

    return expr2tc();
  }

  ieee_floatt n = to_constant_floatbv2t(to_simplify).value;
  if(n.is_NaN() || n.is_zero() || n.is_infinity())
    return typecast_check_return(type, from);

  return expr2tc();
}

expr2tc address_of2t::do_simplify(bool second __attribute__((unused))) const
{
  // NB: address of never has its operands simplified below its feet for sanitys
  // sake.
  // Only attempt to simplify indexes. Whatever we're taking the address of,
  // we can't simplify away the symbol.
  if(is_index2t(ptr_obj))
  {
    const index2t &idx = to_index2t(ptr_obj);
    const pointer_type2t &ptr_type = to_pointer_type(type);

    // Don't simplify &a[0]
    if(
      is_constant_int2t(idx.index) &&
      to_constant_int2t(idx.index).value.is_zero())
      return expr2tc();

    expr2tc new_index = try_simplification(idx.index);
    expr2tc zero = constant_int2tc(index_type2(), BigInt(0));
    expr2tc new_idx = index2tc(idx.type, idx.source_value, zero);
    expr2tc sub_addr_of = address_of2tc(ptr_type.subtype, new_idx);

    return add2tc(type, sub_addr_of, new_index);
  }

  return expr2tc();
}

template <template <typename> class TFunctor, typename constructor>
static expr2tc simplify_relations(
  const type2tc &type,
  const expr2tc &side_1,
  const expr2tc &side_2)
{
  if(!is_number_type(type))
    return expr2tc();

  // Try to recursively simplify nested operations both sides, if any
  expr2tc simplied_side_1 = try_simplification(side_1);
  expr2tc simplied_side_2 = try_simplification(side_2);

  if(!is_constant_expr(simplied_side_1) || !is_constant_expr(simplied_side_2))
  {
    // Were we able to simplify the sides?
    if((side_1 != simplied_side_1) || (side_2 != simplied_side_2))
    {
      expr2tc new_op =
        expr2tc(new constructor(simplied_side_1, simplied_side_2));

      return typecast_check_return(type, new_op);
    }

    return expr2tc();
  }

  expr2tc simpl_res;

  if(is_bv_type(simplied_side_1) || is_bv_type(simplied_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_int2t;

    std::function<BigInt &(expr2tc &)> get_value = [](expr2tc &c) -> BigInt & {
      return to_constant_int2t(c).value;
    };

    simpl_res = TFunctor<BigInt>::simplify(
      simplied_side_1, simplied_side_2, is_constant, get_value);
  }
  else if(is_fixedbv_type(simplied_side_1) || is_fixedbv_type(simplied_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_fixedbv2t;

    std::function<fixedbvt &(expr2tc &)> get_value =
      [](expr2tc &c) -> fixedbvt & { return to_constant_fixedbv2t(c).value; };

    simpl_res = TFunctor<fixedbvt>::simplify(
      simplied_side_1, simplied_side_2, is_constant, get_value);
  }
  else if(is_floatbv_type(simplied_side_1) || is_floatbv_type(simplied_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_floatbv2t;

    std::function<ieee_floatt &(expr2tc &)> get_value = [](
      expr2tc &c) -> ieee_floatt & { return to_constant_floatbv2t(c).value; };

    simpl_res = TFunctor<ieee_floatt>::simplify(
      simplied_side_1, simplied_side_2, is_constant, get_value);
  }
  else if(is_bool_type(simplied_side_1) || is_bool_type(simplied_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_bool2t;

    std::function<bool &(expr2tc &)> get_value = [](expr2tc &c) -> bool & {
      return to_constant_bool2t(c).value;
    };

    simpl_res = TFunctor<bool>::simplify(
      simplied_side_1, simplied_side_2, is_constant, get_value);
  }
  else
    return expr2tc();

  return typecast_check_return(type, simpl_res);
}

template <template <typename> class TFunctor, typename constructor>
static expr2tc simplify_floatbv_relations(
  const type2tc &type,
  const expr2tc &side_1,
  const expr2tc &side_2)
{
  if(!is_number_type(type))
    return expr2tc();

  // Try to recursively simplify nested operations both sides, if any
  expr2tc simplied_side_1 = try_simplification(side_1);
  expr2tc simplied_side_2 = try_simplification(side_2);

  if(
    is_constant_expr(simplied_side_1) || is_constant_expr(simplied_side_2) ||
    (simplied_side_1 == simplied_side_2))
  {
    expr2tc simpl_res = expr2tc();

    if(is_floatbv_type(simplied_side_1) || is_floatbv_type(simplied_side_2))
    {
      std::function<bool(const expr2tc &)> is_constant =
        (bool (*)(const expr2tc &)) & is_constant_floatbv2t;

      std::function<ieee_floatt &(expr2tc &)> get_value = [](
        expr2tc &c) -> ieee_floatt & { return to_constant_floatbv2t(c).value; };

      simpl_res = TFunctor<ieee_floatt>::simplify(
        simplied_side_1, simplied_side_2, is_constant, get_value);
    }
    else
      assert(0);

    return typecast_check_return(type, simpl_res);
  }

  // Were we able to simplify the sides?
  if((side_1 != simplied_side_1) || (side_2 != simplied_side_2))
  {
    expr2tc new_op = expr2tc(new constructor(simplied_side_1, simplied_side_2));

    return typecast_check_return(type, new_op);
  }

  return expr2tc();
}

template <class constant_type>
struct IEEE_equalitytor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    // Two constants? Simplify to result of the comparison
    if(is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      return constant_bool2tc(get_value(c1) == get_value(c2));
    }

    if(op1 == op2)
    {
      // x == x is the same as saying !isnan(x)
      expr2tc is_nan(new isnan2t(op1));
      expr2tc is_not_nan = not2tc(is_nan);
      return try_simplification(is_not_nan);
    }

    return expr2tc();
  }
};

template <class constant_type>
struct Equalitytor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    if(is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      return constant_bool2tc(get_value(c1) == get_value(c2));
    }

    return expr2tc();
  }
};

expr2tc equality2t::do_simplify(bool second __attribute__((unused))) const
{
  // If we're dealing with floatbvs, call IEEE_equalitytor instead
  if(is_floatbv_type(side_1) || is_floatbv_type(side_2))
    return simplify_floatbv_relations<IEEE_equalitytor, equality2t>(
      type, side_1, side_2);

  return simplify_relations<Equalitytor, equality2t>(type, side_1, side_2);
}

template <class constant_type>
struct Notequaltor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    if(is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      return constant_bool2tc(get_value(c1) != get_value(c2));
    }

    return expr2tc();
  }
};

expr2tc notequal2t::do_simplify(bool second __attribute__((unused))) const
{
  return simplify_relations<Notequaltor, notequal2t>(type, side_1, side_2);
}

template <class constant_type>
struct Lessthantor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    // op1 < zero and op2 is unsigned: always true
    if(is_constant(op1))
    {
      expr2tc c1 = op1;
      if((get_value(c1) < 0) && is_unsignedbv_type(op2))
        return gen_true_expr();
    }

    if(is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      return constant_bool2tc(get_value(c1) < get_value(c2));
    }

    return expr2tc();
  }
};

expr2tc lessthan2t::do_simplify(bool second __attribute__((unused))) const
{
  return simplify_relations<Lessthantor, lessthan2t>(type, side_1, side_2);
}

template <class constant_type>
struct Greaterthantor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    // op2 < zero and op1 is unsigned: always true
    if(is_constant(op2))
    {
      expr2tc c2 = op2;
      if((get_value(c2) < 0) && is_unsignedbv_type(op1))
        return gen_true_expr();
    }

    if(is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      return constant_bool2tc(get_value(c1) > get_value(c2));
    }

    return expr2tc();
  }
};

expr2tc greaterthan2t::do_simplify(bool second __attribute__((unused))) const
{
  return simplify_relations<Greaterthantor, greaterthan2t>(
    type, side_1, side_2);
}

template <class constant_type>
struct Lessthanequaltor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    // op1 <= zero and op2 is unsigned: always true
    if(is_constant(op1))
    {
      expr2tc c1 = op1;
      if((get_value(c1) <= 0) && is_unsignedbv_type(op2))
        return gen_true_expr();
    }

    if(is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      return constant_bool2tc(get_value(c1) <= get_value(c2));
    }

    return expr2tc();
  }
};

expr2tc lessthanequal2t::do_simplify(bool second __attribute__((unused))) const
{
  return simplify_relations<Lessthanequaltor, lessthanequal2t>(
    type, side_1, side_2);
}

template <class constant_type>
struct Greaterthanequaltor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    // op2 <= zero and op1 is unsigned: always true
    if(is_constant(op2))
    {
      expr2tc c2 = op2;
      if((get_value(c2) <= 0) && is_unsignedbv_type(op1))
        return gen_true_expr();
    }

    if(is_constant(op1) && is_constant(op2))
    {
      expr2tc c1 = op1, c2 = op2;
      return constant_bool2tc(get_value(c1) >= get_value(c2));
    }

    return expr2tc();
  }
};

expr2tc greaterthanequal2t::do_simplify(bool second
                                        __attribute__((unused))) const
{
  return simplify_relations<Greaterthanequaltor, greaterthanequal2t>(
    type, side_1, side_2);
}

expr2tc if2t::do_simplify(bool second __attribute__((unused))) const
{
  if(is_constant_expr(cond))
  {
    // We can simplify this.
    if(is_constant_bool2t(cond))
    {
      if(to_constant_bool2t(cond).value)
      {
        return true_value;
      }

      return false_value;
    }
    else
    {
      // Cast towards a bool type.
      expr2tc cast = typecast2tc(type_pool.get_bool(), cond);
      cast = cast->simplify();
      assert(
        !is_nil_expr(cast) &&
        "We should always be able to cast a "
        "constant value to a constant bool");

      if(to_constant_bool2t(cast).value)
      {
        return true_value;
      }

      return false_value;
    }
  }
  else
  {
    return expr2tc();
  }
}

expr2tc overflow_cast2t::do_simplify(bool second __attribute__((unused))) const
{
  return expr2tc();
}

expr2tc overflow2t::do_simplify(bool second __attribute__((unused))) const
{
  return expr2tc();
}

// Heavily inspired by cbmc's simplify_exprt::objects_equal_address_of
static expr2tc obj_equals_addr_of(const expr2tc &a, const expr2tc &b)
{
  if(is_symbol2t(a) && is_symbol2t(b))
  {
    if(a == b)
      return gen_true_expr();
  }
  else if(is_index2t(a) && is_index2t(b))
  {
    return obj_equals_addr_of(
      to_index2t(a).source_value, to_index2t(b).source_value);
  }
  else if(is_member2t(a) && is_member2t(b))
  {
    return obj_equals_addr_of(
      to_member2t(a).source_value, to_member2t(b).source_value);
  }
  else if(is_constant_string2t(a) && is_constant_string2t(b))
  {
    bool val = (to_constant_string2t(a).value == to_constant_string2t(b).value);
    if(val)
      return gen_true_expr();

    return gen_false_expr();
  }

  return expr2tc();
}

expr2tc same_object2t::do_simplify(bool second __attribute__((unused))) const
{
  if(is_address_of2t(side_1) && is_address_of2t(side_2))
    return obj_equals_addr_of(
      to_address_of2t(side_1).ptr_obj, to_address_of2t(side_2).ptr_obj);

  if(
    is_symbol2t(side_1) && is_symbol2t(side_2) &&
    to_symbol2t(side_1).get_symbol_name() == "NULL" &&
    to_symbol2t(side_1).get_symbol_name() == "NULL")
    return gen_true_expr();

  return expr2tc();
}

expr2tc concat2t::do_simplify(bool second __attribute__((unused))) const
{
  if(!is_constant_int2t(side_1) || !is_constant_int2t(side_2))
    return expr2tc();

  const mp_integer &value1 = to_constant_int2t(side_1).value;
  const mp_integer &value2 = to_constant_int2t(side_2).value;

  // k; Take the values, and concatenate. Side 1 has higher end bits.
  mp_integer accuml = value1;
  accuml *= (1ULL << side_2->type->get_width());
  accuml += value2;

  return constant_int2tc(type, accuml);
}

expr2tc extract2t::do_simplify(bool second __attribute__((unused))) const
{
  assert(is_bv_type(type));

  if(!is_constant_int2t(from))
    return expr2tc();

  // If you're hitting this, a non-bitfield related piece of code is now
  // generating extracts, and you have to consider performing extracts on
  // negative numbers.
  assert(is_unsignedbv_type(from->type));
  const constant_int2t &cint = to_constant_int2t(from);
  const BigInt &theint = cint.value;
  assert(theint.is_positive());
  assert(theint.get_len() <= 2);

  // Take the value, mask and shift.
  uint64_t theval = theint.to_uint64();
  theval >>= lower;
  theval &= (2 << upper) - 1;
  bool isneg = (1 << (upper)) & theval;

  if(is_signedbv_type(type) && isneg)
  {
    // Type punning.
    union {
      int64_t sign;
      uint64_t nosign;
    } totallytmp;

    theval |= 0xFFFFFFFFFFFFFFFFULL << upper;
    totallytmp.nosign = theval;
    return constant_int2tc(type, BigInt(totallytmp.sign));
  }

  return constant_int2tc(type, BigInt(theval));
}

template <template <typename> class TFunctor, typename constructor>
static expr2tc simplify_floatbv_1op(const type2tc &type, const expr2tc &value)
{
  if(!is_number_type(type))
    return expr2tc();

  // Try to recursively simplify nested operation, if any
  expr2tc to_simplify = try_simplification(value);
  if(!is_constant_expr(to_simplify))
  {
    // Were we able to simplify anything?
    if(value != to_simplify)
    {
      expr2tc new_neg = expr2tc(new constructor(to_simplify));
      return typecast_check_return(type, new_neg);
    }

    return expr2tc();
  }

  expr2tc simpl_res = expr2tc();

  if(is_fixedbv_type(value))
  {
    std::function<constant_fixedbv2t &(expr2tc &)> to_constant =
      (constant_fixedbv2t & (*)(expr2tc &)) to_constant_fixedbv2t;

    simpl_res =
      TFunctor<constant_fixedbv2t>::simplify(to_simplify, to_constant);
  }
  else if(is_floatbv_type(value))
  {
    std::function<constant_floatbv2t &(expr2tc &)> to_constant =
      (constant_floatbv2t & (*)(expr2tc &)) to_constant_floatbv2t;

    simpl_res =
      TFunctor<constant_floatbv2t>::simplify(to_simplify, to_constant);
  }
  else
    return expr2tc();

  return typecast_check_return(type, simpl_res);
}

template <class constant_type>
struct Isnantor
{
  static expr2tc simplify(
    const expr2tc &number,
    std::function<constant_type &(expr2tc &)> to_constant)
  {
    expr2tc c = number;
    return constant_bool2tc(to_constant(c).value.is_NaN());
  }
};

expr2tc isnan2t::do_simplify(bool second __attribute__((unused))) const
{
  return simplify_floatbv_1op<Isnantor, isnan2t>(type, value);
}

template <class constant_type>
struct Isinftor
{
  static expr2tc simplify(
    const expr2tc &number,
    std::function<constant_type &(expr2tc &)> to_constant)
  {
    expr2tc c = number;
    return constant_bool2tc(to_constant(c).value.is_infinity());
  }
};

expr2tc isinf2t::do_simplify(bool second __attribute__((unused))) const
{
  return simplify_floatbv_1op<Isinftor, isinf2t>(type, value);
}

template <class constant_type>
struct Isnormaltor
{
  static expr2tc simplify(
    const expr2tc &number,
    std::function<constant_type &(expr2tc &)> to_constant)
  {
    expr2tc c = number;
    return constant_bool2tc(to_constant(c).value.is_normal());
  }
};

expr2tc isnormal2t::do_simplify(bool second __attribute__((unused))) const
{
  return simplify_floatbv_1op<Isnormaltor, isnormal2t>(type, value);
}

template <class constant_type>
struct Isfinitetor
{
  static expr2tc simplify(
    const expr2tc &number,
    std::function<constant_type &(expr2tc &)> to_constant)
  {
    expr2tc c = number;
    return constant_bool2tc(to_constant(c).value.is_finite());
  }
};

expr2tc isfinite2t::do_simplify(bool second __attribute__((unused))) const
{
  return simplify_floatbv_1op<Isfinitetor, isfinite2t>(type, value);
}

template <class constant_type>
struct Signbittor
{
  static expr2tc simplify(
    const expr2tc &number,
    std::function<constant_type &(expr2tc &)> to_constant)
  {
    auto c = number;
    return constant_bool2tc(to_constant(c).value.get_sign());
  }
};

expr2tc signbit2t::do_simplify(bool second __attribute__((unused))) const
{
  return simplify_floatbv_1op<Signbittor, signbit2t>(type, operand);
}

template <template <typename> class TFunctor, typename constructor>
static expr2tc simplify_floatbv_2ops(
  const type2tc &type,
  const expr2tc &side_1,
  const expr2tc &side_2,
  const expr2tc &rounding_mode)
{
  assert(is_floatbv_type(type));

  if(!is_number_type(type) && !is_pointer_type(type))
    return expr2tc();

  // Try to recursively simplify nested operations both sides, if any
  expr2tc simplied_side_1 = try_simplification(side_1);
  expr2tc simplied_side_2 = try_simplification(side_2);

  // Try to handle NaN
  if(is_constant_floatbv2t(simplied_side_1))
    if(to_constant_floatbv2t(simplied_side_1).value.is_NaN())
      return expr2tc(simplied_side_1->clone());

  if(is_constant_floatbv2t(simplied_side_2))
    if(to_constant_floatbv2t(simplied_side_2).value.is_NaN())
      return expr2tc(simplied_side_2->clone());

  if(
    !is_constant_expr(simplied_side_1) || !is_constant_expr(simplied_side_2) ||
    !is_constant_int2t(rounding_mode))
  {
    // Were we able to simplify the sides?
    if((side_1 != simplied_side_1) || (side_2 != simplied_side_2))
    {
      expr2tc new_op = expr2tc(
        new constructor(type, simplied_side_1, simplied_side_2, rounding_mode));

      return typecast_check_return(type, new_op);
    }

    return expr2tc();
  }

  expr2tc simpl_res = expr2tc();

  if(is_floatbv_type(simplied_side_1) || is_floatbv_type(simplied_side_2))
  {
    std::function<bool(const expr2tc &)> is_constant =
      (bool (*)(const expr2tc &)) & is_constant_floatbv2t;

    std::function<ieee_floatt &(expr2tc &)> get_value = [](
      expr2tc &c) -> ieee_floatt & { return to_constant_floatbv2t(c).value; };

    simpl_res = TFunctor<ieee_floatt>::simplify(
      simplied_side_1, simplied_side_2, rounding_mode, is_constant, get_value);
  }
  else
    assert(0);

  return typecast_check_return(type, simpl_res);
}

template <class constant_type>
struct IEEE_addtor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const expr2tc &rm,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    // Two constants? Simplify to result of the addition
    if(is_constant(op1) && is_constant(op2))
    {
      ieee_floatt::rounding_modet mode =
        static_cast<ieee_floatt::rounding_modet>(
          to_constant_int2t(rm).value.to_long());

      expr2tc c1 = op1;
      get_value(c1).rounding_mode = mode;

      expr2tc c2 = op2;
      get_value(c2).rounding_mode = mode;

      get_value(c1) += get_value(c2);
      return c1;
    }

    return expr2tc();
  }
};

expr2tc ieee_add2t::do_simplify(bool second __attribute__((unused))) const
{
  return simplify_floatbv_2ops<IEEE_addtor, ieee_add2t>(
    type, side_1, side_2, rounding_mode);
}

template <class constant_type>
struct IEEE_subtor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const expr2tc &rm,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    // Two constants? Simplify to result of the subtraction
    if(is_constant(op1) && is_constant(op2))
    {
      ieee_floatt::rounding_modet mode =
        static_cast<ieee_floatt::rounding_modet>(
          to_constant_int2t(rm).value.to_long());

      expr2tc c1 = op1;
      get_value(c1).rounding_mode = mode;

      expr2tc c2 = op2;
      get_value(c2).rounding_mode = mode;

      get_value(c1) -= get_value(c2);
      return c1;
    }

    return expr2tc();
  }
};

expr2tc ieee_sub2t::do_simplify(bool second __attribute__((unused))) const
{
  return simplify_floatbv_2ops<IEEE_subtor, ieee_sub2t>(
    type, side_1, side_2, rounding_mode);
}

template <class constant_type>
struct IEEE_multor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const expr2tc &rm,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    // Two constants? Simplify to result of the multiplication
    if(is_constant(op1) && is_constant(op2))
    {
      ieee_floatt::rounding_modet mode =
        static_cast<ieee_floatt::rounding_modet>(
          to_constant_int2t(rm).value.to_long());

      expr2tc c1 = op1;
      get_value(c1).rounding_mode = mode;

      expr2tc c2 = op2;
      get_value(c2).rounding_mode = mode;

      get_value(c1) *= get_value(c2);
      return c1;
    }

    return expr2tc();
  }
};

expr2tc ieee_mul2t::do_simplify(bool second __attribute__((unused))) const
{
  return simplify_floatbv_2ops<IEEE_multor, ieee_mul2t>(
    type, side_1, side_2, rounding_mode);
}

template <class constant_type>
struct IEEE_divtor
{
  static expr2tc simplify(
    const expr2tc &op1,
    const expr2tc &op2,
    const expr2tc &rm,
    const std::function<bool(const expr2tc &)> &is_constant,
    std::function<constant_type &(expr2tc &)> get_value)
  {
    // Two constants? Simplify to result of the division
    if(is_constant(op1) && is_constant(op2))
    {
      ieee_floatt::rounding_modet mode =
        static_cast<ieee_floatt::rounding_modet>(
          to_constant_int2t(rm).value.to_long());

      expr2tc c1 = op1;
      get_value(c1).rounding_mode = mode;

      expr2tc c2 = op2;
      get_value(c2).rounding_mode = mode;

      get_value(c1) /= get_value(c2);
      return c1;
    }

    if(is_constant(op2))
    {
      ieee_floatt::rounding_modet mode =
        static_cast<ieee_floatt::rounding_modet>(
          to_constant_int2t(rm).value.to_long());

      expr2tc c2 = op2;
      get_value(c2).rounding_mode = mode;

      // Denominator is one? Exact for all rounding modes.
      if(get_value(c2) == 1)
        return op1;
    }

    return expr2tc();
  }
};

expr2tc ieee_div2t::do_simplify(bool second __attribute__((unused))) const
{
  return simplify_floatbv_2ops<IEEE_divtor, ieee_div2t>(
    type, side_1, side_2, rounding_mode);
}

expr2tc ieee_fma2t::do_simplify(bool second __attribute__((unused))) const
{
  assert(is_floatbv_type(type));

  if(!is_number_type(type) && !is_pointer_type(type))
    return expr2tc();

  // Try to recursively simplify nested operations both sides, if any
  expr2tc simplied_value_1 = try_simplification(value_1);
  expr2tc simplied_value_2 = try_simplification(value_2);
  expr2tc simplied_value_3 = try_simplification(value_3);

  if(
    !is_constant_expr(simplied_value_1) ||
    !is_constant_expr(simplied_value_2) ||
    !is_constant_expr(simplied_value_3) || !is_constant_int2t(rounding_mode))
  {
    // Were we able to simplify the sides?
    if(
      (value_1 != simplied_value_1) || (value_2 != simplied_value_2) ||
      (value_3 != simplied_value_3))
    {
      expr2tc new_op = ieee_fma2tc(
        type,
        simplied_value_1,
        simplied_value_2,
        simplied_value_3,
        rounding_mode);

      return typecast_check_return(type, new_op);
    }

    return expr2tc();
  }

  ieee_floatt n1 = to_constant_floatbv2t(simplied_value_1).value;
  ieee_floatt n2 = to_constant_floatbv2t(simplied_value_2).value;
  ieee_floatt n3 = to_constant_floatbv2t(simplied_value_3).value;

  // If x or y are NaN, NaN is returned
  if(n1.is_NaN() || n2.is_NaN())
  {
    n1.make_NaN();
    return constant_floatbv2tc(n1);
  }

  // If x is zero and y is infinite or if x is infinite and y is zero,
  // and z is not a NaN, a domain error shall occur, and either a NaN,
  // or an implementation-defined value shall be returned.

  // If x is zero and y is infinite or if x is infinite and y is zero,
  // and z is a NaN, then NaN is returned and FE_INVALID may be raised
  if((n1.is_zero() && n2.is_infinity()) || (n2.is_zero() && n1.is_infinity()))
  {
    n1.make_NaN();
    return constant_floatbv2tc(n1);
  }

  // If z is NaN, and x*y aren't 0*Inf or Inf*0, then NaN is returned
  // (without FE_INVALID)
  if(n3.is_NaN())
  {
    n1.make_NaN();
    return constant_floatbv2tc(n1);
  }

  // If x*y is an exact infinity and z is an infinity with the opposite sign,
  // NaN is returned and FE_INVALID is raised
  n1 *= n2;
  if((n1.is_infinity() && n3.is_infinity()) && (n1.get_sign() != n3.get_sign()))
  {
    n1.make_NaN();
    return constant_floatbv2tc(n1);
  }

  return expr2tc();
}

expr2tc ieee_sqrt2t::do_simplify(bool second __attribute__((unused))) const
{
  if(!is_number_type(type))
    return expr2tc();

  // Try to recursively simplify nested operation, if any
  expr2tc to_simplify = try_simplification(value);
  if(!is_constant_floatbv2t(to_simplify))
  {
    // Were we able to simplify anything?
    if(value != to_simplify)
      return typecast_check_return(
        type, ieee_sqrt2tc(type, to_simplify, rounding_mode));

    return expr2tc();
  }

  ieee_floatt n = to_constant_floatbv2t(to_simplify).value;
  if(n < 0)
  {
    n.make_NaN();
    return constant_floatbv2tc(n);
  }

  if(n.is_NaN() || n.is_zero() || n.is_infinity())
    return typecast_check_return(type, value);

  return expr2tc();
}
