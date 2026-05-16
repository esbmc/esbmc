#include <irep2/irep2_utils.h>
#include <util/c_types.h>

void make_not(expr2tc &expr)
{
  if (is_true(expr))
  {
    expr = gen_false_expr();
    return;
  }

  if (is_false(expr))
  {
    expr = gen_true_expr();
    return;
  }

  expr2tc new_expr;
  if (is_not2t(expr))
    new_expr = to_not2t(expr).value;
  else
    new_expr = not2tc(expr);

  expr.swap(new_expr);
}

expr2tc conjunction(std::vector<expr2tc> cs)
{
  if (cs.empty())
    return gen_true_expr();

  expr2tc res = cs[0];
  for (std::size_t i = 1; i < cs.size(); ++i)
    res = and2tc(res, cs[i]);

  return res;
}

expr2tc disjunction(std::vector<expr2tc> cs)
{
  if (cs.empty())
    return gen_true_expr();

  expr2tc res = cs[0];
  for (std::size_t i = 1; i < cs.size(); ++i)
    res = or2tc(res, cs[i]);

  return res;
}

expr2tc gen_nondet(const type2tc &type)
{
  return sideeffect2tc(
    type,
    expr2tc(),
    expr2tc(),
    std::vector<expr2tc>(),
    type2tc(),
    sideeffect2t::allockind::nondet);
}

expr2tc gen_zero(const type2tc &type, bool array_as_array_of)
{
  switch (type->type_id)
  {
  case type2t::bool_id:
    return gen_false_expr();

  case type2t::unsignedbv_id:
  case type2t::signedbv_id:
    return constant_int2tc(type, BigInt(0));

  case type2t::fixedbv_id:
    return constant_fixedbv2tc(fixedbvt(fixedbv_spect(to_fixedbv_type(type))));

  case type2t::floatbv_id:
    return constant_floatbv2tc(
      ieee_floatt(ieee_float_spect(to_floatbv_type(type))));

  case type2t::vector_id:
  {
    auto vec_type = to_vector_type(type);
    assert(is_constant_int2t(vec_type.array_size));
    auto s = to_constant_int2t(vec_type.array_size);

    std::vector<expr2tc> members;
    for (long int i = 0; i < s.as_long(); i++)
      members.push_back(
        gen_zero(to_vector_type(type).subtype, array_as_array_of));

    return constant_vector2tc(type, members);
  }
  case type2t::array_id:
  {
    if (array_as_array_of)
      return constant_array_of2tc(type, gen_zero(to_array_type(type).subtype));

    auto arr_type = to_array_type(type);

    assert(is_constant_int2t(arr_type.array_size));
    auto s = to_constant_int2t(arr_type.array_size);

    std::vector<expr2tc> members;
    for (long int i = 0; i < s.as_long(); i++)
      members.push_back(
        gen_zero(to_array_type(type).subtype, array_as_array_of));

    return constant_array2tc(type, members);
  }

  case type2t::pointer_id:
    return symbol2tc(type, "NULL");

  case type2t::struct_id:
  {
    auto struct_type = to_struct_type(type);

    std::vector<expr2tc> members;
    for (auto const &member_type : struct_type.members)
      members.push_back(gen_zero(member_type, array_as_array_of));

    return constant_struct2tc(type, members);
  }

  case type2t::union_id:
  {
    auto union_type = to_union_type(type);

    assert(!union_type.members.empty());
    std::vector<expr2tc> members = {
      gen_zero(union_type.members.front(), array_as_array_of)};

    return constant_union2tc(type, union_type.member_names.front(), members);
  }

  default:
    break;
  }

  log_error("Can't generate zero for type {}", get_type_id(type));
  abort();
}

expr2tc gen_one(const type2tc &type)
{
  switch (type->type_id)
  {
  case type2t::bool_id:
    return gen_true_expr();

  case type2t::unsignedbv_id:
  case type2t::signedbv_id:
    return constant_int2tc(type, BigInt(1));

  case type2t::fixedbv_id:
  {
    fixedbvt f(fixedbv_spect(to_fixedbv_type(type)));
    f.from_integer(BigInt(1));
    return constant_fixedbv2tc(f);
  }

  case type2t::floatbv_id:
  {
    ieee_floatt f(ieee_float_spect(to_floatbv_type(type)));
    f.from_integer(BigInt(1));
    return constant_floatbv2tc(f);
  }

  default:
    break;
  }

  log_error("Can't generate one for type {}", get_type_id(type));
  abort();
}

expr2tc distribute_vector_operation(
  expr2t::expr_ids id,
  expr2tc op1,
  expr2tc op2,
  expr2tc rm)
{
#ifndef NDEBUG
  assert(is_vector_type(op1) || (op2 && is_vector_type(op2)));
#endif
  auto is_op1_vector = is_vector_type(op1);
  auto vector_length = is_op1_vector ? to_vector_type(op1->type).array_size
                                     : to_vector_type(op2->type).array_size;
  assert(is_constant_int2t(vector_length));

  auto result = is_op1_vector ? gen_zero(op1->type) : gen_zero(op2->type);

  for (size_t i = 0; i < to_constant_int2t(vector_length).as_ulong(); i++)
  {
    BigInt position(i);
    type2tc vector_type =
      to_vector_type(is_vector_type(op1->type) ? op1->type : op2->type).subtype;
    expr2tc local_op1 = op1;
    if (is_vector_type(op1->type))
    {
      local_op1 = index2tc(
        to_vector_type(op1->type).subtype,
        op1,
        constant_int2tc(get_uint32_type(), position));
    }

    expr2tc local_op2 = op2;
    if (op2 && is_vector_type(op2->type))
    {
      local_op2 = index2tc(
        to_vector_type(op2->type).subtype,
        op2,
        constant_int2tc(get_uint32_type(), position));
    }

    expr2tc to_add;
    switch (id)
    {
    case expr2t::neg_id:
      to_add = neg2tc(vector_type, local_op1);
      break;
    case expr2t::bitnot_id:
      to_add = bitnot2tc(vector_type, local_op1);
      break;
    case expr2t::sub_id:
      to_add = sub2tc(vector_type, local_op1, local_op2);
      break;
    case expr2t::mul_id:
      to_add = mul2tc(vector_type, local_op1, local_op2);
      break;
    case expr2t::div_id:
      to_add = div2tc(vector_type, local_op1, local_op2);
      break;
    case expr2t::modulus_id:
      to_add = modulus2tc(vector_type, local_op1, local_op2);
      break;
    case expr2t::add_id:
      to_add = add2tc(vector_type, local_op1, local_op2);
      break;
    case expr2t::shl_id:
      to_add = shl2tc(vector_type, local_op1, local_op2);
      break;
    case expr2t::bitxor_id:
      to_add = bitxor2tc(vector_type, local_op1, local_op2);
      break;
    case expr2t::bitor_id:
      to_add = bitor2tc(vector_type, local_op1, local_op2);
      break;
    case expr2t::bitand_id:
      to_add = bitand2tc(vector_type, local_op1, local_op2);
      break;
    case expr2t::ieee_add_id:
      to_add = ieee_add2tc(vector_type, local_op1, local_op2, rm);
      break;
    case expr2t::ieee_div_id:
      to_add = ieee_div2tc(vector_type, local_op1, local_op2, rm);
      break;
    case expr2t::ieee_sub_id:
      to_add = ieee_sub2tc(vector_type, local_op1, local_op2, rm);
      break;
    case expr2t::ieee_mul_id:
      to_add = ieee_mul2tc(vector_type, local_op1, local_op2, rm);
      break;
    default:
      assert(0 && "Unsupported operation for Vector");
      abort();
    }
    to_constant_vector2t(result).datatype_members[i] = to_add;
  }
  return result;
}

expr2tc make_cmp_value(const type2tc &t, int v)
{
  if (!is_struct_type(t))
    return expr2tc();
  const struct_type2t &st = to_struct_type(t);
  if (st.members.empty())
    return expr2tc();
  std::vector<expr2tc> ops;
  ops.reserve(st.members.size());
  ops.push_back(constant_int2tc(st.members[0], BigInt(v)));
  for (size_t i = 1; i < st.members.size(); ++i)
    ops.push_back(gen_zero(st.members[i]));
  return constant_struct2tc(t, std::move(ops));
}

void get_symbols(
  const expr2tc &expr,
  std::unordered_set<expr2tc, irep2_hash> &symbols)
{
  if (is_nil_expr(expr))
    return;

  if (is_symbol2t(expr))
  {
    symbol2t s = to_symbol2t(expr);
    if (s.thename.as_string().find("__ESBMC_") != std::string::npos)
      return;
    symbols.insert(expr);
  }

  expr->foreach_operand(
    [&symbols](const expr2tc &e) -> void { get_symbols(e, symbols); });
}
