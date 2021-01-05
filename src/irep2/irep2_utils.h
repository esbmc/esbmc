#ifndef UTIL_IREP2_UTILS_H_
#define UTIL_IREP2_UTILS_H_

#include <util/c_types.h>
#include <util/message/default_message.h>
#include <util/message/format.h>
#include <irep2/irep2_expr.h>
#include <util/migrate.h>

std::string indent_str_irep2(unsigned int indent);

// Map a base type to it's list of names
template <typename T>
class base_to_names;

template <class T>
std::string pretty_print_func(unsigned int indent, std::string ident, T obj)
{
  list_of_memberst memb = obj.tostring(indent + 2);

  std::string indentstr = indent_str_irep2(indent);
  std::string exprstr = std::move(ident);

  for(list_of_memberst::const_iterator it = memb.begin(); it != memb.end();
      it++)
  {
    exprstr += "\n" + indentstr + "* " + it->first + " : " + it->second;
  }

  return exprstr;
}

/** Test whether type is an integer. */
inline bool is_bv_type(const type2tc &t)
{
  return t->type_id == type2t::unsignedbv_id ||
         t->type_id == type2t::signedbv_id;
}

inline bool is_bv_type(const expr2tc &e)
{
  return is_bv_type(e->type);
}

/** Test whether type is a float/double. */
inline bool is_fractional_type(const type2tc &t)
{
  return t->type_id == type2t::fixedbv_id || t->type_id == type2t::floatbv_id;
}

inline bool is_fractional_type(const expr2tc &e)
{
  return is_bv_type(e->type);
}

/** Test whether type is a number type - bv, fixedbv or floatbv. */
inline bool is_number_type(const type2tc &t)
{
  return t->type_id == type2t::unsignedbv_id ||
         t->type_id == type2t::signedbv_id ||
         t->type_id == type2t::fixedbv_id || t->type_id == type2t::floatbv_id ||
         t->type_id == type2t::bool_id;
}

inline bool is_number_type(const expr2tc &e)
{
  return is_number_type(e->type);
}

inline bool is_scalar_type(const type2tc &t)
{
  return is_number_type(t) || is_pointer_type(t) || is_empty_type(t) ||
         is_code_type(t);
}

inline bool is_scalar_type(const expr2tc &e)
{
  return is_scalar_type(e->type);
}

inline bool is_multi_dimensional_array(const type2tc &t)
{
  if(is_array_type(t))
    return is_array_type(to_array_type(t).subtype);

  return false;
}

inline bool is_multi_dimensional_array(const expr2tc &e)
{
  return is_multi_dimensional_array(e->type);
}

inline bool is_constant_number(const expr2tc &t)
{
  return t->expr_id == expr2t::constant_int_id ||
         t->expr_id == expr2t::constant_fixedbv_id ||
         t->expr_id == expr2t::constant_floatbv_id ||
         t->expr_id == expr2t::constant_bool_id;
}

inline bool is_constant_expr(const expr2tc &t)
{
  return t->expr_id == expr2t::constant_int_id ||
         t->expr_id == expr2t::constant_fixedbv_id ||
         t->expr_id == expr2t::constant_floatbv_id ||
         t->expr_id == expr2t::constant_bool_id ||
         t->expr_id == expr2t::constant_string_id ||
         t->expr_id == expr2t::constant_struct_id ||
         t->expr_id == expr2t::constant_union_id ||
         t->expr_id == expr2t::constant_array_id ||
         t->expr_id == expr2t::constant_array_of_id;
}

inline bool is_structure_type(const type2tc &t)
{
  return t->type_id == type2t::struct_id || t->type_id == type2t::union_id;
}

inline bool is_structure_type(const expr2tc &e)
{
  return is_structure_type(e->type);
}

inline bool is_arith_expr(const expr2tc &expr)
{
  return expr->expr_id == expr2t::neg_id || expr->expr_id == expr2t::abs_id ||
         expr->expr_id == expr2t::add_id || expr->expr_id == expr2t::sub_id ||
         expr->expr_id == expr2t::mul_id || expr->expr_id == expr2t::div_id ||
         expr->expr_id == expr2t::modulus_id;
}

inline bool is_comp_expr(const expr2tc &expr)
{
  return expr->expr_id == expr2t::lessthan_id ||
         expr->expr_id == expr2t::lessthanequal_id ||
         expr->expr_id == expr2t::greaterthan_id ||
         expr->expr_id == expr2t::greaterthanequal_id ||
         expr->expr_id == expr2t::equality_id ||
         expr->expr_id == expr2t::notequal_id;
}

/** Test if expr is true. First checks whether the expr is a constant bool, and
 *  then whether it's true-valued. If these are both true, return true,
 *  otherwise return false.
 *  @param expr Expression to check for true value.
 *  @return Whether or not expr is true-valued.
 */
inline bool is_true(const expr2tc &expr)
{
  if(is_constant_bool2t(expr) && to_constant_bool2t(expr).value)
    return true;

  if(is_constant_int2t(expr) && !to_constant_int2t(expr).value.is_zero())
    return true;

  if(
    is_constant_floatbv2t(expr) && !to_constant_floatbv2t(expr).value.is_zero())
    return true;

  if(
    is_constant_fixedbv2t(expr) && !to_constant_fixedbv2t(expr).value.is_zero())
    return true;

  return false;
}

/** Test if expr is false. First checks whether the expr is a constant bool, and
 *  then whether it's false-valued. If these are both true, return true,
 *  otherwise return false.
 *  @param expr Expression to check for false value.
 *  @return Whether or not expr is true-valued.
 */
inline bool is_false(const expr2tc &expr)
{
  if(is_constant_bool2t(expr) && !to_constant_bool2t(expr).value)
    return true;

  if(is_constant_int2t(expr) && to_constant_int2t(expr).value.is_zero())
    return true;

  if(is_constant_floatbv2t(expr) && to_constant_floatbv2t(expr).value.is_zero())
    return true;

  if(is_constant_fixedbv2t(expr) && to_constant_fixedbv2t(expr).value.is_zero())
    return true;

  return false;
}

inline expr2tc gen_true_expr()
{
  static constant_bool2tc c(true);
  return c;
}

inline expr2tc gen_false_expr()
{
  static constant_bool2tc c(false);
  return c;
}

inline expr2tc gen_ulong(unsigned long val)
{
  constant_int2tc v(get_uint_type(config.ansi_c.word_size), BigInt(val));
  return v;
}

inline const type2tc &get_array_subtype(const type2tc &type)
{
  return to_array_type(type).subtype;
}

inline const type2tc &get_vector_subtype(const type2tc &type)
{
  return to_vector_type(type).subtype;
}

inline const type2tc &get_base_array_subtype(const type2tc &type)
{
  const auto &subtype = to_array_type(type).subtype;
  if(is_array_type(subtype))
    return get_base_array_subtype(subtype);

  return subtype;
}

inline bool simplify(expr2tc &expr)
{
  expr2tc tmp = expr->simplify();
  if(!is_nil_expr(tmp))
  {
    expr = tmp;
    return true;
  }

  return false;
}

inline void make_not(expr2tc &expr)
{
  if(is_true(expr))
  {
    expr = gen_false_expr();
    return;
  }

  if(is_false(expr))
  {
    expr = gen_true_expr();
    return;
  }

  expr2tc new_expr;
  if(is_not2t(expr))
    new_expr = to_not2t(expr).value;
  else
    new_expr = not2tc(expr);

  expr.swap(new_expr);
}

inline expr2tc conjunction(std::vector<expr2tc> cs)
{
  if(cs.empty())
    return gen_true_expr();

  expr2tc res = cs[0];
  for(std::size_t i = 1; i < cs.size(); ++i)
    res = and2tc(res, cs[i]);

  return res;
}

inline expr2tc gen_nondet(const type2tc &type)
{
  return sideeffect2tc(
    type,
    expr2tc(),
    expr2tc(),
    std::vector<expr2tc>(),
    type2tc(),
    sideeffect2t::nondet);
}

inline expr2tc gen_zero(const type2tc &type, bool array_as_array_of = false)
{
  switch(type->type_id)
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

  case type2t::array_id:
  {
    if(array_as_array_of)
      return constant_array_of2tc(type, gen_zero(to_array_type(type).subtype));

    auto arr_type = to_array_type(type);

    assert(is_constant_int2t(arr_type.array_size));
    auto s = to_constant_int2t(arr_type.array_size);

    std::vector<expr2tc> members;
    for(long int i = 0; i < s.as_long(); i++)
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
    for(auto const &member_type : struct_type.members)
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

  default_message msg;
  msg.error(fmt::format("Can't generate zero for type {}", get_type_id(type)));
  abort();
}

inline expr2tc gen_one(const type2tc &type)
{
  switch(type->type_id)
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

  default_message msg;
  msg.error(fmt::format("Can't generate one for type {}", get_type_id(type)));
  abort();
}

#endif /* UTIL_IREP2_UTILS_H_ */
