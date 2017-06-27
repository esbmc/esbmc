#ifndef UTIL_IREP2_UTILS_H_
#define UTIL_IREP2_UTILS_H_

#include <util/irep2_expr.h>
#include <util/c_types.h>

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
         t->type_id == type2t::fixedbv_id ||
         t->type_id == type2t::floatbv_id ||
         t->type_id == type2t::bool_id;
}

inline bool is_number_type(const expr2tc &e)
{
  return is_number_type(e->type);
}

inline bool is_scalar_type(const type2tc &t)
{
  return is_number_type(t) ||
         is_pointer_type(t) ||
         is_empty_type(t) ||
         is_code_type(t);
}

inline bool is_scalar_type(const expr2tc &e)
{
  return is_scalar_type(e->type);
}

inline bool is_multi_dimensional_array(const type2tc &t)
{
  if (is_array_type(t))
    return is_array_type(to_array_type(t).subtype);

  return false;
}

inline bool is_multi_dimensional_array(const expr2tc &e)
{
  return is_multi_dimensional_array(e->type);
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

inline bool is_arith_type(const expr2tc &t)
{
  return t->expr_id == expr2t::neg_id ||
         t->expr_id == expr2t::abs_id ||
         t->expr_id == expr2t::add_id ||
         t->expr_id == expr2t::sub_id ||
         t->expr_id == expr2t::mul_id ||
         t->expr_id == expr2t::modulus_id ||
         t->expr_id == expr2t::div_id;
}

/** Test if expr is true. First checks whether the expr is a constant bool, and
 *  then whether it's true-valued. If these are both true, return true,
 *  otherwise return false.
 *  @param expr Expression to check for true value.
 *  @return Whether or not expr is true-valued.
 */
inline bool
is_true(const expr2tc &expr)
{
  if (is_constant_bool2t(expr) && to_constant_bool2t(expr).value)
    return true;

  return false;
}

/** Test if expr is false. First checks whether the expr is a constant bool, and
 *  then whether it's false-valued. If these are both true, return true,
 *  otherwise return false.
 *  @param expr Expression to check for false value.
 *  @return Whether or not expr is true-valued.
 */
inline bool
is_false(const expr2tc &expr)
{
  if (is_constant_bool2t(expr) && !to_constant_bool2t(expr).value)
    return true;

  return false;
}

inline expr2tc
gen_true_expr()
{
  return constant_bool2tc(true);
}

inline expr2tc
gen_false_expr()
{
  return constant_bool2tc(false);
}

inline expr2tc
gen_ulong(unsigned long val)
{
  constant_int2tc v(type_pool.get_uint(config.ansi_c.word_size), BigInt(val));
  return v;
}

inline const type2tc &
get_uint8_type()
{
  return type_pool.get_uint8();
}

inline const type2tc &
get_uint16_type()
{
  return type_pool.get_uint16();
}

inline const type2tc &
get_uint32_type()
{
  return type_pool.get_uint32();
}

inline const type2tc &
get_uint64_type()
{
  return type_pool.get_uint64();
}

inline const type2tc &
get_int8_type()
{
  return type_pool.get_int8();
}

inline const type2tc &
get_int16_type()
{
  return type_pool.get_int16();
}

inline const type2tc &
get_int32_type()
{
  return type_pool.get_int32();
}

inline const type2tc &
get_int64_type()
{
  return type_pool.get_int64();
}

inline const type2tc &
get_uint_type(unsigned int sz)
{
  return type_pool.get_uint(sz);
}

inline const type2tc &
get_int_type(unsigned int sz)
{
  return type_pool.get_int(sz);
}

inline const type2tc &
get_bool_type()
{
  return type_pool.get_bool();
}

inline const type2tc &
get_empty_type()
{
  return type_pool.get_empty();
}

inline const type2tc &
get_pointer_type(const typet &val)
{
  return type_pool.get_pointer(val);
}

inline const type2tc &
get_array_subtype(const type2tc &type)
{
  return to_array_type(type).subtype;
}

inline const type2tc &
get_base_array_subtype(const type2tc &type)
{
  const auto &subtype = to_array_type(type).subtype;
  if (is_array_type(subtype))
    return get_base_array_subtype(subtype);

  return subtype;
}

inline bool simplify(expr2tc &expr)
{
  expr2tc tmp = expr->simplify();
  if (!is_nil_expr(tmp))
  {
    expr = tmp;
    return true;
  }

  return false;
}

inline void make_not(expr2tc &expr)
{
  if (is_constant_bool2t(expr))
  {
    constant_bool2t &b = to_constant_bool2t(expr);
    b.value = !b.value;
    return;
  }

  expr2tc new_expr;
  if (is_not2t(expr))
    new_expr = to_not2t(expr).value;
  else
    new_expr = not2tc(expr);

  expr.swap(new_expr);
}

inline expr2tc gen_zero(
  const type2tc &type,
  bool array_as_array_of = false)
{
  switch(type->type_id)
  {
    case type2t::bool_id:
      return gen_false_expr();

    case type2t::unsignedbv_id:
    case type2t::signedbv_id:
      return constant_int2tc(type, BigInt(0));

    case type2t::fixedbv_id:
      return constant_fixedbv2tc(
        fixedbvt(fixedbv_spect(to_fixedbv_type(type))));

    case type2t::floatbv_id:
      return constant_floatbv2tc(
        ieee_floatt(ieee_float_spect(to_floatbv_type(type))));

    case type2t::array_id:
    {
      if(array_as_array_of)
        return constant_array_of2tc(
          type, gen_zero(to_array_type(type).subtype));

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
      return symbol2tc(pointer_type2(), "NULL");

    case type2t::struct_id:
    {
      auto struct_type = to_struct_type(type);

      std::vector<expr2tc> members;
      for(auto const& member_type : struct_type.members)
        members.push_back(gen_zero(member_type, array_as_array_of));

      return constant_struct2tc(type, members);
    }

    case type2t::union_id:
    {
      auto union_type = to_union_type(type);

      std::vector<expr2tc> members;
      for(auto const& member_type : union_type.members)
        members.push_back(gen_zero(member_type, array_as_array_of));

      return constant_union2tc(type, members);
    }

    default:
      break;
  }

  abort();
}

inline expr2tc gen_one(
  const type2tc &type)
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

  abort();
}

#endif /* UTIL_IREP2_UTILS_H_ */
