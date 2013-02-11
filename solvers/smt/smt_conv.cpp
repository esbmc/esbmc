#include "smt_conv.h"

smt_convt::smt_convt(bool enable_cache, bool intmode)
  : caching(enable_cache), int_encoding(intmode)
{
}

smt_convt::~smt_convt(void)
{
}

void
smt_convt::push_ctx(void)
{
  prop_convt::push_ctx();
}

void
smt_convt::pop_ctx(void)
{
  prop_convt::pop_ctx();

  union_varst::nth_index<1>::type &union_numindex = union_vars.get<1>();
  union_numindex.erase(ctx_level);
  smt_cachet::nth_index<1>::type &cache_numindex = smt_cache.get<1>();
  cache_numindex.erase(ctx_level);
}

void
smt_convt::set_to(const expr2tc &expr, bool value)
{

  l_set_to(convert(expr), value);

  // Workaround for the fact that we don't have a good way of encoding unions
  // into SMT. Just work out what the last assigned field is.
  if (is_equality2t(expr) && value) {
    const equality2t eq = to_equality2t(expr);
    if (is_union_type(eq.side_1->type) && is_with2t(eq.side_2)) {
      const symbol2t sym = to_symbol2t(eq.side_1);
      const with2t with = to_with2t(eq.side_2);
      const union_type2t &type = to_union_type(eq.side_1->type);
      const std::string &ref = sym.get_symbol_name();
      const constant_string2t &str = to_constant_string2t(with.update_field);

      unsigned int idx = 0;
      forall_names(it, type.member_names) {
        if (*it == str.value)
          break;
        idx++;
      }

      assert(idx != type.member_names.size() &&
             "Member name of with expr not found in struct/union type");

      union_var_mapt mapentry = { ref, idx, 0 };
      union_vars.insert(mapentry);
    }
  }
}

literalt
smt_convt::convert_expr(const expr2tc &expr)
{
  smt_ast *args[3];

  if (caching) {
    smt_cachet::const_iterator cache_result = smt_cache.find(expr);
    if (cache_result != smt_cache.end())
      return mk_lit(cache_result->ast);
  }

  smt_ast *a;
  switch (expr->expr_id) {
  case expr2t::constant_int_id:
  case expr2t::constant_fixedbv_id:
  case expr2t::constant_bool_id:
  case expr2t::symbol_id:
    a = convert_terminal(expr);
  default:
    a = mk_func_app(NULL, SMT_FUNC_HACKS, &args[0], 0, expr);
  }

  struct smt_cache_entryt entry = { expr, a, ctx_level };
  smt_cache.insert(entry);

  literalt l = mk_lit(a);
  return l;
}

smt_sort *
smt_convt::convert_sort(const type2tc &type)
{
  switch (type->type_id) {
  case type2t::bool_id:
    return mk_sort(SMT_SORT_BOOL);
  case type2t::struct_id:
    return mk_struct_sort(type);
  case type2t::union_id:
    return mk_union_sort(type);
  case type2t::pointer_id:
    assert(0);
  case type2t::unsignedbv_id:
  case type2t::signedbv_id:
  {
    unsigned int width = type->get_width();
    if (int_encoding)
      return mk_sort(SMT_SORT_INT);
    else
      return mk_sort(SMT_SORT_BV, width);
  }
  case type2t::fixedbv_id:
  {
    unsigned int width = type->get_width();
    if (int_encoding)
      return mk_sort(SMT_SORT_REAL);
    else
      return mk_sort(SMT_SORT_BV, width);
  }
  case type2t::code_id:
  case type2t::string_id:
  case type2t::cpp_name_id:
  case type2t::symbol_id:
  case type2t::empty_id:
  default:
    assert(0 && "Unexpected type ID reached SMT conversion");
  }
}

smt_ast *
smt_convt::convert_terminal(const expr2tc &expr)
{
  switch (expr->expr_id) {
  case expr2t::constant_int_id:
  {
    const constant_int2t &theint = to_constant_int2t(expr);
    unsigned int width = expr->type->get_width();
    if (int_encoding)
      return mk_smt_int(theint.constant_value, expr);
    else
      return mk_smt_bvint(theint.constant_value, width, expr);
  }
  case expr2t::constant_fixedbv_id:
  {
    const constant_fixedbv2t &thereal = to_constant_fixedbv2t(expr);
    return mk_smt_real(thereal.value.to_integer(), expr);
  }
  case expr2t::constant_bool_id:
  {
    const constant_bool2t &thebool = to_constant_bool2t(expr);
    return mk_smt_bool(thebool.constant_value, expr);
  }
  case expr2t::symbol_id:
  {
    // Can't do this right now due to not having sort conversion yet.
#if 0
    const symbol2t &sym = to_symbol2t(expr);
    std::string name = sym.get_symbol_name();
    return mk_smt_symbol(name, expr);
#endif
  }
  default:
    return mk_func_app(NULL, SMT_FUNC_HACKS, NULL, 0, expr);
  }
}
