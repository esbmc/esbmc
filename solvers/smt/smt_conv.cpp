#include "smt_conv.h"

smt_convt::smt_convt(void)
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

  // Funky recursion stuff goes here. In the meantime, dummy values.
  smt_ast *a = mk_func_app(NULL, SMT_FUNC_INT, &args[0], 0, expr);
  literalt l = mk_lit(a);
  return l;
}
