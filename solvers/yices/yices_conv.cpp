#include <stddef.h>
#include <stdarg.h>
#include <stdint.h>

#include <sstream>

#include "yices_conv.h"

smt_convt *
create_new_yices_solver(bool int_encoding, const namespacet &ns, bool is_cpp,
                              const optionst &opts __attribute__((unused)),
                              tuple_iface **tuple_api __attribute__((unused)),
                              array_iface **array_api)
{
  yices_convt *conv = new yices_convt(int_encoding, ns, is_cpp);
  *array_api = static_cast<array_iface*>(conv);
  return conv;
}

yices_convt::yices_convt(bool int_encoding, const namespacet &ns, bool is_cpp)
  : smt_convt(int_encoding, ns, is_cpp), array_iface(false, false),
    sat_model(NULL)
{
  yices_init();

  yices_clear_error();

  ctx_config_t *config = yices_new_config();
  if (int_encoding)
    yices_default_config_for_logic(config, "QF_AUFLIRA");
  else
    yices_default_config_for_logic(config, "QF_AUFBV");

  // XXX -- work out mode detection in the future.
  // yices_set_config(config, "mode", "push-pop");
  yices_set_config(config, "mode", "one-shot");

  yices_ctx = yices_new_context(config);
  yices_free_config(config);
}

yices_convt::~yices_convt()
{
  yices_free_context(yices_ctx);
  yices_garbage_collect(NULL, 0, NULL, 0, false);
}

smt_convt::resultt
yices_convt::dec_solve()
{
  clear_model();

  smt_status_t result = yices_check_context(yices_ctx, NULL);

  if (result == STATUS_SAT) {
    sat_model = yices_get_model(yices_ctx, 0);
    return smt_convt::P_SATISFIABLE;
  } else if (result == STATUS_UNSAT) {
    sat_model = NULL;
    return smt_convt::P_UNSATISFIABLE;
  } else {
    sat_model = NULL;
    return smt_convt::P_ERROR;
  }
}

tvt
yices_convt::l_get(smt_astt l)
{
  expr2tc b = get_bool(l);
  if (b == true_expr)
    return tvt(true);
  else if (b == false_expr)
    return tvt(false);
  else
    return tvt(tvt::TV_UNKNOWN);
}

const std::string
yices_convt::solver_text()
{
  std::stringstream ss;
  ss << "Yices version " << yices_version;
  return ss.str();
}

void
yices_convt::assert_ast(smt_astt a)
{
  yices_smt_ast *ast = yices_ast_downcast(a);
  yices_assert_formula(yices_ctx, ast->term);
}

smt_astt
yices_convt::mk_func_app(smt_sortt s, smt_func_kind k,
                             smt_astt const *args,
                             unsigned int numargs)
{
  const yices_smt_ast *asts[4];
  unsigned int i;

  assert(numargs <= 4);
  for (i = 0; i < numargs; i++)
    asts[i] = yices_ast_downcast(args[i]);

  switch (k) {
  case SMT_FUNC_EQ:
    if (asts[0]->sort->id == SMT_SORT_BOOL)
      return new_ast(s, yices_eq(asts[0]->term, asts[1]->term));
    else if (int_encoding)
      return new_ast(s, yices_arith_eq_atom(asts[0]->term, asts[1]->term));
    else
      return new_ast(s, yices_bveq_atom(asts[0]->term, asts[1]->term));

  case SMT_FUNC_GT:
    return new_ast(s, yices_arith_gt_atom(asts[0]->term, asts[1]->term));
  case SMT_FUNC_GTE:
    return new_ast(s, yices_arith_geq_atom(asts[0]->term, asts[1]->term));
  case SMT_FUNC_LT:
    return new_ast(s, yices_arith_lt_atom(asts[0]->term, asts[1]->term));
  case SMT_FUNC_LTE:
    return new_ast(s, yices_arith_leq_atom(asts[0]->term, asts[1]->term));

  case SMT_FUNC_BVUGT:
    return new_ast(s, yices_bvgt_atom(asts[0]->term, asts[1]->term));
  case SMT_FUNC_BVUGTE:
    return new_ast(s, yices_bvge_atom(asts[0]->term, asts[1]->term));
  case SMT_FUNC_BVULT:
    return new_ast(s, yices_bvlt_atom(asts[0]->term, asts[1]->term));
  case SMT_FUNC_BVULTE:
    return new_ast(s, yices_bvle_atom(asts[0]->term, asts[1]->term));

  case SMT_FUNC_BVSGT:
    return new_ast(s, yices_bvsgt_atom(asts[0]->term, asts[1]->term));
  case SMT_FUNC_BVSGTE:
    return new_ast(s, yices_bvsge_atom(asts[0]->term, asts[1]->term));
  case SMT_FUNC_BVSLT:
    return new_ast(s, yices_bvslt_atom(asts[0]->term, asts[1]->term));
  case SMT_FUNC_BVSLTE:
    return new_ast(s, yices_bvsle_atom(asts[0]->term, asts[1]->term));

  case SMT_FUNC_AND:
    return new_ast(s, yices_and2(asts[0]->term, asts[1]->term));
  case SMT_FUNC_OR:
    return new_ast(s, yices_or2(asts[0]->term, asts[1]->term));
  case SMT_FUNC_NOT:
    return new_ast(s, yices_not(asts[0]->term));
  case SMT_FUNC_IMPLIES:
    return new_ast(s, yices_implies(asts[0]->term, asts[1]->term));

  case SMT_FUNC_STORE:
    // Crazy "function update" situation.
    return new_ast(s, yices_update(asts[0]->term, 1, &asts[1]->term,
                                   asts[2]->term));
  case SMT_FUNC_SELECT:
    return new_ast(s, yices_application(asts[0]->term, 1, &asts[1]->term));

  case SMT_FUNC_ADD:
    return new_ast(s, yices_add(asts[0]->term, asts[1]->term));
  case SMT_FUNC_SUB:
    return new_ast(s, yices_sub(asts[0]->term, asts[1]->term));
  case SMT_FUNC_MUL:
    return new_ast(s, yices_mul(asts[0]->term, asts[1]->term));
  case SMT_FUNC_DIV:
    return new_ast(s, yices_division(asts[0]->term, asts[1]->term));
  case SMT_FUNC_MOD:

  case SMT_FUNC_BVADD:
    return new_ast(s, yices_bvadd(asts[0]->term, asts[1]->term));
  case SMT_FUNC_BVSUB:
    return new_ast(s, yices_bvsub(asts[0]->term, asts[1]->term));
  case SMT_FUNC_BVMUL:
    return new_ast(s, yices_bvmul(asts[0]->term, asts[1]->term));
  case SMT_FUNC_CONCAT:
    return new_ast(s, yices_bvconcat(asts[0]->term, asts[1]->term));
  default:
    std::cerr << "Unimplemented SMT function '" << smt_func_name_table[k]
              << "' in yices_convt::mk_func_app" << std::endl;
    abort();
  }
}

smt_sortt
yices_convt::mk_sort(const smt_sort_kind k, ...)
{
  va_list ap;
  unsigned long uint;

  va_start(ap, k);
  switch(k) {
  case SMT_SORT_BOOL:
  {
    return new yices_smt_sort(k, yices_bool_type());
  }
  case SMT_SORT_INT:
  {
    return new yices_smt_sort(k, yices_int_type());
  }
  case SMT_SORT_REAL:
  {
    return new yices_smt_sort(k, yices_real_type());
  }
  case SMT_SORT_ARRAY:
  {
    // Arrays are uninterpreted functions with updates. Create an array with
    // the given domain as a single dimension.
    yices_smt_sort *dom = va_arg(ap, yices_smt_sort*);
    yices_smt_sort *range = va_arg(ap, yices_smt_sort*);
    type_t t = yices_function_type(1, &dom->type, range->type);
    return new yices_smt_sort(k, t, range->data_width, dom->data_width);
  }
  case SMT_SORT_BV:
  {
    uint = va_arg(ap, unsigned long);
    return new yices_smt_sort(k, yices_bv_type(uint), uint);
  }
  default:
    std::cerr << "Unimplemented sort " << k << " in yices mk_sort" << std::endl;
    abort();
  }
}

smt_astt
yices_convt::mk_smt_int(const mp_integer &theint, bool sign)
{
  term_t term = yices_int64(theint.to_int64());
  smt_sortt s = mk_sort(SMT_SORT_INT);
  return new_ast(s, term);
}

smt_astt
yices_convt::mk_smt_real(const std::string &str)
{
  abort();
}

smt_astt
yices_convt::mk_smt_bvint(const mp_integer &theint,
      bool sign __attribute__((unused)),
      unsigned int w)
{
  assert(w != 0);
  term_t term = yices_bvconst_uint64(w, theint.to_uint64());
  smt_sortt s = mk_sort(SMT_SORT_BV, w, false);
  return new yices_smt_ast(this, s, term);
}

smt_astt
yices_convt::mk_smt_bool(bool val)
{
  smt_sortt s = mk_sort(SMT_SORT_BOOL);
  if (val)
    return new_ast(s, yices_true());
  else
    return new_ast(s, yices_false());
}

smt_astt
yices_convt::mk_smt_symbol(const std::string &name, smt_sortt s)
{
  // Is this term already in the symbol table?
  term_t term = yices_get_term_by_name(name.c_str());
  if (term == NULL_TERM) {
    // No: create a new one.
    yices_smt_sort *sort = yices_sort_downcast(s);
    term = yices_new_uninterpreted_term(sort->type);
    yices_set_term_name(term, name.c_str());
  }

  return new yices_smt_ast(this, s, term);
}

smt_astt
yices_convt::mk_array_symbol(const std::string &name, smt_sortt s)
{
  return mk_smt_symbol(name, s);
}

smt_sortt
yices_convt::mk_struct_sort(const type2tc &type)
{
  abort();
}

smt_sortt
yices_convt::mk_union_sort(const type2tc &type)
{
  abort();
}

smt_astt
yices_convt::mk_extract(smt_astt a, unsigned int high,
                            unsigned int low, smt_sortt s)
{
  yices_smt_ast *ast = yices_ast_downcast(a);
  term_t term = yices_bvextract(ast->term, low, high);
  return new_ast(s, term);
}

smt_astt
yices_convt::convert_array_of(const expr2tc &init_val,
                                unsigned long domain_width)
{
  abort();
}

void
yices_convt::add_array_constraints_for_solving()
{
  abort();
}

expr2tc
yices_convt::get_bool(smt_astt a)
{
  int32_t val;
  yices_smt_ast *ast = yices_ast_downcast(a);
  if (yices_get_bool_value(sat_model, ast->term, &val) != 0)
    return expr2tc();

  if (val)
    return true_expr;
  else
    return false_expr;
}

expr2tc
yices_convt::get_bv(const type2tc &t, smt_astt a)
{
  int32_t data[64];
  yices_smt_ast *ast = yices_ast_downcast(a);

  unsigned int width = t->get_width();
  assert(width <= 64);
  if (yices_get_bv_value(sat_model, ast->term, data) != 0)
    return expr2tc();

  uint64_t val = 0;
  unsigned int i;
  for (i = 0; i < width; i++) {
    val <<= 1;
    val |= data[i];
  }

  return constant_int2tc(t, BigInt(val));
}

expr2tc
yices_convt::get_array_elem(smt_astt array, uint64_t index,
                       const type2tc &subtype)
{
  abort();
}
