#include <sstream>
#include <set>
#include <iomanip>

#include <base_type.h>
#include <arith_tools.h>

#include "smt_conv.h"
#include <solvers/prop/literal.h>

// Helpers extracted from z3_convt.

static std::string
extract_magnitude(std::string v, unsigned width)
{
    return integer2string(binary2integer(v.substr(0, width / 2), true), 10);
}

static std::string
extract_fraction(std::string v, unsigned width)
{
    return integer2string(binary2integer(v.substr(width / 2, width), false), 10);
}

static std::string
double2string(double d)
{
  std::ostringstream format_message;
  format_message << std::setprecision(12) << d;
  return format_message.str();
}

static std::string
itos(int64_t i)
{
  std::stringstream ss;
  ss << i;
  return ss.str();
}

static unsigned int
get_member_name_field(const type2tc &t, const irep_idt &name)
{
  unsigned int idx = 0;
  const struct_union_data &data_ref =
          dynamic_cast<const struct_union_data &>(*t);

  forall_names(it, data_ref.member_names) {
    if (*it == name)
      break;
    idx++;
  }
  assert(idx != data_ref.member_names.size() &&
         "Member name of with expr not found in struct/union type");

  return idx;
}

static unsigned int
get_member_name_field(const type2tc &t, const expr2tc &name)
{
  const constant_string2t &str = to_constant_string2t(name);
  return get_member_name_field(t, str.value);
}

smt_convt::smt_convt(bool enable_cache, bool intmode, const namespacet &_ns,
                     bool is_cpp, bool _tuple_support, bool _nobools)
  : caching(enable_cache), int_encoding(intmode), ns(_ns),
    tuple_support(_tuple_support), no_bools_in_arrays(_nobools)
{
  std::vector<type2tc> members;
  std::vector<irep_idt> names;

  no_variables = 1;

  members.push_back(type_pool.get_uint(config.ansi_c.pointer_width));
  members.push_back(type_pool.get_uint(config.ansi_c.pointer_width));
  names.push_back(irep_idt("pointer_object"));
  names.push_back(irep_idt("pointer_offset"));

  struct_type2t *tmp = new struct_type2t(members, names, "pointer_struct");
  pointer_type_data = tmp;
  pointer_struct = type2tc(tmp);

  pointer_logic.push_back(pointer_logict());

  addr_space_sym_num.push_back(0);

  members.clear();
  names.clear();
  members.push_back(type_pool.get_uint(config.ansi_c.pointer_width));
  members.push_back(type_pool.get_uint(config.ansi_c.pointer_width));
  names.push_back(irep_idt("start"));
  names.push_back(irep_idt("end"));
  tmp = new struct_type2t(members, names, "addr_space_type");
  addr_space_type_data = tmp;
  addr_space_type = type2tc(tmp);

  addr_space_arr_type = type2tc(new array_type2t(addr_space_type,
                                                 expr2tc(), true)) ;

  addr_space_data.push_back(std::map<unsigned, unsigned>());

  machine_int = type2tc(new signedbv_type2t(config.ansi_c.int_width));
  machine_uint = type2tc(new unsignedbv_type2t(config.ansi_c.int_width));
  machine_ptr = type2tc(new unsignedbv_type2t(config.ansi_c.pointer_width));

  // Pick a modelling array to shoehorn initialization data into. Because
  // we don't yet have complete data for whether pointers are dynamic or not,
  // this is the one modelling array that absolutely _has_ to be initialized
  // to false for each element, which is going to be shoved into
  // convert_identifier_pointer.
  if (is_cpp) {
    dyn_info_arr_name = "cpp::__ESBMC_is_dynamic&0#1";
  } else {
    dyn_info_arr_name = "c::__ESBMC_is_dynamic&0#1";
  }
}

smt_convt::~smt_convt(void)
{
}

void
smt_convt::smt_post_init(void)
{
  if (int_encoding) {
    machine_int_sort = mk_sort(SMT_SORT_INT, false);
    machine_uint_sort = machine_int_sort;
  } else {
    machine_int_sort = mk_sort(SMT_SORT_BV, config.ansi_c.int_width, true);
    machine_uint_sort = mk_sort(SMT_SORT_BV, config.ansi_c.int_width, false);
  }

  init_addr_space_array();
}

void
smt_convt::push_ctx(void)
{
  addr_space_data.push_back(addr_space_data.back());
  addr_space_sym_num.push_back(addr_space_sym_num.back());
  pointer_logic.push_back(pointer_logic.back());
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
  pointer_logic.pop_back();
  addr_space_sym_num.pop_back();
  addr_space_data.pop_back();
}

literalt
smt_convt::new_variable()
{
  literalt l;

  l.set(no_variables, false);

  no_variables = no_variables + 1;

  return l;
}

bool
smt_convt::process_clause(const bvt &bv, bvt &dest)
{

  dest.clear();

  // empty clause! this is UNSAT
  if (bv.empty()) return false;

  std::set<literalt> s;

  dest.reserve(bv.size());

  for (bvt::const_iterator it = bv.begin();
       it != bv.end();
       it++)
  {
    literalt l = *it;

    // we never use index 0
    assert(l.var_no() != 0);

    if (l.is_true())
      return true;  // clause satisfied

    if (l.is_false())
      continue;

    assert(l.var_no() < no_variables);

    // prevent duplicate literals
    if (s.insert(l).second)
      dest.push_back(l);

    if (s.find(lnot(l)) != s.end())
      return true;  // clause satisfied
  }

  return false;
}

void
smt_convt::lcnf(const bvt &bv)
{

  bvt new_bv;

  if (process_clause(bv, new_bv))
    return;

  if (new_bv.size() == 0)
    return;

  literalt l = lor(new_bv);
  assert_lit(l);
}

literalt
smt_convt::lor(const bvt &bv)
{
  const smt_ast *args[bv.size()];
  unsigned int i = 0;

  for (bvt::const_iterator it = bv.begin(); it != bv.end(); it++, i++) {
    args[i] = lit_to_ast(*it);
  }

  // Chain these.
  if (i > 1) {
    unsigned int j;
    const smt_ast *argstwo[2];
    const smt_sort *sort = mk_sort(SMT_SORT_BOOL);
    argstwo[0] = args[0];
    for (j = 1; j < i; j++) {
      argstwo[1] = args[j];
      argstwo[0] = mk_func_app(sort, SMT_FUNC_OR, argstwo, 2);
    }
    literalt tmp = mk_lit(argstwo[0]);
    return tmp;
  } else {
    literalt tmp = mk_lit(args[0]);
    return tmp;
  }
}

literalt
smt_convt::lor(literalt a, literalt b)
{
  if (a == const_literal(false)) return b;
  if (b == const_literal(false)) return a;
  if (a == const_literal(true)) return const_literal(true);
  if (b == const_literal(true)) return const_literal(true);
  if (a == b) return a;

  const smt_ast *args[2];
  args[0] = lit_to_ast(a);
  args[1] = lit_to_ast(b);
  const smt_sort *sort = mk_sort(SMT_SORT_BOOL);
  const smt_ast *c = mk_func_app(sort, SMT_FUNC_OR, args, 2);
  return mk_lit(c);
}

literalt
smt_convt::land(const bvt &bv)
{
  literalt l = new_variable();
  uint size = bv.size();
  const smt_ast *args[size];
  unsigned int i;

  for (i = 0; i < bv.size(); i++) {
    args[i] = lit_to_ast(bv[i]);
  }

  // Chain these.
  if (i > 1) {
    unsigned int j;
    const smt_ast *argstwo[2];
    const smt_sort *sort = mk_sort(SMT_SORT_BOOL);
    argstwo[0] = args[0];
    for (j = 1; j < i; j++) {
      argstwo[1] = args[j];
      argstwo[0] = mk_func_app(sort, SMT_FUNC_AND, argstwo, 2);
    }
    l = mk_lit(argstwo[0]);
  } else {
    l = mk_lit(args[0]);
  }

  return l;
}

literalt
smt_convt::land(literalt a, literalt b)
{
  if (a == const_literal(true)) return b;
  if (b == const_literal(true)) return a;
  if (a == const_literal(false)) return const_literal(false);
  if (b == const_literal(false)) return const_literal(false);
  if (a == b) return a;

  const smt_ast *args[2];
  args[0] = lit_to_ast(a);
  args[1] = lit_to_ast(b);
  const smt_sort *sort = mk_sort(SMT_SORT_BOOL);
  const smt_ast *c = mk_func_app(sort, SMT_FUNC_AND, args, 2);
  return mk_lit(c);
}

literalt
smt_convt::lnot(literalt a)
{
  a.invert();
  return a;
}

literalt
smt_convt::limplies(literalt a, literalt b)
{
  const smt_ast *args[2];
  args[0] = lit_to_ast(a);
  args[1] = lit_to_ast(b);
  const smt_sort *sort = mk_sort(SMT_SORT_BOOL);
  const smt_ast *c = mk_func_app(sort, SMT_FUNC_IMPLIES, args, 2);
  return mk_lit(c);
}

const smt_ast *
smt_convt::lit_to_ast(const literalt &l)
{
  if (l.var_no() == literalt::const_var_no()) {
    // Then don't turn this into a literal, turn it into a bool.
    if (l.sign()) {
      return mk_smt_bool(true);
    } else {
      return mk_smt_bool(false);
    }
  }

  std::stringstream ss;
  ss << "l" << l.var_no();
  std::string name = ss.str();
  symbol2tc sym(get_bool_type(), name);
  if (l.sign()) {
    not2tc anot(sym);
    return convert_ast(anot);
  } else {
    return convert_ast(sym);
  }
}

uint64_t
smt_convt::get_no_variables() const
{
  return no_variables;
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
  const smt_ast *a = convert_ast(expr);
  return mk_lit(a);
}

const smt_ast *
smt_convt::convert_ast(const expr2tc &expr)
{
  // Variable length array; constant array's and so forth can have hundreds
  // of fields.
  const smt_ast *args[expr->get_num_sub_exprs()];
  const smt_sort *sort;
  const smt_ast *a;
  unsigned int num_args, used_sorts = 0;
  bool seen_signed_operand = false;
  bool make_ints_reals = false;

  if (caching) {
    smt_cachet::const_iterator cache_result = smt_cache.find(expr);
    if (cache_result != smt_cache.end())
      return (cache_result->ast);
  }

  // Second fail -- comparisons are turning up in 01_cbmc_abs1 that compare
  // ints and reals, which is invalid. So convert ints up to reals.
  if (int_encoding && expr->get_num_sub_exprs() >= 2 &&
                      (is_fixedbv_type((*expr->get_sub_expr(0))->type) ||
                       is_fixedbv_type((*expr->get_sub_expr(1))->type))) {
    make_ints_reals = true;
  }

  // Convert /all the arguments/.
  unsigned int i = 0;
  forall_operands2(it, idx, expr) {
    args[i] = convert_ast(*it);

    if (make_ints_reals && args[i]->sort->id == SMT_SORT_INT) {
      args[i] = mk_func_app(mk_sort(SMT_SORT_REAL), SMT_FUNC_INT2REAL,
                            &args[i], 1);
    }

    used_sorts |= args[i]->sort->id;
    i++;
    if (is_signedbv_type(*it) || is_fixedbv_type(*it))
      seen_signed_operand = true;
  }
  num_args = i;

  sort = convert_sort(expr->type);

  const expr_op_convert *cvt = &smt_convert_table[expr->expr_id];

  // Irritating special case: if we're selecting a bool out of an array, and
  // we're in QF_AUFBV mode, do special handling.
  if (!int_encoding && is_index2t(expr) && is_bool_type(expr->type) &&
      no_bools_in_arrays)
    goto expr_handle_table;

  if ((int_encoding && cvt->int_mode_func > SMT_FUNC_INVALID) ||
      (!int_encoding && cvt->bv_mode_func_signed > SMT_FUNC_INVALID)) {
    assert(cvt->args == num_args);
    // An obvious check, but catches cases where we add a field to a future expr
    // and then fail to update the SMT layer, leading to an ignored field.

    // Now check sort types.
    if ((used_sorts | cvt->permitted_sorts) == cvt->permitted_sorts) {
      // Matches; we can just convert this.
      smt_func_kind k = (int_encoding) ? cvt->int_mode_func
                      : (seen_signed_operand)
                          ? cvt->bv_mode_func_signed
                          : cvt->bv_mode_func_unsigned;
      a = mk_func_app(sort, k, &args[0], cvt->args);
      goto done;
    }
  }

  if ((int_encoding && cvt->int_mode_func == SMT_FUNC_INVALID) ||
      (!int_encoding && cvt->bv_mode_func_signed == SMT_FUNC_INVALID)) {
    std::cerr << "Invalid expression " << get_expr_id(expr) << " for encoding "
      << "mode discovered, refusing to convert to SMT" << std::endl;
    abort();
  }

expr_handle_table:
  switch (expr->expr_id) {
  case expr2t::constant_int_id:
  case expr2t::constant_fixedbv_id:
  case expr2t::constant_bool_id:
  case expr2t::symbol_id:
    a = convert_terminal(expr);
    break;
  case expr2t::constant_string_id:
  {
    const constant_string2t &str = to_constant_string2t(expr);
    expr2tc newarr = str.to_array();
    a = convert_ast(newarr);
    break;
  }
  case expr2t::constant_struct_id:
  {
    a = tuple_create(expr);
    break;
  }
  case expr2t::constant_array_id:
  case expr2t::constant_array_of_id:
  {
    const array_type2t &arr = to_array_type(expr->type);
    if (arr.size_is_infinite) {
      // Don't honour inifinite sized array initializers. Modelling only.
      a = mk_fresh(sort, "inf_array");
      break;
    }

    // Domain sort may be mesed with:
    const smt_sort *domain;
    if (int_encoding) {
      domain = machine_int_sort;
    } else {
      domain = mk_sort(SMT_SORT_BV, calculate_array_domain_width(arr), false);
    }

    if (is_struct_type(arr.subtype) || is_union_type(arr.subtype) ||
        is_pointer_type(arr.subtype))
      a = tuple_array_create_despatch(expr, domain);
    else
      a = array_create(expr);
    break;
  }
  case expr2t::add_id:
  case expr2t::sub_id:
  {
    a = convert_pointer_arith(expr, expr->type);
    break;
  }
  case expr2t::mul_id:
  {
    assert(!int_encoding);

    // Handle BV mode multiplies: for normal integers multiply normally, for
    // fixedbv apply hacks.
    if (is_fixedbv_type(expr)) {
      const mul2t &mul = to_mul2t(expr);
      const fixedbv_type2t &fbvt = to_fixedbv_type(mul.type);
      unsigned int fraction_bits = fbvt.width - fbvt.integer_bits;
      unsigned int topbit = mul.side_1->type->get_width();
      const smt_sort *s1 = convert_sort(mul.side_1->type);
      const smt_sort *s2 = convert_sort(mul.side_2->type);
      args[0] = convert_sign_ext(args[0], s1, topbit, fraction_bits);
      args[1] = convert_sign_ext(args[1], s2, topbit, fraction_bits);
      a = mk_func_app(sort, SMT_FUNC_BVMUL, args, 2);
      a = mk_extract(a, fbvt.width + fraction_bits - 1, fraction_bits, sort);
    } else {
      assert(is_bv_type(expr));
      a = mk_func_app(sort, SMT_FUNC_BVMUL, args, 2);
    }
    break;
  }
  case expr2t::div_id:
  {
    // Handle BV mode divisions. Similar arrangement to multiplies.
    if (int_encoding) {
      a = mk_func_app(sort, SMT_FUNC_DIV, args, 2);
    } else if (is_fixedbv_type(expr)) {
      const div2t &div = to_div2t(expr);
      fixedbvt fbt(migrate_expr_back(expr));

      unsigned int fraction_bits = fbt.spec.get_fraction_bits();
      unsigned int topbit2 = div.side_2->type->get_width();
      const smt_sort *s2 = convert_sort(div.side_2->type);

      args[1] = convert_sign_ext(args[1], s2, topbit2,fraction_bits);
      const smt_ast *zero = mk_smt_bvint(BigInt(0), false, fraction_bits);
      const smt_ast *op0 = args[0];
      const smt_ast *concatargs[2];
      concatargs[0] = op0;
      concatargs[1] = zero;
      args[0] = mk_func_app(s2, SMT_FUNC_CONCAT, concatargs, 2);

      // Sorts.
      a = mk_func_app(s2, SMT_FUNC_BVSDIV, args, 2);
      a = mk_extract(a, fbt.spec.width - 1, 0, s2);
    } else {
      assert(is_bv_type(expr));
      smt_func_kind k = (seen_signed_operand)
              ? cvt->bv_mode_func_signed
              : cvt->bv_mode_func_unsigned;
      a = mk_func_app(sort, k, args, 2);
    }
    break;
  }
  case expr2t::index_id:
  {
    const index2t &index = to_index2t(expr);

    if (is_index2t(index.source_value)) {
      args[1] = handle_select_chain(expr, &args[0]);
    } else {
      args[1] = fix_array_idx(args[1], args[0]->sort);
    }

    // Firstly, if it's a string, shortcircuit.
    if (is_string_type(index.source_value)) {
      a = mk_func_app(sort, SMT_FUNC_SELECT, args, 2);
      break;
    }

    const array_type2t &arrtype = to_array_type(index.source_value->type);
    if (!int_encoding && is_bool_type(arrtype.subtype) && no_bools_in_arrays) {
      // Perform a fix for QF_AUFBV, only arrays of bv's are allowed.
      const smt_sort *tmpsort = mk_sort(SMT_SORT_BV, 1, false);
      a = mk_func_app(tmpsort, SMT_FUNC_SELECT, args, 2);
      a = make_bit_bool(a);
    } else if (is_tuple_array_ast_type(index.source_value->type)) {
      a = tuple_array_select(args[0], sort, args[1]);
    } else {
      a = mk_func_app(sort, SMT_FUNC_SELECT, args, 2);
    }
    break;
  }
  case expr2t::with_id:
  {
    const with2t &with = to_with2t(expr);

    // We reach here if we're with'ing a struct, not an array. Or a bool.
    if (is_struct_type(expr->type) || is_union_type(expr)) {
      unsigned int idx = get_member_name_field(expr->type, with.update_field);
      a = tuple_update(args[0], idx, args[2]);
    } else {
      if (is_with2t(with.source_value)) {
        a = handle_store_chain(expr);
        break;
      }

      args[1] = fix_array_idx(args[1], args[0]->sort);

      assert(is_array_type(expr->type));
      const array_type2t &arrtype = to_array_type(expr->type);
      if (!int_encoding && is_bool_type(arrtype.subtype) && no_bools_in_arrays){
        args[2] = make_bool_bit(args[2]);
        a = mk_func_app(sort, SMT_FUNC_STORE, args, 3);
        break;
      } else if (is_tuple_array_ast_type(with.type)) {
        assert(is_structure_type(arrtype.subtype) ||
               is_pointer_type(arrtype.subtype));
        const smt_sort *sort = convert_sort(with.update_value->type);
        a = tuple_array_update(args[0], args[1], args[2], sort);
      } else {
        // Normal operation
        a = mk_func_app(sort, SMT_FUNC_STORE, args, 3);
      }
    }
    break;
  }
  case expr2t::member_id:
  {
    a = convert_member(expr, args[0]);
    break;
  }
  case expr2t::same_object_id:
  {
    // Two projects, then comparison.
    smt_sort *s = convert_sort(pointer_type_data->members[0]);
    args[0] = tuple_project(args[0], s, 0);
    args[1] = tuple_project(args[1], s, 0);
    a = mk_func_app(sort, SMT_FUNC_EQ, &args[0], 2);
    break;
  }
  case expr2t::pointer_offset_id:
  {
    smt_sort *s = convert_sort(pointer_type_data->members[1]);
    const pointer_offset2t &obj = to_pointer_offset2t(expr);
    // Can you cay super irritating?
    const expr2tc *ptr = &obj.ptr_obj;
    while (is_typecast2t(*ptr) && !is_pointer_type((*ptr)))
      ptr = &to_typecast2t(*ptr).from;

    args[0] = convert_ast(*ptr);
    a = tuple_project(args[0], s, 1);
    break;
  }
  case expr2t::pointer_object_id:
  {
    smt_sort *s = convert_sort(pointer_type_data->members[0]);
    const pointer_object2t &obj = to_pointer_object2t(expr);
    // Can you cay super irritating?
    const expr2tc *ptr = &obj.ptr_obj;
    while (is_typecast2t(*ptr) && !is_pointer_type((*ptr)))
      ptr = &to_typecast2t(*ptr).from;

    args[0] = convert_ast(*ptr);
    a = tuple_project(args[0], s, 0);
    break;
  }
  case expr2t::typecast_id:
  {
    a = convert_typecast(expr);
    break;
  }
  case expr2t::if_id:
  {
    // Only attempt to handle struct.s
    if (is_struct_type(expr) || is_pointer_type(expr)) {
      a = tuple_ite(args[0], args[1], args[2], sort);
    } else {
      assert(is_array_type(expr));
      a = tuple_array_ite(args[0], args[1], args[2], sort);
    }
    break;
  }
  case expr2t::isnan_id:
  {
    a = convert_is_nan(expr, args[0]);
    break;
  }
  case expr2t::overflow_id:
  {
    a = overflow_arith(expr);
    break;
  }
  case expr2t::overflow_cast_id:
  {
    a = overflow_cast(expr);
    break;
  }
  case expr2t::overflow_neg_id:
  {
    a = overflow_neg(expr);
    break;
  }
  case expr2t::zero_length_string_id:
  {
    // Extremely unclear.
    a = tuple_project(args[0], sort, 0);
    break;
  }
  case expr2t::zero_string_id:
  {
    // Actually broken. And always has been.
    a = mk_smt_symbol("zero_string", sort);
    break;
  }
  case expr2t::byte_extract_id:
  {
    a = convert_byte_extract(expr);
    break;
  }
  case expr2t::byte_update_id:
  {
    a = convert_byte_update(expr);
    break;
  }
  case expr2t::address_of_id:
  {
    a = convert_addr_of(expr);
    break;
  }
  case expr2t::equality_id:
  {
    const equality2t &eq = to_equality2t(expr);
    if (is_struct_type(eq.side_1->type) && is_struct_type(eq.side_2->type)) {
      // Struct equality
      a = tuple_equality(args[0], args[1]);
    } else if (is_array_type(eq.side_1->type) &&
               is_array_type(eq.side_2->type)) {
      if (is_structure_type(to_array_type(eq.side_1->type).subtype) ||
          is_pointer_type(to_array_type(eq.side_1->type).subtype)) {
        // Array of structs equality.
        a = tuple_array_equality(args[0], args[1]);
      } else {
        // Normal array equality
        a = mk_func_app(sort, SMT_FUNC_EQ, &args[0], 2);
      }
    } else if (is_pointer_type(eq.side_1) && is_pointer_type(eq.side_2)) {
      // Pointers are tuples
      a = tuple_equality(args[0], args[1]);
    } else if (is_union_type(eq.side_1) && is_union_type(eq.side_2)) {
      // Unions are also tuples
      a = tuple_equality(args[0], args[1]);
    } else {
      std::cerr << "Unrecognized equality form" << std::endl;
      expr->dump();
      abort();
    }
    break;
  }
  case expr2t::ashr_id:
  {
    const ashr2t &ashr = to_ashr2t(expr);

    if (ashr.side_1->type->get_width() != ashr.side_2->type->get_width()) {
      // FIXME: frontend doesn't cast the second operand up to the width of
      // the first, which SMT does not enjoy.
      typecast2tc cast(ashr.side_1->type, ashr.side_2);
      args[1] = convert_ast(cast);
    }

    if (int_encoding) {
      // Raise 2^shift, then divide first operand by that value. If it's
      // negative, I suspect the correct operation is to latch to -1,
      // XXX XXX XXX haven't implemented that yet.
      constant_int2tc two(ashr.type, BigInt(2));
      const smt_ast *powargs[2];
      powargs[0] = args[1];
      powargs[1] = convert_ast(two);
      args[1] = mk_func_app(sort, SMT_FUNC_POW, &powargs[0], 2);
      a = mk_func_app(sort, SMT_FUNC_DIV, &args[0], 2);
    } else {
      a = mk_func_app(sort, SMT_FUNC_BVASHR, &args[0], 2);
    }
    break;
  }
  case expr2t::notequal_id:
  {
    const notequal2t &notequal = to_notequal2t(expr);
    // Handle all kinds of structs by inverted equality. The only that's really
    // going to turn up is pointers though.
    if (is_structure_type(notequal.side_1) ||is_pointer_type(notequal.side_1)) {
      a = tuple_equality(args[0], args[1]);
      a = mk_func_app(sort, SMT_FUNC_NOT, &a, 1);
    } else {
      std::cerr << "Unexpected inequailty operands" << std::endl;
      expr->dump();
      abort();
    }
    break;
  }
  case expr2t::abs_id:
  {
    const abs2t &abs = to_abs2t(expr);
    if (is_unsignedbv_type(abs.value)) {
      // No need to do anything.
      a = args[0];
    } else {
      constant_int2tc zero(abs.value->type, BigInt(0));
      lessthan2tc lt(abs.value, zero);
      sub2tc sub(abs.value->type, zero, abs.value);
      if2tc ite(abs.type, lt, sub, abs.value);
      a = convert_ast(ite);
    }
    break;
  }
  case expr2t::lessthan_id:
  case expr2t::lessthanequal_id:
  case expr2t::greaterthan_id:
  case expr2t::greaterthanequal_id:
  {
    // Pointer relation:
    const expr2tc &side1 = *expr->get_sub_expr(0);
    const expr2tc &side2 = *expr->get_sub_expr(1);
    if (is_pointer_type(side1->type) && is_pointer_type(side2->type)) {
      a = convert_ptr_cmp(side1, side2, expr);
    } else {
      // One operand isn't a pointer; go the slow way, with typecasts.
      type2tc inttype = machine_ptr;
      expr2tc cast1 = (!is_unsignedbv_type(side1))
        ? typecast2tc(inttype, side1)
        : side1;
      expr2tc cast2 = (!is_unsignedbv_type(side2))
        ? typecast2tc(inttype, side2)
        : side2;
      expr2tc new_expr = expr;
      *new_expr.get()->get_sub_expr_nc(0) = cast1;
      *new_expr.get()->get_sub_expr_nc(1) = cast2;
      a = convert_ast(new_expr);
    }
    break;
  }
  default:
    std::cerr << "Couldn't convert expression in unrecognized format"
              << std::endl;
    expr->dump();
    abort();
  }

done:
  if (caching) {
    struct smt_cache_entryt entry = { expr, a, ctx_level };
    smt_cache.insert(entry);
  }

  return a;
}

void
smt_convt::assert_expr(const expr2tc &e)
{
  const smt_ast *a = convert_ast(e);
  literalt l = mk_lit(a);
  assert_lit(l);
  return;
}

smt_sort *
smt_convt::convert_sort(const type2tc &type)
{
  bool is_signed = true;

  switch (type->type_id) {
  case type2t::bool_id:
    return mk_sort(SMT_SORT_BOOL);
  case type2t::struct_id:
    if (!tuple_support) {
      return new tuple_smt_sort(type);
    } else {
      return mk_struct_sort(type);
    }
  case type2t::union_id:
    if (!tuple_support) {
      return new tuple_smt_sort(type);
    } else {
      return mk_union_sort(type);
    }
  case type2t::code_id:
  case type2t::pointer_id:
    if (!tuple_support) {
      return new tuple_smt_sort(pointer_struct);
    } else {
      return mk_struct_sort(pointer_struct);
    }
  case type2t::unsignedbv_id:
    is_signed = false;
    /* FALLTHROUGH */
  case type2t::signedbv_id:
  {
    unsigned int width = type->get_width();
    if (int_encoding)
      return mk_sort(SMT_SORT_INT, is_signed);
    else
      return mk_sort(SMT_SORT_BV, width, is_signed);
  }
  case type2t::fixedbv_id:
  {
    unsigned int width = type->get_width();
    if (int_encoding)
      return mk_sort(SMT_SORT_REAL);
    else
      return mk_sort(SMT_SORT_BV, width, false);
  }
  case type2t::string_id:
  {
    const smt_sort *d = machine_int_sort;
    smt_sort *r = (int_encoding)? mk_sort(SMT_SORT_INT)
                                : mk_sort(SMT_SORT_BV, 8,
                                          !config.ansi_c.char_is_unsigned);
    return mk_sort(SMT_SORT_ARRAY, d, r);
  }
  case type2t::array_id:
  {
    const array_type2t &arr = to_array_type(type);

    if (!tuple_support &&
        (is_structure_type(arr.subtype) || is_pointer_type(arr.subtype))) {
      return new tuple_smt_sort(type, calculate_array_domain_width(arr));
    }

    // Index arrays by the smallest integer required to represent its size.
    // Unless it's either infinite or dynamic in size, in which case use the
    // machine int size. Also, faff about if it's an array of arrays, extending
    // the domain.
    const smt_sort *d = make_array_domain_sort(arr);

    // Determine the range if we have arrays of arrays.
    type2tc range = arr.subtype;
    while (is_array_type(range))
      range = to_array_type(range).subtype;

    // Work around QF_AUFBV demanding arrays of bitvectors.
    smt_sort *r;
    if (!int_encoding && is_bool_type(range) && no_bools_in_arrays) {
      r = mk_sort(SMT_SORT_BV, 1, false);
    } else {
      r = convert_sort(range);
    }
    return mk_sort(SMT_SORT_ARRAY, d, r);
  }
  case type2t::cpp_name_id:
  case type2t::symbol_id:
  case type2t::empty_id:
  default:
    std::cerr << "Unexpected type ID reached SMT conversion" << std::endl;
    abort();
  }
}

static std::string
fixed_point(std::string v, unsigned width)
{
  const int precision = 1000000;
  std::string i, f, b, result;
  double integer, fraction, base;

  i = extract_magnitude(v, width);
  f = extract_fraction(v, width);
  b = integer2string(power(2, width / 2), 10);

  integer = atof(i.c_str());
  fraction = atof(f.c_str());
  base = (atof(b.c_str()));

  fraction = (fraction / base);

  if (fraction < 0)
    fraction = -fraction;

   fraction = fraction * precision;

  if (fraction == 0)
    result = double2string(integer);
  else  {
    int64_t numerator = (integer*precision + fraction);
    result = itos(numerator) + "/" + double2string(precision);
  }

  return result;
}

smt_ast *
smt_convt::convert_terminal(const expr2tc &expr)
{
  switch (expr->expr_id) {
  case expr2t::constant_int_id:
  {
    bool sign = is_signedbv_type(expr);
    const constant_int2t &theint = to_constant_int2t(expr);
    unsigned int width = expr->type->get_width();
    if (int_encoding)
      return mk_smt_int(theint.constant_value, sign);
    else
      return mk_smt_bvint(theint.constant_value, sign, width);
  }
  case expr2t::constant_fixedbv_id:
  {
    const constant_fixedbv2t &thereal = to_constant_fixedbv2t(expr);
    if (int_encoding) {
      std::string val = thereal.value.to_expr().value().as_string();
      std::string result = fixed_point(val, thereal.value.spec.width);
      return mk_smt_real(result);
    } else {
      assert(thereal.type->get_width() <= 64 && "Converting fixedbv constant to"
             " SMT, too large to fit into a uint64_t");

      uint64_t magnitude, fraction, fin;
      unsigned int bitwidth = thereal.type->get_width();
      std::string m, f, c;
      std::string theval = thereal.value.to_expr().value().as_string();

      m = extract_magnitude(theval, bitwidth);
      f = extract_fraction(theval, bitwidth);
      magnitude = strtoll(m.c_str(), NULL, 10);
      fraction = strtoll(f.c_str(), NULL, 10);

      magnitude <<= (bitwidth / 2);
      fin = magnitude | fraction;

      return mk_smt_bvint(mp_integer(fin), false, bitwidth);
    }
  }
  case expr2t::constant_bool_id:
  {
    const constant_bool2t &thebool = to_constant_bool2t(expr);
    return mk_smt_bool(thebool.constant_value);
  }
  case expr2t::symbol_id:
  {
    // Special case for tuple symbols
    if (!tuple_support &&
        (is_union_type(expr) || is_struct_type(expr) || is_pointer_type(expr))){
      // Perform smt-tuple hacks.
      return mk_tuple_symbol(expr);
    } else if (!tuple_support && is_array_type(expr) &&
                (is_struct_type(to_array_type(expr->type).subtype) ||
                 is_union_type(to_array_type(expr->type).subtype) ||
                 is_pointer_type(to_array_type(expr->type).subtype))) {
      return mk_tuple_array_symbol(expr);
    }
    const symbol2t &sym = to_symbol2t(expr);
    std::string name = sym.get_symbol_name();
    const smt_sort *sort = convert_sort(sym.type);
    return mk_smt_symbol(name, sort);
  }
  default:
    std::cerr << "Converting unrecognized terminal expr to SMT" << std::endl;
    expr->dump();
    abort();
  }
}

std::string
smt_convt::mk_fresh_name(const std::string &tag)
{
  std::string new_name = "smt_conv::" + tag;
  std::stringstream ss;
  ss << new_name << fresh_map[new_name]++;
  return ss.str();
}

smt_ast *
smt_convt::mk_fresh(const smt_sort *s, const std::string &tag)
{
  return mk_smt_symbol(mk_fresh_name(tag), s);
}

const smt_ast *
smt_convt::convert_is_nan(const expr2tc &expr, const smt_ast *operand)
{
  const smt_ast *args[3];
  const isnan2t &isnan = to_isnan2t(expr);
  const smt_sort *bs = mk_sort(SMT_SORT_BOOL);

  // Assumes operand is fixedbv.
  assert(is_fixedbv_type(isnan.value));
  unsigned width = isnan.value->type->get_width();

  const smt_ast *t = mk_smt_bool(true);
  const smt_ast *f = mk_smt_bool(false);

  if (int_encoding) {
    args[0] = round_real_to_int(operand);
    args[1] = mk_smt_int(BigInt(0), false);
    args[0] = mk_func_app(bs, SMT_FUNC_GTE, args, 2);
    args[1] = t;
    args[2] = f;
    return mk_func_app(bs, SMT_FUNC_ITE, args, 3);
  } else {
    args[0] = operand;
    args[1] = mk_smt_bvint(BigInt(0), false, width);
    args[0] = mk_func_app(bs, SMT_FUNC_GTE, args, 2);
    args[1] = t;
    args[2] = f;
    return mk_func_app(bs, SMT_FUNC_ITE, args, 3);
  }
}

const smt_ast *
smt_convt::convert_member(const expr2tc &expr, const smt_ast *src)
{
  const smt_sort *sort = convert_sort(expr->type);
  const member2t &member = to_member2t(expr);
  unsigned int idx = -1;

  if (is_union_type(member.source_value->type)) {
    union_varst::const_iterator cache_result;
    const union_type2t &data_ref = to_union_type(member.source_value->type);

    if (is_symbol2t(member.source_value)) {
      const symbol2t &sym = to_symbol2t(member.source_value);
      cache_result = union_vars.find(sym.get_symbol_name().c_str());
    } else {
      cache_result = union_vars.end();
    }

    if (cache_result != union_vars.end()) {
      const std::vector<type2tc> &members = data_ref.get_structure_members();

      const type2tc source_type = members[cache_result->idx];
      if (source_type == member.type) {
        // Type we're fetching from union matches expected type; just return it.
        idx = cache_result->idx;
      } else {
        // Union field and expected type mismatch. Need to insert a cast.
        // Duplicate expr as we're changing it
        member2tc memb2(source_type, member.source_value, member.member);
        typecast2tc cast(member.type, memb2);
        return convert_ast(cast);
      }
    } else {
      // If no assigned result available, we're probably broken for whatever
      // reason, just go haywire.
      idx = get_member_name_field(member.source_value->type, member.member);
    }
  } else {
    idx = get_member_name_field(member.source_value->type, member.member);
  }

  return tuple_project(src, sort, idx);
}

const smt_ast *
smt_convt::convert_sign_ext(const smt_ast *a, const smt_sort *s,
                            unsigned int topbit, unsigned int topwidth)
{
  const smt_ast *args[4];

  const smt_sort *bit = mk_sort(SMT_SORT_BV, 1, false);
  args[0] = mk_extract(a, topbit-1, topbit-1, bit);
  args[1] = mk_smt_bvint(BigInt(0), false, 1);
  const smt_sort *b = mk_sort(SMT_SORT_BOOL);
  const smt_ast *t = mk_func_app(b, SMT_FUNC_EQ, args, 2);

  const smt_ast *z = mk_smt_bvint(BigInt(0), false, topwidth);

  // Calculate the exact value; SMTLIB text parsers don't like taking an
  // over-full integer literal.
  uint64_t big = 0xFFFFFFFFFFFFFFFFULL;
  unsigned int num_topbits = 64 - topwidth;
  big >>= num_topbits;
  BigInt big_int(big);
  const smt_ast *f = mk_smt_bvint(big_int, false, topwidth);

  args[0] = t;
  args[1] = z;
  args[2] = f;
  const smt_sort *topsort = mk_sort(SMT_SORT_BV, topwidth, false);
  const smt_ast *topbits = mk_func_app(topsort, SMT_FUNC_ITE, args, 3);

  args[0] = topbits;
  args[1] = a;
  return mk_func_app(s, SMT_FUNC_CONCAT, args, 2);
}

const smt_ast *
smt_convt::convert_zero_ext(const smt_ast *a, const smt_sort *s,
                            unsigned int topwidth)
{
  const smt_ast *args[2];

  const smt_ast *z = mk_smt_bvint(BigInt(0), false, topwidth);
  args[0] = z;
  args[1] = a;
  return mk_func_app(s, SMT_FUNC_CONCAT, args, 2);
}

const smt_ast *
smt_convt::round_real_to_int(const smt_ast *a)
{
  // SMT truncates downwards; however C truncates towards zero, which is not
  // the same. (Technically, it's also platform dependant). To get around this,
  // add one to the result in all circumstances, except where the value was
  // already an integer.
  const smt_sort *realsort = mk_sort(SMT_SORT_REAL);
  const smt_sort *intsort = mk_sort(SMT_SORT_INT);
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  const smt_ast *args[3];
  args[0] = a;
  args[1] = mk_smt_real("0");
  const smt_ast *is_lt_zero = mk_func_app(realsort, SMT_FUNC_LT, args, 2);

  // The actual conversion
  const smt_ast *as_int = mk_func_app(intsort, SMT_FUNC_REAL2INT, args, 2);

  const smt_ast *one = mk_smt_int(BigInt(1), false);
  args[0] = one;
  args[1] = as_int;
  const smt_ast *plus_one = mk_func_app(intsort, SMT_FUNC_ADD, args, 2);

  // If it's an integer, just keep it's untruncated value.
  args[0] = mk_func_app(boolsort, SMT_FUNC_IS_INT, &a, 1);
  args[1] = as_int;
  args[2] = plus_one;
  args[1] = mk_func_app(intsort, SMT_FUNC_ITE, args, 3);

  // Switch on whether it's > or < 0.
  args[0] = is_lt_zero;
  args[2] = as_int;
  return mk_func_app(intsort, SMT_FUNC_ITE, args, 3);
}

const smt_ast *
smt_convt::round_fixedbv_to_int(const smt_ast *a, unsigned int fromwidth,
                                unsigned int towidth)
{
  // Perform C rounding: just truncate towards zero. Annoyingly, this isn't
  // that simple for negative numbers, because they're represented as a negative
  // integer _plus_ a positive fraction. So we need to round up if there's a
  // nonzero fraction, and not if there's not.
  const smt_ast *args[3];
  unsigned int frac_width = fromwidth / 2;

  // Sorts
  const smt_sort *bit = mk_sort(SMT_SORT_BV, 1, false);
  const smt_sort *halfwidth = mk_sort(SMT_SORT_BV, frac_width, false);
  const smt_sort *tosort = mk_sort(SMT_SORT_BV, towidth, false);
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);

  // Determine whether the source is signed from its topmost bit.
  const smt_ast *is_neg_bit = mk_extract(a, fromwidth-1, fromwidth-1, bit);
  const smt_ast *true_bit = mk_smt_bvint(BigInt(1), false, 1);

  // Also collect data for dealing with the magnitude.
  const smt_ast *magnitude = mk_extract(a, fromwidth-1, frac_width, halfwidth);
  const smt_ast *intvalue = convert_sign_ext(magnitude, tosort, frac_width,
                                             frac_width);

  // Data for inspecting fraction part
  const smt_ast *frac_part = mk_extract(a, frac_width-1, 0, bit);
  const smt_ast *zero = mk_smt_bvint(BigInt(0), false, frac_width);
  args[0] = frac_part;
  args[1] = zero;
  const smt_ast *is_zero_frac = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);

  // So, we have a base number (the magnitude), and need to decide whether to
  // round up or down. If it's positive, round down towards zero. If it's neg
  // and the fraction is zero, leave it, otherwise round towards zero.

  // We may need a value + 1.
  args[0] = intvalue;
  args[1] = mk_smt_bvint(BigInt(1), false, towidth);
  const smt_ast *intvalue_plus_one =
    mk_func_app(tosort, SMT_FUNC_BVADD, args, 2);

  args[0] = is_zero_frac;
  args[1] = intvalue;
  args[2] = intvalue_plus_one;
  const smt_ast *neg_val = mk_func_app(tosort, SMT_FUNC_ITE, args, 3);

  args[0] = true_bit;
  args[1] = is_neg_bit;
  const smt_ast *is_neg = mk_func_app(bit, SMT_FUNC_EQ, args, 2);

  // final switch
  args[0] = is_neg;
  args[1] = neg_val;
  args[2] = intvalue;
  return mk_func_app(tosort, SMT_FUNC_ITE, args, 3);
}

const smt_ast *
smt_convt::make_bool_bit(const smt_ast *a)
{

  assert(a->sort->id == SMT_SORT_BOOL && "Wrong sort fed to "
         "smt_convt::make_bool_bit");
  const smt_ast *one = mk_smt_bvint(BigInt(1), false, 1);
  const smt_ast *zero = mk_smt_bvint(BigInt(1), false, 1);
  const smt_ast *args[3];
  args[0] = a;
  args[1] = one;
  args[2] = zero;
  return mk_func_app(one->sort, SMT_FUNC_ITE, args, 3);
}

const smt_ast *
smt_convt::make_bit_bool(const smt_ast *a)
{

  assert(a->sort->id == SMT_SORT_BV && "Wrong sort fed to "
         "smt_convt::make_bit_bool");
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  const smt_ast *one = mk_smt_bvint(BigInt(1), false, 1);
  const smt_ast *args[2];
  args[0] = a;
  args[1] = one;
  return mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
}

const smt_ast *
smt_convt::fix_array_idx(const smt_ast *idx, const smt_sort *arrsort)
{
  if (int_encoding)
    return idx;

  unsigned int domain_width = arrsort->get_domain_width();
  if (domain_width == config.ansi_c.int_width)
    return idx;

  // Otherwise, we need to extract the lower bits out of this.
  const smt_sort *domsort = mk_sort(SMT_SORT_BV, domain_width, false);
  return mk_extract(idx, domain_width-1, 0, domsort);
}

unsigned long
smt_convt::calculate_array_domain_width(const array_type2t &arr)
{
  // Index arrays by the smallest integer required to represent its size.
  // Unless it's either infinite or dynamic in size, in which case use the
  // machine int size.
  if (!is_nil_expr(arr.array_size) && is_constant_int2t(arr.array_size)) {
    constant_int2tc thesize = arr.array_size;
    unsigned long sz = thesize->constant_value.to_ulong();
    uint64_t domwidth = 2;
    unsigned int dombits = 1;

    // Shift domwidth up until it's either larger or equal to sz, or we risk
    // overflowing.
    while (domwidth != 0x8000000000000000ULL && domwidth < sz) {
      domwidth <<= 1;
      dombits++;
    }

    if (domwidth == 0x8000000000000000ULL)
      dombits = 64;

    return dombits;
  } else {
    return config.ansi_c.int_width;
  }
}

const smt_sort *
smt_convt::make_array_domain_sort(const array_type2t &arr)
{

  // Start special casing if this is an array of arrays.
  if (!is_array_type(arr.subtype)) {
    // Normal array, work out what the domain sort is.
    if (int_encoding)
      return mk_sort(SMT_SORT_INT);
    else
      return mk_sort(SMT_SORT_BV, calculate_array_domain_width(arr), false);
  } else {
    // This is an array of arrays -- we're going to convert this into a single
    // array that has an extended domain. Work out that width. Firstly, how
    // many levels of array do we have?

    unsigned int how_many_arrays = 1;
    type2tc subarr = arr.subtype;
    while (is_array_type(subarr)) {
      how_many_arrays++;
      subarr = to_array_type(subarr).subtype;
    }

    assert(how_many_arrays < 64 && "Suspiciously large number of array "
                                   "dimensions");
    unsigned int domwidth;
    unsigned int i;
    domwidth = calculate_array_domain_width(arr);
    subarr = arr.subtype;
    for (i = 1; i < how_many_arrays; i++) {
      domwidth += calculate_array_domain_width(to_array_type(arr.subtype));
      subarr = arr.subtype;
    }

    return mk_sort(SMT_SORT_BV, domwidth, false);
  }
}

const smt_ast *
smt_convt::handle_select_chain(const expr2tc &expr, const smt_ast **base)
{
  // So: some series of index exprs will occur here, with some symbol or
  // other expression at the bottom that's actually some symbol, or whatever.
  // So, extract all the indexes, and concat them, with the first (lowest)
  // index at the top, then descending.

  unsigned int how_many_selects = 1, i;
  index2tc idx = expr;
  while (is_index2t(idx->source_value)) {
    how_many_selects++;
    idx = idx->source_value;
  }

  // Give the caller the base array object / thing. So that it can actually
  // select out of the right piece of data.
  *base = convert_ast(idx->source_value);

  assert(how_many_selects < 64 && "Suspiciously large number of array selects");
  const expr2tc *idx_ptrs[how_many_selects];
  const type2tc *arr_types[how_many_selects];
  idx = expr;
  idx_ptrs[0] = &idx->index;
  arr_types[0] = &idx->source_value->type;
  for (i = 1; i < how_many_selects; i++, idx = idx->source_value) {
    idx_ptrs[i] = &idx->index;
    arr_types[i] = &idx->source_value->type;
  }

  const smt_ast *idxes[how_many_selects];
  unsigned int domsizes[how_many_selects];
  for (i = 0; i < how_many_selects; i++) {
    idxes[i] = convert_ast(*idx_ptrs[i]);
    domsizes[i] = calculate_array_domain_width(to_array_type(*arr_types[i]));
    const smt_sort *arrsort = convert_sort(*arr_types[i]);
    idxes[i] = fix_array_idx(idxes[i], arrsort);
  }

  // Now, concatenate them.
  const smt_ast *concat = idxes[0];
  unsigned long bvsize = domsizes[0];
  for (i = 1; i < how_many_selects; i++) {
    bvsize += domsizes[i];
    const smt_sort *bvsort = mk_sort(SMT_SORT_BV, bvsize, false);
    const smt_ast *args[2];
    args[0] = idxes[i];
    args[1] = concat;
    concat = mk_func_app(bvsort, SMT_FUNC_CONCAT, args, 2);
  }

  return concat;
}

const smt_ast *
smt_convt::handle_store_chain(const expr2tc &expr __attribute__((unused)))
{
  abort();
}

const smt_convt::expr_op_convert
smt_convt::smt_convert_table[expr2t::end_expr_id] =  {
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //const int
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //const fixedbv
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //const bool
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //const string
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //const struct
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //const union
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //const array
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //const array_of
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //symbol
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //typecast
{ SMT_FUNC_ITE, SMT_FUNC_ITE, SMT_FUNC_ITE, 3, SMT_SORT_ALLINTS | SMT_SORT_BOOL | SMT_SORT_ARRAY},  //if
{ SMT_FUNC_EQ, SMT_FUNC_EQ, SMT_FUNC_EQ, 2, SMT_SORT_ALLINTS | SMT_SORT_BOOL},  //equality
{ SMT_FUNC_NOTEQ, SMT_FUNC_NOTEQ, SMT_FUNC_NOTEQ, 2, SMT_SORT_ALLINTS | SMT_SORT_BOOL},  //notequal
{ SMT_FUNC_LT, SMT_FUNC_BVSLT, SMT_FUNC_BVULT, 2, SMT_SORT_ALLINTS},  //lt
{ SMT_FUNC_GT, SMT_FUNC_BVSGT, SMT_FUNC_BVUGT, 2, SMT_SORT_ALLINTS},  //gt
{ SMT_FUNC_LTE, SMT_FUNC_BVSLTE, SMT_FUNC_BVULTE, 2, SMT_SORT_ALLINTS},  //lte
{ SMT_FUNC_GTE, SMT_FUNC_BVSGTE, SMT_FUNC_BVUGTE, 2, SMT_SORT_ALLINTS},  //gte
{ SMT_FUNC_NOT, SMT_FUNC_NOT, SMT_FUNC_NOT, 1, SMT_SORT_BOOL},  //not
{ SMT_FUNC_AND, SMT_FUNC_AND, SMT_FUNC_AND, 2, SMT_SORT_BOOL},  //and
{ SMT_FUNC_OR, SMT_FUNC_OR, SMT_FUNC_OR, 2, SMT_SORT_BOOL},  //or
{ SMT_FUNC_XOR, SMT_FUNC_XOR, SMT_FUNC_XOR, 2, SMT_SORT_BOOL},  //xor
{ SMT_FUNC_IMPLIES, SMT_FUNC_IMPLIES, SMT_FUNC_IMPLIES, 2, SMT_SORT_BOOL},//impl
{ SMT_FUNC_INVALID, SMT_FUNC_BVAND, SMT_FUNC_BVAND, 2, SMT_SORT_BV},  //bitand
{ SMT_FUNC_INVALID, SMT_FUNC_BVOR, SMT_FUNC_BVOR, 2, SMT_SORT_BV},  //bitor
{ SMT_FUNC_INVALID, SMT_FUNC_BVXOR, SMT_FUNC_BVXOR, 2, SMT_SORT_BV},  //bitxor
{ SMT_FUNC_INVALID, SMT_FUNC_BVNAND, SMT_FUNC_BVNAND, 2, SMT_SORT_BV},//bitnand
{ SMT_FUNC_INVALID, SMT_FUNC_BVNOR, SMT_FUNC_BVNOR, 2, SMT_SORT_BV},  //bitnor
{ SMT_FUNC_INVALID, SMT_FUNC_BVNXOR, SMT_FUNC_BVNXOR, 2, SMT_SORT_BV}, //bitnxor
{ SMT_FUNC_INVALID, SMT_FUNC_BVNOT, SMT_FUNC_BVNOT, 1, SMT_SORT_BV},  //bitnot
{ SMT_FUNC_INVALID, SMT_FUNC_BVLSHR, SMT_FUNC_BVLSHR, 2, SMT_SORT_BV},  //lshr
{ SMT_FUNC_NEG, SMT_FUNC_BVNEG, SMT_FUNC_BVNEG, 1, SMT_SORT_ALLINTS},  //neg
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //abs
{ SMT_FUNC_ADD, SMT_FUNC_BVADD, SMT_FUNC_BVADD, 2, SMT_SORT_ALLINTS},//add
{ SMT_FUNC_SUB, SMT_FUNC_BVSUB, SMT_FUNC_BVSUB, 2, SMT_SORT_ALLINTS},//sub
{ SMT_FUNC_MUL, SMT_FUNC_BVMUL, SMT_FUNC_BVMUL, 2, SMT_SORT_INT | SMT_SORT_REAL },//mul
{ SMT_FUNC_DIV, SMT_FUNC_BVSDIV, SMT_FUNC_BVUDIV, 2, SMT_SORT_INT | SMT_SORT_REAL },//div
{ SMT_FUNC_MOD, SMT_FUNC_BVSMOD, SMT_FUNC_BVUMOD, 2, SMT_SORT_BV | SMT_SORT_INT},//mod
{ SMT_FUNC_SHL, SMT_FUNC_BVSHL, SMT_FUNC_BVSHL, 2, SMT_SORT_BV | SMT_SORT_INT},  //shl

// Error: C frontend doesn't upcast the 2nd operand to ashr to the 1st operands
// bit width. Therefore this doesn't work. Fall back to backup method.
//{ SMT_FUNC_INVALID, SMT_FUNC_BVASHR, SMT_FUNC_BVASHR, 2, SMT_SORT_BV},  //ashr
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //ashr

{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //dyn_obj_id
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //same_obj_id
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //ptr_offs
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //ptr_obj
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //addr_of
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //byte_extract
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //byte_update
{ SMT_FUNC_STORE, SMT_FUNC_STORE, SMT_FUNC_STORE, 3, 0},  //with
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //member
{ SMT_FUNC_SELECT, SMT_FUNC_SELECT, SMT_FUNC_SELECT, 2, 0},  //index
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //zero_str_id
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //zero_len_str
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //isnan
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //overflow
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //overflow_cast
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //overflow_neg
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //unknown
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //invalid
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //null_obj
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //deref
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //valid_obj
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //deallocated
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //dyn_size
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //sideeffect
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_block
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_assign
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_init
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_decl
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_printf
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_expr
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_return
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_skip
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_free
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_goto
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //obj_desc
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_func_call
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_comma
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //invalid_ptr
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //buffer_sz
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_asm
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //cpp_del_arr
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //cpp_del_id
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //cpp_catch
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //cpp_throw
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //cpp_throw_dec
};

const std::string
smt_convt::smt_func_name_table[expr2t::end_expr_id] =  {
  "hack_func_id",
  "invalid_func_id",
  "int_func_id",
  "bool_func_id",
  "bvint_func_id",
  "real_func_id",
  "symbol_func_id",
  "add",
  "bvadd",
  "sub",
  "bvsub",
  "mul",
  "bvmul",
  "div",
  "bvudiv",
  "bvsdiv",
  "mod",
  "bvsmod",
  "bvumod",
  "shl",
  "bvshl",
  "bvashr",
  "neg",
  "bvneg",
  "bvshlr",
  "bvnot",
  "bvnxor",
  "bvnor",
  "bvnand",
  "bvxor",
  "bvor",
  "bvand",
  "=>",
  "xor",
  "or",
  "and",
  "not",
  "lt",
  "bvslt",
  "bvult",
  "gt",
  "bvsgt",
  "bvugt",
  "lte",
  "bvsle",
  "bvule",
  "gte",
  "bvsge",
  "bvuge",
  "=",
  "distinct",
  "ite",
  "store",
  "select",
  "concat",
  "extract",
  "int2real",
  "real2int",
  "pow",
  "is_int"
};
