#include <sstream>
#include <set>

#include <base_type.h>

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
                     bool is_cpp, bool _tuple_support)
  : caching(enable_cache), int_encoding(intmode), ns(_ns),
    tuple_support(_tuple_support)
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
    if (is_signedbv_type(*it))
      seen_signed_operand = true;
  }
  num_args = i;

  sort = convert_sort(expr->type);

  const expr_op_convert *cvt = &smt_convert_table[expr->expr_id];

  // Irritating special case: if we're selecting a bool out of an array, and
  // we're in QF_AUFBV mode, do special handling.
  if (!int_encoding && is_index2t(expr) && is_bool_type(expr->type))
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

    const smt_sort *domain;
    if (int_encoding)
      domain = mk_sort(SMT_SORT_INT, false);
    else
      domain = mk_sort(SMT_SORT_BV, config.ansi_c.int_width, false);

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
      a = mk_func_app(sort, SMT_FUNC_MUL, args, 2);
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
    const array_type2t &arrtype = to_array_type(index.source_value->type);
    if (!int_encoding && is_bool_type(arrtype.subtype)) {
      // Perform a fix for QF_AUFBV, only arrays of bv's are allowed.
      // XXX sort is wrong
      a = mk_func_app(sort, SMT_FUNC_SELECT, args, 2);
      // Quickie bv-to-bool casting.
      args[0] = a;
      args[1] = mk_smt_bvint(BigInt(1), false, 1);
      a = mk_func_app(sort, SMT_FUNC_EQ, args, 2);
    } else if (is_bool_type(arrtype.subtype)) {
      a = mk_func_app(sort, SMT_FUNC_EQ, args, 2);
    } else {
      a = tuple_array_select(args[0], sort, args[1]);
    }
    break;
  }
  case expr2t::with_id:
  {
    // We reach here if we're with'ing a struct, not an array. Or a bool.
    if (is_struct_type(expr->type) || is_union_type(expr)) {
      const with2t &with = to_with2t(expr);
      unsigned int idx = get_member_name_field(expr->type, with.update_field);
      a = tuple_update(args[0], idx, args[2]);
    } else {
      assert(is_array_type(expr->type));
      const array_type2t &arrtype = to_array_type(expr->type);
      const with2t &with = to_with2t(expr);
      if (!int_encoding && is_bool_type(arrtype.subtype)) {
        // If we're using QF_AUFBV, we need to cast (on the fly) booleans to
        // single bit bv's, because for some reason the logic doesn't support
        // arrays of bools.
        typecast2tc cast(get_uint_type(1), with.update_value);
        args[2] = convert_ast(cast);
        a = mk_func_app(sort, SMT_FUNC_STORE, args, 3);
        break;
      } else if (is_bool_type(arrtype.subtype)) {
        // Normal operation
        a = mk_func_app(sort, SMT_FUNC_STORE, args, 3);
        break;
      } else {
        assert(is_structure_type(arrtype.subtype) ||
               is_pointer_type(arrtype.subtype));
        const smt_sort *sort = convert_sort(with.update_value->type);
        a = tuple_array_update(args[0], args[1], args[2], sort);
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
      type2tc inttype = get_uint_type(config.ansi_c.pointer_width);
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
    smt_sort *d = (int_encoding)? mk_sort(SMT_SORT_INT)
                                : mk_sort(SMT_SORT_BV, config.ansi_c.int_width,
                                          !config.ansi_c.char_is_unsigned);
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
      return new tuple_smt_sort(type);
    }

    // All arrays are indexed by integerse
    smt_sort *d = (int_encoding)? mk_sort(SMT_SORT_INT)
                                : mk_sort(SMT_SORT_BV, config.ansi_c.int_width,
                                          false);

    // Work around QF_AUFBV demanding arrays of bitvectors.
    smt_sort *r;
    if (!int_encoding && is_bool_type(arr.subtype)) {
      r = mk_sort(SMT_SORT_BV, 1, false);
    } else {
      r = convert_sort(arr.subtype);
    }
    return mk_sort(SMT_SORT_ARRAY, d, r);
  }
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
      return mk_smt_real(thereal.value.to_integer());
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
const smt_ast *
smt_convt::array_create(const expr2tc &expr)
{
  const smt_ast *args[3];
  const smt_sort *sort = convert_sort(expr->type);
  std::string new_name = "smt_conv::array_create::";
  std::stringstream ss;
  ss << new_name << fresh_map[new_name]++;
  const smt_ast *newsym = mk_smt_symbol(ss.str(), sort);

  // Check size
  const array_type2t &arr_type =
    static_cast<const array_type2t &>(*expr->type.get());
  if (arr_type.size_is_infinite) {
    // Guarentee nothing, this is modelling only.
    return newsym;
  } else if (!is_constant_int2t(arr_type.array_size)) {
    std::cerr << "Non-constant sized array of type constant_array_of2t"
              << std::endl;
    abort();
  }

  const constant_int2t &thesize = to_constant_int2t(arr_type.array_size);
  uint64_t sz = thesize.constant_value.to_ulong();

  if (is_constant_array_of2t(expr)) {
    const constant_array_of2t &array = to_constant_array_of2t(expr);

    // Repeatedly store things into this.
    const smt_ast *init = convert_ast(array.initializer);
    for (unsigned int i = 0; i < sz; i++) {
      const smt_ast *field = (int_encoding)
        ? mk_smt_int(BigInt(i), false)
        : mk_smt_bvint(BigInt(i), false, config.ansi_c.int_width);
      args[0] = newsym;
      args[1] = field;
      args[2] = init;
      newsym = mk_func_app(sort, SMT_FUNC_STORE, args, 3);
    }

    return newsym;
  } else {
    assert(is_constant_array2t(expr));
    const constant_array2t &array = to_constant_array2t(expr);

    // Repeatedly store things into this.
    for (unsigned int i = 0; i < sz; i++) {
      const smt_ast *field = (int_encoding)
        ? mk_smt_int(BigInt(i), false)
        : mk_smt_bvint(BigInt(i), false, config.ansi_c.int_width);
      args[0] = newsym;
      args[1] = field;
      args[2] = convert_ast(array.datatype_members[i]);
      newsym = mk_func_app(sort, SMT_FUNC_STORE, args, 3);
    }

    return newsym;
  }
}

const smt_ast *
smt_convt::tuple_array_create_despatch(const expr2tc &expr,
                                       const smt_sort *domain)
{

  if (is_constant_array_of2t(expr)) {
    const constant_array_of2t &arr = to_constant_array_of2t(expr);
    const smt_ast *arg = convert_ast(arr.initializer);

    return tuple_array_create(arr.type, &arg, true, domain);
  } else {
    assert(is_constant_array2t(expr));
    const constant_array2t &arr = to_constant_array2t(expr);
    const smt_ast *args[arr.datatype_members.size()];
    unsigned int i = 0;
    for (std::vector<expr2tc>::const_iterator it = arr.datatype_members.begin();
         it != arr.datatype_members.end(); it++, i++) {
      args[i] = convert_ast(*it);
    }

    return tuple_array_create(arr.type, args, false, domain);
  }
}

smt_ast *
smt_convt::tuple_create(const expr2tc &structdef)
{
  std::string name_prefix = "smt_conv::tuple_create::";
  std::stringstream ss;
  ss << name_prefix << fresh_map[name_prefix]++ << ".";
  std::string name = ss.str();

  const smt_ast *args[structdef->get_num_sub_exprs()];
  for (unsigned int i = 0; i < structdef->get_num_sub_exprs(); i++)
    args[i] = convert_ast(*structdef->get_sub_expr(i));

  tuple_create_rec(name, structdef->type, args);

  return new tuple_smt_ast(convert_sort(structdef->type), name);
}

smt_ast *
smt_convt::tuple_fresh(const smt_sort *s)
{
  std::string name_prefix = "smt_conv::tuple_fresh::";
  std::stringstream ss;
  ss << name_prefix << fresh_map[name_prefix]++ << ".";
  std::string name = ss.str();

  smt_ast *a = mk_smt_symbol(name, s);
  a = a;
  return new tuple_smt_ast(s, name);
}

void
smt_convt::tuple_create_rec(const std::string &name, const type2tc &structtype,
                            const smt_ast **inputargs)
{
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  const struct_union_data &data = (is_pointer_type(structtype))
    ? *pointer_type_data
    : dynamic_cast<const struct_union_data &>(*structtype.get());

  unsigned int i = 0;
  for (std::vector<type2tc>::const_iterator it = data.members.begin();
       it != data.members.end(); it++, i++) {
    if (is_tuple_ast_type(*it)) {
      // Do something recursive
      std::string subname = name + data.member_names[i].as_string() + ".";
      // Generate an array of fields to pump in. First, fetch the type. It has
      // to be something struct based.
      const struct_union_data &nextdata = (is_pointer_type(*it))
        ? *pointer_type_data
        : dynamic_cast<const struct_union_data &>(*(*it).get());
      const smt_ast *nextargs[nextdata.members.size()];
      for (unsigned int j = 0; j < nextdata.members.size(); j++)
        nextargs[j] = tuple_project(inputargs[i],
                                    convert_sort(nextdata.members[j]), j);

      tuple_create_rec(subname, *it, nextargs);
    } else if (is_tuple_array_ast_type(*it)) {
      // convert_ast will have already, in fact, created a tuple array.
      // We just need to bind it into this one.
      std::string subname = name + data.member_names[i].as_string() + ".";
      const tuple_smt_ast *target =
        new tuple_smt_ast(convert_sort(*it), subname);
      const smt_ast *src = inputargs[i];
      assert_lit(mk_lit(tuple_array_equality(target, src)));
    } else {
      std::string symname = name + data.member_names[i].as_string();
      const smt_sort *sort = convert_sort(*it);
      const smt_ast *args[2];
      args[0] = mk_smt_symbol(symname, sort);
      args[1] = inputargs[i];
      const smt_ast *eq = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
      literalt l = mk_lit(eq);
      assert_lit(l);
    }
  }
}

smt_ast *
smt_convt::mk_tuple_symbol(const expr2tc &expr)
{
  const symbol2t &sym = to_symbol2t(expr);
  std::string name = sym.get_symbol_name() + ".";
  const smt_sort *sort = convert_sort(sym.type);
  return new tuple_smt_ast(sort, name);
}

smt_ast *
smt_convt::mk_tuple_array_symbol(const expr2tc &expr)
{
  const symbol2t &sym = to_symbol2t(expr);
  std::string name = sym.get_symbol_name() + "[]";
  const smt_sort *sort = convert_sort(sym.type);
  return new tuple_smt_ast(sort, name);
}

smt_ast *
smt_convt::tuple_project(const smt_ast *a, const smt_sort *s, unsigned int i)
{
  const tuple_smt_ast *ta = dynamic_cast<const tuple_smt_ast *>(a);
  assert(ta != NULL && "Non tuple_smt_ast class in smt_convt::tuple_project");

  const tuple_smt_sort *ts = dynamic_cast<const tuple_smt_sort *>(a->sort);
  assert(ts != NULL && "Non tuple_smt_sort class in smt_convt::tuple_project");
  const struct_union_data &data =
    dynamic_cast<const struct_union_data &>(*ts->thetype.get());

  assert(i < data.members.size() && "Out-of-bounds tuple element accessed");
  const std::string &fieldname = data.member_names[i].as_string();
  std::string sym_name = ta->name + fieldname;

  // Cope with recursive structs.
  const type2tc &restype = data.members[i];
  if (is_tuple_ast_type(restype) || is_tuple_array_ast_type(restype)) {
    sym_name = sym_name + ".";
    return new tuple_smt_ast(s, sym_name);
  } else {
    return mk_smt_symbol(sym_name, s);
  }
}

const smt_ast *
smt_convt::tuple_update(const smt_ast *a, unsigned int i, const smt_ast *v)
{
  // Turn a project into an equality with an update.
  const smt_ast *args[2];
  bvt eqs;
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);

  // Create a fresh tuple to store the result in
  std::string name_prefix = "smt_conv::tuple_update::";
  std::stringstream ss;
  ss << name_prefix << fresh_map[name_prefix]++ << ".";
  std::string name = ss.str();
  const tuple_smt_ast *result = new tuple_smt_ast(a->sort, name);

  const tuple_smt_ast *ta = dynamic_cast<const tuple_smt_ast *>(a);
  assert(ta != NULL && "Non tuple_smt_ast class in smt_convt::tuple_update");

  const tuple_smt_sort *ts = dynamic_cast<const tuple_smt_sort *>(ta->sort);
  assert(ts != NULL && "Non tuple_smt_sort class in smt_convt::tuple_update");

  const struct_union_data &data =
    dynamic_cast<const struct_union_data &>(*ts->thetype.get());

  unsigned int j = 0;
  for (std::vector<type2tc>::const_iterator it = data.members.begin();
       it != data.members.end(); it++, j++) {
    if (j == i) {
      const smt_sort *tmp = convert_sort(*it);
      const smt_ast *thefield = tuple_project(result, tmp, j);
      if (is_tuple_ast_type(*it)) {
        eqs.push_back(mk_lit(tuple_equality(thefield, v)));
      } else {
        args[0] = thefield;
        args[1] = v;
        eqs.push_back(mk_lit(mk_func_app(boolsort, SMT_FUNC_EQ, args, 2)));
      }
    } else {
      if (is_tuple_ast_type(*it)) {
        std::stringstream ss2;
        ss2 << name << data.member_names[j] << ".";
        std::string field_name = ss.str();
        const smt_sort *tmp = convert_sort(*it);
        const smt_ast *field1 = tuple_project(ta, tmp, j);
        const smt_ast *field2 = tuple_project(result, tmp, j);
        eqs.push_back(mk_lit(tuple_equality(field1, field2)));
      } else {
        const smt_sort *tmp = convert_sort(*it);
        args[0] = tuple_project(ta, tmp, j);
        args[1] = tuple_project(result, tmp, j);
        eqs.push_back(mk_lit(mk_func_app(boolsort, SMT_FUNC_EQ, args, 2)));
      }
    }
  }

  assert_lit(land(eqs));
  return result;
}

const smt_ast *
smt_convt::tuple_equality(const smt_ast *a, const smt_ast *b)
{
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  const tuple_smt_ast *ta = dynamic_cast<const tuple_smt_ast *>(a);
  const tuple_smt_ast *tb = dynamic_cast<const tuple_smt_ast *>(b);
  assert(ta != NULL && "Non tuple_smt_ast class in smt_convt::tuple_equality");
  assert(tb != NULL && "Non tuple_smt_ast class in smt_convt::tuple_equality");

  const tuple_smt_sort *ts = dynamic_cast<const tuple_smt_sort *>(ta->sort);
  assert(ts != NULL && "Non tuple_smt_sort class in smt_convt::tuple_equality");

  const struct_union_data &data =
    dynamic_cast<const struct_union_data &>(*ts->thetype.get());

  std::vector<literalt> lits;
  lits.reserve(data.members.size());

  // Iterate through each field and encode an equality.
  unsigned int i = 0;
  for (std::vector<type2tc>::const_iterator it = data.members.begin();
       it != data.members.end(); it++, i++) {
    if (is_tuple_ast_type(*it)) {
      // Recurse.
      const smt_ast *args[2];
      const smt_sort *sort = convert_sort(*it);
      args[0] = tuple_project(a, sort, i);
      args[1] = tuple_project(b, sort, i);
      const smt_ast *eq = tuple_equality(args[0], args[1]);
      literalt l = mk_lit(eq);
      lits.push_back(l);
    } else if (is_tuple_array_ast_type(*it)) {
      const smt_ast *args[2];
      const smt_sort *sort = convert_sort(*it);
      args[0] = tuple_project(a, sort, i);
      args[1] = tuple_project(b, sort, i);
      const smt_ast *eq = tuple_array_equality(args[0], args[1]);
      literalt l = mk_lit(eq);
      lits.push_back(l);
    } else {
      const smt_ast *args[2];
      const smt_sort *sort = convert_sort(*it);
      args[0] = tuple_project(a, sort, i);
      args[1] = tuple_project(b, sort, i);
      const smt_ast *eq = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
      literalt l = mk_lit(eq);
      lits.push_back(l);
    }
  }

  literalt l = land(lits);
  return lit_to_ast(l);
}

const smt_ast *
smt_convt::tuple_ite(const smt_ast *cond, const smt_ast *true_val,
                     const smt_ast *false_val, const smt_sort *sort)
{
  // Encode as an ite of each element.
  const tuple_smt_ast *trueast = dynamic_cast<const tuple_smt_ast *>(true_val);
  const tuple_smt_ast *falseast = dynamic_cast<const tuple_smt_ast*>(false_val);
  assert(trueast != NULL && "Non tuple_smt_ast class in smt_convt::tuple_ite");
  assert(falseast != NULL && "Non tuple_smt_ast class in smt_convt::tuple_ite");

  // Create a fresh tuple to store the result in
  std::string name_prefix = "smt_conv::tuple_ite::";
  std::stringstream ss;
  ss << name_prefix << fresh_map[name_prefix]++ << ".";
  std::string name = ss.str();
  const tuple_smt_ast *result = new tuple_smt_ast(sort, name);

  tuple_ite_rec(result, cond, trueast, falseast);
  return result;
}

void
smt_convt::tuple_ite_rec(const tuple_smt_ast *result, const smt_ast *cond,
                         const tuple_smt_ast *true_val,
                         const tuple_smt_ast *false_val)
{
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  const tuple_smt_sort *ts =
    dynamic_cast<const tuple_smt_sort *>(true_val->sort);
  assert(ts != NULL && "Non tuple_smt_sort class in smt_convt::tuple_ite");

  const struct_union_data &data =
    dynamic_cast<const struct_union_data &>(*ts->thetype.get());

  // Iterate through each field and encode an ite.
  unsigned int i = 0;
  for (std::vector<type2tc>::const_iterator it = data.members.begin();
       it != data.members.end(); it++, i++) {
    if (is_tuple_ast_type(*it)) {
      // Recurse.
      const tuple_smt_ast *args[3];
      const smt_sort *sort = convert_sort(*it);
      args[0] =
        static_cast<const tuple_smt_ast *>(tuple_project(result, sort, i));
      args[1] =
        static_cast<const tuple_smt_ast *>(tuple_project(true_val, sort, i));
      args[2] =
        static_cast<const tuple_smt_ast *>(tuple_project(false_val, sort, i));
      tuple_ite_rec(args[0], cond, args[1], args[2]);
    } else if (is_tuple_array_ast_type(*it)) {
      // Same deal, but with arrays
      const tuple_smt_ast *args[3];
      const smt_sort *sort = convert_sort(*it);
      args[0] =
        static_cast<const tuple_smt_ast *>(tuple_project(result, sort, i));
      args[1] =
        static_cast<const tuple_smt_ast *>(tuple_project(true_val, sort, i));
      args[2] =
        static_cast<const tuple_smt_ast *>(tuple_project(false_val, sort, i));
      args[1] = static_cast<const tuple_smt_ast*>
          (tuple_array_ite(cond, args[1], args[2], args[1]->sort));
      assert_lit(mk_lit(tuple_array_equality(args[0], args[1])));
    } else {
      const smt_ast *args[3], *eqargs[2];
      const smt_sort *sort = convert_sort(*it);
      args[0] = cond;
      args[1] = tuple_project(true_val, sort, i);
      args[2] = tuple_project(false_val, sort, i);
      eqargs[0] = mk_func_app(sort, SMT_FUNC_ITE, args, 3);
      eqargs[1] = tuple_project(result, sort, i);
      const smt_ast *eq = mk_func_app(boolsort, SMT_FUNC_EQ, eqargs, 2);
      literalt l = mk_lit(eq);
      assert_lit(l);
    }
  }
}

const smt_ast *
smt_convt::tuple_array_create(const type2tc &array_type,
                              const smt_ast **inputargs,
                              bool const_array,
                              const smt_sort *domain __attribute__((unused)))
{
  const smt_sort *sort = convert_sort(array_type);
  std::string new_name = "smt_conv::tuple_array_create::";
  std::stringstream ss;
  ss << new_name << fresh_map[new_name]++;
  const smt_ast *newsym = new tuple_smt_ast(sort, ss.str());

  // Check size
  const array_type2t &arr_type =
    static_cast<const array_type2t &>(*array_type.get());
  if (arr_type.size_is_infinite) {
    // Guarentee nothing, this is modelling only.
    return newsym;
  } else if (!is_constant_int2t(arr_type.array_size)) {
    std::cerr << "Non-constant sized array of type constant_array_of2t"
              << std::endl;
    abort();
  }

  const smt_sort *fieldsort = convert_sort(arr_type.subtype);
  const constant_int2t &thesize = to_constant_int2t(arr_type.array_size);
  uint64_t sz = thesize.constant_value.to_ulong();

  if (const_array) {
    // Repeatedly store things into this.
    const smt_ast *init = inputargs[0];
    for (unsigned int i = 0; i < sz; i++) {
      const smt_ast *field = (int_encoding)
        ? mk_smt_int(BigInt(i), false)
        : mk_smt_bvint(BigInt(i), false, config.ansi_c.int_width);
      newsym = tuple_array_update(newsym, field, init, fieldsort);
    }

    return newsym;
  } else {
    // Repeatedly store things into this.
    for (unsigned int i = 0; i < sz; i++) {
      const smt_ast *field = (int_encoding)
        ? mk_smt_int(BigInt(i), false)
        : mk_smt_bvint(BigInt(i), false, config.ansi_c.int_width);
      newsym = tuple_array_update(newsym, field, inputargs[i], fieldsort);
    }

    return newsym;
  }
}

const smt_ast *
smt_convt::tuple_array_select(const smt_ast *a, const smt_sort *s,
                              const smt_ast *field)
{
  // Select everything at the given element into a fresh tulple. Don't attempt
  // to support selecting array fields. In the future we can arrange something
  // whereby tuple operations are aware of this array situation and don't
  // have to take this inefficient approach.
  const tuple_smt_ast *ta = dynamic_cast<const tuple_smt_ast *>(a);
  assert(ta != NULL &&
         "Non tuple_smt_ast class in smt_convt::tuple_array_select");
  const tuple_smt_sort *ts = dynamic_cast<const tuple_smt_sort *>(a->sort);
  assert(ts != NULL &&
         "Non tuple_smt_sort class in smt_convt::tuple_array_select");

  std::string new_name = "smt_conv::tuple_array_select::";
  std::stringstream ss;
  ss << new_name << fresh_map[new_name]++;
  const tuple_smt_ast *result = new tuple_smt_ast(s, ss.str());

  const array_type2t &array_type = to_array_type(ts->thetype);
  tuple_array_select_rec(ta, array_type.subtype, result, field);
  return result;
}

void
smt_convt::tuple_array_select_rec(const tuple_smt_ast *ta,
                                  const type2tc &subtype,
                                  const tuple_smt_ast *result,
                                  const smt_ast *field)
{
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  const struct_union_data &struct_type = (is_pointer_type(subtype))
    ? *pointer_type_data
    : static_cast<const struct_union_data &>(*subtype.get());

  unsigned int i = 0;
  for (std::vector<type2tc>::const_iterator it = struct_type.members.begin();
       it != struct_type.members.end(); it++, i++) {
    if (is_tuple_ast_type(*it)) {
      const smt_sort *sort = convert_sort(*it);
      const tuple_smt_ast *result_field =
        static_cast<const tuple_smt_ast *>(tuple_project(result, sort, i));
      std::string substruct_name =
        ta->name + struct_type.member_names[i].as_string() + ".";
      const tuple_smt_ast *array_name = new tuple_smt_ast(sort, substruct_name);
      tuple_array_select_rec(array_name, *it, result_field, field);
    } else {
      std::string name = ta->name + struct_type.member_names[i].as_string();
      const smt_ast *args[2];
      const smt_sort *field_sort = convert_sort(*it);
      const smt_sort *arrsort = mk_sort(SMT_SORT_ARRAY, field->sort,field_sort);
      args[0] = mk_smt_symbol(name, arrsort);
      args[1] = field;
      args[0] = mk_func_app(field_sort, SMT_FUNC_SELECT, args, 2);
      args[1] = tuple_project(result, field_sort, i);
      const smt_ast *res = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
      literalt l = mk_lit(res);
      assert_lit(l);
    }
  }
}

const smt_ast *
smt_convt::tuple_array_update(const smt_ast *a, const smt_ast *index,
                              const smt_ast *val,
                              const smt_sort *fieldsort __attribute__((unused)))
{
  const tuple_smt_ast *ta = dynamic_cast<const tuple_smt_ast *>(a);
  assert(ta != NULL &&
         "Non tuple_smt_ast class in smt_convt::tuple_array_update");
  const tuple_smt_ast *tv = dynamic_cast<const tuple_smt_ast *>(val);
  assert(tv != NULL &&
         "Non tuple_smt_ast class in smt_convt::tuple_array_update");
  const tuple_smt_sort *ts = dynamic_cast<const tuple_smt_sort *>(ta->sort);
  assert(ts != NULL &&
         "Non tuple_smt_sort class in smt_convt::tuple_array_update");

  std::string new_name = "smt_conv::tuple_array_select[]";
  std::stringstream ss;
  ss << new_name << fresh_map[new_name]++;
  const tuple_smt_ast *result = new tuple_smt_ast(a->sort, ss.str());

  const array_type2t &array_type = to_array_type(ts->thetype);
  tuple_array_update_rec(ta, tv, index, result, array_type.subtype);
  return result;
}

void
smt_convt::tuple_array_update_rec(const tuple_smt_ast *ta,
                                  const tuple_smt_ast *tv, const smt_ast *idx,
                                  const tuple_smt_ast *result,
                                  const type2tc &subtype)
{
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  const struct_union_data &struct_type = (is_pointer_type(subtype))
    ? *pointer_type_data
    : static_cast<const struct_union_data &>(*subtype.get());

  unsigned int i = 0;
  for (std::vector<type2tc>::const_iterator it = struct_type.members.begin();
       it != struct_type.members.end(); it++, i++) {
    if (is_tuple_ast_type(*it)) {
      const smt_sort *tmp = convert_sort(*it);
      std::string resname = result->name +
                            struct_type.member_names[i].as_string() +
                            ".";
      std::string srcname = ta->name + struct_type.member_names[i].as_string() +
                            ".";
      std::string valname = tv->name + struct_type.member_names[i].as_string() +
                            ".";
      const tuple_smt_ast *target = new tuple_smt_ast(tmp, resname);
      const tuple_smt_ast *src = new tuple_smt_ast(tmp, srcname);
      const tuple_smt_ast *val = new tuple_smt_ast(tmp, valname);

      tuple_array_update_rec(src, val, idx, target, *it);
    } else {
      std::string arrname = ta->name + struct_type.member_names[i].as_string();
      std::string valname = tv->name + struct_type.member_names[i].as_string();
      std::string resname = result->name +
                            struct_type.member_names[i].as_string();
      const smt_ast *args[3];
      const smt_sort *idx_sort = convert_sort(*it);
      const smt_sort *arrsort = mk_sort(SMT_SORT_ARRAY, idx->sort, idx_sort);
      args[0] = mk_smt_symbol(arrname, arrsort);
      args[1] = idx;
      args[2] = mk_smt_symbol(valname, idx_sort);
      args[0] = mk_func_app(arrsort, SMT_FUNC_STORE, args, 3);
      args[1] = mk_smt_symbol(resname, arrsort);
      const smt_ast *res = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
      literalt l = mk_lit(res);
      assert_lit(l);
    }
  }
}

const smt_ast *
smt_convt::tuple_array_equality(const smt_ast *a, const smt_ast *b)
{

  const tuple_smt_ast *ta = dynamic_cast<const tuple_smt_ast *>(a);
  assert(ta != NULL &&
         "Non tuple_smt_ast class in smt_convt::tuple_array_equality");
  const tuple_smt_ast *tb = dynamic_cast<const tuple_smt_ast *>(b);
  assert(tb != NULL &&
         "Non tuple_smt_ast class in smt_convt::tuple_array_equality");
  const tuple_smt_sort *ts = dynamic_cast<const tuple_smt_sort *>(a->sort);
  assert(ts != NULL &&
         "Non tuple_smt_sort class in smt_convt::tuple_array_equality");

  const array_type2t &array_type = to_array_type(ts->thetype);
  return tuple_array_equality_rec(ta, tb, array_type.subtype);
}

const smt_ast *
smt_convt::tuple_array_equality_rec(const tuple_smt_ast *a,
                                    const tuple_smt_ast *b,
                                    const type2tc &subtype)
{
  bvt eqs;
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  const struct_union_data &struct_type = (is_pointer_type(subtype))
    ? *pointer_type_data
    : static_cast<const struct_union_data &>(*subtype.get());

  unsigned int i = 0;
  for (std::vector<type2tc>::const_iterator it = struct_type.members.begin();
       it != struct_type.members.end(); it++, i++) {
    if (is_tuple_ast_type(*it)) {
      const smt_sort *tmp = convert_sort(*it);
      std::string name1 = a->name + struct_type.member_names[i].as_string()+".";
      std::string name2 = b->name + struct_type.member_names[i].as_string()+".";
      const tuple_smt_ast *new1 = new tuple_smt_ast(tmp, name1);
      const tuple_smt_ast *new2 = new tuple_smt_ast(tmp, name2);
      eqs.push_back(mk_lit(tuple_array_equality_rec(new1, new2, *it)));
    } else {
      std::string name1 = a->name + struct_type.member_names[i].as_string();
      std::string name2 = b->name + struct_type.member_names[i].as_string();
      const smt_ast *args[2];
      const smt_sort *idx_sort = convert_sort(*it);
      const smt_sort *dom_sort = mk_sort(SMT_SORT_BV,
                                         config.ansi_c.int_width, false);
      const smt_sort *arrsort = mk_sort(SMT_SORT_ARRAY, dom_sort, idx_sort);
      args[0] = mk_smt_symbol(name1, arrsort);
      args[1] = mk_smt_symbol(name2, arrsort);
      eqs.push_back(mk_lit(mk_func_app(boolsort, SMT_FUNC_EQ, args, 2)));
    }
  }

  return lit_to_ast(land(eqs));
}

const smt_ast *
smt_convt::tuple_array_ite(const smt_ast *cond, const smt_ast *trueval,
                           const smt_ast *false_val,
                           const smt_sort *sort __attribute__((unused)))
{

  const tuple_smt_ast *tv = dynamic_cast<const tuple_smt_ast *>(trueval);
  assert(tv != NULL &&
         "Non tuple_smt_ast class in smt_convt::tuple_array_update");
  const tuple_smt_ast *fv = dynamic_cast<const tuple_smt_ast *>(false_val);
  assert(fv != NULL &&
         "Non tuple_smt_ast class in smt_convt::tuple_array_update");
  const tuple_smt_sort *ts = dynamic_cast<const tuple_smt_sort *>(tv->sort);
  assert(ts != NULL &&
         "Non tuple_smt_sort class in smt_convt::tuple_array_update");

  std::string new_name = "smt_conv::tuple_array_ite[]";
  std::stringstream ss;
  ss << new_name << fresh_map[new_name]++;
  const tuple_smt_ast *result = new tuple_smt_ast(tv->sort, new_name);

  tuple_array_ite_rec(tv, fv, cond, ts->thetype, result);
  return result;
}

void
smt_convt::tuple_array_ite_rec(const tuple_smt_ast *tv, const tuple_smt_ast *fv,
                               const smt_ast *cond, const type2tc &type,
                               const tuple_smt_ast *res)
{

  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  const array_type2t &array_type = to_array_type(type);
  const struct_union_data &struct_type = (is_pointer_type(array_type.subtype))
    ? *pointer_type_data
    : static_cast<const struct_union_data &>(*array_type.subtype.get());

  unsigned int i = 0;
  for (std::vector<type2tc>::const_iterator it = struct_type.members.begin();
       it != struct_type.members.end(); it++, i++) {
    if (is_tuple_ast_type(*it)) {
      std::cerr << "XXX struct struct array ite unimplemented" << std::endl;
      abort();
    } else {
      std::string tname = tv->name + struct_type.member_names[i].as_string();
      std::string fname = fv->name + struct_type.member_names[i].as_string();
      std::string rname = res->name + struct_type.member_names[i].as_string();
      const smt_ast *args[3];
      const smt_sort *idx_sort = convert_sort(*it);
      const smt_sort *dom_sort = mk_sort(SMT_SORT_BV,
                                         config.ansi_c.int_width, false);
      const smt_sort *arrsort = mk_sort(SMT_SORT_ARRAY, dom_sort, idx_sort);
      args[0] = cond;
      args[1] = mk_smt_symbol(tname, arrsort);
      args[2] = mk_smt_symbol(fname, arrsort);
      args[0] = mk_func_app(idx_sort, SMT_FUNC_ITE, args, 3);
      args[1] = mk_smt_symbol(rname, arrsort);
      args[0] = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
      literalt l = mk_lit(args[0]);
      assert_lit(l);
    }
  }
}

expr2tc
smt_convt::tuple_get(const expr2tc &expr)
{
  assert(is_symbol2t(expr) && "Non-symbol in smtlib expr get()");
  const symbol2t &sym = to_symbol2t(expr);
  std::string name = sym.get_symbol_name();

  const type2tc &thetype = (is_structure_type(expr->type))
    ? expr->type : pointer_struct;
  const struct_union_data &strct =
    static_cast<const struct_union_data &>(*thetype.get());

  // XXX - what's the correct type to return here.
  constant_struct2tc outstruct(expr->type, std::vector<expr2tc>());

  // Run through all fields and despatch to 'get' again.
  unsigned int i = 0;
  for (std::vector<type2tc>::const_iterator it = strct.members.begin();
       it != strct.members.end(); it++, i++) {
    std::stringstream ss;
    ss << name << "." << strct.member_names[i];
    symbol2tc sym(*it, ss.str());
    outstruct.get()->datatype_members.push_back(get(sym));
  }

  return outstruct;
}

smt_ast *
smt_convt::mk_fresh(const smt_sort *s, const std::string &tag)
{
  std::string new_name = "smt_conv::" + tag;
  std::stringstream ss;
  ss << new_name << fresh_map[new_name]++;
  return mk_smt_symbol(ss.str(), s);
}

const smt_ast *
smt_convt::overflow_arith(const expr2tc &expr)
{
  // If in integer mode, this is completely pointless. Return false.
  if (int_encoding)
    return mk_smt_bool(false);

  const overflow2t &overflow = to_overflow2t(expr);
  const arith_2ops &opers = static_cast<const arith_2ops &>(*overflow.operand);
  constant_int2tc zero(opers.side_1->type, BigInt(0));
  lessthan2tc op1neg(opers.side_1, zero);
  lessthan2tc op2neg(opers.side_2, zero);

  equality2tc op1iszero(opers.side_1, zero);
  equality2tc op2iszero(opers.side_2, zero);
  or2tc containszero(op1iszero, op2iszero);

  // Guess whether we're performing a signed or unsigned comparison.
  bool is_signed = (is_signedbv_type(opers.side_1) ||
                    is_signedbv_type(opers.side_2));

  if (is_add2t(overflow.operand)) {
    if (is_signed) {
      // Three cases; pos/pos, pos/neg, neg/neg, each with their own failure
      // modes.
      // First, if both pos, usual constraint.
      greaterthanequal2tc c1(overflow.operand, zero);

      // If pos/neg, result needs to be in the range posval > x >= negval
      lessthan2tc foo(overflow.operand, opers.side_1);
      lessthanequal2tc bar(opers.side_2, overflow.operand);
      and2tc c2_1(foo, bar);

      // And vice versa for neg/pos
      lessthan2tc oof(overflow.operand, opers.side_2);
      lessthanequal2tc rab(opers.side_1, overflow.operand);
      and2tc c2_2(oof, rab);

      // neg/neg: result should be below 0.
      lessthan2tc c3(overflow.operand, zero);

      // Finally, encode this into a series of implies that must always be true
      or2tc ncase1(op1neg, op2neg);
      not2tc case1(ncase1);
      implies2tc f1(case1, c1);
      
      equality2tc e1(op1neg, false_expr);
      equality2tc e2(op2neg, true_expr);
      and2tc case2_1(e1, e2);
      implies2tc f2(case2_1, c2_1);

      equality2tc e3(op1neg, true_expr);
      equality2tc e4(op2neg, false_expr);
      and2tc case2_2(e3, e4);
      implies2tc f3(case2_2, c2_2);

      and2tc case3(op1neg, op2neg);
      implies2tc f4(case3, c3);

      // Link them up.
      and2tc f5(f1, f2);
      and2tc f6(f3, f4);
      and2tc f7(f5, f6);
      not2tc inv(f7);
      return convert_ast(inv);
    } else {
      // Just ensure the result is >= both operands.
      greaterthanequal2tc ge1(overflow.operand, opers.side_1);
      greaterthanequal2tc ge2(overflow.operand, opers.side_2);
      and2tc res(ge1, ge2);
      not2tc inv(res);
      return convert_ast(inv);
    }
  } else if (is_sub2t(overflow.operand)) {
    if (is_signed) {
      // Same deal as with add. Enumerate the cases.
      // plus/plus, only failure mode is underflowing:
      lessthanequal2tc c1(overflow.operand, opers.side_1);

      // pos/neg, could overflow.
      greaterthan2tc c2(overflow.operand, opers.side_1);

      // neg/pos - already covered by c1

      // neg/neg - impossible to get wrong.

      equality2tc e1(op1neg, false_expr);
      equality2tc e2(op2neg, false_expr);
      equality2tc e3(op1neg, true_expr);
      equality2tc e4(op2neg, true_expr);

      and2tc cond1(e1, e2);
      and2tc cond3(e3, e2);
      or2tc dualcond(cond1, cond3);
      implies2tc f1(dualcond, c1);

      and2tc cond2(e1, e4);
      implies2tc f2(cond2, c2);

      // No encoding for neg/neg on account of how it's impossible to be wrong

      // Combine
      and2tc f3(f1, f2);
      not2tc inv(f3);
      return convert_ast(inv);
    } else {
      // Just ensure the result is >= the operands.
      lessthanequal2tc le1(overflow.operand, opers.side_1);
      lessthanequal2tc le2(overflow.operand, opers.side_2);
      and2tc res(le1, le2);
      not2tc inv(res);
      return convert_ast(inv);
    }
  } else {
    assert(is_mul2t(overflow.operand) && "unexpected overflow_arith operand");

    // Zero extend; multiply; Make a decision based on the top half.
    const smt_ast *args[3], *mulargs[2];
    unsigned int sz = zero->type->get_width();
    const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
    const smt_sort *normalsort = mk_sort(SMT_SORT_BV, sz, false);
    const smt_sort *bigsort = mk_sort(SMT_SORT_BV, sz * 2, false);

    // All one bit vector is tricky, might be 64 bits wide for all we know.
    constant_int2tc allonesexpr(zero->type, BigInt((sz == 64)
                                                 ? 0xFFFFFFFFFFFFFFFFULL
                                                 : ((1ULL << sz) - 1)));
    const smt_ast *allonesvector = convert_ast(allonesexpr);

    const smt_ast *arg1_ext, *arg2_ext;
    if (is_signed) {
      // sign extend top bits.
      arg1_ext = convert_ast(opers.side_1);
      arg1_ext = convert_sign_ext(arg1_ext, bigsort, sz - 1, sz);
      arg2_ext = convert_ast(opers.side_2);
      arg2_ext = convert_sign_ext(arg2_ext, bigsort, sz - 1, sz);
    } else {
      // Zero extend the top parts
      arg1_ext = convert_ast(opers.side_1);
      arg1_ext = convert_zero_ext(arg1_ext, bigsort, sz);
      arg2_ext = convert_ast(opers.side_2);
      arg2_ext = convert_zero_ext(arg2_ext, bigsort, sz);
    }

    mulargs[0] = arg1_ext;
    mulargs[1] = arg2_ext;
    const smt_ast *result = mk_func_app(bigsort, SMT_FUNC_MUL, mulargs, 2);

    // Extract top half.
    const smt_ast *toppart = mk_extract(result, (sz * 2) - 1, sz, normalsort);

    if (is_signed) {
      // It should either be zero or all one's; which depends on what
      // configuration of signs it had. If both pos / both neg, then the top
      // should all be zeros, otherwise all ones. Implement with xor.
      args[0] = convert_ast(op1neg);
      args[1] = convert_ast(op2neg);
      const smt_ast *allonescond = mk_func_app(boolsort, SMT_FUNC_XOR, args, 2);
      const smt_ast *zerovector = convert_ast(zero);

      args[0] = allonescond;
      args[1] = allonesvector;
      args[2] = zerovector;
      args[2] = mk_func_app(normalsort, SMT_FUNC_ITE, args, 3);

      // either value being zero means the top must be zero.
      args[0] = convert_ast(containszero);
      args[1] = zerovector;
      args[0] = mk_func_app(normalsort, SMT_FUNC_ITE, args, 3);

      args[1] = toppart;
      args[0] = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
      return mk_func_app(boolsort, SMT_FUNC_NOT, args, 1);
    } else {
      // It should be zero; if not, overflow
      args[0] = toppart;
      args[1] = convert_ast(zero);
      args[0] = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
      return mk_func_app(boolsort, SMT_FUNC_NOT, args, 1);
    }
  }

  return NULL;
}

smt_ast *
smt_convt::overflow_cast(const expr2tc &expr)
{
  const overflow_cast2t &ocast = to_overflow_cast2t(expr);
  unsigned int width = ocast.operand->type->get_width();
  unsigned int bits = ocast.bits;
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);

  if (ocast.bits >= width || ocast.bits == 0) {
    std::cerr << "SMT conversion: overflow-typecast got wrong number of bits"
              << std::endl;
    abort();
  }

  // Basically: if it's positive in the first place, ensure all the top bits
  // are zero. If neg, then all the top are 1's /and/ the next bit, so that
  // it's considered negative in the next interpretation.

  constant_int2tc zero(ocast.operand->type, BigInt(0));
  lessthan2tc isnegexpr(ocast.operand, zero);
  const smt_ast *isneg = convert_ast(isnegexpr);
  const smt_ast *orig_val = convert_ast(ocast.operand);

  // Difference bits
  unsigned int pos_zero_bits = width - bits;
  unsigned int neg_one_bits = (width - bits) + 1;

  const smt_sort *pos_zero_bits_sort =
    mk_sort(SMT_SORT_BV, pos_zero_bits, false);
  const smt_sort *neg_one_bits_sort =
    mk_sort(SMT_SORT_BV, neg_one_bits, false);

  const smt_ast *pos_bits = mk_smt_bvint(BigInt(0), false, pos_zero_bits);
  const smt_ast *neg_bits = mk_smt_bvint(BigInt((1 << neg_one_bits) - 1),
                                         false, neg_one_bits);

  const smt_ast *pos_sel = mk_extract(orig_val, width - 1,
                                      width - pos_zero_bits,
                                      pos_zero_bits_sort);
  const smt_ast *neg_sel = mk_extract(orig_val, width - 1,
                                      width - neg_one_bits,
                                      neg_one_bits_sort);

  const smt_ast *args[2];
  args[0] = pos_bits;
  args[1] = pos_sel;
  const smt_ast *pos_eq = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
  args[0] = neg_bits;
  args[1] = neg_sel;
  const smt_ast *neg_eq = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);

  // isneg -> neg_eq, !isneg -> pos_eq
  const smt_ast *notisneg = mk_func_app(boolsort, SMT_FUNC_NOT, &isneg, 1);
  args[0] = isneg;
  args[1] = neg_eq;
  const smt_ast *c1 = mk_func_app(boolsort, SMT_FUNC_IMPLIES, args, 2);
  args[0] = notisneg;
  args[1] = pos_eq;
  const smt_ast *c2 = mk_func_app(boolsort, SMT_FUNC_IMPLIES, args, 2);

  args[0] = c1;
  args[1] = c2;
  const smt_ast *nooverflow = mk_func_app(boolsort, SMT_FUNC_AND, args, 2);
  return mk_func_app(boolsort, SMT_FUNC_NOT, &nooverflow, 1);
}

const smt_ast *
smt_convt::overflow_neg(const expr2tc &expr)
{
  // Single failure mode: MIN_INT can't be neg'd
  const overflow_neg2t &neg = to_overflow_neg2t(expr);
  unsigned int width = neg.operand->type->get_width();

  constant_int2tc min_int(neg.operand->type, BigInt(1 << (width - 1)));
  equality2tc val(neg.operand, min_int);
  return convert_ast(val);
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
    const smt_sort *tmpsort = mk_sort(SMT_SORT_INT, false);
    args[0] = mk_func_app(tmpsort, SMT_FUNC_REAL2INT, &operand, 1);
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
smt_convt::convert_byte_extract(const expr2tc &expr)
{
  const byte_extract2t &data = to_byte_extract2t(expr);

  if (!is_constant_int2t(data.source_offset)) {
    std::cerr << "byte_extract expects constant 2nd arg";
    abort();
  }

  const constant_int2t &intref = to_constant_int2t(data.source_offset);

  unsigned width;
  width = data.source_value->type->get_width();
  // XXXjmorse - looks like this only ever reads a single byte, not the desired
  // number of bytes to fill the type.

  uint64_t upper, lower;
  if (!data.big_endian) {
    upper = ((intref.constant_value.to_long() + 1) * 8) - 1; //((i+1)*w)-1;
    lower = intref.constant_value.to_long() * 8; //i*w;
  } else {
    uint64_t max = width - 1;
    upper = max - (intref.constant_value.to_long() * 8); //max-(i*w);
    lower = max - ((intref.constant_value.to_long() + 1) * 8 - 1); //max-((i+1)*w-1);
  }

  const smt_ast *source = convert_ast(data.source_value);;

  if (int_encoding) {
    std::cerr << "Refusing to byte extract in integer mode; re-run in "
                 "bitvector mode" << std::endl;
    abort();
  } else {
    if (is_struct_type(data.source_value)) {
      const struct_type2t &struct_type =to_struct_type(data.source_value->type);
      unsigned i = 0, num_elems = struct_type.members.size();
      const smt_ast *struct_elem[num_elems + 1], *struct_elem_inv[num_elems +1];

      forall_types(it, struct_type.members) {
        struct_elem[i] = tuple_project(source, convert_sort(*it), i);
        i++;
      }

      for (unsigned k = 0; k < num_elems; k++)
        struct_elem_inv[(num_elems - 1) - k] = struct_elem[k];

      // Concat into one massive vector.
      const smt_ast *args[2];
      for (unsigned k = 0; k < num_elems; k++)
      {
        if (k == 1) {
          args[0] = struct_elem_inv[k - 1];
          args[1] = struct_elem_inv[k];
          // FIXME: sorts
          struct_elem_inv[num_elems] = mk_func_app(NULL, SMT_FUNC_CONCAT, args,
                                                   2);
        } else if (k > 1) {
          args[0] = struct_elem_inv[num_elems];
          args[1] = struct_elem_inv[k];
          // FIXME: sorts
          struct_elem_inv[num_elems] = mk_func_app(NULL, SMT_FUNC_CONCAT, args,
                                                   2);
        }
      }

      source = struct_elem_inv[num_elems];
    }

    unsigned int sort_sz = data.source_value->type->get_width();
    if (sort_sz < upper) {
      // Extends past the end of this data item. Should be fixed in some other
      // dedicated feature branch, in the meantime stop Z3 from crashing
      const smt_sort *s = mk_sort(SMT_SORT_BV, 8, false);
      return mk_smt_symbol("out_of_bounds_byte_extract", s);
    } else {
      return mk_extract(source, upper, lower, convert_sort(expr->type));
    }
  }

  std::cerr << "Unsupported byte extract operand" << std::endl;
  abort();
}

const smt_ast *
smt_convt::convert_byte_update(const expr2tc &expr)
{
  const byte_update2t &data = to_byte_update2t(expr);

  // op0 is the object to update
  // op1 is the byte number
  // op2 is the value to update with

  if (!is_constant_int2t(data.source_offset)) {
    std::cerr << "byte_extract expects constant 2nd arg";
    abort();
  }

  const constant_int2t &intref = to_constant_int2t(data.source_offset);

  const smt_ast *tuple, *value;
  uint width_op0, width_op2;

  tuple = convert_ast(data.source_value);
  value = convert_ast(data.update_value);

  width_op2 = data.update_value->type->get_width();

  if (int_encoding) {
    std::cerr << "Can't byte update in integer mode; rerun in bitvector mode"
              << std::endl;
    abort();
  }

  if (is_struct_type(data.source_value)) {
    const struct_type2t &struct_type = to_struct_type(data.source_value->type);
    bool has_field = false;

    // XXXjmorse, this isn't going to be the case if it's a with.

    forall_types(it, struct_type.members) {
      width_op0 = (*it)->get_width();

      if (((*it)->type_id == data.update_value->type->type_id) &&
          (width_op0 == width_op2))
	has_field = true;
    }

    if (has_field)
      return tuple_update(tuple, intref.constant_value.to_long(), value);
    else
      return tuple;
  } else if (is_signedbv_type(data.source_value->type)) {
    width_op0 = data.source_value->type->get_width();

    if (width_op0 == 0) {
      // XXXjmorse - can this ever happen now?
      std::cerr << "failed to get width of byte_update operand";
      abort();
    }

    if (width_op0 > width_op2) {
      return convert_sign_ext(value, convert_sort(expr->type), width_op2,
                              width_op0 - width_op2);
    } else {
      std::cerr << "unsupported irep for conver_byte_update" << std::endl;
      abort();
    }
  }

  std::cerr << "unsupported irep for convert_byte_update" << std::endl;;
  abort();
}

const smt_ast *
smt_convt::convert_ptr_cmp(const expr2tc &side1, const expr2tc &side2,
                           const expr2tc &templ_expr)
{
  // Special handling for pointer comparisons (both ops are pointers; otherwise
  // it's obviously broken). First perform a test as to whether or not the
  // pointer locations are greater or lower; and only involve the ptr offset
  // if the ptr objs are the same.
  type2tc int_type = get_uint_type(config.ansi_c.int_width);

  pointer_object2tc ptr_obj1(int_type, side1);
  pointer_offset2tc ptr_offs1(int_type, side1);
  pointer_object2tc ptr_obj2(int_type, side2);
  pointer_offset2tc ptr_offs2(int_type, side2);

  // Don't ask
  std::vector<type2tc> members;
  std::vector<irep_idt> names;
  members.push_back(int_type);
  members.push_back(int_type);
  names.push_back(irep_idt("start"));
  names.push_back(irep_idt("end"));
  type2tc strct(new struct_type2t(members, names,
                irep_idt("addr_space_tuple")));
  type2tc addrspace_type(new array_type2t(strct, expr2tc((expr2t*)NULL), true));

  symbol2tc addrspacesym(addrspace_type, get_cur_addrspace_ident());
  index2tc obj1_data(strct, addrspacesym, ptr_obj1);
  index2tc obj2_data(strct, addrspacesym, ptr_obj2);

  member2tc obj1_start(int_type, obj1_data, irep_idt("start"));
  member2tc obj2_start(int_type, obj2_data, irep_idt("start"));

  expr2tc start_expr = templ_expr, offs_expr = templ_expr;

  // To ensure we can do this in an operation independant way, we're going to
  // clone the original comparison expression, and replace its operands with
  // new values. Works whatever the expr is, so long as it has two operands.
  *start_expr.get()->get_sub_expr_nc(0) = obj1_start;
  *start_expr.get()->get_sub_expr_nc(1) = obj2_start;
  *offs_expr.get()->get_sub_expr_nc(0) = ptr_offs1;
  *offs_expr.get()->get_sub_expr_nc(1) = ptr_offs2;

  // Those are now boolean type'd relations.
  equality2tc is_same_obj_expr(ptr_obj1, ptr_obj2);

  if2tc res(offs_expr->type, is_same_obj_expr, offs_expr, start_expr);
  return convert_ast(res);
}

const smt_ast *
smt_convt::convert_pointer_arith(const expr2tc &expr, const type2tc &type)
{
  const arith_2ops &expr_ref = static_cast<const arith_2ops &>(*expr);
  const expr2tc &side1 = expr_ref.side_1;
  const expr2tc &side2 = expr_ref.side_2;

  // So eight cases; one for each combination of two operands and the return
  // type, being pointer or nonpointer. So with P=pointer, N= notpointer,
  //    return    op1        op2        action
  //      N        N          N         Will never be fed here
  //      N        P          N         Expected arith option, then cast to int
  //      N        N          P            "
  //      N        P          P         Not permitted by C spec
  //      P        N          N         Return arith action with cast to pointer
  //      P        P          N         Calculate expected ptr arith operation
  //      P        N          P            "
  //      P        P          P         Not permitted by C spec
  //      NPP is the most dangerous - there's the possibility that an integer
  //      arithmatic is going to lead to an invalid pointer, that falls out of
  //      all dereference switch cases. So, we need to verify that all derefs
  //      have a finally case that asserts the val was a valid ptr XXXjmorse.
  int ret_is_ptr, op1_is_ptr, op2_is_ptr;
  ret_is_ptr = (is_pointer_type(type)) ? 4 : 0;
  op1_is_ptr = (is_pointer_type(side1)) ? 2 : 0;
  op2_is_ptr = (is_pointer_type(side2)) ? 1 : 0;

  switch (ret_is_ptr | op1_is_ptr | op2_is_ptr) {
    case 0:
      assert(false);
      break;
    case 3:
    case 7:
      assert(0 && "Pointer arithmetic with two pointer operands");
      break;
    case 4:
      // Artithmetic operation that has the result type of ptr.
      // Should have been handled at a higher level
      assert(0 && "Non-pointer op being interpreted as pointer without cast");
      break;
    case 1:
    case 2:
      { // Block required to give a variable lifetime to the cast/add variables
      expr2tc ptr_op = (op1_is_ptr) ? side1 : side2;
      expr2tc non_ptr_op = (op1_is_ptr) ? side2 : side1;

      add2tc add(ptr_op->type, ptr_op, non_ptr_op);
      // That'll generate the correct pointer arithmatic; now typecast
      typecast2tc cast(type, add);
      return convert_ast(cast);
      }
    case 5:
    case 6:
      {
      expr2tc ptr_op = (op1_is_ptr) ? side1 : side2;
      expr2tc non_ptr_op = (op1_is_ptr) ? side2 : side1;

      // Actually perform some pointer arith
      const pointer_type2t &ptr_type = to_pointer_type(ptr_op->type);
      typet followed_type_old = ns.follow(migrate_type_back(ptr_type.subtype));
      type2tc followed_type;
      migrate_type(followed_type_old, followed_type);
      mp_integer type_size = pointer_offset_size(*followed_type);

      // Generate nonptr * constant.
      type2tc inttype(new unsignedbv_type2t(config.ansi_c.int_width));
      constant_int2tc constant(get_uint_type(32), type_size);
      expr2tc mul = mul2tc(inttype, non_ptr_op, constant);

      // Add or sub that value
      expr2tc ptr_offset = pointer_offset2tc(inttype, ptr_op);

      expr2tc newexpr;
      if (is_add2t(expr)) {
        newexpr = add2tc(inttype, mul, ptr_offset);
      } else {
        // Preserve order for subtraction.
        expr2tc tmp_op1 = (op1_is_ptr) ? ptr_offset : mul;
        expr2tc tmp_op2 = (op1_is_ptr) ? mul : ptr_offset;
        newexpr = sub2tc(inttype, tmp_op1, tmp_op2);
      }

      // Voila, we have our pointer arithmatic
      const smt_ast *a = convert_ast(newexpr);
      const smt_ast *the_ptr = convert_ast(ptr_op);

      // That calculated the offset; update field in pointer.
      return tuple_update(the_ptr, 1, a);
      }
  }

  assert(0 && "Fell through convert_pointer_logic");
}

const smt_ast *
smt_convt::convert_identifier_pointer(const expr2tc &expr, std::string symbol)
{
  const smt_ast *a;
  const smt_sort *s;
  std::string cte, identifier;
  unsigned int obj_num;
  bool got_obj_num = false;

  if (is_symbol2t(expr)) {
    const symbol2t &sym = to_symbol2t(expr);
    if (sym.thename == "NULL" || sym.thename == "0") {
      obj_num = pointer_logic.back().get_null_object();
      got_obj_num = true;
    }
  }

  if (!got_obj_num)
    // add object won't duplicate objs for identical exprs (it's a map)
    obj_num = pointer_logic.back().add_object(expr);

  s = convert_sort(pointer_struct);
  if (!tuple_support)
    a = new tuple_smt_ast(s, symbol);
  else
    a = mk_smt_symbol(symbol, s);

  // If this object hasn't yet been put in the address space record, we need to
  // assert that the symbol has the object ID we've allocated, and then fill out
  // the address space record.
  if (addr_space_data.back().find(obj_num) == addr_space_data.back().end()) {

    std::vector<expr2tc> membs;
    membs.push_back(gen_uint(obj_num));
    membs.push_back(zero_uint);
    constant_struct2tc ptr_val_s(pointer_struct, membs);
    const smt_ast *ptr_val = tuple_create(ptr_val_s);
    const smt_ast *constraint = tuple_equality(a, ptr_val);
    literalt l = mk_lit(constraint);
    assert_lit(l);

    type2tc ptr_loc_type(new unsignedbv_type2t(config.ansi_c.int_width));

    std::stringstream sse1, sse2;
    sse1 << "__ESBMC_ptr_obj_start_" << obj_num;
    sse2 << "__ESBMC_ptr_obj_end_" << obj_num;
    std::string start_name = sse1.str();
    std::string end_name = sse2.str();

    symbol2tc start_sym(ptr_loc_type, start_name);
    symbol2tc end_sym(ptr_loc_type, end_name);

    // Another thing to note is that the end var must be /the size of the obj/
    // from start. Express this in irep.
    expr2tc endisequal;
    try {
      uint64_t type_size = expr->type->get_width() / 8;
      constant_int2tc const_offs(ptr_loc_type, BigInt(type_size));
      add2tc start_plus_offs(ptr_loc_type, start_sym, const_offs);
      endisequal = equality2tc(start_plus_offs, end_sym);
    } catch (array_type2t::dyn_sized_array_excp *e) {
      // Dynamically (nondet) sized array; take that size and use it for the
      // offset-to-end expression.
      const expr2tc size_expr = e->size;
      add2tc start_plus_offs(ptr_loc_type, start_sym, size_expr);
      endisequal = equality2tc(start_plus_offs, end_sym);
    } catch (type2t::symbolic_type_excp *e) {
      // Type is empty or code -- something that we can never have a real size
      // for. In that case, create an object of size 1: this means we have a
      // valid entry in the address map, but that any modification of the
      // pointer leads to invalidness, because there's no size to think about.
      constant_int2tc const_offs(ptr_loc_type, BigInt(1));
      add2tc start_plus_offs(ptr_loc_type, start_sym, const_offs);
      endisequal = equality2tc(start_plus_offs, end_sym);
    }

    // Assert that start + offs == end
    assert_expr(endisequal);

    // Even better, if we're operating in bitvector mode, it's possible that
    // the solver will try to be clever and arrange the pointer range to cross
    // the end of the address space (ie, wrap around). So, also assert that
    // end > start
    greaterthan2tc wraparound(end_sym, start_sym);
    assert_expr(wraparound);

    // Generate address space layout constraints.
    finalize_pointer_chain(obj_num);

    addr_space_data.back()[obj_num] =
          pointer_offset_size(*expr->type.get()).to_long() + 1;

    membs.clear();
    membs.push_back(start_sym);
    membs.push_back(end_sym);
    constant_struct2tc range_struct(addr_space_type, membs);
    std::stringstream ss;
    ss << "__ESBMC_ptr_addr_range_" <<  obj_num;
    symbol2tc range_sym(addr_space_type, ss.str());
    equality2tc eq(range_sym, range_struct);
    assert_expr(eq);

    // Update array
    bump_addrspace_array(obj_num, range_struct);

    // Finally, ensure that the array storing whether this pointer is dynamic,
    // is initialized for this ptr to false. That way, only pointers created
    // through malloc will be marked dynamic.

    type2tc arrtype(new array_type2t(type2tc(new bool_type2t()),
                                     expr2tc((expr2t*)NULL), true));
    symbol2tc allocarr(arrtype, dyn_info_arr_name);
    constant_int2tc objid(get_uint_type(config.ansi_c.int_width),
                          BigInt(obj_num));
    index2tc idx(get_bool_type(), allocarr, objid);
    equality2tc dyn_eq(idx, false_expr);
    assert_expr(dyn_eq);
  }

  return a;
}

void
smt_convt::finalize_pointer_chain(unsigned int objnum)
{
  type2tc inttype = get_uint_type(config.ansi_c.int_width);
  unsigned int num_ptrs = addr_space_data.back().size();
  if (num_ptrs == 0)
    return;

  std::stringstream start1, end1;
  start1 << "__ESBMC_ptr_obj_start_" << objnum;
  end1 << "__ESBMC_ptr_obj_end_" << objnum;
  symbol2tc start_i(inttype, start1.str());
  symbol2tc end_i(inttype, end1.str());

  for (unsigned int j = 0; j < objnum; j++) {
    // Obj1 is designed to overlap
    if (j == 1)
      continue;

    std::stringstream startj, endj;
    startj << "__ESBMC_ptr_obj_start_" << j;
    endj << "__ESBMC_ptr_obj_end_" << j;
    symbol2tc start_j(inttype, startj.str());
    symbol2tc end_j(inttype, endj.str());

    // Formula: (i_end < j_start) || (i_start > j_end)
    // Previous assertions ensure start < end for all objs.
    lessthan2tc lt1(end_i, start_j);
    greaterthan2tc gt1(start_i, end_j);
    or2tc or1(lt1, gt1);
    assert_expr(or1);
  }

  return;
}

const smt_ast *
smt_convt::convert_addr_of(const expr2tc &expr)
{
  const address_of2t &obj = to_address_of2t(expr);

  std::string symbol_name, out;

  if (is_index2t(obj.ptr_obj)) {
    const index2t &idx = to_index2t(obj.ptr_obj);

    if (!is_string_type(idx.source_value)) {
      const array_type2t &arr = to_array_type(idx.source_value->type);

      // Pick pointer-to array subtype; need to make pointer arith work.
      address_of2tc addrof(arr.subtype, idx.source_value);
      add2tc plus(addrof->type, addrof, idx.index);
      return convert_ast(plus);
    } else {
      // Strings; convert with slightly different types.
      type2tc stringtype(new unsignedbv_type2t(8));
      address_of2tc addrof(stringtype, idx.source_value);
      add2tc plus(addrof->type, addrof, idx.index);
      return convert_ast(plus);
    }
  } else if (is_member2t(obj.ptr_obj)) {
    const member2t &memb = to_member2t(obj.ptr_obj);

    int64_t offs;
    if (is_struct_type(memb.source_value)) {
      const struct_type2t &type = to_struct_type(memb.source_value->type);
      offs = member_offset(type, memb.member).to_long();
    } else {
      offs = 0; // Offset is always zero for unions.
    }

    address_of2tc addr(type2tc(new pointer_type2t(memb.source_value->type)),
                       memb.source_value);

    const smt_ast *a = convert_ast(addr);

    // Update pointer offset to offset to that field.
    constant_int2tc offset(get_int_type(config.ansi_c.int_width), BigInt(offs));
    const smt_ast *o = convert_ast(offset);
    return tuple_update(a, 1, o);
  } else if (is_symbol2t(obj.ptr_obj)) {
// XXXjmorse             obj.ptr_obj->expr_id == expr2t::code_id) {

    const symbol2t &symbol = to_symbol2t(obj.ptr_obj);
    return convert_identifier_pointer(obj.ptr_obj, symbol.get_symbol_name());
  } else if (is_constant_string2t(obj.ptr_obj)) {
    // XXXjmorse - we should avoid encoding invalid characters in the symbol,
    // but this works for now.
    const constant_string2t &str = to_constant_string2t(obj.ptr_obj);
    std::string identifier =
      "address_of_str_const(" + str.value.as_string() + ")";
    return convert_identifier_pointer(obj.ptr_obj, identifier);
  } else if (is_if2t(obj.ptr_obj)) {
    // We can't nondeterministically take the address of something; So instead
    // rewrite this to be if (cond) ? &a : &b;.

    const if2t &ifval = to_if2t(obj.ptr_obj);

    address_of2tc addrof1(obj.type, ifval.true_value);
    address_of2tc addrof2(obj.type, ifval.false_value);
    if2tc newif(obj.type, ifval.cond, addrof1, addrof2);
    return convert_ast(newif);
  } else if (is_typecast2t(obj.ptr_obj)) {
    // Take the address of whatevers being casted. Either way, they all end up
    // being of a pointer_tuple type, so this should be fine.
    address_of2tc tmp(type2tc(), to_typecast2t(obj.ptr_obj).from);
    tmp.get()->type = obj.type;
    return convert_ast(tmp);
  }

  assert(0 && "Unrecognized address_of operand");
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

void
smt_convt::init_addr_space_array(void)
{
  addr_space_sym_num.back() = 1;

  type2tc ptr_int_type = get_uint_type(config.ansi_c.pointer_width);
  symbol2tc obj0_start(ptr_int_type, "__ESBMC_ptr_obj_start_0");
  symbol2tc obj0_end(ptr_int_type, "__ESBMC_ptr_obj_end_0");
  equality2tc obj0_start_eq(obj0_start, zero_uint);
  equality2tc obj0_end_eq(obj0_start, zero_uint);

  assert_expr(obj0_start_eq);
  assert_expr(obj0_end_eq);

  symbol2tc obj1_start(ptr_int_type, "__ESBMC_ptr_obj_start_1");
  symbol2tc obj1_end(ptr_int_type, "__ESBMC_ptr_obj_end_1");
  constant_int2tc obj1_end_const(ptr_int_type, BigInt(0xFFFFFFFFFFFFFFFFULL));
  equality2tc obj1_start_eq(obj1_start, one_uint);
  equality2tc obj1_end_eq(obj1_end, obj1_end_const);

  assert_expr(obj1_start_eq);
  assert_expr(obj1_end_eq);

  std::vector<expr2tc> membs;
  membs.push_back(obj0_start);
  membs.push_back(obj0_end);
  constant_struct2tc addr0_tuple(addr_space_type, membs);
  symbol2tc addr0_range(addr_space_type, "__ESBMC_ptr_addr_range_0");
  equality2tc addr0_range_eq(addr0_tuple, addr0_range);
  assert_expr(addr0_range_eq);

  membs.clear();
  membs.push_back(obj1_start);
  membs.push_back(obj1_end);
  constant_struct2tc addr1_tuple(addr_space_type, membs);
  symbol2tc addr1_range(addr_space_type, "__ESBMC_ptr_addr_range_1");
  equality2tc addr1_range_eq(addr1_tuple, addr1_range);
  assert_expr(addr1_range_eq);

  bump_addrspace_array(pointer_logic.back().get_null_object(), addr0_tuple);
  bump_addrspace_array(pointer_logic.back().get_invalid_object(), addr1_tuple);

  // Give value to '0', 'NULL', 'INVALID' symbols
  symbol2tc zero_ptr(pointer_struct, "0");
  symbol2tc null_ptr(pointer_struct, "NULL");
  symbol2tc invalid_ptr(pointer_struct, "INVALID");

  membs.clear();
  membs.push_back(zero_uint);
  membs.push_back(zero_uint);
  constant_struct2tc null_ptr_tuple(pointer_struct, membs);
  membs.clear();
  membs.push_back(one_uint);
  membs.push_back(zero_uint);
  constant_struct2tc invalid_ptr_tuple(pointer_struct, membs);

  equality2tc zero_eq(zero_ptr, null_ptr_tuple);
  equality2tc null_eq(null_ptr, null_ptr_tuple);
  equality2tc invalid_eq(invalid_ptr, invalid_ptr_tuple);

  assert_expr(zero_eq);
  assert_expr(null_eq);
  assert_expr(invalid_eq);

  addr_space_data.back()[0] = 0;
  addr_space_data.back()[1] = 0;
}

void
smt_convt::bump_addrspace_array(unsigned int idx, const expr2tc &val)
{
  std::stringstream ss, ss2;
  std::string str, new_str;
  type2tc ptr_int_type = get_uint_type(config.ansi_c.pointer_width);

  ss << "__ESBMC_addrspace_arr_" << addr_space_sym_num.back()++;
  symbol2tc oldname(addr_space_arr_type, ss.str());
  constant_int2tc ptr_idx(ptr_int_type, BigInt(idx));

  with2tc store(addr_space_arr_type, oldname, ptr_idx, val);
  ss2 << "__ESBMC_addrspace_arr_" << addr_space_sym_num.back();
  symbol2tc newname(addr_space_arr_type, ss2.str());
  equality2tc eq(newname, store);
  assert_expr(eq);
  return;
}

std::string
smt_convt::get_cur_addrspace_ident(void)
{
  std::stringstream ss;
  ss << "__ESBMC_addrspace_arr_" << addr_space_sym_num.back();
  return ss.str();
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
  const smt_ast *f = mk_smt_bvint(BigInt(0xFFFFFFFFFFFFFFFFULL), false,
                                  topwidth);

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
smt_convt::convert_typecast_bool(const typecast2t &cast)
{

  if (is_bv_type(cast.from)) {
    notequal2tc neq(cast.from, zero_uint);
    return convert_ast(neq);
  } else if (is_pointer_type(cast.from)) {
    // Convert to two casts.
    typecast2tc to_int(get_uint_type(config.ansi_c.pointer_width), cast.from);
    constant_int2tc zero(get_uint_type(config.ansi_c.pointer_width), BigInt(0));
    equality2tc as_bool(zero, to_int);
    return convert_ast(as_bool);
  } else {
    assert(0 && "Unimplemented bool typecast");
  }
}

const smt_ast *
smt_convt::convert_typecast_fixedbv_nonint(const expr2tc &expr)
{
  const smt_ast *args[4];
  const typecast2t &cast = to_typecast2t(expr);
  const fixedbv_type2t &fbvt = to_fixedbv_type(cast.type);
  unsigned to_fraction_bits = fbvt.width - fbvt.integer_bits;
  unsigned to_integer_bits = fbvt.integer_bits;

  if (is_pointer_type(cast.from)) {
    std::cerr << "Converting pointer to a float is unsupported" << std::endl;
    abort();
  }

  const smt_ast *a = convert_ast(cast.from);
  const smt_sort *s = convert_sort(cast.type);

  if (is_bv_type(cast.from)) {
    unsigned from_width = cast.from->type->get_width();

    if (from_width == to_integer_bits) {
      // Just concat fraction ozeros at the bottom
      args[0] = a;
    } else if (from_width > to_integer_bits) {
      const smt_sort *tmp = mk_sort(SMT_SORT_BV, from_width - to_integer_bits,
                                    false);
      args[0] = mk_extract(a, (from_width - 1), to_integer_bits, tmp);
    } else {
      assert(from_width < to_integer_bits);
      const smt_sort *tmp = mk_sort(SMT_SORT_BV, to_integer_bits, false);
      args[0] = convert_sign_ext(a, tmp, from_width,
                                 to_integer_bits - from_width);
    }

    // Make all zeros fraction bits
    args[1] = mk_smt_bvint(BigInt(0), false, to_fraction_bits);
    return mk_func_app(s, SMT_FUNC_CONCAT, args, 2);
  } else if (is_bool_type(cast.from)) {
    const smt_ast *args[3];
    const smt_sort *intsort;
    args[0] = a;
    args[1] = mk_smt_bvint(BigInt(0), false, to_integer_bits);
    args[2] = mk_smt_bvint(BigInt(1), false, to_integer_bits);
    intsort = mk_sort(SMT_SORT_BV, to_integer_bits, false);
    args[0] = mk_func_app(intsort, SMT_FUNC_ITE, args, 3);
    args[1] = mk_smt_bvint(BigInt(0), false, to_integer_bits);
    return mk_func_app(s, SMT_FUNC_CONCAT, args, 2);
  } else if (is_fixedbv_type(cast.from)) {
    // FIXME: conversion here for to_int_bits > from_int_bits is factually
    // broken, run 01_cbmc_Fixedbv8 with --no-simplify
    const smt_ast *magnitude, *fraction;

    const fixedbv_type2t &from_fbvt = to_fixedbv_type(cast.from->type);

    unsigned from_fraction_bits = from_fbvt.width - from_fbvt.integer_bits;
    unsigned from_integer_bits = from_fbvt.integer_bits;
    unsigned from_width = from_fbvt.width;

    if (to_integer_bits <= from_integer_bits) {
      const smt_sort *tmp_sort = mk_sort(SMT_SORT_BV, to_integer_bits, false);
      magnitude = mk_extract(a, (from_fraction_bits + to_integer_bits - 1),
                             from_fraction_bits, tmp_sort);
    } else   {
      assert(to_integer_bits > from_integer_bits);
      const smt_sort *tmp_sort = mk_sort(SMT_SORT_BV,
                                        from_width - from_fraction_bits, false);
      const smt_ast *ext = mk_extract(a, from_width - 1, from_fraction_bits,
                                      tmp_sort);

      tmp_sort = mk_sort(SMT_SORT_BV, (from_width - from_fraction_bits)
                                      + (to_integer_bits - from_integer_bits),
                                      false);
      magnitude = convert_sign_ext(ext, tmp_sort,
                                   from_width - from_fraction_bits,
                                   to_integer_bits - from_integer_bits);
    }

    if (to_fraction_bits <= from_fraction_bits) {
      const smt_sort *tmp_sort = mk_sort(SMT_SORT_BV, to_fraction_bits, false);
      fraction = mk_extract(a, from_fraction_bits - 1,
                            from_fraction_bits - to_fraction_bits, tmp_sort);
    } else {
      const smt_ast *args[2];
      assert(to_fraction_bits > from_fraction_bits);
      const smt_sort *tmp_sort = mk_sort(SMT_SORT_BV, from_fraction_bits,
                                         false);
      args[0] = mk_extract(a, from_fraction_bits -1, 0, tmp_sort);
      args[1] = mk_smt_bvint(BigInt(0), false,
                             to_fraction_bits - from_fraction_bits);

      tmp_sort = mk_sort(SMT_SORT_BV, to_fraction_bits, false);
      fraction = mk_func_app(tmp_sort, SMT_FUNC_CONCAT, args, 2);
    }

    const smt_ast *args[2];
    args[0] = magnitude;
    args[1] = fraction;
    return mk_func_app(s, SMT_FUNC_CONCAT, args, 2);
  }

  std::cerr << "unexpected typecast to fixedbv" << std::endl;
  abort();
}

const smt_ast *
smt_convt::convert_typecast_to_ints(const typecast2t &cast)
{
  unsigned to_width = cast.type->get_width();
  const smt_sort *s = convert_sort(cast.type);
  const smt_ast *a = convert_ast(cast.from);

  if (is_signedbv_type(cast.from) || is_fixedbv_type(cast.from)) {
    unsigned from_width = cast.from->type->get_width();

    if (from_width == to_width) {
      if (int_encoding && is_signedbv_type(cast.from) &&
               is_fixedbv_type(cast.type)) {
        return mk_func_app(s, SMT_FUNC_INT2REAL, &a, 1);
      } else if (int_encoding && is_fixedbv_type(cast.from) &&
               is_signedbv_type(cast.type)) {
        return mk_func_app(s, SMT_FUNC_REAL2INT, &a, 1);
      } else if (int_encoding && is_unsignedbv_type(cast.from) &&
                 is_signedbv_type(cast.type)) {
        // Unsigned -> Signed. Seeing how integer mode is an approximation,
        // just return the original value, and if it would have wrapped around,
        // too bad.
        return convert_ast(cast.from);
      } else if (int_encoding && is_signedbv_type(cast.from) &&
                 is_unsignedbv_type(cast.type)) {
        // XXX XXX XXX seriously rethink what this code attempts to do,
        // implementing something that tries to look like twos compliment.
        constant_int2tc maxint(cast.type, BigInt(0xFFFFFFFF));
        add2tc add(cast.type, maxint, cast.from);

        constant_int2tc zero(cast.from->type, BigInt(0));
        lessthan2tc lt(cast.from, zero);
        if2tc ite(cast.type, lt, add, cast.from);
        return convert_ast(ite);
      } else if (!int_encoding) {
        // Just return the bit representation. It's fffiiiiiiinnneeee.
        return convert_ast(cast.from);
      } else {
        std::cerr << "Unrecognized equal-width int typecast format" <<std::endl;
        abort();
      }
    } else if (from_width < to_width) {
      if (int_encoding &&
          ((is_fixedbv_type(cast.type) && is_signedbv_type(cast.from)))) {
        return mk_func_app(s, SMT_FUNC_INT2REAL, &a, 1);
      } else if (int_encoding) {
	return a; // output = output
      } else {
        return convert_sign_ext(a, s, from_width, (to_width - from_width));
      }
    } else if (from_width > to_width) {
      if (int_encoding &&
          ((is_signedbv_type(cast.from) && is_fixedbv_type(cast.type)))) {
        return mk_func_app(s, SMT_FUNC_INT2REAL, &a, 1);
      } else if (int_encoding &&
                (is_fixedbv_type(cast.from) && is_signedbv_type(cast.type))) {
        return mk_func_app(s, SMT_FUNC_REAL2INT, &a, 1);
      } else if (int_encoding) {
        return a; // output = output
      } else {
	if (!to_width)
          to_width = config.ansi_c.int_width;

        return mk_extract(a, to_width-1, 0, s);
      }
    }
  } else if (is_unsignedbv_type(cast.from)) {
    unsigned from_width = cast.from->type->get_width();

    if (from_width == to_width) {
      return a; // output = output
    } else if (from_width < to_width) {
      if (int_encoding) {
	return a; // output = output
      } else {
        return convert_zero_ext(a, s, (to_width - from_width));
      }
    } else if (from_width > to_width) {
      if (int_encoding) {
	return a; // output = output
      } else {
        return mk_extract(a, to_width - 1, 0, s);
      }
    }
  } else if (is_bool_type(cast.from)) {
    const smt_ast *zero, *one;
    unsigned width = cast.type->get_width();

    if (is_bv_type(cast.type)) {
      if (int_encoding) {
        zero = mk_smt_int(BigInt(0), false);
        one = mk_smt_int(BigInt(1), false);
      } else {
        zero = mk_smt_bvint(BigInt(0), false, width);
        one = mk_smt_bvint(BigInt(1), false, width);
      }
    } else if (is_fixedbv_type(cast.type)) {
      zero = mk_smt_real(BigInt(0));
      one = mk_smt_real(BigInt(1));
    } else {
      std::cerr << "Unexpected type in typecast of bool" << std::endl;
      abort();
    }

    const smt_ast *args[3];
    args[0] = a;
    args[1] = one;
    args[2] = zero;
    return mk_func_app(s, SMT_FUNC_ITE, args, 3);
  }

  std::cerr << "Unexpected type in int/ptr typecast" << std::endl;
  abort();
}

const smt_ast *
smt_convt::convert_typecast_to_ptr(const typecast2t &cast)
{

  // First, sanity check -- typecast from one kind of a pointer to another kind
  // is a simple operation. Check for that first.
  if (is_pointer_type(cast.from)) {
    return convert_ast(cast.from);
  }

  // Unpleasentness; we don't know what pointer this integer is going to
  // correspond to, and there's no way of telling statically, so we have
  // to enumerate all pointers it could point at. IE, all of them. Which
  // is expensive, but here we are.

  // First cast it to an unsignedbv
  type2tc int_type = get_uint_type(config.ansi_c.int_width);
  typecast2tc cast_to_unsigned(int_type, cast.from);
  expr2tc target = cast_to_unsigned;

  // Construct array for all possible object outcomes
  expr2tc is_in_range[addr_space_data.back().size()];
  expr2tc obj_ids[addr_space_data.back().size()];
  expr2tc obj_starts[addr_space_data.back().size()];

  std::map<unsigned,unsigned>::const_iterator it;
  unsigned int i;
  for (it = addr_space_data.back().begin(), i = 0;
       it != addr_space_data.back().end(); it++, i++)
  {
    unsigned id = it->first;
    obj_ids[i] = constant_int2tc(int_type, BigInt(id));

    std::stringstream ss1, ss2;
    ss1 << "__ESBMC_ptr_obj_start_" << id;
    symbol2tc ptr_start(int_type, ss1.str());
    ss2 << "__ESBMC_ptr_obj_end_" << id;
    symbol2tc ptr_end(int_type, ss2.str());

    obj_starts[i] = ptr_start;

    greaterthanequal2tc ge(target, ptr_start);
    lessthanequal2tc le(target, ptr_end);
    and2tc theand(ge, le);
    is_in_range[i] = theand;
  }

  // Generate a big ITE chain, selecing a particular pointer offset. A
  // significant question is what happens when it's neither; in which case I
  // suggest the ptr becomes invalid_object. However, this needs frontend
  // support to check for invalid_object after all dereferences XXXjmorse.

  // So, what's the default value going to be if it doesn't match any existing
  // pointers? Answer, it's going to be the invalid object identifier, but with
  // an offset that calculates to the integer address of this object.
  // That's so that we can store an invalid pointer in a pointer type, that
  // eventually can be converted back via some mechanism to a valid pointer.
  expr2tc id, offs;
  id = constant_int2tc(int_type, pointer_logic.back().get_invalid_object());

  // Calculate ptr offset - target minus start of invalid range, ie 1
  offs = sub2tc(int_type, target, one_uint);

  std::vector<expr2tc> membs;
  membs.push_back(id);
  membs.push_back(offs);
  expr2tc prev_in_chain = constant_struct2tc(pointer_struct, membs);

  // Now that big ite chain,
  for (i = 0; i < addr_space_data.back().size(); i++) {
    membs.clear();

    // Calculate ptr offset were it this
    offs = sub2tc(int_type, target, obj_starts[i]);

    membs.push_back(obj_ids[i]);
    membs.push_back(offs);
    constant_struct2tc selected_tuple(pointer_struct, membs);

    prev_in_chain = if2tc(pointer_struct, is_in_range[i],
                          selected_tuple, prev_in_chain);
  }

  // Finally, we're now at the point where prev_in_chain represents a pointer
  // object. Hurrah.
  return convert_ast(prev_in_chain);
}

const smt_ast *
smt_convt::convert_typecast_from_ptr(const typecast2t &cast)
{

  type2tc int_type(new unsignedbv_type2t(config.ansi_c.int_width));

  // The plan: index the object id -> address-space array and pick out the
  // start address, then add it to any additional pointer offset.

  pointer_object2tc obj_num(int_type, cast.from);

  symbol2tc addrspacesym(addr_space_arr_type, get_cur_addrspace_ident());
  index2tc idx(addr_space_type, addrspacesym, obj_num);

  // We've now grabbed the pointer struct, now get first element. Represent
  // as fetching the first element of the struct representation.
  member2tc memb(int_type, idx, addr_space_type_data->member_names[0]);

  pointer_offset2tc ptr_offs(int_type, cast.from);
  add2tc add(int_type, memb, ptr_offs);

  // Finally, replace typecast
  typecast2tc new_cast(cast.type, add);
  return convert_ast(new_cast);
}

const smt_ast *
smt_convt::convert_typecast_struct(const typecast2t &cast)
{

  const struct_type2t &struct_type_from = to_struct_type(cast.from->type);
  const struct_type2t &struct_type_to = to_struct_type(cast.type);

  u_int i = 0, i2 = 0;

  std::vector<type2tc> new_members;
  std::vector<irep_idt> new_names;
  new_members.reserve(struct_type_to.members.size());
  new_names.reserve(struct_type_to.members.size());

  i = 0;
  // This all goes to pot when we consider polymorphism, and in particular,
  // multiple inheritance. So, for normal structs, as usual check that each
  // field has a compatible type. But for classes, check that either they're
  // the same class, or the source is a subclass of the target type. If so,
  // we just select out the common fields, which drops any additional data in
  // the subclass.

  bool same_format = true;
  if (is_subclass_of(cast.from->type, cast.type, ns)) {
    same_format = false; // then we're fine
  } else if (struct_type_from.name == struct_type_to.name) {
    ; // Also fine
  } else {
    // Check that these two different structs have the same format.
    forall_types(it, struct_type_to.members) {
      if (!base_type_eq(struct_type_from.members[i], *it, ns)) {
        std::cerr << "Incompatible struct in cast-to-struct" << std::endl;
        abort();
      }

      i++;
    }
  }

  smt_sort *fresh_sort = convert_sort(cast.type);
  smt_ast *fresh = tuple_fresh(fresh_sort);
  const smt_ast *src_ast = convert_ast(cast.from);
  smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);

  if (same_format) {
    // Alas, Z3 considers field names as being part of the type, so we can't
    // just consider the source expression to be the casted expression.
    i2 = 0;
    forall_types(it, struct_type_to.members) {
      const smt_ast *args[2];
      smt_sort *this_sort = convert_sort(*it);
      args[0] = tuple_project(src_ast, this_sort, i2);
      args[1] = tuple_project(fresh, this_sort, i2);
      const smt_ast *eq;
      if (is_struct_type(*it) || is_union_type(*it) || is_pointer_type(*it))
        eq = tuple_equality(args[0], args[1]);
      else
        eq = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
      assert_lit(mk_lit(eq));
      i2++;
    }
  } else {
    // Due to inheritance, these structs don't have the same format. Therefore
    // we have to look up source fields by matching the field names between
    // structs, then using their index numbers construct equalities between
    // fields in the source value and a fresh value.
    i2 = 0;
    forall_names(it, struct_type_to.member_names) {
      // Linear search, yay :(
      unsigned int i3 = 0;
      forall_names(it2, struct_type_from.member_names) {
        if (*it == *it2)
          break;
        i3++;
      }

      assert(i3 != struct_type_from.member_names.size() &&
             "Superclass field doesn't exist in subclass during conversion "
             "cast");
      // Could assert that the types are the same, however Z3 is going to
      // complain mightily if we get it wrong.

      const smt_ast *args[2];
      const type2tc &thetype = struct_type_from.members[i3];
      smt_sort *this_sort = convert_sort(thetype);
      args[0] = tuple_project(src_ast, this_sort, i3);
      args[1] = tuple_project(fresh, this_sort, i2);

      const smt_ast *eq;
      if (is_struct_type(thetype) || is_union_type(thetype) ||
          is_pointer_type(thetype))
        eq = tuple_equality(args[0], args[1]);
      else
        eq = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
      assert_lit(mk_lit(eq));
      i2++;
    }
   }

  return fresh;
}

const smt_ast *
smt_convt::convert_typecast(const expr2tc &expr)
{

  const typecast2t &cast = to_typecast2t(expr);

  if (is_pointer_type(cast.type)) {
    return convert_typecast_to_ptr(cast);
  } else if (is_pointer_type(cast.from)) {
    return convert_typecast_from_ptr(cast);
  } else if (is_bool_type(cast.type)) {
    return convert_typecast_bool(cast);
  } else if (is_fixedbv_type(cast.type) && !int_encoding)      {
    return convert_typecast_fixedbv_nonint(expr);
  } else if (is_bv_type(cast.type) ||
             is_fixedbv_type(cast.type) ||
             is_pointer_type(cast.type)) {
    return convert_typecast_to_ints(cast);
  } else if (is_struct_type(cast.type))     {
    return convert_typecast_struct(cast);
  } else if (is_union_type(cast.type)) {
    if (base_type_eq(cast.type, cast.from->type, namespacet(contextt()))) {
      return convert_ast(cast.from); // No additional conversion required
    } else {
      std::cerr << "Can't typecast between unions" << std::endl;
      abort();
    }
  }

  // XXXjmorse -- what about all other types, eh?
  std::cerr << "Typecast for unexpected type" << std::endl;
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
{ SMT_FUNC_STORE, SMT_FUNC_STORE, SMT_FUNC_STORE, 3, SMT_SORT_ARRAY | SMT_SORT_ALLINTS },  //with
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //member
{ SMT_FUNC_SELECT, SMT_FUNC_SELECT, SMT_FUNC_SELECT, 2, SMT_SORT_ARRAY | SMT_SORT_INT | SMT_SORT_BV},  //index
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
  "pow"
};
