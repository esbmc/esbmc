#include <sstream>
#include <set>
#include <iomanip>

#include <base_type.h>
#include <arith_tools.h>
#include <c_types.h>

#include "smt_conv.h"
#include <solvers/prop/literal.h>

#include "smt_tuple_flat.h"

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

unsigned int
smt_convt::get_member_name_field(const type2tc &t, const irep_idt &name) const
{
  unsigned int idx = 0;
  const struct_union_data &data_ref = get_type_def(t);

  forall_names(it, data_ref.member_names) {
    if (*it == name)
      break;
    idx++;
  }
  assert(idx != data_ref.member_names.size() &&
         "Member name of with expr not found in struct type");

  return idx;
}

unsigned int
smt_convt::get_member_name_field(const type2tc &t, const expr2tc &name) const
{
  const constant_string2t &str = to_constant_string2t(name);
  return get_member_name_field(t, str.value);
}

smt_convt::smt_convt(bool intmode, const namespacet &_ns, bool is_cpp)
  : ctx_level(0), boolean_sort(NULL), int_encoding(intmode), ns(_ns)
{
  tuple_api = NULL;
  array_api = NULL;

  std::vector<type2tc> members;
  std::vector<irep_idt> names;

  members.push_back(type_pool.get_uint(config.ansi_c.pointer_width));
  members.push_back(type_pool.get_uint(config.ansi_c.pointer_width));
  names.push_back(irep_idt("pointer_object"));
  names.push_back(irep_idt("pointer_offset"));

  struct_type2t *tmp = new struct_type2t(members, names, "pointer_struct");
  pointer_type_data = tmp;
  pointer_struct = type2tc(tmp);

  pointer_logic.push_back(pointer_logict());

  addr_space_sym_num.push_back(0);

  renumber_map.push_back(renumber_mapt());

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

  ptr_foo_inited = false;
}

smt_convt::~smt_convt(void)
{
}

void
smt_convt::set_tuple_iface(tuple_iface *iface)
{
  assert(tuple_api == NULL && "set_tuple_iface should only be called once");
  tuple_api = iface;
}

void
smt_convt::set_array_iface(array_iface *iface)
{
  assert(array_api == NULL && "set_array_iface should only be called once");
  array_api = iface;
}

void
smt_convt::delete_all_asts()
{

  // Erase all the remaining asts in the live ast vector.
  for (smt_ast *ast : live_asts)
    delete ast;
  live_asts.clear();
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

  boolean_sort = mk_sort(SMT_SORT_BOOL);

  init_addr_space_array();

  if (int_encoding) {
    std::vector<expr2tc> power_array_data;
    uint64_t pow;
    unsigned int count = 0;
    type2tc powarr_elemt = get_uint_type(64);
    for (pow = 1ULL; count < 64; pow <<= 1, count++)
      power_array_data.push_back(constant_int2tc(powarr_elemt, BigInt(pow)));

    type2tc power_array_type(new array_type2t(powarr_elemt,
                                              gen_ulong(64), false));

    constant_array2tc power_array(power_array_type, power_array_data);
    int_shift_op_array = convert_ast(power_array);
  }

  ptr_foo_inited = true;
}

void
smt_convt::push_ctx(void)
{
  tuple_api->push_tuple_ctx();
  array_api->push_array_ctx();

  addr_space_data.push_back(addr_space_data.back());
  addr_space_sym_num.push_back(addr_space_sym_num.back());
  pointer_logic.push_back(pointer_logic.back());
  renumber_map.push_back(renumber_map.back());

  live_asts_sizes.push_back(live_asts.size());

  ctx_level++;
}

void
smt_convt::pop_ctx(void)
{

  // Erase everything in caches added in the current context level. Everything
  // before the push is going to disappear.
  smt_cachet::nth_index<1>::type &cache_numindex = smt_cache.get<1>();
  cache_numindex.erase(ctx_level);
  pointer_logic.pop_back();
  addr_space_sym_num.pop_back();
  addr_space_data.pop_back();
  renumber_map.pop_back();

  ctx_level--;

  // Go through all the asts created since the last push and delete them.

  for (unsigned int idx = live_asts_sizes.back(); idx < live_asts.size(); idx++)
    delete live_asts[idx];

  // And reset the storage back to that point.
  live_asts.resize(live_asts_sizes.back());
  live_asts_sizes.pop_back();

  array_api->pop_array_ctx();
  tuple_api->pop_tuple_ctx();
}

smt_astt
smt_convt::make_disjunct(const ast_vec &v)
{
  smt_astt args[v.size()];
  smt_astt result = NULL;
  unsigned int i = 0;

  // This is always true.
  if (v.size() == 0)
    return mk_smt_bool(true);

  // Slightly funky due to being morphed from lor:
  for (ast_vec::const_iterator it = v.begin(); it != v.end(); it++, i++) {
    args[i] = *it;
  }

  // Chain these.
  if (i > 1) {
    unsigned int j;
    smt_astt accuml = args[0];
    for (j = 1; j < i; j++) {
      accuml = mk_func_app(boolean_sort, SMT_FUNC_OR, accuml, args[j]);
    }
    result = accuml;
  } else {
    result = args[0];
  }

  return result;
}

smt_astt
smt_convt::make_conjunct(const ast_vec &v)
{
  smt_astt args[v.size()];
  smt_astt result;
  unsigned int i;

  assert(v.size() != 0);

  // Funky on account of conversion from land...
  for (i = 0; i < v.size(); i++) {
    args[i] = v[i];
  }

  // Chain these.
  if (i > 1) {
    unsigned int j;
    smt_astt accuml = args[0];
    for (j = 1; j < i; j++) {
      accuml = mk_func_app(boolean_sort, SMT_FUNC_AND, accuml, args[j]);
    }
    result = accuml;
  } else {
    result = args[0];
  }

  return result;
}

smt_astt
smt_convt::invert_ast(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_BOOL);
  return mk_func_app(a->sort, SMT_FUNC_NOT, a);
}

smt_astt
smt_convt::imply_ast(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return mk_func_app(a->sort, SMT_FUNC_IMPLIES, a, b);
}

void
smt_convt::set_to(const expr2tc &expr, bool value)
{

  smt_astt a = convert_ast(expr);
  if (value == false)
    a = invert_ast(a);
  assert_ast(a);
}

void
smt_convt::convert_assign(const expr2tc &expr)
{
  const equality2t &eq = to_equality2t(expr);
  smt_astt side1 = convert_ast(eq.side_1);
  smt_astt side2 = convert_ast(eq.side_2);
  side2->assign(this, side1);

  // Put that into the smt cache, thus preserving the assigned symbols value.
  // IMPORTANT: the cache is now a fundemental part of how some flatteners work,
  // in that one can chose to create a set of expressions and their ASTs, then
  // store them in the cache, rather than have a more sophisticated conversion.
  smt_cache_entryt e = { eq.side_1, side1, ctx_level };
  smt_cache.insert(e);
}

smt_astt
smt_convt::convert_ast(const expr2tc &expr)
{
  // Variable length array; constant array's and so forth can have hundreds
  // of fields.
  smt_astt args[expr->get_num_sub_exprs()];
  smt_sortt sort;
  smt_astt a;
  unsigned int used_sorts = 0;
  bool seen_signed_operand = false;
  bool make_ints_reals = false;
  bool special_cases = true;

  smt_cachet::const_iterator cache_result = smt_cache.find(expr);
  if (cache_result != smt_cache.end())
    return (cache_result->ast);

  // Second fail -- comparisons are turning up in 01_cbmc_abs1 that compare
  // ints and reals, which is invalid. So convert ints up to reals.
  if (int_encoding && expr->get_num_sub_exprs() >= 2 &&
                      (is_fixedbv_type((*expr->get_sub_expr(0))->type) ||
                       is_fixedbv_type((*expr->get_sub_expr(1))->type))) {
    make_ints_reals = true;
  }

  unsigned int i = 0;

  // FIXME: turn this into a lookup table
  if (is_constant_array2t(expr) || is_with2t(expr) || is_index2t(expr) ||
      is_address_of2t(expr) ||
      (is_equality2t(expr) && is_array_type(to_equality2t(expr).side_1->type)))
    // Nope; needs special handling
    goto nocvt;
  special_cases = false;

  // Convert /all the arguments/. Via magical delegates.
  expr->foreach_operand(
      [this, &args, &i, &used_sorts, &seen_signed_operand, make_ints_reals]
      (const expr2tc &e)
    {
    args[i] = convert_ast(e);

    if (make_ints_reals && args[i]->sort->id == SMT_SORT_INT) {
      args[i] = mk_func_app(mk_sort(SMT_SORT_REAL), SMT_FUNC_INT2REAL, args[i]);
    }

    used_sorts |= args[i]->sort->id;
    i++;
    if (is_signedbv_type(e) || is_fixedbv_type(e))
      seen_signed_operand = true;
    }
  );
nocvt:

  sort = convert_sort(expr->type);

  const expr_op_convert *cvt = &smt_convert_table[expr->expr_id];

  // Irritating special case: if we're selecting a bool out of an array, and
  // we're in QF_AUFBV mode, do special handling.
  if ((!int_encoding && is_index2t(expr) && is_bool_type(expr->type) &&
       !array_api->supports_bools_in_arrays) ||
       special_cases)
    goto expr_handle_table;

  if ((int_encoding && cvt->int_mode_func > SMT_FUNC_INVALID) ||
      (!int_encoding && cvt->bv_mode_func_signed > SMT_FUNC_INVALID)) {
    assert(cvt->args == i);
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
    a = tuple_api->tuple_create(expr);
    break;
  }
  case expr2t::constant_union_id:
    std::cerr << "Post-parse union literals are deprecated and broken, sorry";
    std::cerr << std::endl;
    abort();
  case expr2t::constant_array_id:
  case expr2t::constant_array_of_id:
  {
    const array_type2t &arr = to_array_type(expr->type);
    if (!array_api->can_init_infinite_arrays && arr.size_is_infinite) {
      // Don't honour inifinite sized array initializers. Modelling only.
      // If we have an array of tuples and no tuple support, use tuple_fresh.
      // Otherwise, mk_fresh.
      if (is_tuple_ast_type(arr.subtype))
        a = tuple_api->tuple_fresh(sort);
      else
        a = mk_fresh(sort, "inf_array",
                     convert_sort(get_flattened_array_subtype(expr->type)));
      break;
    }

    // Domain sort may be mesed with:
    smt_sortt domain;
    if (int_encoding) {
      domain = machine_int_sort;
    } else {
      domain = mk_sort(SMT_SORT_BV, calculate_array_domain_width(arr), false);
    }

    expr2tc flat_expr = expr;
    if (is_array_type(get_array_subtype(expr->type)) &&
        is_constant_array2t(expr))
      flat_expr = flatten_array_body(expr);

    if (is_struct_type(arr.subtype) || is_pointer_type(arr.subtype))
      a = tuple_array_create_despatch(flat_expr, domain);
    else
      a = array_create(flat_expr);
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
      smt_sortt s1 = convert_sort(mul.side_1->type);
      smt_sortt s2 = convert_sort(mul.side_2->type);
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
      smt_sortt s2 = convert_sort(div.side_2->type);

      args[1] = convert_sign_ext(args[1], s2, topbit2,fraction_bits);
      smt_astt zero = mk_smt_bvint(BigInt(0), false, fraction_bits);
      smt_astt op0 = args[0];
      args[0] = mk_func_app(s2, SMT_FUNC_CONCAT, op0, zero);

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
    a = convert_array_index(expr);
    break;
  }
  case expr2t::with_id:
  {
    const with2t &with = to_with2t(expr);

    // We reach here if we're with'ing a struct, not an array. Or a bool.
    if (is_struct_type(expr) || is_pointer_type(expr)) {
      unsigned int idx = get_member_name_field(expr->type, with.update_field);
      smt_astt srcval = convert_ast(with.source_value);

#ifndef NDEBUG
      const struct_union_data &data = get_type_def(with.type);
      assert(idx < data.members.size() && "Out of bounds with expression");
      // Base type eq examines pointer types to closely
      assert((base_type_eq(data.members[idx], with.update_value->type, ns) ||
              (is_pointer_type(data.members[idx]) && is_pointer_type(with.update_value)))
                && "Assigned tuple member has type mismatch");
#endif

      a = srcval->update(this, convert_ast(with.update_value), idx);
    } else {
      a = convert_array_store(expr);
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
    args[0] = args[0]->project(this, 0);
    args[1] = args[1]->project(this, 0);
    a = mk_func_app(sort, SMT_FUNC_EQ, &args[0], 2);
    break;
  }
  case expr2t::pointer_offset_id:
  {
    const pointer_offset2t &obj = to_pointer_offset2t(expr);
    // Potentially walk through some typecasts
    const expr2tc *ptr = &obj.ptr_obj;
    while (is_typecast2t(*ptr) && !is_pointer_type((*ptr)))
      ptr = &to_typecast2t(*ptr).from;

    args[0] = convert_ast(*ptr);
    a = args[0]->project(this, 1);
    break;
  }
  case expr2t::pointer_object_id:
  {
    const pointer_object2t &obj = to_pointer_object2t(expr);
    // Potentially walk through some typecasts
    const expr2tc *ptr = &obj.ptr_obj;
    while (is_typecast2t(*ptr) && !is_pointer_type((*ptr)))
      ptr = &to_typecast2t(*ptr).from;

    args[0] = convert_ast(*ptr);
    a = args[0]->project(this, 0);
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
    const if2t &if_ref = to_if2t(expr);
    args[0] = convert_ast(if_ref.cond);
    args[1] = convert_ast(if_ref.true_value);
    args[2] = convert_ast(if_ref.false_value);
    a = args[1]->ite(this, args[0], args[2]);
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
    smt_astt b = convert_ast(eq.side_1);
    smt_astt c = convert_ast(eq.side_2);
    a = b->eq(this, c);
    break;
  }
  case expr2t::shl_id:
  {
    const shl2t &shl = to_shl2t(expr);

    if (shl.side_1->type->get_width() != shl.side_2->type->get_width()) {
      // FIXME: frontend doesn't cast the second operand up to the width of
      // the first, which SMT does not enjoy.
      typecast2tc cast(shl.side_1->type, shl.side_2);
      args[1] = convert_ast(cast);
    }

    if (int_encoding) {
      // Raise 2^shift, then multiply first operand by that value. If it's
      // negative, what to do? FIXME.
      smt_astt powval = int_shift_op_array->select(this, shl.side_2);
      args[1] = powval;
      a = mk_func_app(sort, SMT_FUNC_MUL, &args[0], 2);
    } else {
      a = mk_func_app(sort, SMT_FUNC_BVSHL, &args[0], 2);
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
      smt_astt powval = int_shift_op_array->select(this, ashr.side_2);
      args[1] = powval;
      a = mk_func_app(sort, SMT_FUNC_DIV, &args[0], 2);
    } else {
      a = mk_func_app(sort, SMT_FUNC_BVASHR, &args[0], 2);
    }
    break;
  }
  case expr2t::lshr_id:
  {
    // Like ashr. Haven't got around to cleaning this up yet.
    const lshr2t &lshr = to_lshr2t(expr);

    if (lshr.side_1->type->get_width() != lshr.side_2->type->get_width()) {
      // FIXME: frontend doesn't cast the second operand up to the width of
      // the first, which SMT does not enjoy.
      typecast2tc cast(lshr.side_1->type, lshr.side_2);
      args[1] = convert_ast(cast);
    }

    if (int_encoding) {
      // Raise 2^shift, then divide first operand by that value. If it's
      // negative, I suspect the correct operation is to latch to -1,
      smt_astt powval = int_shift_op_array->select(this, lshr.side_2);
      args[1] = powval;
      a = mk_func_app(sort, SMT_FUNC_DIV, &args[0], 2);
    } else {
      a = mk_func_app(sort, SMT_FUNC_BVLSHR, &args[0], 2);
    }
    break;
  }
  case expr2t::notequal_id:
  {
    // Handle all kinds of structs by inverted equality. The only that's really
    // going to turn up is pointers though.
    a = args[0]->eq(this, args[1]);
    a = mk_func_app(sort, SMT_FUNC_NOT, &a, 1);
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
  case expr2t::concat_id:
  {
    assert(!int_encoding && "Concatonate encountered in integer mode; "
           "unimplemented (and funky)");
    const concat2t &cat = to_concat2t(expr);
    args[0] = convert_ast(cat.side_1);
    args[1] = convert_ast(cat.side_2);

    unsigned long accuml_side =
      cat.side_1->type->get_width() + cat.side_2->type->get_width();
    smt_sortt s = mk_sort(SMT_SORT_BV, accuml_side, false);
    a = mk_func_app(s, SMT_FUNC_CONCAT, args, 2);

    break;
  }
  default:
    std::cerr << "Couldn't convert expression in unrecognized format"
              << std::endl;
    expr->dump();
    abort();
  }

done:
  struct smt_cache_entryt entry = { expr, a, ctx_level };
  smt_cache.insert(entry);

  return a;
}

void
smt_convt::assert_expr(const expr2tc &e)
{
  assert_ast(convert_ast(e));
  return;
}

smt_sortt
smt_convt::convert_sort(const type2tc &type)
{

  smt_sort_cachet::const_iterator it = sort_cache.find(type);
  if (it != sort_cache.end()) {
    return it->second;
  }

  smt_sortt result = NULL;
  switch (type->type_id) {
  case type2t::bool_id:
    result = boolean_sort;
    break;
  case type2t::struct_id:
    result = tuple_api->mk_struct_sort(type);
    break;
  case type2t::code_id:
  case type2t::pointer_id:
    result = tuple_api->mk_struct_sort(pointer_struct);
    break;
  case type2t::unsignedbv_id:
    /* FALLTHROUGH */
  case type2t::signedbv_id:
  {
    unsigned int width = type->get_width();
    assert(width != 0);
    result = mk_int_bv_sort(width);
  }
  break;
  case type2t::fixedbv_id:
  {
    unsigned int width = type->get_width();
    if (int_encoding)
      result = mk_sort(SMT_SORT_REAL);
    else
      result = mk_sort(SMT_SORT_BV, width, false);
  }
  break;
  case type2t::string_id:
  {
    const string_type2t &str_type = to_string_type(type);
    constant_int2tc width(get_uint_type(config.ansi_c.int_width),
                          BigInt(str_type.width));
    type2tc new_type(new array_type2t(get_uint8_type(), width, false));
    result = convert_sort(new_type);
    break;
  }
  case type2t::array_id:
  {
    const array_type2t &arr = to_array_type(type);

    // Index arrays by the smallest integer required to represent its size.
    // Unless it's either infinite or dynamic in size, in which case use the
    // machine int size. Also, faff about if it's an array of arrays, extending
    // the domain.
    smt_sortt d = make_array_domain_sort(arr);

    // Determine the range if we have arrays of arrays.
    type2tc range = arr.subtype;
    while (is_array_type(range))
      range = to_array_type(range).subtype;

    if (is_tuple_ast_type(range)) {
      type2tc thetype = flatten_array_type(type);
      result = tuple_api->mk_struct_sort(thetype);
      break;
    }

    // Work around QF_AUFBV demanding arrays of bitvectors.
    smt_sortt r;
    if (is_bool_type(range) && !array_api->supports_bools_in_arrays) {
      r = mk_int_bv_sort(1);
    } else {
      r = convert_sort(range);
    }

    result = mk_sort(SMT_SORT_ARRAY, d, r);
    break;
  }
  case type2t::cpp_name_id:
  case type2t::symbol_id:
  case type2t::empty_id:
  default:
    std::cerr << "Unexpected type ID " << get_type_id(type);
    std::cerr << " reached SMT conversion" << std::endl;
    abort();
  }

  sort_cache.insert(smt_sort_cachet::value_type(type, result));
  return result;
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

smt_astt
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
    if (is_tuple_ast_type(expr)) {
      const symbol2t &sym = to_symbol2t(expr);
      return tuple_api->mk_tuple_symbol(sym.get_symbol_name(),
                                        convert_sort(sym.type));
    } else if (is_array_type(expr)) {
      // Determine the range if we have arrays of arrays.
      const array_type2t &arr = to_array_type(expr->type);
      type2tc range = arr.subtype;
      while (is_array_type(range))
        range = to_array_type(range).subtype;

      // If this is an array of structs, we have a tuple array sym.
      if (is_structure_type(range) || is_pointer_type(range)) {
        return tuple_api->mk_tuple_array_symbol(expr);
      } else {
        ; // continue onwards;
      }
    }

    // Just a normal symbol. Possibly an array symbol.
    const symbol2t &sym = to_symbol2t(expr);
    std::string name = sym.get_symbol_name();
    smt_sortt sort = convert_sort(sym.type);
    if (is_array_type(expr)) {
      smt_sortt subtype = convert_sort(get_flattened_array_subtype(sym.type));
      return array_api->mk_array_symbol(name, sort, subtype);
    } else {
      return mk_smt_symbol(name, sort);
    }
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

smt_astt
smt_convt::mk_fresh(smt_sortt s, const std::string &tag,
                    smt_sortt array_subtype)
{
  std::string newname = mk_fresh_name(tag);

  if (s->id == SMT_SORT_UNION || s->id == SMT_SORT_STRUCT) {
    return tuple_api->mk_tuple_symbol(newname, s);
  } else if (s->id == SMT_SORT_ARRAY) {
    assert(array_subtype != NULL && "Must call mk_fresh for arrays with a subtype");
    return array_api->mk_array_symbol(newname, s, array_subtype);
  } else {
    return mk_smt_symbol(newname, s);
  }
}

smt_astt
smt_convt::convert_is_nan(const expr2tc &expr, smt_astt operand)
{
  const isnan2t &isnan = to_isnan2t(expr);
  smt_sortt bs = boolean_sort;

  // Assumes operand is fixedbv.
  assert(is_fixedbv_type(isnan.value));
  unsigned width = isnan.value->type->get_width();

  smt_astt t = mk_smt_bool(true);
  smt_astt f = mk_smt_bool(false);

  if (int_encoding) {
    smt_astt asint = round_real_to_int(operand);
    smt_astt zero = mk_smt_int(BigInt(0), false);
    smt_astt gte = mk_func_app(bs, SMT_FUNC_GTE, asint, zero);
    return mk_func_app(bs, SMT_FUNC_ITE, gte, t, f);
  } else {
    smt_astt zero =  mk_smt_bvint(BigInt(0), false, width);
    smt_astt gte = mk_func_app(bs, SMT_FUNC_GTE, operand, zero);
    return mk_func_app(bs, SMT_FUNC_ITE, gte, t, f);
  }
}

smt_astt
smt_convt::convert_member(const expr2tc &expr, smt_astt src)
{
  const member2t &member = to_member2t(expr);
  unsigned int idx = -1;

  assert(is_struct_type(member.source_value) ||
         is_pointer_type(member.source_value));
  idx = get_member_name_field(member.source_value->type, member.member);

  return src->project(this, idx);
}

smt_astt
smt_convt::convert_sign_ext(smt_astt a, smt_sortt s,
                            unsigned int topbit, unsigned int topwidth)
{

  smt_sortt bit = mk_sort(SMT_SORT_BV, 1, false);
  smt_astt the_top_bit = mk_extract(a, topbit-1, topbit-1, bit);
  smt_astt zero_bit = mk_smt_bvint(BigInt(0), false, 1);
  smt_sortt b = boolean_sort;
  smt_astt t = mk_func_app(b, SMT_FUNC_EQ, the_top_bit, zero_bit);

  smt_astt z = mk_smt_bvint(BigInt(0), false, topwidth);

  // Calculate the exact value; SMTLIB text parsers don't like taking an
  // over-full integer literal.
  uint64_t big = 0xFFFFFFFFFFFFFFFFULL;
  unsigned int num_topbits = 64 - topwidth;
  big >>= num_topbits;
  BigInt big_int(big);
  smt_astt f = mk_smt_bvint(big_int, false, topwidth);

  smt_sortt topsort = mk_sort(SMT_SORT_BV, topwidth, false);
  smt_astt topbits = mk_func_app(topsort, SMT_FUNC_ITE, t, z, f);

  return mk_func_app(s, SMT_FUNC_CONCAT, topbits, a);
}

smt_astt
smt_convt::convert_zero_ext(smt_astt a, smt_sortt s,
                            unsigned int topwidth)
{

  smt_astt z = mk_smt_bvint(BigInt(0), false, topwidth);
  return mk_func_app(s, SMT_FUNC_CONCAT, z, a);
}

smt_astt
smt_convt::round_real_to_int(smt_astt a)
{
  // SMT truncates downwards; however C truncates towards zero, which is not
  // the same. (Technically, it's also platform dependant). To get around this,
  // add one to the result in all circumstances, except where the value was
  // already an integer.
  smt_sortt realsort = mk_sort(SMT_SORT_REAL);
  smt_sortt intsort = mk_sort(SMT_SORT_INT);
  smt_sortt boolsort = boolean_sort;
  smt_astt is_lt_zero = mk_func_app(realsort, SMT_FUNC_LT, a, mk_smt_real("0"));

  // The actual conversion
  smt_astt as_int = mk_func_app(intsort, SMT_FUNC_REAL2INT, a);

  smt_astt one = mk_smt_int(BigInt(1), false);
  smt_astt plus_one = mk_func_app(intsort, SMT_FUNC_ADD, one, as_int);

  // If it's an integer, just keep it's untruncated value.
  smt_astt is_int = mk_func_app(boolsort, SMT_FUNC_IS_INT, &a, 1);
  smt_astt selected = mk_func_app(intsort, SMT_FUNC_ITE, is_int, as_int, plus_one);

  // Switch on whether it's > or < 0.
  return mk_func_app(intsort, SMT_FUNC_ITE, is_lt_zero, selected, as_int);
}

smt_astt
smt_convt::round_fixedbv_to_int(smt_astt a, unsigned int fromwidth,
                                unsigned int towidth)
{
  // Perform C rounding: just truncate towards zero. Annoyingly, this isn't
  // that simple for negative numbers, because they're represented as a negative
  // integer _plus_ a positive fraction. So we need to round up if there's a
  // nonzero fraction, and not if there's not.
  unsigned int frac_width = fromwidth / 2;

  // Sorts
  smt_sortt bit = mk_sort(SMT_SORT_BV, 1, false);
  smt_sortt halfwidth = mk_sort(SMT_SORT_BV, frac_width, false);
  smt_sortt tosort = mk_sort(SMT_SORT_BV, towidth, false);
  smt_sortt boolsort = boolean_sort;

  // Determine whether the source is signed from its topmost bit.
  smt_astt is_neg_bit = mk_extract(a, fromwidth-1, fromwidth-1, bit);
  smt_astt true_bit = mk_smt_bvint(BigInt(1), false, 1);

  // Also collect data for dealing with the magnitude.
  smt_astt magnitude = mk_extract(a, fromwidth-1, frac_width, halfwidth);
  smt_astt intvalue = convert_sign_ext(magnitude, tosort, frac_width,
                                             frac_width);

  // Data for inspecting fraction part
  smt_astt frac_part = mk_extract(a, frac_width-1, 0, bit);
  smt_astt zero = mk_smt_bvint(BigInt(0), false, frac_width);
  smt_astt is_zero_frac = mk_func_app(boolsort, SMT_FUNC_EQ, frac_part, zero);

  // So, we have a base number (the magnitude), and need to decide whether to
  // round up or down. If it's positive, round down towards zero. If it's neg
  // and the fraction is zero, leave it, otherwise round towards zero.

  // We may need a value + 1.
  smt_astt one = mk_smt_bvint(BigInt(1), false, towidth);
  smt_astt intvalue_plus_one =
    mk_func_app(tosort, SMT_FUNC_BVADD, intvalue, one);

  smt_astt neg_val =
    mk_func_app(tosort, SMT_FUNC_ITE, is_zero_frac, intvalue,intvalue_plus_one);

  smt_astt is_neg = mk_func_app(boolsort, SMT_FUNC_EQ, true_bit, is_neg_bit);

  // final switch
  return mk_func_app(tosort, SMT_FUNC_ITE, is_neg, neg_val, intvalue);
}

smt_astt
smt_convt::make_bool_bit(smt_astt a)
{

  assert(a->sort->id == SMT_SORT_BOOL && "Wrong sort fed to "
         "smt_convt::make_bool_bit");
  smt_astt one = (int_encoding) ? mk_smt_int(BigInt(1), false)
                                : mk_smt_bvint(BigInt(1), false, 1);
  smt_astt zero = (int_encoding) ? mk_smt_int(BigInt(0), false)
                                 : mk_smt_bvint(BigInt(0), false, 1);
  return mk_func_app(one->sort, SMT_FUNC_ITE, a, one, zero);
}

smt_astt
smt_convt::make_bit_bool(smt_astt a)
{

  assert(((!int_encoding && a->sort->id == SMT_SORT_BV) ||
          (int_encoding && a->sort->id == SMT_SORT_INT)) &&
        "Wrong sort fed to " "smt_convt::make_bit_bool");

  smt_sortt boolsort = boolean_sort;
  smt_astt one = (int_encoding) ? mk_smt_int(BigInt(1), false)
                                : mk_smt_bvint(BigInt(1), false, 1);
  return mk_func_app(boolsort, SMT_FUNC_EQ, a, one);
}

expr2tc
smt_convt::fix_array_idx(const expr2tc &idx, const type2tc &arr_sort)
{
  if (int_encoding)
    return idx;

  smt_sortt s = convert_sort(arr_sort);
  unsigned int domain_width = s->domain_width;
  if (domain_width == config.ansi_c.int_width)
    return idx;

  // Otherwise, we need to extract the lower bits out of this.
  return typecast2tc(get_uint_type(domain_width), idx);
}

unsigned long
smt_convt::size_to_bit_width(unsigned long sz)
{
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
}

unsigned long
smt_convt::calculate_array_domain_width(const array_type2t &arr)
{
  // Index arrays by the smallest integer required to represent its size.
  // Unless it's either infinite or dynamic in size, in which case use the
  // machine word size.
  if (!is_nil_expr(arr.array_size) && is_constant_int2t(arr.array_size)) {
    constant_int2tc thesize = arr.array_size;
    return size_to_bit_width(thesize->constant_value.to_ulong());
  } else {
    return config.ansi_c.word_size;
  }
}

smt_sortt
smt_convt::make_array_domain_sort(const array_type2t &arr)
{

  // Start special casing if this is an array of arrays.
  if (!is_array_type(arr.subtype)) {
    // Normal array, work out what the domain sort is.
    unsigned int domain_width = calculate_array_domain_width(arr);
    return mk_int_bv_sort(domain_width);
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

type2tc
smt_convt::make_array_domain_sort_exp(const array_type2t &arr)
{

  // Start special casing if this is an array of arrays.
  if (!is_array_type(arr.subtype)) {
    // Normal array, work out what the domain sort is.
    if (int_encoding)
      return get_uint_type(config.ansi_c.int_width);
    else
      return get_uint_type(calculate_array_domain_width(arr));
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

    return get_uint_type(domwidth);
  }
}

expr2tc
smt_convt::array_domain_to_width(const type2tc &type)
{
  const unsignedbv_type2t &uint = to_unsignedbv_type(type);
  uint64_t sz = 1ULL << uint.width;
  return constant_int2tc(index_type2(), BigInt(sz));
}

expr2tc
smt_convt::twiddle_index_width(const expr2tc &expr, const type2tc &type)
{
  const array_type2t &arrtype = to_array_type(type);
  unsigned int width = calculate_array_domain_width(arrtype);
  typecast2tc t(type2tc(new unsignedbv_type2t(width)), expr);
  expr2tc tmp = t->simplify();
  if (is_nil_expr(tmp))
    return t;
  else
    return tmp;
}

expr2tc
smt_convt::decompose_select_chain(const expr2tc &expr, expr2tc &base)
{
  // So: some series of index exprs will occur here, with some symbol or
  // other expression at the bottom that's actually some symbol, or whatever.
  // So, extract all the indexes, and concat them, with the first (lowest)
  // index at the top, then descending.

  unsigned long accuml_size = 0;
  index2tc idx = expr;
  expr2tc output = twiddle_index_width(idx->index, idx->source_value->type);
  accuml_size += output->type->get_width();
  while (is_index2t(idx->source_value)) {
    idx = idx->source_value;
    expr2tc tmp = twiddle_index_width(idx->index, idx->source_value->type);
    accuml_size += tmp->type->get_width();
    output = concat2tc(get_uint_type(accuml_size), output, tmp);
  }

  // Give the caller the base array object / thing. So that it can actually
  // select out of the right piece of data.
  base = idx->source_value;
  return output;
}

expr2tc
smt_convt::decompose_store_chain(const expr2tc &expr, expr2tc &base)
{
  // Just like handle_select_chain, we have some kind of multidimensional
  // array, which we're representing as a single array with an extended domain,
  // and using different segments of the domain to represent different
  // dimensions of it. Concat all of the indexs into one index; also give the
  // caller the base object that this is being applied to.

  unsigned long accuml_size = 0;
  with2tc with = expr;
  expr2tc output = twiddle_index_width(with->update_field, with->type);
  accuml_size += output->type->get_width();
  while (is_with2t(with->update_value)) {
    with = with->update_value;
    expr2tc tmp = twiddle_index_width(with->update_field, with->type);
    accuml_size += tmp->type->get_width();

    // NB: order is reversed from indexes.
    output = concat2tc(get_uint_type(accuml_size), tmp, output);
  }

  // Give the caller the actual value we're updating with.
  base = with->update_value;
  return output;
}

smt_astt
smt_convt::convert_array_index(const expr2tc &expr)
{
  smt_astt a;
  const index2t &index = to_index2t(expr);
  expr2tc src_value = index.source_value;
  expr2tc newidx;

  if (is_index2t(index.source_value)) {
    newidx = decompose_select_chain(expr, src_value);
  } else {
    newidx = fix_array_idx(index.index, index.source_value->type);
  }

  expr2tc tmp_idx = newidx->simplify();
  if (!is_nil_expr(tmp_idx))
    newidx = tmp_idx;

  // Firstly, if it's a string, shortcircuit.
  if (is_string_type(index.source_value)) {
    smt_astt tmp = convert_ast(src_value);
    return tmp->select(this, newidx);
  }

  a = convert_ast(src_value);
  a = a->select(this, newidx);

  const array_type2t &arrtype = to_array_type(index.source_value->type);
  if (is_bool_type(arrtype.subtype) && !array_api->supports_bools_in_arrays) {
    return make_bit_bool(a);
  } else {
    return a;
  }
}

smt_astt
smt_convt::convert_array_store(const expr2tc &expr)
{
  const with2t &with = to_with2t(expr);
  expr2tc update_val = with.update_value;
  expr2tc newidx;

  if (is_array_type(with.type) &&
      is_array_type(to_array_type(with.type).subtype) &&
      is_with2t(with.update_value)) {
    newidx = decompose_store_chain(expr, update_val);
  } else {
    newidx = fix_array_idx(with.update_field, with.type);
  }

  expr2tc tmp_idx = newidx->simplify();
  if (!is_nil_expr(tmp_idx))
    newidx = tmp_idx;

  assert(is_array_type(expr->type));
  smt_astt src, update;
  const array_type2t &arrtype = to_array_type(expr->type);

  // Workaround for bools-in-arrays.
  if (is_bool_type(arrtype.subtype) && !array_api->supports_bools_in_arrays) {
    typecast2tc cast(get_uint_type(1), update_val);
    update = convert_ast(cast);
  } else {
    update = convert_ast(update_val);
  }

  src = convert_ast(with.source_value);
  return src->update(this, update, 0, newidx);
}

type2tc
smt_convt::flatten_array_type(const type2tc &type)
{
  unsigned long arrbits = 0;

  if (to_array_type(type).size_is_infinite)
    // Don't touch these
    return type;

  // Otherwise, accumulate sufficient domain bits to represent all dimensions
  // of the given array, in one dimension.
  type2tc type_rec = type;
  while (is_array_type(type_rec)) {
    arrbits += calculate_array_domain_width(to_array_type(type_rec));
    type_rec = to_array_type(type_rec).subtype;
  }

  // type_rec is now the base type.
  uint64_t arr_size = 1ULL << arrbits;
  constant_int2tc arr_size_expr(index_type2(), BigInt(arr_size));

  return type2tc(new array_type2t(type_rec, arr_size_expr, false));
}

expr2tc
smt_convt::flatten_array_body(const expr2tc &expr)
{
  assert(is_constant_array2t(expr));
  const constant_array2t &the_array = to_constant_array2t(expr);

#ifndef NDEBUG
  for (const auto &elem : the_array.datatype_members)
    // Must only contain constant arrays, for now. No indirection should be
    // expressable at this level.
    assert(is_constant_array2t(elem) && "Sub-member of constant array must be "
        "constant array");
#endif

  std::vector<expr2tc> sub_expr_list;
  for (const auto &elem : the_array.datatype_members) {
    expr2tc tmp_container;
    const constant_array2t *sub_array = &to_constant_array2t(elem);

    // Possibly flatten an inner layer
    if (is_array_type(get_array_subtype(elem->type))) {
      tmp_container = flatten_array_body(elem);
      sub_array = &to_constant_array2t(tmp_container);
    }

    sub_expr_list.insert(sub_expr_list.end(),
                         sub_array->datatype_members.begin(),
                         sub_array->datatype_members.end());
  }

  return constant_array2tc(expr->type, sub_expr_list);
}

type2tc
smt_convt::get_flattened_array_subtype(const type2tc &type)
{
  // Get the subtype of an array, ensuring that any intermediate arrays have
  // been flattened.

  type2tc type_rec = type;
  while (is_array_type(type_rec)) {
    type_rec = to_array_type(type_rec).subtype;
  }

  // type_rec is now the base type.
  return type_rec;
}

std::string
smt_convt::get_fixed_point(const unsigned width, std::string value) const
{
  std::string m, f, tmp;
  size_t found, size;
  double v, magnitude, fraction, expoent;

  found = value.find_first_of("/");
  size = value.size();
  m = value.substr(0, found);
  if (found != std::string::npos)
    f = value.substr(found + 1, size);
  else
		f = "1";

  if (m.compare("0") == 0 && f.compare("0") == 0)
    return "0";

  v = atof(m.c_str()) / atof(f.c_str());

  magnitude = (int)v;
  fraction = v - magnitude;
  tmp = integer2string(power(2, width / 2), 10);
  expoent = atof(tmp.c_str());
  fraction = fraction * expoent;
  fraction = floor(fraction);

  std::string integer_str, fraction_str;
  integer_str = integer2binary(string2integer(double2string(magnitude), 10), width / 2);

  fraction_str = integer2binary(string2integer(double2string(fraction), 10), width / 2);

  value = integer_str + fraction_str;

  if (magnitude == 0 && v<0) {
    value = integer2binary(string2integer("-1", 10) - binary2integer(integer_str, true), width)
          + integer2binary(string2integer(double2string(fraction), 10), width / 2);
  }

  return value;
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
{ SMT_FUNC_ITE, SMT_FUNC_ITE, SMT_FUNC_ITE, 3, SMT_SORT_ALLINTS | SMT_SORT_BOOL },  //if
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
  // See comment below about shifts
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0}, // lshl
{ SMT_FUNC_NEG, SMT_FUNC_BVNEG, SMT_FUNC_BVNEG, 1, SMT_SORT_ALLINTS},  //neg
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //abs
{ SMT_FUNC_ADD, SMT_FUNC_BVADD, SMT_FUNC_BVADD, 2, SMT_SORT_ALLINTS},//add
{ SMT_FUNC_SUB, SMT_FUNC_BVSUB, SMT_FUNC_BVSUB, 2, SMT_SORT_ALLINTS},//sub
{ SMT_FUNC_MUL, SMT_FUNC_BVMUL, SMT_FUNC_BVMUL, 2, SMT_SORT_INT | SMT_SORT_REAL },//mul
{ SMT_FUNC_DIV, SMT_FUNC_BVSDIV, SMT_FUNC_BVUDIV, 2, SMT_SORT_INT | SMT_SORT_REAL },//div
{ SMT_FUNC_MOD, SMT_FUNC_BVSMOD, SMT_FUNC_BVUMOD, 2, SMT_SORT_BV | SMT_SORT_INT},//mod
// Error: C frontend doesn't upcast the 2nd operand to shift to the 1st operands
// bit width. Therefore this doesn't work. Fall back to backup method.
//{ SMT_FUNC_INVALID, SMT_FUNC_BVASHR, SMT_FUNC_BVASHR, 2, SMT_SORT_BV},  //ashr
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0}, // shl
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
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //code_asm
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //cpp_del_arr
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //cpp_del_id
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //cpp_catch
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //cpp_throw
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //cpp_throw_dec
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //isinf
{ SMT_FUNC_INVALID, SMT_FUNC_INVALID, SMT_FUNC_INVALID, 0, 0},  //isnormal
{ SMT_FUNC_HACKS, SMT_FUNC_HACKS, SMT_FUNC_HACKS, 0, 0},  //concat
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
  "+",
  "bvadd",
  "-",
  "bvsub",
  "*",
  "bvmul",
  "/",
  "bvudiv",
  "bvsdiv",
  "%",
  "bvsmod",
  "bvurem",
  "shl",
  "bvshl",
  "bvashr",
  "-",
  "bvneg",
  "bvlshr",
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
  "<",
  "bvslt",
  "bvult",
  ">",
  "bvsgt",
  "bvugt",
  "<=",
  "bvsle",
  "bvule",
  ">=",
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
  "is_int"
};

// Debris from prop_convt: to be reorganized.

void
smt_convt::pre_solve()
{
  // NB: always perform tuple constraint adding first, as it covers tuple
  // arrays too, and might end up generating more ASTs to be encoded in
  // the array api class.
  tuple_api->add_tuple_constraints_for_solving();
  array_api->add_array_constraints_for_solving();
  return;
}

expr2tc
smt_convt::get(const expr2tc &expr)
{
  switch (expr->type->type_id) {
  case type2t::bool_id:
    return get_bool(convert_ast(expr));
  case type2t::unsignedbv_id:
  case type2t::signedbv_id:
    return get_bv(expr->type, convert_ast(expr));
  case type2t::fixedbv_id:
  {
    // XXX -- again, another candidate for refactoring.
    expr2tc tmp = get_bv(expr->type, convert_ast(expr));
    if (is_nil_expr(tmp))
      return tmp;

    const constant_int2t &intval = to_constant_int2t(tmp);
    uint64_t val = intval.constant_value.to_ulong();
    std::stringstream ss;
    ss << val;
    constant_exprt value_expr(migrate_type_back(expr->type));
    value_expr.set_value(get_fixed_point(expr->type->get_width(), ss.str()));
    fixedbvt fbv;
    fbv.from_expr(value_expr);
    return constant_fixedbv2tc(expr->type, fbv);
  }
  case type2t::array_id:
    return get_array(convert_ast(expr), expr->type);
  case type2t::struct_id:
  case type2t::pointer_id:
    return tuple_api->tuple_get(expr);
  default:
    std::cerr << "Unimplemented type'd expression (" << expr->type->type_id
              << ") in smt get" << std::endl;
    abort();
  }
}

expr2tc
smt_convt::get_array(smt_astt array, const type2tc &t)
{
  // XXX -- printing multidimensional arrays?

  type2tc newtype = flatten_array_type(t);

  const array_type2t &ar = to_array_type(newtype);
  if (is_tuple_ast_type(ar.subtype)) {
    std::cerr << "Tuple array getting not implemented yet, sorry" << std::endl;
    return expr2tc();
  }

  // Fetch the array bounds, if it's huge then assume this is a 1024 element
  // array. Then fetch all elements and formulate a constant_array.
  size_t w = array->sort->domain_width;
  if (w > 10)
    w = 10;

  constant_int2tc arr_size(index_type2(), BigInt(1 << w));
  type2tc arr_type = type2tc(new array_type2t(ar.subtype, arr_size, false));
  std::vector<expr2tc> fields;

  for (size_t i = 0; i < (1ULL << w); i++) {
    fields.push_back(array_api->get_array_elem(array, i, ar.subtype));
  }

  return constant_array2tc(arr_type, fields);
}

const struct_union_data &
smt_convt::get_type_def(const type2tc &type) const
{

  return (is_pointer_type(type))
        ? *pointer_type_data
        : dynamic_cast<const struct_union_data &>(*type.get());
}

smt_astt
smt_convt::array_create(const expr2tc &expr)
{
  if (is_constant_array_of2t(expr))
    return convert_array_of_prep(expr);

  // Handle constant array expressions: these don't have tuple type and so
  // don't need funky handling, but we need to create a fresh new symbol and
  // repeatedly store the desired data into it, to create an SMT array
  // representing the expression we're converting.
  std::string name = mk_fresh_name("array_create::") + ".";
  expr2tc newsym = symbol2tc(expr->type, name);

  // Check size
  const array_type2t &arr_type = to_array_type(expr->type);
  if (arr_type.size_is_infinite) {
    // Guarentee nothing, this is modelling only.
    return convert_ast(newsym);
  } else if (!is_constant_int2t(arr_type.array_size)) {
    std::cerr << "Non-constant sized array of type constant_array_of2t"
              << std::endl;
    abort();
  }

  const constant_int2t &thesize = to_constant_int2t(arr_type.array_size);
  uint64_t sz = thesize.constant_value.to_ulong();

  assert(is_constant_array2t(expr));
  const constant_array2t &array = to_constant_array2t(expr);

  // Repeatedly store things into this.
  smt_astt newsym_ast = convert_ast(newsym);
  for (unsigned int i = 0; i < sz; i++) {
    expr2tc init = array.datatype_members[i];

    // Workaround for bools-in-arrays
    if (is_bool_type(array.datatype_members[i]->type) && !int_encoding &&
        !array_api->supports_bools_in_arrays)
      init = typecast2tc(type2tc(new unsignedbv_type2t(1)), init);

    newsym_ast = newsym_ast->update(this, convert_ast(init), i);
  }

  return newsym_ast;
}

smt_astt
smt_convt::convert_array_of_prep(const expr2tc &expr)
{
  const constant_array_of2t &arrof = to_constant_array_of2t(expr);
  const array_type2t &arrtype = to_array_type(arrof.type);
  expr2tc base_init;
  unsigned long array_size = 0;

  // So: we have an array_of, that we have to convert into a bunch of stores.
  // However, it might be a nested array. If that's the case, then we're
  // guaranteed to have another array_of in the initializer which we can flatten
  // to a single array of whatever's at the bottom of the array_of. Or, it's
  // a constant_array, in which case we can just copy the contents.
  if (is_array_type(arrtype.subtype)) {
    expr2tc rec_expr = expr;

    if (is_constant_array_of2t(to_constant_array_of2t(rec_expr).initializer)) {
      type2tc flat_type = flatten_array_type(expr->type);
      const array_type2t &arrtype2 = to_array_type(flat_type);
      array_size = calculate_array_domain_width(arrtype2);

      while (is_constant_array_of2t(rec_expr))
        rec_expr = to_constant_array_of2t(rec_expr).initializer;

      base_init = rec_expr;
    } else {
      const constant_array_of2t &arrof = to_constant_array_of2t(rec_expr);
      assert(is_constant_array2t(arrof.initializer));
      const constant_array2t &constarray =
        to_constant_array2t(arrof.initializer);
      const array_type2t &constarray_type = to_array_type(constarray.type);

      // Duplicate contents repeatedly.
      assert(is_constant_int2t(arrtype.array_size) &&
          "Cannot have complex nondet-sized array_of initializers");
      const BigInt &size = to_constant_int2t(arrtype.array_size).constant_value;

      std::vector<expr2tc> new_contents;
      for (uint64_t i = 0; i < size.to_uint64(); i++)
        new_contents.insert(new_contents.end(),
            constarray.datatype_members.begin(),
            constarray.datatype_members.end());

      // Create new expression, convert and return that.
      mul2tc newsize(arrtype.array_size->type, arrtype.array_size,
          constarray_type.array_size);
      expr2tc simplified = newsize->simplify();
      assert(!is_nil_expr(simplified));
      type2tc new_arr_type(new array_type2t(constarray_type.subtype,
            simplified,false));
      constant_array2tc new_const_array(new_arr_type, new_contents);
      return convert_ast(new_const_array);
    }
  } else {
    base_init = arrof.initializer;
    array_size = calculate_array_domain_width(arrtype);
  }

  if (is_structure_type(base_init->type))
    return tuple_api->tuple_array_of(base_init, array_size);
  else if (is_pointer_type(base_init->type))
    return pointer_array_of(base_init, array_size);
  else
    return array_api->convert_array_of(convert_ast(base_init), array_size);
}

smt_astt
array_iface::default_convert_array_of(smt_astt init_val,
                                          unsigned long array_size,
                                          smt_convt *ctx)
{
  // We now an initializer, and a size of array to build. So:
  // Repeatedly store things into this.
  // XXX int mode

  if (init_val->sort->id == SMT_SORT_BOOL && !supports_bools_in_arrays) {
    smt_astt zero = ctx->mk_smt_bvint(BigInt(0), false, 1);
    smt_astt one = ctx->mk_smt_bvint(BigInt(0), false, 1);
    smt_sortt result_sort = ctx->mk_sort(SMT_SORT_BV, 1, false);
    init_val = ctx->mk_func_app(result_sort, SMT_FUNC_ITE, init_val, one, zero);
  }

  smt_sortt domwidth = ctx->mk_int_bv_sort(array_size);
  smt_sortt arrsort = ctx->mk_sort(SMT_SORT_ARRAY, domwidth, init_val->sort);
  smt_astt newsym_ast =
    ctx->mk_fresh(arrsort, "default_array_of::", init_val->sort);

  unsigned long sz = 1ULL << array_size;
  for (unsigned long i = 0; i < sz; i++) {
    newsym_ast = newsym_ast->update(ctx, init_val, i);
  }

  return newsym_ast;
}

smt_astt
smt_convt::pointer_array_of(const expr2tc &init_val __attribute__((unused)),
    unsigned long array_width)
{
  // Actually a tuple, but the operand is going to be a symbol, null.
  assert(is_symbol2t(init_val) && "Pointer type'd array_of can only be an "
         "array of null");

#ifndef NDEBUG
  const symbol2t &sym = to_symbol2t(init_val);
  assert(sym.thename == "NULL" && "Pointer type'd array_of can only be an "
         "array of null");
#endif

  // Well known value; zero and zero.
  constant_int2tc zero_val(machine_ptr, BigInt(0));
  std::vector<expr2tc> operands;
  operands.reserve(2);
  operands.push_back(zero_val);
  operands.push_back(zero_val);

  constant_struct2tc strct(pointer_struct, operands);
  return tuple_api->tuple_array_of(strct, array_width);
}

smt_astt
smt_convt::tuple_array_create_despatch(const expr2tc &expr, smt_sortt domain)
{
  // Take a constant_array2t or an array_of, and format the data from them into
  // a form palatable to tuple_array_create.

  if (is_constant_array_of2t(expr)) {
    const constant_array_of2t &arr = to_constant_array_of2t(expr);
    smt_astt arg = convert_ast(arr.initializer);

    return tuple_api->tuple_array_create(arr.type, &arg, true, domain);
  } else {
    assert(is_constant_array2t(expr));
    const constant_array2t &arr = to_constant_array2t(expr);
    smt_astt args[arr.datatype_members.size()];
    unsigned int i = 0;
    forall_exprs(it, arr.datatype_members) {
      args[i] = convert_ast(*it);
      i++;
    }

    return tuple_api->tuple_array_create(arr.type, args, false, domain);
  }
}

// Default behaviours for SMT AST's

void
smt_ast::assign(smt_convt *ctx, smt_astt sym) const
{
  ctx->assert_ast(eq(ctx, sym));
}

smt_astt
smt_ast::ite(smt_convt *ctx, smt_astt cond, smt_astt falseop) const
{
  return ctx->mk_func_app(sort, SMT_FUNC_ITE, cond, this, falseop);
}

smt_astt
smt_ast::eq(smt_convt *ctx, smt_astt other) const
{
  // Simple approach: this is a leaf piece of SMT, compute a basic equality.
  return ctx->mk_func_app(ctx->boolean_sort, SMT_FUNC_EQ, this, other);
}

smt_astt
smt_ast::update(smt_convt *ctx, smt_astt value, unsigned int idx,
    expr2tc idx_expr) const
{
  // If we're having an update applied to us, then the only valid situation
  // this can occur in is if we're an array.
  assert(sort->id == SMT_SORT_ARRAY);

  // We're an array; just generate a 'with' operation.
  expr2tc index;
  if (is_nil_expr(idx_expr)) {
    assert(sort->domain_width != 0 && "Array sort with zero-sized domain "
           "width");
    index = constant_int2tc(type2tc(new unsignedbv_type2t(sort->domain_width)),
          BigInt(idx));
  } else {
    index = idx_expr;
  }

  return ctx->mk_func_app(sort, SMT_FUNC_STORE,
                          this, ctx->convert_ast(index), value);
}

smt_astt
smt_ast::select(smt_convt *ctx, const expr2tc &idx) const
{
  assert(sort->id == SMT_SORT_ARRAY && "Select operation applied to non-array "
         "scalar AST");

  // Just apply a select operation to the current array. Index should be fixed.

  // Guess the resulting sort. This could be a lot, lot better.
  smt_sortt range_sort = NULL;
  if (sort->data_width == 1 && ctx->array_api->supports_bools_in_arrays)
    range_sort = ctx->boolean_sort;
  else
    range_sort = ctx->mk_int_bv_sort(sort->data_width);

  return ctx->mk_func_app(range_sort, SMT_FUNC_SELECT,
                          this, ctx->convert_ast(idx));
}

smt_astt
smt_ast::project(smt_convt *ctx __attribute__((unused)),
    unsigned int idx __attribute__((unused))) const
{
  std::cerr << "Projecting from non-tuple based AST" << std::endl;
  abort();
}
