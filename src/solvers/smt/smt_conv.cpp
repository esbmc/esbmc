#include <iomanip>
#include <set>
#include <solvers/prop/literal.h>
#include <solvers/smt/smt_conv.h>
#include <sstream>
#include <util/arith_tools.h>
#include <util/base_type.h>
#include <util/c_types.h>
#include <util/expr_util.h>

// Helpers extracted from z3_convt.

static std::string extract_magnitude(const std::string &v, unsigned width)
{
  return integer2string(binary2integer(v.substr(0, width / 2), true), 10);
}

static std::string extract_fraction(const std::string &v, unsigned width)
{
  return integer2string(binary2integer(v.substr(width / 2, width), false), 10);
}

static std::string double2string(double d)
{
  std::ostringstream format_message;
  format_message << std::setprecision(12) << d;
  return format_message.str();
}

static std::string itos(int64_t i)
{
  std::stringstream ss;
  ss << i;
  return ss.str();
}

unsigned int smt_convt::get_member_name_field(
  const type2tc &t,
  const irep_idt &name) const
{
  unsigned int idx = 0;
  const struct_union_data &data_ref = get_type_def(t);

  for(auto const &it : data_ref.member_names)
  {
    if(it == name)
      break;
    idx++;
  }
  assert(
    idx != data_ref.member_names.size() &&
    "Member name of with expr not found in struct type");

  return idx;
}

unsigned int smt_convt::get_member_name_field(
  const type2tc &t,
  const expr2tc &name) const
{
  const constant_string2t &str = to_constant_string2t(name);
  return get_member_name_field(t, str.value);
}

smt_convt::smt_convt(bool intmode, const namespacet &_ns)
  : ctx_level(0), boolean_sort(nullptr), int_encoding(intmode), ns(_ns)
{
  tuple_api = nullptr;
  array_api = nullptr;
  fp_api = nullptr;

  std::vector<type2tc> members;
  std::vector<irep_idt> names;

  members.push_back(type_pool.get_uint(config.ansi_c.pointer_width));
  members.push_back(type_pool.get_uint(config.ansi_c.pointer_width));
  names.emplace_back("pointer_object");
  names.emplace_back("pointer_offset");

  struct_type2t *tmp =
    new struct_type2t(members, names, names, "pointer_struct");
  pointer_type_data = tmp;
  pointer_struct = type2tc(tmp);

  pointer_logic.emplace_back();

  addr_space_sym_num.push_back(0);

  renumber_map.emplace_back();

  members.clear();
  names.clear();
  members.push_back(type_pool.get_uint(config.ansi_c.pointer_width));
  members.push_back(type_pool.get_uint(config.ansi_c.pointer_width));
  names.emplace_back("start");
  names.emplace_back("end");
  tmp = new struct_type2t(members, names, names, "addr_space_type");
  addr_space_type_data = tmp;
  addr_space_type = type2tc(tmp);

  addr_space_arr_type =
    type2tc(new array_type2t(addr_space_type, expr2tc(), true));

  addr_space_data.emplace_back();

  machine_int = type2tc(new signedbv_type2t(config.ansi_c.int_width));
  machine_uint = type2tc(new unsignedbv_type2t(config.ansi_c.int_width));
  machine_ptr = type2tc(new unsignedbv_type2t(config.ansi_c.pointer_width));

  // Pick a modelling array to shoehorn initialization data into. Because
  // we don't yet have complete data for whether pointers are dynamic or not,
  // this is the one modelling array that absolutely _has_ to be initialized
  // to false for each element, which is going to be shoved into
  // convert_identifier_pointer.
  dyn_info_arr_name = "__ESBMC_is_dynamic&0#1";

  ptr_foo_inited = false;
}

void smt_convt::set_tuple_iface(tuple_iface *iface)
{
  assert(tuple_api == nullptr && "set_tuple_iface should only be called once");
  tuple_api = iface;
}

void smt_convt::set_array_iface(array_iface *iface)
{
  assert(array_api == nullptr && "set_array_iface should only be called once");
  array_api = iface;
}

void smt_convt::set_fp_conv(fp_convt *iface)
{
  assert(fp_api == NULL && "set_fp_iface should only be called once");
  fp_api = iface;
}

void smt_convt::delete_all_asts()
{
  // Erase all the remaining asts in the live ast vector.
  for(smt_ast *ast : live_asts)
    delete ast;
  live_asts.clear();
}

void smt_convt::smt_post_init()
{
  machine_int_sort = mk_int_bv_sort(config.ansi_c.int_width);
  machine_uint_sort = mk_int_bv_sort(config.ansi_c.int_width);

  boolean_sort = mk_bool_sort();

  init_addr_space_array();

  if(int_encoding)
  {
    std::vector<expr2tc> power_array_data;
    uint64_t pow;
    unsigned int count = 0;
    type2tc powarr_elemt = get_uint_type(64);
    for(pow = 1ULL; count < 64; pow <<= 1, count++)
      power_array_data.push_back(constant_int2tc(powarr_elemt, BigInt(pow)));

    type2tc power_array_type(
      new array_type2t(powarr_elemt, gen_ulong(64), false));

    constant_array2tc power_array(power_array_type, power_array_data);
    int_shift_op_array = convert_ast(power_array);
  }

  ptr_foo_inited = true;
}

void smt_convt::push_ctx()
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

void smt_convt::pop_ctx()
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

  for(unsigned int idx = live_asts_sizes.back(); idx < live_asts.size(); idx++)
    delete live_asts[idx];

  // And reset the storage back to that point.
  live_asts.resize(live_asts_sizes.back());
  live_asts_sizes.pop_back();

  array_api->pop_array_ctx();
  tuple_api->pop_tuple_ctx();
}

smt_astt smt_convt::invert_ast(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_BOOL);
  return mk_not(a);
}

smt_astt smt_convt::imply_ast(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return mk_implies(a, b);
}

void smt_convt::set_to(const expr2tc &expr, bool value)
{
  smt_astt a = convert_ast(expr);
  if(value == false)
    a = invert_ast(a);
  assert_ast(a);
}

smt_astt smt_convt::convert_assign(const expr2tc &expr)
{
  const equality2t &eq = to_equality2t(expr);
  smt_astt side1 = convert_ast(eq.side_1);
  smt_astt side2 = convert_ast(eq.side_2);
  side2->assign(this, side1);

  // Put that into the smt cache, thus preserving the assigned symbols value.
  // IMPORTANT: the cache is now a fundemental part of how some flatteners work,
  // in that one can chose to create a set of expressions and their ASTs, then
  // store them in the cache, rather than have a more sophisticated conversion.
  smt_cache_entryt e = {eq.side_1, side1, ctx_level};
  smt_cache.insert(e);

  return side2;
}

smt_astt smt_convt::convert_ast(const expr2tc &expr)
{
  smt_cachet::const_iterator cache_result = smt_cache.find(expr);
  if(cache_result != smt_cache.end())
    return (cache_result->ast);
  std::vector<smt_astt> args;
  args.reserve(expr->get_num_sub_exprs());

  switch(expr->expr_id)
  {
  case expr2t::with_id:
  case expr2t::constant_array_id:
  case expr2t::constant_array_of_id:
  case expr2t::index_id:
  case expr2t::address_of_id:
  case expr2t::ieee_add_id:
  case expr2t::ieee_sub_id:
  case expr2t::ieee_mul_id:
  case expr2t::ieee_div_id:
  case expr2t::ieee_fma_id:
  case expr2t::ieee_sqrt_id:
    break; // Don't convert their operands

  default:
  {
    // Convert /all the arguments/. Via magical delegates.
    unsigned int i = 0;
    expr->foreach_operand(
      [this, &args, &i](const expr2tc &e) { args[i++] = convert_ast(e); });
  }
  }

  smt_astt a;
  switch(expr->expr_id)
  {
  case expr2t::constant_int_id:
  case expr2t::constant_fixedbv_id:
  case expr2t::constant_floatbv_id:
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
    std::cerr << "Post-parse union literals are deprecated and broken, sorry\n";
    abort();
  case expr2t::constant_array_id:
  case expr2t::constant_array_of_id:
  {
    const array_type2t &arr = to_array_type(expr->type);
    if(!array_api->can_init_infinite_arrays && arr.size_is_infinite)
    {
      smt_sortt sort = convert_sort(expr->type);

      // Don't honour inifinite sized array initializers. Modelling only.
      // If we have an array of tuples and no tuple support, use tuple_fresh.
      // Otherwise, mk_fresh.
      if(is_tuple_ast_type(arr.subtype))
        a = tuple_api->tuple_fresh(sort);
      else
        a = mk_fresh(
          sort,
          "inf_array",
          convert_sort(get_flattened_array_subtype(expr->type)));
      break;
    }

    // Domain sort may be mesed with:
    smt_sortt domain;
    if(int_encoding)
    {
      domain = machine_int_sort;
    }
    else
    {
      domain = mk_int_bv_sort(calculate_array_domain_width(arr));
    }

    expr2tc flat_expr = expr;
    if(
      is_array_type(get_array_subtype(expr->type)) && is_constant_array2t(expr))
      flat_expr = flatten_array_body(expr);

    if(is_struct_type(arr.subtype) || is_pointer_type(arr.subtype))
      a = tuple_array_create_despatch(flat_expr, domain);
    else
      a = array_create(flat_expr);
    break;
  }
  case expr2t::add_id:
  {
    const add2t &add = to_add2t(expr);
    if(
      is_pointer_type(expr->type) || is_pointer_type(add.side_1) ||
      is_pointer_type(add.side_2))
    {
      a = convert_pointer_arith(expr, expr->type);
    }
    else if(int_encoding)
    {
      a = mk_add(args[0], args[1]);
    }
    else
    {
      a = mk_bvadd(args[0], args[1]);
    }
    break;
  }
  case expr2t::sub_id:
  {
    const sub2t &sub = to_sub2t(expr);
    if(
      is_pointer_type(expr->type) || is_pointer_type(sub.side_1) ||
      is_pointer_type(sub.side_2))
    {
      a = convert_pointer_arith(expr, expr->type);
    }
    else if(int_encoding)
    {
      a = mk_sub(args[0], args[1]);
    }
    else
    {
      a = mk_bvsub(args[0], args[1]);
    }
    break;
  }
  case expr2t::mul_id:
  {
    // Fixedbvs are handled separately
    if(is_fixedbv_type(expr) && !int_encoding)
    {
      auto mul = to_mul2t(expr);
      auto fbvt = to_fixedbv_type(mul.type);

      unsigned int fraction_bits = fbvt.width - fbvt.integer_bits;

      args[0] = mk_sign_ext(convert_ast(mul.side_1), fraction_bits);
      args[1] = mk_sign_ext(convert_ast(mul.side_2), fraction_bits);

      a = mk_bvmul(args[0], args[1]);
      a = mk_extract(a, fbvt.width + fraction_bits - 1, fraction_bits);
    }
    else if(int_encoding)
    {
      a = mk_mul(args[0], args[1]);
    }
    else
    {
      a = mk_bvmul(args[0], args[1]);
    }
    break;
  }
  case expr2t::div_id:
  {
    auto d = to_div2t(expr);

    // Fixedbvs are handled separately
    if(is_fixedbv_type(expr) && !int_encoding)
    {
      auto fbvt = to_fixedbv_type(d.type);

      unsigned int fraction_bits = fbvt.width - fbvt.integer_bits;

      args[1] = mk_sign_ext(convert_ast(d.side_2), fraction_bits);

      smt_astt zero = mk_smt_bv(BigInt(0), fraction_bits);
      smt_astt op0 = convert_ast(d.side_1);

      args[0] = mk_concat(op0, zero);

      // Sorts.
      a = mk_bvsdiv(args[0], args[1]);
      a = mk_extract(a, fbvt.width - 1, 0);
    }
    else if(int_encoding)
    {
      a = mk_div(args[0], args[1]);
    }
    else if(is_unsignedbv_type(d.side_1) && is_unsignedbv_type(d.side_2))
    {
      a = mk_bvudiv(args[0], args[1]);
    }
    else
    {
      assert(is_signedbv_type(d.side_1) && is_signedbv_type(d.side_2));
      a = mk_bvsdiv(args[0], args[1]);
    }
    break;
  }
  case expr2t::ieee_add_id:
  {
    assert(is_floatbv_type(expr));
    a = fp_api->mk_smt_fpbv_add(
      convert_ast(to_ieee_add2t(expr).side_1),
      convert_ast(to_ieee_add2t(expr).side_2),
      convert_rounding_mode(to_ieee_add2t(expr).rounding_mode));
    break;
  }
  case expr2t::ieee_sub_id:
  {
    assert(is_floatbv_type(expr));
    a = fp_api->mk_smt_fpbv_sub(
      convert_ast(to_ieee_sub2t(expr).side_1),
      convert_ast(to_ieee_sub2t(expr).side_2),
      convert_rounding_mode(to_ieee_sub2t(expr).rounding_mode));
    break;
  }
  case expr2t::ieee_mul_id:
  {
    assert(is_floatbv_type(expr));
    a = fp_api->mk_smt_fpbv_mul(
      convert_ast(to_ieee_mul2t(expr).side_1),
      convert_ast(to_ieee_mul2t(expr).side_2),
      convert_rounding_mode(to_ieee_mul2t(expr).rounding_mode));
    break;
  }
  case expr2t::ieee_div_id:
  {
    assert(is_floatbv_type(expr));
    a = fp_api->mk_smt_fpbv_div(
      convert_ast(to_ieee_div2t(expr).side_1),
      convert_ast(to_ieee_div2t(expr).side_2),
      convert_rounding_mode(to_ieee_div2t(expr).rounding_mode));
    break;
  }
  case expr2t::ieee_fma_id:
  {
    assert(is_floatbv_type(expr));
    a = fp_api->mk_smt_fpbv_fma(
      convert_ast(to_ieee_fma2t(expr).value_1),
      convert_ast(to_ieee_fma2t(expr).value_2),
      convert_ast(to_ieee_fma2t(expr).value_3),
      convert_rounding_mode(to_ieee_fma2t(expr).rounding_mode));
    break;
  }
  case expr2t::ieee_sqrt_id:
  {
    assert(is_floatbv_type(expr));
    a = fp_api->mk_smt_fpbv_sqrt(
      convert_ast(to_ieee_sqrt2t(expr).value),
      convert_rounding_mode(to_ieee_sqrt2t(expr).rounding_mode));
    break;
  }
  case expr2t::modulus_id:
  {
    auto m = to_modulus2t(expr);

    if(int_encoding)
    {
      a = mk_mod(args[0], args[1]);
    }
    else if(is_fixedbv_type(m.side_1) && is_fixedbv_type(m.side_2))
    {
      a = mk_bvsmod(args[0], args[1]);
    }
    else if(is_unsignedbv_type(m.side_1) && is_unsignedbv_type(m.side_2))
    {
      a = mk_bvumod(args[0], args[1]);
    }
    else
    {
      assert(is_signedbv_type(m.side_1) || is_signedbv_type(m.side_2));
      a = mk_bvsmod(args[0], args[1]);
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
    if(is_struct_type(expr) || is_pointer_type(expr))
    {
      unsigned int idx = get_member_name_field(expr->type, with.update_field);
      smt_astt srcval = convert_ast(with.source_value);

#ifndef NDEBUG
      const struct_union_data &data = get_type_def(with.type);
      assert(idx < data.members.size() && "Out of bounds with expression");
      // Base type eq examines pointer types to closely
      assert(
        (base_type_eq(data.members[idx], with.update_value->type, ns) ||
         (is_pointer_type(data.members[idx]) &&
          is_pointer_type(with.update_value))) &&
        "Assigned tuple member has type mismatch");
#endif

      a = srcval->update(this, convert_ast(with.update_value), idx);
    }
    else
    {
      a = convert_array_store(expr);
    }
    break;
  }
  case expr2t::member_id:
  {
    a = convert_member(expr);
    break;
  }
  case expr2t::same_object_id:
  {
    // Two projects, then comparison.
    args[0] = args[0]->project(this, 0);
    args[1] = args[1]->project(this, 0);
    a = mk_eq(args[0], args[1]);
    break;
  }
  case expr2t::pointer_offset_id:
  {
    const pointer_offset2t &obj = to_pointer_offset2t(expr);
    // Potentially walk through some typecasts
    const expr2tc *ptr = &obj.ptr_obj;
    while(is_typecast2t(*ptr) && !is_pointer_type((*ptr)))
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
    while(is_typecast2t(*ptr) && !is_pointer_type((*ptr)))
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
  case expr2t::nearbyint_id:
  {
    assert(is_floatbv_type(expr));
    a = fp_api->mk_smt_nearbyint_from_float(
      convert_ast(to_nearbyint2t(expr).from),
      convert_rounding_mode(to_nearbyint2t(expr).rounding_mode));
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
    a = convert_is_nan(expr);
    break;
  }
  case expr2t::isinf_id:
  {
    a = convert_is_inf(expr);
    break;
  }
  case expr2t::isnormal_id:
  {
    a = convert_is_normal(expr);
    break;
  }
  case expr2t::isfinite_id:
  {
    a = convert_is_finite(expr);
    break;
  }
  case expr2t::signbit_id:
  {
    a = convert_signbit(expr);
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
    auto eq = to_equality2t(expr);

    if(is_floatbv_type(eq.side_1) && is_floatbv_type(eq.side_2))
      a = fp_api->mk_smt_fpbv_eq(args[0], args[1]);
    else
      a = args[0]->eq(this, args[1]);
    break;
  }
  case expr2t::notequal_id:
  {
    // Handle all kinds of structs by inverted equality. The only that's really
    // going to turn up is pointers though.

    auto neq = to_notequal2t(expr);

    if(is_floatbv_type(neq.side_1) && is_floatbv_type(neq.side_2))
      a = fp_api->mk_smt_fpbv_eq(args[0], args[1]);
    else
      a = args[0]->eq(this, args[1]);
    a = mk_not(a);
    break;
  }
  case expr2t::shl_id:
  {
    const shl2t &shl = to_shl2t(expr);

    if(int_encoding)
    {
      // Raise 2^shift, then multiply first operand by that value. If it's
      // negative, what to do? FIXME.
      smt_astt powval = int_shift_op_array->select(this, shl.side_2);
      args[1] = powval;
      a = mk_mul(args[0], args[1]);
    }
    else
    {
      a = mk_bvshl(args[0], args[1]);
    }
    break;
  }
  case expr2t::ashr_id:
  {
    const ashr2t &ashr = to_ashr2t(expr);

    if(int_encoding)
    {
      // Raise 2^shift, then divide first operand by that value. If it's
      // negative, I suspect the correct operation is to latch to -1,
      smt_astt powval = int_shift_op_array->select(this, ashr.side_2);
      args[1] = powval;
      a = mk_div(args[0], args[1]);
    }
    else
    {
      a = mk_bvashr(args[0], args[1]);
    }
    break;
  }
  case expr2t::lshr_id:
  {
    // Like ashr. Haven't got around to cleaning this up yet.
    const lshr2t &lshr = to_lshr2t(expr);

    if(int_encoding)
    {
      // Raise 2^shift, then divide first operand by that value. If it's
      // negative, I suspect the correct operation is to latch to -1,
      smt_astt powval = int_shift_op_array->select(this, lshr.side_2);
      args[1] = powval;
      a = mk_div(args[0], args[1]);
    }
    else
    {
      a = mk_bvlshr(args[0], args[1]);
    }
    break;
  }
  case expr2t::abs_id:
  {
    const abs2t &abs = to_abs2t(expr);
    if(is_unsignedbv_type(abs.value))
    {
      // No need to do anything.
      a = args[0];
    }
    else if(is_floatbv_type(abs.value))
    {
      a = fp_api->mk_smt_fpbv_abs(args[0]);
    }
    else
    {
      lessthan2tc lt(abs.value, gen_zero(abs.value->type));
      neg2tc neg(abs.value->type, abs.value);
      if2tc ite(abs.type, lt, neg, abs.value);

      a = convert_ast(ite);
    }
    break;
  }
  case expr2t::lessthan_id:
  {
    const lessthan2t &lt = to_lessthan2t(expr);
    // Pointer relation:
    if(is_pointer_type(lt.side_1))
    {
      a = convert_ptr_cmp(lt.side_1, lt.side_2, expr);
    }
    else if(is_floatbv_type(lt.side_1) && is_floatbv_type(lt.side_2))
    {
      a = fp_api->mk_smt_fpbv_lt(args[0], args[1]);
    }
    else if(int_encoding)
    {
      a = mk_lt(args[0], args[1]);
    }
    else if(is_fixedbv_type(lt.side_1) && is_fixedbv_type(lt.side_2))
    {
      a = mk_bvslt(args[0], args[1]);
    }
    else if(is_unsignedbv_type(lt.side_1) && is_unsignedbv_type(lt.side_2))
    {
      a = mk_bvult(args[0], args[1]);
    }
    else
    {
      assert(is_signedbv_type(lt.side_1) && is_signedbv_type(lt.side_2));
      a = mk_bvslt(args[0], args[1]);
    }
    break;
  }
  case expr2t::lessthanequal_id:
  {
    const lessthanequal2t &lte = to_lessthanequal2t(expr);
    // Pointer relation:
    if(is_pointer_type(lte.side_1))
    {
      a = convert_ptr_cmp(lte.side_1, lte.side_2, expr);
    }
    else if(is_floatbv_type(lte.side_1) && is_floatbv_type(lte.side_2))
    {
      a = fp_api->mk_smt_fpbv_lte(args[0], args[1]);
    }
    else if(int_encoding)
    {
      a = mk_le(args[0], args[1]);
    }
    else if(is_fixedbv_type(lte.side_1) && is_fixedbv_type(lte.side_2))
    {
      a = mk_bvsle(args[0], args[1]);
    }
    else if(is_unsignedbv_type(lte.side_1) && is_unsignedbv_type(lte.side_2))
    {
      a = mk_bvule(args[0], args[1]);
    }
    else
    {
      assert(is_signedbv_type(lte.side_1) && is_signedbv_type(lte.side_2));
      a = mk_bvsle(args[0], args[1]);
    }
    break;
  }
  case expr2t::greaterthan_id:
  {
    const greaterthan2t &gt = to_greaterthan2t(expr);
    // Pointer relation:
    if(is_pointer_type(gt.side_1))
    {
      a = convert_ptr_cmp(gt.side_1, gt.side_2, expr);
    }
    else if(is_floatbv_type(gt.side_1) && is_floatbv_type(gt.side_2))
    {
      a = fp_api->mk_smt_fpbv_gt(args[0], args[1]);
    }
    else if(int_encoding)
    {
      a = mk_gt(args[0], args[1]);
    }
    else if(is_fixedbv_type(gt.side_1) && is_fixedbv_type(gt.side_2))
    {
      a = mk_bvsgt(args[0], args[1]);
    }
    else if(is_unsignedbv_type(gt.side_1) && is_unsignedbv_type(gt.side_2))
    {
      a = mk_bvugt(args[0], args[1]);
    }
    else
    {
      assert(is_signedbv_type(gt.side_1) && is_signedbv_type(gt.side_2));
      a = mk_bvsgt(args[0], args[1]);
    }
    break;
  }
  case expr2t::greaterthanequal_id:
  {
    const greaterthanequal2t &gte = to_greaterthanequal2t(expr);
    // Pointer relation:
    if(is_pointer_type(gte.side_1))
    {
      a = convert_ptr_cmp(gte.side_1, gte.side_2, expr);
    }
    else if(is_floatbv_type(gte.side_1) && is_floatbv_type(gte.side_2))
    {
      a = fp_api->mk_smt_fpbv_gte(args[0], args[1]);
    }
    else if(int_encoding)
    {
      a = mk_ge(args[0], args[1]);
    }
    else if(is_fixedbv_type(gte.side_1) && is_fixedbv_type(gte.side_2))
    {
      a = mk_bvsge(args[0], args[1]);
    }
    else if(is_unsignedbv_type(gte.side_1) && is_unsignedbv_type(gte.side_2))
    {
      a = mk_bvuge(args[0], args[1]);
    }
    else
    {
      assert(is_signedbv_type(gte.side_1) && is_signedbv_type(gte.side_2));
      a = mk_bvsge(args[0], args[1]);
    }
    break;
  }
  case expr2t::concat_id:
  {
    assert(
      !int_encoding &&
      "Concatonate encountered in integer mode; unimplemented (and funky)");
    a = mk_concat(args[0], args[1]);
    break;
  }
  case expr2t::implies_id:
  {
    a = mk_implies(args[0], args[1]);
    break;
  }
  case expr2t::bitand_id:
  {
    assert(!int_encoding);
    a = mk_bvand(args[0], args[1]);
    break;
  }
  case expr2t::bitor_id:
  {
    assert(!int_encoding);
    a = mk_bvor(args[0], args[1]);
    break;
  }
  case expr2t::bitxor_id:
  {
    assert(!int_encoding);
    a = mk_bvxor(args[0], args[1]);
    break;
  }
  case expr2t::bitnand_id:
  {
    assert(!int_encoding);
    a = mk_bvnand(args[0], args[1]);
    break;
  }
  case expr2t::bitnor_id:
  {
    assert(!int_encoding);
    a = mk_bvnor(args[0], args[1]);
    break;
  }
  case expr2t::bitnxor_id:
  {
    assert(!int_encoding);
    a = mk_bvnxor(args[0], args[1]);
    break;
  }
  case expr2t::bitnot_id:
  {
    assert(!int_encoding);
    a = mk_bvnot(args[0]);
    break;
  }
  case expr2t::not_id:
  {
    assert(is_bool_type(expr));
    a = mk_not(args[0]);
    break;
  }
  case expr2t::neg_id:
  {
    const neg2t &neg = to_neg2t(expr);
    if(is_floatbv_type(neg.value))
    {
      a = fp_api->mk_smt_fpbv_neg(args[0]);
    }
    else if(int_encoding)
    {
      a = mk_neg(args[0]);
    }
    else
    {
      a = mk_bvneg(args[0]);
    }
    break;
  }
  case expr2t::and_id:
  {
    a = mk_and(args[0], args[1]);
    break;
  }
  case expr2t::or_id:
  {
    a = mk_or(args[0], args[1]);
    break;
  }
  case expr2t::xor_id:
  {
    a = mk_xor(args[0], args[1]);
    break;
  }
  case expr2t::bitcast_id:
  {
    const bitcast2t &cast = to_bitcast2t(expr);
    assert(is_scalar_type(cast.type) && is_scalar_type(cast.from));

    // As it stands, the only circusmtance where bitcast can make a difference
    // is where we're casting to or from a float, where casting by value means
    // something different. Filter that case out, pass everything else to normal
    // cast.
    bool to_float = is_floatbv_type(cast.type);
    bool from_float = is_floatbv_type(cast.from);

    if((to_float && !from_float) || (!to_float && from_float))
    {
      a = to_float ? fp_api->mk_from_bv_to_fp(args[0], convert_sort(cast.type))
                   : fp_api->mk_from_fp_to_bv(args[0]);
    }
    else
    {
      // Cast by value is fine
      typecast2tc tcast(cast.type, cast.from);
      a = convert_ast(tcast);
    }
    break;
  }
  case expr2t::extract_id:
  {
    const extract2t &ex = to_extract2t(expr);
    a = convert_ast(ex.from);
    a = mk_extract(a, ex.upper, ex.lower);
    break;
  }
  default:
    std::cerr << "Couldn't convert expression in unrecognised format\n";
    expr->dump();
    abort();
  }

  struct smt_cache_entryt entry = {expr, a, ctx_level};
  smt_cache.insert(entry);

  return a;
}

void smt_convt::assert_expr(const expr2tc &e)
{
  assert_ast(convert_ast(e));
}

smt_sortt smt_convt::convert_sort(const type2tc &type)
{
  smt_sort_cachet::const_iterator it = sort_cache.find(type);
  if(it != sort_cache.end())
  {
    return it->second;
  }

  smt_sortt result = nullptr;
  switch(type->type_id)
  {
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
  {
    result = mk_int_bv_sort(type->get_width());
    break;
  }
  case type2t::signedbv_id:
  {
    result = mk_int_bv_sort(type->get_width());
    break;
  }
  case type2t::fixedbv_id:
  {
    unsigned int int_bits = to_fixedbv_type(type).integer_bits;
    unsigned int width = type->get_width();
    result = mk_real_fp_sort(int_bits, width - int_bits);
    break;
  }
  case type2t::floatbv_id:
  {
    unsigned int sw = to_floatbv_type(type).fraction;
    unsigned int ew = to_floatbv_type(type).exponent;
    result = mk_real_fp_sort(ew, sw);
    break;
  }
  case type2t::string_id:
  {
    const string_type2t &str_type = to_string_type(type);
    constant_int2tc width(
      get_uint_type(config.ansi_c.int_width), BigInt(str_type.width));
    type2tc new_type(new array_type2t(get_uint8_type(), width, false));
    result = convert_sort(new_type);
    break;
  }
  case type2t::array_id:
  {
    // Index arrays by the smallest integer required to represent its size.
    // Unless it's either infinite or dynamic in size, in which case use the
    // machine int size. Also, faff about if it's an array of arrays, extending
    // the domain.
    type2tc t = make_array_domain_type(to_array_type(flatten_array_type(type)));
    smt_sortt d = mk_int_bv_sort(t->get_width());

    // Determine the range if we have arrays of arrays.
    type2tc range = get_flattened_array_subtype(type);

    if(is_tuple_ast_type(range))
    {
      type2tc thetype = flatten_array_type(type);
      rewrite_ptrs_to_structs(thetype);
      result = tuple_api->mk_struct_sort(thetype);
      break;
    }

    // Work around QF_AUFBV demanding arrays of bitvectors.
    smt_sortt r;
    if(is_bool_type(range) && !array_api->supports_bools_in_arrays)
    {
      r = mk_int_bv_sort(1);
    }
    else
    {
      r = convert_sort(range);
    }

    result = mk_array_sort(d, r);
    break;
  }
  default:
    std::cerr << "Unexpected type ID " << get_type_id(type);
    std::cerr << " reached SMT conversion\n";
    abort();
  }

  sort_cache.insert(smt_sort_cachet::value_type(type, result));
  return result;
}

static std::string fixed_point(const std::string &v, unsigned width)
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

  if(fraction < 0)
    fraction = -fraction;

  fraction = fraction * precision;

  if(fraction == 0)
    result = double2string(integer);
  else
  {
    int64_t numerator = (integer * precision + fraction);
    result = itos(numerator) + "/" + double2string(precision);
  }

  return result;
}

smt_astt smt_convt::convert_terminal(const expr2tc &expr)
{
  switch(expr->expr_id)
  {
  case expr2t::constant_int_id:
  {
    const constant_int2t &theint = to_constant_int2t(expr);
    unsigned int width = expr->type->get_width();
    if(int_encoding)
      return mk_smt_int(theint.value, is_signedbv_type(expr));

    return mk_smt_bv(theint.value, width);
  }
  case expr2t::constant_fixedbv_id:
  {
    const constant_fixedbv2t &thereal = to_constant_fixedbv2t(expr);
    if(int_encoding)
    {
      std::string val = thereal.value.to_expr().value().as_string();
      std::string result = fixed_point(val, thereal.value.spec.width);
      return mk_smt_real(result);
    }

    assert(
      thereal.type->get_width() <= 64 &&
      "Converting fixedbv constant to"
      " SMT, too large to fit into a uint64_t");

    uint64_t magnitude, fraction, fin;
    unsigned int bitwidth = thereal.type->get_width();
    std::string m, f, c;
    std::string theval = thereal.value.to_expr().value().as_string();

    m = extract_magnitude(theval, bitwidth);
    f = extract_fraction(theval, bitwidth);
    magnitude = strtoll(m.c_str(), nullptr, 10);
    fraction = strtoll(f.c_str(), nullptr, 10);

    magnitude <<= (bitwidth / 2);
    fin = magnitude | fraction;

    return mk_smt_bv(mp_integer(fin), bitwidth);
  }
  case expr2t::constant_floatbv_id:
  {
    const constant_floatbv2t &thereal = to_constant_floatbv2t(expr);
    if(int_encoding)
    {
      std::string val = thereal.value.to_expr().value().as_string();
      std::string result = fixed_point(val, thereal.value.spec.width());
      return mk_smt_real(result);
    }

    unsigned int fraction_width = to_floatbv_type(thereal.type).fraction;
    unsigned int exponent_width = to_floatbv_type(thereal.type).exponent;
    if(thereal.value.is_NaN())
      return fp_api->mk_smt_fpbv_nan(exponent_width, fraction_width + 1);

    bool sign = thereal.value.get_sign();
    if(thereal.value.is_infinity())
      return fp_api->mk_smt_fpbv_inf(sign, exponent_width, fraction_width + 1);

    return fp_api->mk_smt_fpbv(thereal.value);
  }
  case expr2t::constant_bool_id:
  {
    const constant_bool2t &thebool = to_constant_bool2t(expr);
    return mk_smt_bool(thebool.value);
  }
  case expr2t::symbol_id:
  {
    // Special case for tuple symbols
    if(is_tuple_ast_type(expr))
    {
      const symbol2t &sym = to_symbol2t(expr);
      return tuple_api->mk_tuple_symbol(
        sym.get_symbol_name(), convert_sort(sym.type));
    }

    if(is_array_type(expr))
    {
      // Determine the range if we have arrays of arrays.
      type2tc range = get_flattened_array_subtype(expr->type);

      // If this is an array of structs, we have a tuple array sym.
      if(is_structure_type(range) || is_pointer_type(range))
        return tuple_api->mk_tuple_array_symbol(expr);
    }

    // Just a normal symbol. Possibly an array symbol.
    const symbol2t &sym = to_symbol2t(expr);
    std::string name = sym.get_symbol_name();

    smt_sortt sort = convert_sort(sym.type);

    if(is_array_type(expr))
    {
      smt_sortt subtype = convert_sort(get_flattened_array_subtype(sym.type));
      return array_api->mk_array_symbol(name, sort, subtype);
    }

    return mk_smt_symbol(name, sort);
  }

  default:
    std::cerr << "Converting unrecognized terminal expr to SMT\n";
    expr->dump();
    abort();
  }
}

std::string smt_convt::mk_fresh_name(const std::string &tag)
{
  std::string new_name = "smt_conv::" + tag;
  std::stringstream ss;
  ss << new_name << fresh_map[new_name]++;
  return ss.str();
}

smt_astt smt_convt::mk_fresh(
  smt_sortt s,
  const std::string &tag,
  smt_sortt array_subtype)
{
  std::string newname = mk_fresh_name(tag);

  if(s->id == SMT_SORT_STRUCT)
    return tuple_api->mk_tuple_symbol(newname, s);

  if(s->id == SMT_SORT_ARRAY)
  {
    assert(
      array_subtype != nullptr &&
      "Must call mk_fresh for arrays with a subtype");
    return array_api->mk_array_symbol(newname, s, array_subtype);
  }

  return mk_smt_symbol(newname, s);
}

smt_astt smt_convt::convert_is_nan(const expr2tc &expr)
{
  const isnan2t &isnan = to_isnan2t(expr);

  // Anything other than floats will never be NaNs
  if(!is_floatbv_type(isnan.value))
    return mk_smt_bool(false);

  smt_astt operand = convert_ast(isnan.value);
  return fp_api->mk_smt_fpbv_is_nan(operand);
}

smt_astt smt_convt::convert_is_inf(const expr2tc &expr)
{
  const isinf2t &isinf = to_isinf2t(expr);

  // Anything other than floats will never be infs
  if(!is_floatbv_type(isinf.value))
    return mk_smt_bool(false);

  smt_astt operand = convert_ast(isinf.value);
  return fp_api->mk_smt_fpbv_is_inf(operand);
}

smt_astt smt_convt::convert_is_normal(const expr2tc &expr)
{
  const isnormal2t &isnormal = to_isnormal2t(expr);

  // Anything other than floats will always be normal
  if(!is_floatbv_type(isnormal.value))
    return mk_smt_bool(true);

  smt_astt operand = convert_ast(isnormal.value);
  return fp_api->mk_smt_fpbv_is_normal(operand);
}

smt_astt smt_convt::convert_is_finite(const expr2tc &expr)
{
  const isfinite2t &isfinite = to_isfinite2t(expr);

  // Anything other than floats will always be finite
  if(!is_floatbv_type(isfinite.value))
    return mk_smt_bool(true);

  smt_astt value = convert_ast(isfinite.value);

  // isfinite = !(isinf || isnan)
  smt_astt isinf = fp_api->mk_smt_fpbv_is_inf(value);
  smt_astt isnan = fp_api->mk_smt_fpbv_is_nan(value);

  smt_astt or_op = mk_or(isinf, isnan);
  return mk_not(or_op);
}

smt_astt smt_convt::convert_signbit(const expr2tc &expr)
{
  const signbit2t &signbit = to_signbit2t(expr);

  // Since we can't extract the top bit, from the fpbv, we'll
  // convert it to return if(is_neg) ? 1 : 0;
  auto value = convert_ast(signbit.operand);

  // Create is_neg
  smt_astt is_neg;
  if(!config.ansi_c.use_fixed_for_float && !int_encoding)
    is_neg = fp_api->mk_smt_fpbv_is_negative(value);
  else
    // For fixedbvs, we check if it's < 0
    is_neg = mk_lt(value, convert_ast(gen_zero(signbit.operand->type)));

  // If it's true, return 1. Return 0, othewise.
  return mk_ite(
    is_neg,
    convert_ast(gen_one(signbit.type)),
    convert_ast(gen_zero(signbit.type)));
}

smt_astt smt_convt::convert_rounding_mode(const expr2tc &expr)
{
  // Easy case, we know the rounding mode
  if(is_constant_int2t(expr))
  {
    ieee_floatt::rounding_modet rm = static_cast<ieee_floatt::rounding_modet>(
      to_constant_int2t(expr).value.to_int64());
    return fp_api->mk_smt_fpbv_rm(rm);
  }

  assert(is_symbol2t(expr));
  // 0 is round to Nearest/even
  // 2 is round to +oo
  // 3 is round to -oo
  // 4 is round to zero

  smt_astt symbol = convert_ast(expr);

  smt_astt is_0 =
    mk_eq(symbol, mk_smt_bv(BigInt(0), symbol->sort->get_data_width()));

  smt_astt is_2 =
    mk_eq(symbol, mk_smt_bv(BigInt(2), symbol->sort->get_data_width()));

  smt_astt is_3 =
    mk_eq(symbol, mk_smt_bv(BigInt(3), symbol->sort->get_data_width()));

  smt_astt ne = fp_api->mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_EVEN);
  smt_astt mi = fp_api->mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_MINUS_INF);
  smt_astt pi = fp_api->mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_PLUS_INF);
  smt_astt ze = fp_api->mk_smt_fpbv_rm(ieee_floatt::ROUND_TO_ZERO);

  smt_astt ite2 = mk_ite(is_3, mi, ze);
  smt_astt ite1 = mk_ite(is_2, pi, ite2);
  smt_astt ite0 = mk_ite(is_0, ne, ite1);

  return ite0;
}

smt_astt smt_convt::convert_member(const expr2tc &expr)
{
  const member2t &member = to_member2t(expr);
  unsigned int idx = -1;

  assert(
    is_struct_type(member.source_value) ||
    is_pointer_type(member.source_value));
  idx = get_member_name_field(member.source_value->type, member.member);

  smt_astt src = convert_ast(member.source_value);
  return src->project(this, idx);
}

smt_astt smt_convt::round_real_to_int(smt_astt a)
{
  // SMT truncates downwards; however C truncates towards zero, which is not
  // the same. (Technically, it's also platform dependant). To get around this,
  // add one to the result in all circumstances, except where the value was
  // already an integer.
  smt_astt is_lt_zero = mk_lt(a, mk_smt_real("0"));

  // The actual conversion
  smt_astt as_int = mk_real2int(a);

  smt_astt one = mk_smt_int(BigInt(1), false);
  smt_astt plus_one = mk_add(one, as_int);

  // If it's an integer, just keep it's untruncated value.
  smt_astt is_int = mk_isint(a);
  smt_astt selected = mk_ite(is_int, as_int, plus_one);

  // Switch on whether it's > or < 0.
  return mk_ite(is_lt_zero, selected, as_int);
}

smt_astt smt_convt::round_fixedbv_to_int(
  smt_astt a,
  unsigned int fromwidth,
  unsigned int towidth)
{
  // Perform C rounding: just truncate towards zero. Annoyingly, this isn't
  // that simple for negative numbers, because they're represented as a negative
  // integer _plus_ a positive fraction. So we need to round up if there's a
  // nonzero fraction, and not if there's not.
  unsigned int frac_width = fromwidth / 2;

  // Determine whether the source is signed from its topmost bit.
  smt_astt is_neg_bit = mk_extract(a, fromwidth - 1, fromwidth - 1);
  smt_astt true_bit = mk_smt_bv(BigInt(1), 1);

  // Also collect data for dealing with the magnitude.
  smt_astt magnitude = mk_extract(a, fromwidth - 1, frac_width);
  smt_astt intvalue = mk_sign_ext(magnitude, frac_width);

  // Data for inspecting fraction part
  smt_astt frac_part = mk_extract(a, frac_width - 1, 0);
  smt_astt zero = mk_smt_bv(BigInt(0), frac_width);
  smt_astt is_zero_frac = mk_eq(frac_part, zero);

  // So, we have a base number (the magnitude), and need to decide whether to
  // round up or down. If it's positive, round down towards zero. If it's neg
  // and the fraction is zero, leave it, otherwise round towards zero.

  // We may need a value + 1.
  smt_astt one = mk_smt_bv(BigInt(1), towidth);
  smt_astt intvalue_plus_one = mk_bvadd(intvalue, one);

  smt_astt neg_val = mk_ite(is_zero_frac, intvalue, intvalue_plus_one);

  smt_astt is_neg = mk_eq(true_bit, is_neg_bit);

  // final switch
  return mk_ite(is_neg, neg_val, intvalue);
}

smt_astt smt_convt::make_bool_bit(smt_astt a)
{
  assert(
    a->sort->id == SMT_SORT_BOOL &&
    "Wrong sort fed to "
    "smt_convt::make_bool_bit");
  smt_astt one =
    (int_encoding) ? mk_smt_int(BigInt(1), false) : mk_smt_bv(BigInt(1), 1);
  smt_astt zero =
    (int_encoding) ? mk_smt_int(BigInt(0), false) : mk_smt_bv(BigInt(0), 1);
  return mk_ite(a, one, zero);
}

smt_astt smt_convt::make_bit_bool(smt_astt a)
{
  assert(
    ((!int_encoding && a->sort->id == SMT_SORT_BV) ||
     (int_encoding && a->sort->id == SMT_SORT_INT)) &&
    "Wrong sort fed to smt_convt::make_bit_bool");

  smt_astt one =
    (int_encoding) ? mk_smt_int(BigInt(1), false) : mk_smt_bv(BigInt(1), 1);
  return mk_eq(a, one);
}

expr2tc smt_convt::fix_array_idx(const expr2tc &idx, const type2tc &arr_sort)
{
  if(int_encoding)
    return idx;

  smt_sortt s = convert_sort(arr_sort);
  size_t domain_width = s->get_domain_width();
  if(domain_width == config.ansi_c.int_width)
    return idx;

  // Otherwise, we need to extract the lower bits out of this.
  return typecast2tc(
    get_uint_type(domain_width), idx, gen_zero(get_int32_type()));
}

unsigned long smt_convt::size_to_bit_width(unsigned long sz)
{
  uint64_t domwidth = 2;
  unsigned int dombits = 1;

  // Shift domwidth up until it's either larger or equal to sz, or we risk
  // overflowing.
  while(domwidth != 0x8000000000000000ULL && domwidth < sz)
  {
    domwidth <<= 1;
    dombits++;
  }

  if(domwidth == 0x8000000000000000ULL)
    dombits = 64;

  return dombits;
}

unsigned long smt_convt::calculate_array_domain_width(const array_type2t &arr)
{
  // Index arrays by the smallest integer required to represent its size.
  // Unless it's either infinite or dynamic in size, in which case use the
  // machine word size.
  if(!is_nil_expr(arr.array_size) && is_constant_int2t(arr.array_size))
  {
    constant_int2tc thesize = arr.array_size;
    return size_to_bit_width(thesize->value.to_ulong());
  }

  return config.ansi_c.word_size;
}

type2tc smt_convt::make_array_domain_type(const array_type2t &arr)
{
  // Start special casing if this is an array of arrays.
  if(!is_array_type(arr.subtype))
  {
    // Normal array, work out what the domain sort is.
    if(int_encoding)
      return get_uint_type(config.ansi_c.int_width);

    return get_uint_type(calculate_array_domain_width(arr));
  }

  // This is an array of arrays -- we're going to convert this into a single
  // array that has an extended domain. Work out that width.

  unsigned int domwidth = calculate_array_domain_width(arr);

  type2tc subarr = arr.subtype;
  while(is_array_type(subarr))
  {
    domwidth += calculate_array_domain_width(to_array_type(subarr));
    subarr = to_array_type(subarr).subtype;
  }

  return get_uint_type(domwidth);
}

expr2tc smt_convt::array_domain_to_width(const type2tc &type)
{
  const unsignedbv_type2t &uint = to_unsignedbv_type(type);
  uint64_t sz = 1ULL << uint.width;
  return constant_int2tc(index_type2(), BigInt(sz));
}

static expr2tc gen_additions(const type2tc &type, std::vector<expr2tc> &exprs)
{
  // Reached end of recursion
  if(exprs.size() == 2)
    return add2tc(type, exprs[0], exprs[1]);

  // Remove last two exprs
  expr2tc side1 = exprs.back();
  exprs.pop_back();

  expr2tc side2 = exprs.back();
  exprs.pop_back();

  // Add them together, push back to the vector and recurse
  exprs.push_back(add2tc(type, side1, side2));

  return gen_additions(type, exprs);
}

expr2tc smt_convt::decompose_select_chain(const expr2tc &expr, expr2tc &base)
{
  index2tc idx = expr;

  // First we need to find the flatten_array_type, to cast symbols/constants
  // with different types during the addition and multiplication. They'll be
  // casted to the flattened array index type
  while(is_index2t(idx->source_value))
    idx = idx->source_value;

  type2tc subtype = make_array_domain_type(
    to_array_type(flatten_array_type(idx->source_value->type)));

  // Rewrite the store chain as additions and multiplications
  idx = expr;

  // Multiplications will hold of the mult2tc terms, we have to
  // add them together in the end
  std::vector<expr2tc> multiplications;
  multiplications.push_back(typecast2tc(subtype, idx->index));

  while(is_index2t(idx->source_value))
  {
    idx = idx->source_value;

    type2tc t = flatten_array_type(idx->type);
    assert(is_array_type(t));

    multiplications.push_back(mul2tc(
      subtype,
      typecast2tc(subtype, to_array_type(t).array_size),
      typecast2tc(subtype, idx->index)));
  }

  // We should only enter this method when handling multidimensional arrays
  assert(multiplications.size() != 1);

  // Add them together
  expr2tc output = gen_additions(subtype, multiplications);

  // Try to simplify the expression
  simplify(output);

  // Give the caller the base array object / thing. So that it can actually
  // select out of the right piece of data.
  base = idx->source_value;
  return output;
}

expr2tc smt_convt::decompose_store_chain(
  const expr2tc &expr,
  expr2tc &update_val)
{
  with2tc with = expr;

  // First we need to find the flatten_array_type, to cast symbols/constants
  // with different types during the addition and multiplication. They'll be
  // casted to the flattened array index type
  type2tc subtype = make_array_domain_type(
    to_array_type(flatten_array_type(with->source_value->type)));

  // Multiplications will hold of the mult2tc terms, we have to
  // add them together in the end
  std::vector<expr2tc> multiplications;

  assert(is_array_type(with->update_value));
  multiplications.push_back(mul2tc(
    subtype,
    typecast2tc(
      subtype,
      to_array_type(flatten_array_type(with->update_value->type)).array_size),
    typecast2tc(subtype, with->update_field)));

  while(is_with2t(with->update_value) && is_array_type(with->update_value))
  {
    with = with->update_value;

    type2tc t = flatten_array_type(with->update_value->type);

    multiplications.push_back(mul2tc(
      subtype,
      typecast2tc(subtype, to_array_type(t).array_size),
      typecast2tc(subtype, with->update_field)));
  }

  // We should only enter this method when handling multidimensional arrays
  assert(multiplications.size() != 1);

  // Add them together
  expr2tc output = gen_additions(subtype, multiplications);

  // Try to simplify the expression
  simplify(output);

  // Fix base expr
  update_val = with->update_value;
  return output;
}

smt_astt smt_convt::convert_array_index(const expr2tc &expr)
{
  smt_astt a;
  const index2t &index = to_index2t(expr);
  expr2tc src_value = index.source_value;

  expr2tc newidx;
  if(is_index2t(index.source_value))
  {
    newidx = decompose_select_chain(expr, src_value);
  }
  else
  {
    newidx = fix_array_idx(index.index, index.source_value->type);
  }

  // Firstly, if it's a string, shortcircuit.
  if(is_string_type(index.source_value))
  {
    smt_astt tmp = convert_ast(src_value);
    return tmp->select(this, newidx);
  }

  a = convert_ast(src_value);
  a = a->select(this, newidx);

  const array_type2t &arrtype = to_array_type(index.source_value->type);
  if(is_bool_type(arrtype.subtype) && !array_api->supports_bools_in_arrays)
    return make_bit_bool(a);

  return a;
}

smt_astt smt_convt::convert_array_store(const expr2tc &expr)
{
  const with2t &with = to_with2t(expr);
  expr2tc update_val = with.update_value;
  expr2tc newidx;

  if(
    is_array_type(with.type) && is_array_type(to_array_type(with.type).subtype))
  {
    newidx = decompose_store_chain(expr, update_val);
  }
  else
  {
    newidx = fix_array_idx(with.update_field, with.type);
  }

  assert(is_array_type(expr->type));
  smt_astt src, update;
  const array_type2t &arrtype = to_array_type(expr->type);

  // Workaround for bools-in-arrays.
  if(is_bool_type(arrtype.subtype) && !array_api->supports_bools_in_arrays)
  {
    typecast2tc cast(get_uint_type(1), update_val);
    update = convert_ast(cast);
  }
  else
  {
    update = convert_ast(update_val);
  }

  src = convert_ast(with.source_value);
  return src->update(this, update, 0, newidx);
}

type2tc smt_convt::flatten_array_type(const type2tc &type)
{
  // If this is not an array, we return an array of size 1
  if(!is_array_type(type))
    return array_type2tc(type, gen_one(int_type2()), false);

  // Don't touch these
  if(to_array_type(type).size_is_infinite)
    return type;

  // No need to handle one dimensional arrays
  if(!is_array_type(to_array_type(type).subtype))
    return type;

  type2tc subtype = get_flattened_array_subtype(type);
  assert(is_array_type(to_array_type(type).subtype));

  type2tc type_rec = type;
  expr2tc arr_size1 = to_array_type(type_rec).array_size;

  type_rec = to_array_type(type_rec).subtype;
  expr2tc arr_size2 = to_array_type(type_rec).array_size;

  assert(arr_size1->type == arr_size2->type);

  expr2tc arr_size = mul2tc(arr_size1->type, arr_size1, arr_size2);

  while(is_array_type(to_array_type(type_rec).subtype))
  {
    arr_size =
      mul2tc(arr_size1->type, to_array_type(type_rec).array_size, arr_size);
    type_rec = to_array_type(type_rec).subtype;
  }
  simplify(arr_size);
  return array_type2tc(subtype, arr_size, false);
}

expr2tc smt_convt::flatten_array_body(const expr2tc &expr)
{
  assert(is_constant_array2t(expr));
  const constant_array2t &the_array = to_constant_array2t(expr);
  const array_type2t &arr_type = to_array_type(the_array.type);

  // inner most level, just return the array
  if(!is_array_type(arr_type.subtype))
    return expr;

// This should be an array of arrays, glue the sub arrays together
#ifndef NDEBUG
  for(auto const &elem : the_array.datatype_members)
    // Must only contain constant arrays, for now. No indirection should be
    // expressable at this level.
    assert(
      is_constant_array2t(elem) &&
      "Sub-member of constant array must be "
      "constant array");
#endif

  std::vector<expr2tc> sub_expr_list;
  for(auto const &elem : the_array.datatype_members)
  {
    expr2tc flatten_elem = flatten_array_body(elem);

    sub_expr_list.insert(
      sub_expr_list.end(),
      to_constant_array2t(flatten_elem).datatype_members.begin(),
      to_constant_array2t(flatten_elem).datatype_members.end());
  }

  return constant_array2tc(flatten_array_type(expr->type), sub_expr_list);
}

type2tc smt_convt::get_flattened_array_subtype(const type2tc &type)
{
  // Get the subtype of an array, ensuring that any intermediate arrays have
  // been flattened.

  type2tc type_rec = type;
  while(is_array_type(type_rec))
  {
    type_rec = to_array_type(type_rec).subtype;
  }

  // type_rec is now the base type.
  return type_rec;
}

void smt_convt::pre_solve()
{
  // NB: always perform tuple constraint adding first, as it covers tuple
  // arrays too, and might end up generating more ASTs to be encoded in
  // the array api class.
  tuple_api->add_tuple_constraints_for_solving();
  array_api->add_array_constraints_for_solving();
}

expr2tc smt_convt::get(const expr2tc &expr)
{
  if(is_constant_number(expr))
    return expr;

  if(is_symbol2t(expr) && to_symbol2t(expr).thename == "NULL")
    return expr;

  expr2tc res = expr;

  // Special cases:
  switch(res->expr_id)
  {
  case expr2t::index_id:
  {
    // If we try to get an index from the solver, it will first
    // return the whole array and then get the index, we can
    // do better and call get_array_element directly
    index2t index = to_index2t(res);
    expr2tc src_value = index.source_value;

    expr2tc newidx;
    if(is_index2t(index.source_value))
    {
      newidx = decompose_select_chain(expr, src_value);
    }
    else
    {
      newidx = fix_array_idx(index.index, index.source_value->type);
    }

    // if the source value is a constant, there's no need to
    // call the array api
    if(is_constant_number(src_value))
      return src_value;

    // Convert the idx, it must be an integer
    expr2tc idx = get(newidx);
    if(is_constant_int2t(idx))
    {
      // Convert the array so we can call the array api
      smt_astt array = convert_ast(src_value);

      // Retrieve the element
      res = array_api->get_array_elem(
        array,
        to_constant_int2t(idx).value.to_uint64(),
        get_flattened_array_subtype(res->type));
    }
    break;
  }

  case expr2t::with_id:
  {
    // This will be converted
    with2t with = to_with2t(res);
    expr2tc update_val = with.update_value;

    if(
      is_array_type(with.type) &&
      is_array_type(to_array_type(with.type).subtype))
    {
      decompose_store_chain(expr, update_val);
    }

    return get(update_val);
  }

  case expr2t::address_of_id:
    return res;

  case expr2t::symbol_id:
    if(is_structure_type(res))
      return res;

    // Query symbol value from the solver
    return get_by_type(res);

  default:
    break;
  }

  // Recurse on operands
  res->Foreach_operand([this](expr2tc &e) {
    expr2tc new_e = get(e);
    e = new_e;
  });

  // And simplify
  simplify(res);
  return res;
}

expr2tc smt_convt::get_by_ast(const type2tc &type, smt_astt a)
{
  switch(type->type_id)
  {
  case type2t::bool_id:
    return get_bool(a) ? gen_true_expr() : gen_false_expr();

  case type2t::unsignedbv_id:
  case type2t::signedbv_id:
  case type2t::fixedbv_id:
    return build_bv(type, get_bv(a));

  case type2t::floatbv_id:
    return constant_floatbv2tc(fp_api->get_fpbv(a));

  default:
    std::cerr << "Unimplemented type'd expression (" << type->type_id
              << ") in smt get\n";
    abort();
  }
}

expr2tc smt_convt::get_by_type(const expr2tc &expr)
{
  switch(expr->type->type_id)
  {
  case type2t::bool_id:
  case type2t::unsignedbv_id:
  case type2t::signedbv_id:
  case type2t::fixedbv_id:
  case type2t::floatbv_id:
    return get_by_ast(expr->type, convert_ast(expr));

  case type2t::array_id:
    return get_array(expr);

  case type2t::struct_id:
  case type2t::pointer_id:
    return tuple_api->tuple_get(expr);

  default:
    std::cerr << "Unimplemented type'd expression (" << expr->type->type_id
              << ") in smt get\n";
    abort();
  }
}

expr2tc smt_convt::get_array(const expr2tc &expr)
{
  // XXX -- printing multidimensional arrays?

  smt_astt array = convert_ast(expr);

  // Fetch the array bounds, if it's huge then assume this is a 1024 element
  // array. Then fetch all elements and formulate a constant_array.
  size_t w = array->sort->get_domain_width();
  if(w > 10)
    w = 10;

  const array_type2t &ar = to_array_type(flatten_array_type(expr->type));
  constant_int2tc arr_size(index_type2(), BigInt(1 << w));
  type2tc arr_type = type2tc(new array_type2t(ar.subtype, arr_size, false));
  std::vector<expr2tc> fields;

  for(size_t i = 0; i < (1ULL << w); i++)
  {
    fields.push_back(array_api->get_array_elem(array, i, ar.subtype));
  }

  return constant_array2tc(arr_type, fields);
}

const struct_union_data &smt_convt::get_type_def(const type2tc &type) const
{
  return (is_pointer_type(type))
           ? *pointer_type_data
           : dynamic_cast<const struct_union_data &>(*type.get());
}

smt_astt smt_convt::array_create(const expr2tc &expr)
{
  if(is_constant_array_of2t(expr))
    return convert_array_of_prep(expr);

  // Handle constant array expressions: these don't have tuple type and so
  // don't need funky handling, but we need to create a fresh new symbol and
  // repeatedly store the desired data into it, to create an SMT array
  // representing the expression we're converting.
  std::string name = mk_fresh_name("array_create::") + ".";
  expr2tc newsym = symbol2tc(expr->type, name);

  // Check size
  const array_type2t &arr_type = to_array_type(expr->type);

  // Guarentee nothing, this is modelling only.
  if(arr_type.size_is_infinite)
    return convert_ast(newsym);

  if(!is_constant_int2t(arr_type.array_size))
  {
    std::cerr << "Non-constant sized array of type constant_array_of2t\n";
    abort();
  }

  const constant_int2t &thesize = to_constant_int2t(arr_type.array_size);
  uint64_t sz = thesize.value.to_ulong();

  assert(is_constant_array2t(expr));
  const constant_array2t &array = to_constant_array2t(expr);

  // Repeatedly store things into this.
  smt_astt newsym_ast = convert_ast(newsym);
  for(unsigned int i = 0; i < sz; i++)
  {
    expr2tc init = array.datatype_members[i];

    // Workaround for bools-in-arrays
    if(
      is_bool_type(array.datatype_members[i]->type) && !int_encoding &&
      !array_api->supports_bools_in_arrays)
      init = typecast2tc(type2tc(new unsignedbv_type2t(1)), init);

    newsym_ast = newsym_ast->update(this, convert_ast(init), i);
  }

  return newsym_ast;
}

smt_astt smt_convt::convert_array_of_prep(const expr2tc &expr)
{
  const constant_array_of2t &arrof = to_constant_array_of2t(expr);
  const array_type2t &arrtype = to_array_type(arrof.type);
  expr2tc base_init;
  unsigned long array_size = 0;

  // So: we have an array_of, that we have to convert into a bunch of stores.
  // However, it might be a nested array. If that's the case, then we're
  // guarenteed to have another array_of in the initializer which we can flatten
  // to a single array of whatever's at the bottom of the array_of. Or, it's
  // a constant_array, in which case we can just copy the contents.
  if(is_array_type(arrtype.subtype))
  {
    expr2tc rec_expr = expr;

    if(is_constant_array_of2t(to_constant_array_of2t(rec_expr).initializer))
    {
      type2tc flat_type = flatten_array_type(expr->type);
      const array_type2t &arrtype2 = to_array_type(flat_type);
      array_size = calculate_array_domain_width(arrtype2);

      while(is_constant_array_of2t(rec_expr))
        rec_expr = to_constant_array_of2t(rec_expr).initializer;

      base_init = rec_expr;
    }
    else
    {
      const constant_array_of2t &arrof = to_constant_array_of2t(rec_expr);
      assert(is_constant_array2t(arrof.initializer));
      const constant_array2t &constarray =
        to_constant_array2t(arrof.initializer);
      const array_type2t &constarray_type = to_array_type(constarray.type);

      // Duplicate contents repeatedly.
      assert(
        is_constant_int2t(arrtype.array_size) &&
        "Cannot have complex nondet-sized array_of initializers");
      const BigInt &size = to_constant_int2t(arrtype.array_size).value;

      std::vector<expr2tc> new_contents;
      for(uint64_t i = 0; i < size.to_uint64(); i++)
        new_contents.insert(
          new_contents.end(),
          constarray.datatype_members.begin(),
          constarray.datatype_members.end());

      // Create new expression, convert and return that.
      mul2tc newsize(
        arrtype.array_size->type,
        arrtype.array_size,
        constarray_type.array_size);
      simplify(newsize);

      type2tc new_arr_type(
        new array_type2t(constarray_type.subtype, newsize, false));
      constant_array2tc new_const_array(new_arr_type, new_contents);
      return convert_ast(new_const_array);
    }
  }
  else
  {
    base_init = arrof.initializer;
    array_size = calculate_array_domain_width(arrtype);
  }

  if(is_structure_type(base_init->type))
    return tuple_api->tuple_array_of(base_init, array_size);
  if(is_pointer_type(base_init->type))
    return pointer_array_of(base_init, array_size);
  else
    return array_api->convert_array_of(convert_ast(base_init), array_size);
}

smt_astt array_iface::default_convert_array_of(
  smt_astt init_val,
  unsigned long array_size,
  smt_convt *ctx)
{
  // We now an initializer, and a size of array to build. So:
  // Repeatedly store things into this.
  // XXX int mode

  if(init_val->sort->id == SMT_SORT_BOOL && !supports_bools_in_arrays)
  {
    smt_astt zero = ctx->mk_smt_bv(BigInt(0), 1);
    smt_astt one = ctx->mk_smt_bv(BigInt(0), 1);
    init_val = ctx->mk_ite(init_val, one, zero);
  }

  smt_sortt domwidth = ctx->mk_int_bv_sort(array_size);
  smt_sortt arrsort = ctx->mk_array_sort(domwidth, init_val->sort);
  smt_astt newsym_ast =
    ctx->mk_fresh(arrsort, "default_array_of::", init_val->sort);

  unsigned long sz = 1ULL << array_size;
  for(unsigned long i = 0; i < sz; i++)
  {
    newsym_ast = newsym_ast->update(ctx, init_val, i);
  }

  return newsym_ast;
}

smt_astt smt_convt::pointer_array_of(
  const expr2tc &init_val __attribute__((unused)),
  unsigned long array_width)
{
  // Actually a tuple, but the operand is going to be a symbol, null.
  assert(
    is_symbol2t(init_val) &&
    "Pointer type'd array_of can only be an "
    "array of null");

#ifndef NDEBUG
  const symbol2t &sym = to_symbol2t(init_val);
  assert(
    sym.thename == "NULL" &&
    "Pointer type'd array_of can only be an "
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

smt_astt smt_convt::tuple_array_create_despatch(
  const expr2tc &expr,
  smt_sortt domain)
{
  // Take a constant_array2t or an array_of, and format the data from them into
  // a form palatable to tuple_array_create.

  // Strip out any pointers
  type2tc arr_type = expr->type;
  rewrite_ptrs_to_structs(arr_type);

  if(is_constant_array_of2t(expr))
  {
    const constant_array_of2t &arr = to_constant_array_of2t(expr);
    smt_astt arg = convert_ast(arr.initializer);

    return tuple_api->tuple_array_create(arr_type, &arg, true, domain);
  }

  assert(is_constant_array2t(expr));
  const constant_array2t &arr = to_constant_array2t(expr);
  smt_astt args[arr.datatype_members.size()];
  unsigned int i = 0;
  for(auto const &it : arr.datatype_members)
  {
    args[i] = convert_ast(it);
    i++;
  }

  return tuple_api->tuple_array_create(arr_type, args, false, domain);
}

void smt_convt::rewrite_ptrs_to_structs(type2tc &type)
{
  // Type may contain pointers; replace those with the structure equivalent.
  // Ideally the real solver will never see pointer types.
  // Create a delegate that recurses over all subtypes, replacing pointers
  // as we go. Extra scaffolding is to work around the fact we can't refer
  // to replace_w_ptr until after it's been defined, ho hum.
  type2t::subtype_delegate *delegate = nullptr;
  auto replace_w_ptr = [this, &delegate](type2tc &e) {
    if(is_pointer_type(e))
    {
      // Replace this field of the expr with a pointer struct :O:O:O:O
      e = pointer_struct;
    }
    else
    {
      // Recurse
      e->Foreach_subtype(*delegate);
    }
  };

  type2t::subtype_delegate del_wrap(std::ref(replace_w_ptr));
  delegate = &del_wrap;
  type->Foreach_subtype(replace_w_ptr);
}

// Default behaviours for SMT AST's

void smt_ast::assign(smt_convt *ctx, smt_astt sym) const
{
  ctx->assert_ast(eq(ctx, sym));
}

smt_astt smt_ast::ite(smt_convt *ctx, smt_astt cond, smt_astt falseop) const
{
  return ctx->mk_ite(cond, this, falseop);
}

smt_astt smt_ast::eq(smt_convt *ctx, smt_astt other) const
{
  // Simple approach: this is a leaf piece of SMT, compute a basic equality.
  return ctx->mk_eq(this, other);
}

smt_astt smt_ast::update(
  smt_convt *ctx,
  smt_astt value,
  unsigned int idx,
  expr2tc idx_expr) const
{
  // If we're having an update applied to us, then the only valid situation
  // this can occur in is if we're an array.
  assert(sort->id == SMT_SORT_ARRAY);

  // We're an array; just generate a 'with' operation.
  expr2tc index;
  if(is_nil_expr(idx_expr))
  {
    size_t dom_width =
      ctx->int_encoding ? config.ansi_c.int_width : sort->get_domain_width();
    index = constant_int2tc(unsignedbv_type2tc(dom_width), BigInt(idx));
  }
  else
  {
    index = idx_expr;
  }

  return ctx->mk_store(this, ctx->convert_ast(index), value);
}

smt_astt smt_ast::select(smt_convt *ctx, const expr2tc &idx) const
{
  assert(
    sort->id == SMT_SORT_ARRAY &&
    "Select operation applied to non-array scalar AST");

  const smt_ast *args[2];
  args[0] = this;
  args[1] = ctx->convert_ast(idx);
  return ctx->mk_select(args[0], args[1]);
}

smt_astt smt_ast::project(
  smt_convt *ctx __attribute__((unused)),
  unsigned int idx __attribute__((unused))) const
{
  std::cerr << "Projecting from non-tuple based AST\n";
  abort();
}

void smt_convt::dump_smt()
{
  std::cerr << "SMT dump not implemented for " << solver_text() << "\n";
}

void smt_convt::print_model()
{
  std::cerr << "SMT model printing not implemented for " << solver_text()
            << "\n";
}

tvt smt_convt::l_get(smt_astt a)
{
  return get_bool(a) ? tvt(true) : tvt(false);
}

expr2tc smt_convt::build_bv(const type2tc &type, BigInt value)
{
  if(is_fixedbv_type(type))
  {
    fixedbvt fbv(constant_exprt(
      integer2binary(value, type->get_width()),
      integer2string(value),
      migrate_type_back(type)));
    return constant_fixedbv2tc(fbv);
  }
  assert(is_bv_type(type));
  return constant_int2tc(type, value);
}

smt_sortt smt_convt::mk_bool_sort()
{
  std::cerr << "Chosen solver doesn't support boolean sorts\n";
  abort();
}

smt_sortt smt_convt::mk_real_sort()
{
  std::cerr << "Chosen solver doesn't support real sorts\n";
  abort();
}

smt_sortt smt_convt::mk_int_sort()
{
  std::cerr << "Chosen solver doesn't support integer sorts\n";
  abort();
}

smt_sortt smt_convt::mk_bv_sort(std::size_t width)
{
  std::cerr << "Chosen solver doesn't support bit vector sorts\n";
  (void)width;
  abort();
}

smt_sortt smt_convt::mk_fbv_sort(std::size_t width)
{
  std::cerr << "Chosen solver doesn't support bit vector sorts\n";
  (void)width;
  abort();
}

smt_sortt smt_convt::mk_array_sort(smt_sortt domain, smt_sortt range)
{
  std::cerr << "Chosen solver doesn't support array sorts\n";
  (void)domain;
  (void)range;
  abort();
}

smt_sortt smt_convt::mk_bvfp_sort(std::size_t ew, std::size_t sw)
{
  std::cerr << "Chosen solver doesn't support bit vector sorts\n";
  (void)ew;
  (void)sw;
  abort();
}

smt_sortt smt_convt::mk_bvfp_rm_sort()
{
  std::cerr << "Chosen solver doesn't support bit vector sorts\n";
  abort();
}

smt_astt smt_convt::mk_bvredor(smt_astt op)
{
  // bvredor = bvnot(bvcomp(x,0)) ? bv1 : bv0;

  smt_astt comp = mk_eq(op, mk_smt_bv(BigInt(0), op->sort->get_data_width()));

  smt_astt ncomp = mk_not(comp);

  // If it's true, return 1. Return 0, othewise.
  return mk_ite(ncomp, mk_smt_bv(1, 1), mk_smt_bv(BigInt(0), 1));
}

smt_astt smt_convt::mk_bvredand(smt_astt op)
{
  // bvredand = bvcomp(x,-1) ? bv1 : bv0;

  smt_astt comp =
    mk_eq(op, mk_smt_bv(BigInt(ULLONG_MAX), op->sort->get_data_width()));

  // If it's true, return 1. Return 0, othewise.
  return mk_ite(comp, mk_smt_bv(1, 1), mk_smt_bv(BigInt(0), 1));
}

smt_astt smt_convt::mk_add(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_bvadd(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_sub(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_bvsub(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_mul(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_bvmul(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_mod(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_bvsmod(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_bvumod(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_div(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_bvsdiv(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_bvudiv(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_shl(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_bvshl(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_bvashr(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_bvlshr(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_neg(smt_astt a)
{
  (void)a;
  abort();
}

smt_astt smt_convt::mk_bvneg(smt_astt a)
{
  (void)a;
  abort();
}

smt_astt smt_convt::mk_bvnot(smt_astt a)
{
  (void)a;
  abort();
}

smt_astt smt_convt::mk_bvnxor(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_bvnor(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_bvnand(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_bvxor(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_bvor(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_bvand(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_implies(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_xor(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_or(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_and(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_not(smt_astt a)
{
  (void)a;
  abort();
}

smt_astt smt_convt::mk_lt(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_bvult(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_bvslt(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_gt(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  return mk_lt(b, a);
}

smt_astt smt_convt::mk_bvugt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return mk_not(mk_bvule(a, b));
}

smt_astt smt_convt::mk_bvsgt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return mk_not(mk_bvsle(a, b));
}

smt_astt smt_convt::mk_le(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_bvule(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_bvsle(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_ge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return mk_not(mk_lt(a, b));
}

smt_astt smt_convt::mk_bvuge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return mk_not(mk_bvult(a, b));
}

smt_astt smt_convt::mk_bvsge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return mk_not(mk_bvslt(a, b));
}

smt_astt smt_convt::mk_eq(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_neq(smt_astt a, smt_astt b)
{
  return mk_not(mk_eq(a, b));
}

smt_astt smt_convt::mk_store(smt_astt a, smt_astt b, smt_astt c)
{
  (void)a;
  (void)b;
  (void)c;
  abort();
}

smt_astt smt_convt::mk_select(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_convt::mk_real2int(smt_astt a)
{
  (void)a;
  abort();
}

smt_astt smt_convt::mk_int2real(smt_astt a)
{
  (void)a;
  abort();
}

smt_astt smt_convt::mk_isint(smt_astt a)
{
  (void)a;
  abort();
}
