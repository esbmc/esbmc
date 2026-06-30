#include "irep2/irep2_expr.h"
#include <cfloat>
#include <iomanip>
#include <solvers/prop/literal.h>
#include <solvers/smt/smt_solver.h>
#include <solvers/smt/fp/ir_ieee_conv.h>
#include <solvers/smt/smt_fp_rounding_utils.h>
#include <sstream>
#include <util/arith_tools.h>
#include <util/base_type.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/message.h>
#include <util/message/format.h>
#include <util/type_byte_size.h>
#include <cmath>
#include <limits>

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

unsigned int smt_solver_baset::get_member_name_field(
  const type2tc &t,
  const irep_idt &name) const
{
  unsigned int idx = 0;
  // Pointer types lower to the synthetic pointer_struct tuple in SMT;
  // for them the named lookup uses pointer_struct's member_names.
  const std::vector<irep_idt> &names =
    struct_union_member_names(is_pointer_type(t) ? pointer_struct : t);

  for (const irep_idt &it : names)
  {
    if (it == name)
      break;
    idx++;
  }
  assert(
    idx != names.size() && "Member name of with expr not found in struct type");

  return idx;
}

unsigned int smt_solver_baset::get_member_name_field(
  const type2tc &t,
  const expr2tc &name) const
{
  const constant_string2t &str = to_constant_string2t(name);
  return get_member_name_field(t, str.value);
}

smt_solver_baset::smt_solver_baset(
  const namespacet &_ns,
  const optionst &_options)
  : ctx_level(0), boolean_sort(nullptr), ns(_ns), options(_options)
{
  int_encoding = options.get_bool_option("int-encoding");
  ir_ieee = options.get_bool_option("ir-ieee");
  tuple_api = nullptr;
  array_api = nullptr;
  fp_api = nullptr;
  ra_api = nullptr;
  ir_ieee_api = std::make_unique<ir_ieee_convt>(this);

  std::vector<type2tc> members;
  std::vector<irep_idt> names;

  /* TODO: pointer_object is actually identified by an 'unsigned int' number */
  members.push_back(ptraddr_type2()); /* CHERI-TODO */
  members.push_back(ptraddr_type2());
  names.emplace_back("pointer_object");
  names.emplace_back("pointer_offset");
  if (config.ansi_c.cheri)
  {
    members.push_back(ptraddr_type2());
    names.emplace_back("pointer_cap_info");
  }

  pointer_struct = struct_type2tc(members, names, names, "pointer_struct");

  pointer_logic.emplace_back();

  addr_space_sym_num.push_back(0);

  renumber_map.emplace_back();

  members.clear();
  names.clear();
  members.push_back(ptraddr_type2()); /* CHERI-TODO */
  members.push_back(ptraddr_type2()); /* CHERI-TODO */
  names.emplace_back("start");
  names.emplace_back("end");
  addr_space_type = struct_type2tc(members, names, names, "addr_space_type");

  /* indexed by pointer_object2t expressions */
  addr_space_arr_type = array_type2tc(addr_space_type, expr2tc(), true);

  addr_space_data.emplace_back();

  machine_ptr = get_uint_type(config.ansi_c.pointer_width()); /* CHERI-TODO */

  ptr_foo_inited = false;
}

smt_solver_baset::~smt_solver_baset() = default;

void smt_solver_baset::set_tuple_iface(tuple_iface *iface)
{
  assert(tuple_api == nullptr && "set_tuple_iface should only be called once");
  tuple_api = iface;
}

void smt_solver_baset::set_array_iface(array_iface *iface)
{
  assert(array_api == nullptr && "set_array_iface should only be called once");
  array_api = iface;
}

void smt_solver_baset::set_fp_conv(fp_convt *iface)
{
  assert(fp_api == NULL && "set_fp_iface should only be called once");
  fp_api = iface;
}

void smt_solver_baset::set_ra_conv(ra_apit *iface)
{
  assert(ra_api == NULL && "set_ra_conv should only be called once");
  ra_api = iface;
}

void smt_solver_baset::delete_all_asts()
{
  // Erase all the remaining asts in the live ast vector.
  for (auto *ast : live_asts)
    delete ast;
  live_asts.clear();
}

void smt_solver_baset::smt_post_init()
{
  boolean_sort = mk_bool_sort();

  init_addr_space_array();

  if (int_encoding)
  {
    std::vector<expr2tc> power_array_data;
    uint64_t pow;
    unsigned int count = 0;
    type2tc powarr_elemt = get_uint_type(64);
    for (pow = 1ULL; count < 64; pow <<= 1, count++)
      power_array_data.push_back(constant_int2tc(powarr_elemt, BigInt(pow)));

    type2tc power_array_type =
      array_type2tc(powarr_elemt, gen_ulong(64), false);

    expr2tc power_array = constant_array2tc(power_array_type, power_array_data);
    int_shift_op_array = convert_ast(power_array);
  }

  ptr_foo_inited = true;
}

void smt_solver_baset::push_ctx()
{
  // Any context change can change the model; drop memoised l_get values.
  l_get_cache.clear();

  tuple_api->push_tuple_ctx();
  array_api->push_array_ctx();

  addr_space_data.push_back(addr_space_data.back());
  addr_space_sym_num.push_back(addr_space_sym_num.back());
  pointer_logic.push_back(pointer_logic.back());
  renumber_map.push_back(renumber_map.back());

  live_asts_sizes.push_back(live_asts.size());

  ctx_level++;
}

/** Replace all occurrences of the named symbol @p lhs with @p replacement
 *  throughout @p body (in-place). */
static void replace_name_in_body(
  const expr2tc &lhs,
  const expr2tc &replacement,
  expr2tc &body)
{
  assert(is_symbol2t(lhs));
  if (is_symbol2t(body))
  {
    if (to_symbol2t(body).thename == to_symbol2t(lhs).thename)
      body = replacement;
    return;
  }
  body->Foreach_operand([&lhs, &replacement](expr2tc &e) {
    replace_name_in_body(lhs, replacement, e);
  });
}

/** Recursively expand any symbol in @p e that is a key in @p defs, replacing
 *  it with (a clone of) its associated forall/exists expression.  This
 *  inlines nested quantifier bodies so that replace_name_in_body can
 *  substitute the outer bound variable into their predicates. */
static void expand_quantifier_defs_in(
  expr2tc &e,
  const std::unordered_map<irep_idt, expr2tc, irep_id_hash> &defs)
{
  if (is_symbol2t(e))
  {
    auto it = defs.find(to_symbol2t(e).thename);
    if (it != defs.end())
    {
      e = it->second->clone();
      expand_quantifier_defs_in(e, defs);
    }
    return;
  }
  e->Foreach_operand(
    [&defs](expr2tc &sub) { expand_quantifier_defs_in(sub, defs); });
}

void smt_solver_baset::pop_ctx()
{
  // Any context change can change the model; drop memoised l_get values.
  l_get_cache.clear();

  // Erase everything in caches added in the current context level. Everything
  // before the push is going to disappear.
  smt_cachet::nth_index<1>::type &cache_numindex = smt_cache.get<1>();
  cache_numindex.erase(ctx_level);

  // Drop Ackermann-fallback history recorded at this level; pop_ctx is about to
  // delete the asts those entries point at (see below), so they must not be
  // referenced by a later application of the same function.
  for (auto it = uf_ackermann_history.begin();
       it != uf_ackermann_history.end();)
  {
    auto &entries = it->second;
    std::erase_if(entries, [this](const uf_ackermann_entry &e) {
      return e.level >= ctx_level;
    });
    it = entries.empty() ? uf_ackermann_history.erase(it) : std::next(it);
  }

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

smt_astt smt_solver_baset::invert_ast(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_BOOL);
  return mk_not(a);
}

smt_astt smt_solver_baset::imply_ast(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  return mk_implies(a, b);
}

smt_astt smt_solver_baset::convert_concat_int_mode(
  smt_astt left_ast,
  smt_astt right_ast,
  const expr2tc &expr)
{
  const concat2t &concat_expr = to_concat2t(expr);

  // Get the widths of the operands
  unsigned int left_width = concat_expr.side_1->type->get_width();
  unsigned int right_width = concat_expr.side_2->type->get_width();
  unsigned int result_width = left_width + right_width;

  // Create the result type
  type2tc result_type = get_uint_type(result_width);

  // Convert concatenation to mathematical operations:
  // result = left * (2^right_width) + right

  // Calculate 2^right_width
  BigInt multiplier = BigInt(1);
  for (unsigned int i = 0; i < right_width; i++)
    multiplier = multiplier * BigInt(2);

  // Create the multiplier constant
  expr2tc multiplier_expr = constant_int2tc(result_type, multiplier);
  smt_astt multiplier_ast = convert_ast(multiplier_expr);

  // Convert operands to the result type if needed
  smt_astt left_converted = left_ast;
  smt_astt right_converted = right_ast;

  if (concat_expr.side_1->type->get_width() != result_width)
  {
    expr2tc left_extended = typecast2tc(result_type, concat_expr.side_1);
    left_converted = convert_ast(left_extended);
  }

  if (concat_expr.side_2->type->get_width() != result_width)
  {
    expr2tc right_extended = typecast2tc(result_type, concat_expr.side_2);
    right_converted = convert_ast(right_extended);
  }

  // Perform left * multiplier
  smt_astt shifted_left = mk_mul(left_converted, multiplier_ast);

  // Add the right operand: (left * 2^right_width) + right
  smt_astt result = mk_add(shifted_left, right_converted);

  return result;
}

smt_astt smt_solver_baset::convert_assign(const expr2tc &expr)
{
  const equality2t &eq = to_equality2t(expr);

  // Record forall/exists assignments so nested quantifier handlers can
  // inline the inner body before substituting the outer bound variable.
  if (
    is_symbol2t(eq.side_1) &&
    (is_forall2t(eq.side_2) || is_exists2t(eq.side_2)))
    forall_defs_[to_symbol2t(eq.side_1).thename] = eq.side_2;

  smt_astt side1 = convert_ast(eq.side_1); // LHS
  smt_astt side2 = convert_ast(eq.side_2); // RHS
  side2->assign(this, side1);

  // Put that into the smt cache, thus preserving the value of the assigned symbols.
  // IMPORTANT: the cache is now a fundamental part of how some flatteners work,
  // in that one can choose to create a set of expressions and their ASTs, then
  // store them in the cache, rather than have a more sophisticated conversion.
  {
    const smt_cache_entryt e = {eq.side_1, side2, ctx_level};
    // Lock automatically released when it goes out of scope
    std::lock_guard lock(smt_cache_mutex);
    smt_cache.insert(e);
  }

  // Propagate ir-ieee interval metadata from RHS to LHS SSA variable so
  // that get_interval() lookups on the LHS variable find the stored interval
  // for compositional lifting.
  ir_ieee_api->propagate_interval(side1, side2);
  ir_ieee_api->propagate_nan_pred(side1, side2);

  return side2;
}

// Returns true when convert_ast_node() would recurse into expr's operands
// via the default operand walk. Mirrors the switch in convert_ast_node():
// vector-typed nodes take dedicated rewrite paths, and the listed expr_ids
// explicitly skip their operands ("Don't convert their operands"). Keep this
// in sync with that switch.
static bool walks_operands(const expr2tc &expr)
{
  if (is_vector_type(expr))
    return false;

  switch (expr->expr_id)
  {
  case expr2t::with_id:
  case expr2t::constant_array_id:
  case expr2t::constant_vector_id:
  case expr2t::constant_array_of_id:
  case expr2t::index_id:
  case expr2t::address_of_id:
  case expr2t::ieee_add_id:
  case expr2t::ieee_sub_id:
  case expr2t::ieee_mul_id:
  case expr2t::ieee_div_id:
  case expr2t::ieee_fma_id:
  case expr2t::ieee_sqrt_id:
  case expr2t::pointer_offset_id:
  case expr2t::pointer_object_id:
  case expr2t::pointer_capability_id:
    return false;
  default:
    return true;
  }
}

smt_astt smt_solver_baset::convert_ast(const expr2tc &expr)
{
  // Fast path: a hit returns without building a worklist. convert_ast_node
  // re-enters convert_ast for every operand/sub-expression it converts, and on
  // the warmed post-order path those re-entrant calls are guaranteed hits;
  // checking the cache up front keeps them O(1) instead of paying a worklist
  // allocation per call.
  {
    std::lock_guard lock(smt_cache_mutex);
    smt_cachet::const_iterator hit = smt_cache.find(expr);
    if (hit != smt_cache.end())
      return hit->ast;
  }

  // Warm the cache bottom-up with an explicit-stack post-order walk so that
  // the per-node body (convert_ast_node) never has to recurse through a long
  // chain of operands. This keeps deeply left-nested associative expressions
  // (e.g. a disjunction of tens of thousands of clauses) from overflowing the
  // C++ stack: each node's operands are already converted by the time we
  // convert the node itself. We only descend through nodes whose operands the
  // body would actually convert (walks_operands), so the cache contents match
  // exactly what recursive conversion would have produced.
  std::vector<std::pair<expr2tc, bool>> stack; // (node, children_pushed)
  stack.emplace_back(expr, false);

  while (!stack.empty())
  {
    // Copy out of the stack: emplace_back below may reallocate, so we must not
    // hold a reference into the vector across a push.
    const expr2tc node = stack.back().first;
    const bool expanded = stack.back().second;

    {
      std::lock_guard lock(smt_cache_mutex);
      if (smt_cache.find(node) != smt_cache.end())
      {
        stack.pop_back();
        continue;
      }
    }

    if (!expanded && walks_operands(node))
    {
      stack.back().second = true;
      // Push operands in reverse so they pop (and convert) left-to-right,
      // matching the recursive foreach_operand order — fresh-symbol numbering
      // and cache-insertion order then stay identical to recursive conversion.
      // get_sub_expr() visits the same slots in the same order as
      // foreach_operand (both fold over K::fields), so no operand is skipped.
      const size_t n = node->get_num_sub_exprs();
      for (size_t i = n; i-- > 0;)
        stack.emplace_back(*node->get_sub_expr(i), false);
      continue;
    }

    // All operands (if any) are cached now: convert this node. Some
    // convert_ast_node paths return early without populating the cache
    // themselves (e.g. they tail-call convert_ast on a rewritten expr), so we
    // must record the result here — later iterations and the final lookup rely
    // on every converted node being present in the cache.
    smt_astt a = convert_ast_node(node);
    {
      std::lock_guard lock(smt_cache_mutex);
      if (smt_cache.find(node) == smt_cache.end())
        smt_cache.insert(smt_cache_entryt{node, a, ctx_level});
    }
    stack.pop_back();
  }

  std::lock_guard lock(smt_cache_mutex);
  return smt_cache.find(expr)->ast;
}

smt_astt smt_solver_baset::convert_ast_node(const expr2tc &expr)
{
  {
    std::lock_guard lock(smt_cache_mutex);
    smt_cachet::const_iterator cache_result = smt_cache.find(expr);
    if (cache_result != smt_cache.end())
      return (cache_result->ast);
  }

  // A sizeof(T) node lowers to its eagerly-computed byte-size value. do_simplify
  // normally folds it away, but under --no-simplify it survives to here, so
  // lower it explicitly rather than hitting the unrecognised-format abort
  // (esbmc/esbmc#5337).
  if (is_sizeof2t(expr))
    return convert_ast(to_sizeof2t(expr).value);
  /* Vectors!
   *
   * Here we need special attention for Vectors, because of the way
   * they are encoded, an vector expression can reach here with binary
   * operations that weren't done.
   *
   * The simplification module take care of all the operations, but if
   * for some reason we would like to run ESBMC without simplifications
   * then we need to apply it here.
  */
  if (is_vector_type(expr))
  {
    if (is_neg2t(expr))
    {
      return convert_ast(
        distribute_vector_operation(expr->expr_id, to_neg2t(expr).value));
    }
    if (is_bitnot2t(expr))
    {
      return convert_ast(
        distribute_vector_operation(expr->expr_id, to_bitnot2t(expr).value));
    }

    switch (expr->expr_id)
    {
    case expr2t::ieee_add_id:
    case expr2t::ieee_sub_id:
    case expr2t::ieee_mul_id:
    case expr2t::ieee_div_id:
      return convert_ast(distribute_vector_operation(
        expr->expr_id,
        *expr->get_sub_expr(1),   // side_1
        *expr->get_sub_expr(2),   // side_2
        *expr->get_sub_expr(0))); // rounding_mode
    case expr2t::add_id:
    case expr2t::sub_id:
    case expr2t::mul_id:
    case expr2t::div_id:
    case expr2t::modulus_id:
    case expr2t::bitand_id:
    case expr2t::bitor_id:
    case expr2t::bitxor_id:
    case expr2t::shl_id:
    case expr2t::ashr_id:
    case expr2t::lshr_id:
      return convert_ast(distribute_vector_operation(
        expr->expr_id, *expr->get_sub_expr(0), *expr->get_sub_expr(1)));
    default:
      break;
    }
  }

  std::vector<smt_astt> args;

  switch (expr->expr_id)
  {
  case expr2t::with_id:
  case expr2t::constant_array_id:
  case expr2t::constant_vector_id:
  case expr2t::constant_array_of_id:
  case expr2t::index_id:
  case expr2t::address_of_id:
  case expr2t::ieee_add_id:
  case expr2t::ieee_sub_id:
  case expr2t::ieee_mul_id:
  case expr2t::ieee_div_id:
  case expr2t::ieee_fma_id:
  case expr2t::ieee_sqrt_id:
  case expr2t::pointer_offset_id:
  case expr2t::pointer_object_id:
  case expr2t::pointer_capability_id:
    break; // Don't convert their operands

  default:
  {
    // Convert all the arguments and store them in 'args'.
    args.reserve(expr->get_num_sub_exprs());
    expr->foreach_operand(
      [this, &args](const expr2tc &e) { args.push_back(convert_ast(e)); });
  }
  }

  smt_astt a;
  switch (expr->expr_id)
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
  {
    // Get size
    const constant_union2t &cu = to_constant_union2t(expr);
    const std::vector<expr2tc> &dt_memb = cu.datatype_members;
    expr2tc src_expr =
      dt_memb.empty() ? gen_zero(get_uint_type(0)) : dt_memb[0];
#ifndef NDEBUG
    if (!cu.init_field.empty())
    {
      const union_type2t &ut = to_union_type(expr->type);
      unsigned c =
        struct_union_get_component_number(expr->type, cu.init_field).value();
      /* Can only initialize unions by expressions of same type as init_field */
      assert(src_expr->type->type_id == ut.members[c]->type_id);
    }
#endif
    a = convert_ast(typecast2tc(
      get_uint_type(type_byte_size_bits(expr->type).to_uint64()),
      bitcast2tc(
        get_uint_type(type_byte_size_bits(src_expr->type).to_uint64()),
        src_expr)));
    break;
  }
  case expr2t::constant_vector_id:
  {
    a = array_create(expr);
    break;
  }
  case expr2t::constant_array_id:
  case expr2t::constant_array_of_id:
  {
    const array_type2t &arr = to_array_type(expr->type);
    if (!array_api->can_init_infinite_arrays && arr.size_is_infinite)
    {
      smt_sortt sort = convert_sort(expr->type);

      // Don't honor inifinite sized array initializers. Modelling only.
      // If we have an array of tuples and no tuple support, use tuple_fresh.
      // Otherwise, mk_fresh.
      if (is_tuple_ast_type(arr.subtype))
        a = tuple_api->tuple_fresh(sort);
      else
        a = mk_fresh(sort, "inf_array", convert_sort(arr.subtype));
      break;
    }

    expr2tc flat_expr = expr;
    if (
      is_array_type(to_array_type(expr->type).subtype) &&
      is_constant_array2t(expr))
      flat_expr = flatten_array_body(expr);

    if (is_struct_type(arr.subtype) || is_pointer_type(arr.subtype))
    {
      // Domain sort may be mesed with:
      smt_sortt domain = mk_int_bv_sort(
        int_encoding ? config.ansi_c.int_width
                     : array_domain_width_or_word_size(arr));

      a = tuple_array_create_despatch(flat_expr, domain);
    }
    else
      a = array_create(flat_expr);
    break;
  }
  case expr2t::add_id:
  {
    const add2t &add = to_add2t(expr);
    if (
      is_pointer_type(expr->type) || is_pointer_type(add.side_1) ||
      is_pointer_type(add.side_2))
    {
      a = convert_pointer_arith(expr, expr->type);
    }
    else if (int_encoding)
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
    if (
      is_pointer_type(expr->type) || is_pointer_type(sub.side_1) ||
      is_pointer_type(sub.side_2))
    {
      a = convert_pointer_arith(expr, expr->type);
    }
    else if (int_encoding)
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
    if (is_fixedbv_type(expr) && !int_encoding)
    {
      auto mul = to_mul2t(expr);
      auto fbvt = to_fixedbv_type(mul.type);

      unsigned int fraction_bits = fbvt.width - fbvt.integer_bits;

      args[0] = mk_sign_ext(convert_ast(mul.side_1), fraction_bits);
      args[1] = mk_sign_ext(convert_ast(mul.side_2), fraction_bits);

      a = mk_bvmul(args[0], args[1]);
      a = mk_extract(a, fbvt.width + fraction_bits - 1, fraction_bits);
    }
    else if (int_encoding)
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
    if (is_fixedbv_type(expr) && !int_encoding)
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
    else if (int_encoding)
    {
      a = mk_div(args[0], args[1]);
    }
    else if (is_unsignedbv_type(d.side_1) && is_unsignedbv_type(d.side_2))
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
    if (int_encoding)
      a = ir_ieee_api->encode_ieee_add(expr);
    else
      a = fp_api->mk_smt_fpbv_add(
        convert_ast(to_ieee_add2t(expr).side_1),
        convert_ast(to_ieee_add2t(expr).side_2),
        convert_rounding_mode(to_ieee_add2t(expr).rounding_mode));
    break;
  }
  case expr2t::ieee_sub_id:
  {
    assert(is_floatbv_type(expr));
    if (int_encoding)
      a = ir_ieee_api->encode_ieee_sub(expr);
    else
      a = fp_api->mk_smt_fpbv_sub(
        convert_ast(to_ieee_sub2t(expr).side_1),
        convert_ast(to_ieee_sub2t(expr).side_2),
        convert_rounding_mode(to_ieee_sub2t(expr).rounding_mode));
    break;
  }
  case expr2t::ieee_mul_id:
  {
    assert(is_floatbv_type(expr));
    if (int_encoding)
      a = ir_ieee_api->encode_ieee_mul(expr);
    else
      a = fp_api->mk_smt_fpbv_mul(
        convert_ast(to_ieee_mul2t(expr).side_1),
        convert_ast(to_ieee_mul2t(expr).side_2),
        convert_rounding_mode(to_ieee_mul2t(expr).rounding_mode));
    break;
  }
  case expr2t::ieee_div_id:
  {
    assert(is_floatbv_type(expr));
    if (int_encoding)
      a = ir_ieee_api->encode_ieee_div(expr);
    else
      a = fp_api->mk_smt_fpbv_div(
        convert_ast(to_ieee_div2t(expr).side_1),
        convert_ast(to_ieee_div2t(expr).side_2),
        convert_rounding_mode(to_ieee_div2t(expr).rounding_mode));
    break;
  }
  case expr2t::ieee_fma_id:
  {
    assert(is_floatbv_type(expr));
    if (int_encoding)
      a = ir_ieee_api->encode_ieee_fma(expr);
    else
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
    if (int_encoding)
    {
      smt_astt operand = convert_ast(to_ieee_sqrt2t(expr).value);
      const floatbv_type2t &fbv_type = to_floatbv_type(expr->type);
      const expr2tc &rounding_mode = to_ieee_sqrt2t(expr).rounding_mode;

      // Two fresh reals are introduced:
      //
      //   sqrt_pos  (tagged "ra_sqrt::")   — for operand >= 0.
      //     Pinned by the guarded quadratic:
      //       operand >= 0  →  sqrt_pos >= 0  ∧  sqrt_pos² = operand
      //     The enclosure is applied to this symbol only.
      //
      //   sqrt_nan  (tagged "ra_sqrt_nan::")  — for operand < 0.
      //     Completely unconstrained.  Models the IEEE 754 NaN result
      //     without making the formula inconsistent.
      //
      // The publicly returned result is  ite(operand >= 0, sqrt_pos, sqrt_nan).
      // This ensures that for negative operands the enclosure constraints on
      // sqrt_pos do not propagate to the observable result.
      //
      // Under --ir-ieee, a NaN predicate  not(operand >= 0)  is stored for
      // the result so that floating-point comparisons involving this value
      // can apply IEEE 754 NaN comparison semantics (ordered comparisons
      // return false; != returns true).
      smt_sortt rs = mk_real_sort();
      smt_astt zero = mk_smt_real("0.0");
      smt_astt op_nonneg = mk_le(zero, operand);

      smt_astt sqrt_pos = mk_fresh(rs, "ra_sqrt::", nullptr);
      // operand >= 0 → sqrt_pos >= 0
      assert_ast(mk_or(mk_not(op_nonneg), mk_le(zero, sqrt_pos)));
      // operand >= 0 → sqrt_pos² = operand
      assert_ast(
        mk_or(mk_not(op_nonneg), mk_eq(mk_mul(sqrt_pos, sqrt_pos), operand)));

      // Unconstrained result used when operand < 0.
      smt_astt sqrt_nan = mk_fresh(rs, "ra_sqrt_nan::", nullptr);

      // Interval-lifted enclosure for ieee_sqrt (--ir-ieee only).
      // sqrt is monotone increasing on [0, ∞), so the exact hull is:
      //   lo_r = sqrt(iv.lo),  hi_r = sqrt(iv.hi)
      // Each bound is a fresh real pinned by the same quadratic axiom.
      // Mode-dispatch follows the same five-case pattern as add/sub/mul/div.
      bool interval_lifted = false;
      if (
        options.get_bool_option("ir-ieee") &&
        (smt_fp_rounding_utils::is_nearest_rounding_mode(rounding_mode) ||
         smt_fp_rounding_utils::is_round_to_away(rounding_mode) ||
         smt_fp_rounding_utils::is_round_to_plus_inf(rounding_mode) ||
         smt_fp_rounding_utils::is_round_to_minus_inf(rounding_mode) ||
         smt_fp_rounding_utils::is_round_to_zero(rounding_mode)))
      {
        const auto double_spec = ieee_float_spect::double_precision();
        const auto single_spec = ieee_float_spect::single_precision();
        if (
          (fbv_type.exponent == double_spec.e &&
           fbv_type.fraction == double_spec.f) ||
          (fbv_type.exponent == single_spec.e &&
           fbv_type.fraction == single_spec.f))
        {
          ir_ieee_convt::ra_interval_t iv = ir_ieee_api->get_interval(operand);

          // Fresh reals for sqrt of interval bounds.
          // Clamp bounds to zero: the RNE/RNA enclosures from prior operations
          // can make iv.lo slightly negative (by eps_abs) even when the float
          // operand is always non-negative.  sqrt is only defined for x >= 0,
          // so we tighten to max(0, iv.lo) and max(0, iv.hi).
          smt_astt iv_lo_pos = mk_ite(mk_lt(iv.lo, zero), zero, iv.lo);
          smt_astt iv_hi_pos = mk_ite(mk_lt(iv.hi, zero), zero, iv.hi);

          smt_astt lo_r = mk_fresh(rs, "ra_sqrt_lo::", nullptr);
          smt_astt hi_r = mk_fresh(rs, "ra_sqrt_hi::", nullptr);
          assert_ast(mk_le(zero, lo_r));
          assert_ast(mk_eq(mk_mul(lo_r, lo_r), iv_lo_pos));
          assert_ast(mk_le(zero, hi_r));
          assert_ast(mk_eq(mk_mul(hi_r, hi_r), iv_hi_pos));

          // Enclosure is applied to sqrt_pos, not to the final ITE result.
          // The containment assertions (ra_lo <= sqrt_pos <= ra_hi) therefore
          // do not constrain the result when operand < 0.
          std::pair<smt_astt, smt_astt> bounds;
          if (smt_fp_rounding_utils::is_nearest_rounding_mode(rounding_mode))
            bounds = ir_ieee_api->apply_ieee754_rne_enclosure(
              sqrt_pos, lo_r, hi_r, fbv_type);
          else if (smt_fp_rounding_utils::is_round_to_away(rounding_mode))
            bounds = ir_ieee_api->apply_ieee754_rna_enclosure(
              sqrt_pos, lo_r, hi_r, fbv_type);
          else if (smt_fp_rounding_utils::is_round_to_plus_inf(rounding_mode))
            bounds = ir_ieee_api->apply_ieee754_rup_enclosure(
              sqrt_pos, lo_r, hi_r, fbv_type);
          else if (smt_fp_rounding_utils::is_round_to_minus_inf(rounding_mode))
            bounds = ir_ieee_api->apply_ieee754_rdn_enclosure(
              sqrt_pos, lo_r, hi_r, fbv_type);
          else
            bounds = ir_ieee_api->apply_ieee754_rtz_enclosure(
              sqrt_pos, lo_r, hi_r, fbv_type);

          smt_astt sqrt_result = mk_ite(op_nonneg, sqrt_pos, sqrt_nan);
          smt_astt map_lo = mk_ite(op_nonneg, bounds.first, sqrt_result);
          smt_astt map_hi = mk_ite(op_nonneg, bounds.second, sqrt_result);
          ir_ieee_api->store_interval(sqrt_result, map_lo, map_hi);
          a = sqrt_result;
          interval_lifted = true;
        }
      }
      if (!interval_lifted)
      {
        smt_astt pos_result =
          apply_ieee754_semantics(sqrt_pos, fbv_type, nullptr, rounding_mode);
        a = mk_ite(op_nonneg, pos_result, sqrt_nan);
      }
      if (ir_ieee)
        ir_ieee_api->store_nan_pred(
          a,
          ir_ieee_api->combine_nan_preds(
            ir_ieee_api->get_nan_pred(operand), mk_not(op_nonneg)));
    }
    else
    {
      a = fp_api->mk_smt_fpbv_sqrt(
        convert_ast(to_ieee_sqrt2t(expr).value),
        convert_rounding_mode(to_ieee_sqrt2t(expr).rounding_mode));
    }
    break;
  }
  case expr2t::modulus_id:
  {
    auto m = to_modulus2t(expr);

    if (int_encoding)
    {
      a = mk_mod(args[0], args[1]);
    }
    else if (is_fixedbv_type(m.side_1) && is_fixedbv_type(m.side_2))
    {
      a = mk_bvsmod(args[0], args[1]);
    }
    else if (is_unsignedbv_type(m.side_1) && is_unsignedbv_type(m.side_2))
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
    if (is_struct_type(expr) || is_pointer_type(expr) || is_complex_type(expr))
    {
      unsigned int idx = get_member_name_field(expr->type, with.update_field);
      smt_astt srcval = convert_ast(with.source_value);

#ifndef NDEBUG
      // Pointer with's lower into pointer_struct tuple updates.
      const std::vector<type2tc> &members = struct_union_members(
        is_pointer_type(with.type) ? pointer_struct : with.type);
      assert(idx < members.size() && "Out of bounds with expression");
      // Base type eq examines pointer types to closely
      assert(
        (base_type_eq(members[idx], with.update_value->type, ns) ||
         (is_pointer_type(members[idx]) &&
          is_pointer_type(with.update_value))) &&
        "Assigned tuple member has type mismatch");
#endif

      a = srcval->update(this, convert_ast(with.update_value), idx);
    }
    else if (is_union_type(expr))
    {
      uint64_t bits = type_byte_size_bits(expr->type).to_uint64();
      const union_type2t &tu = to_union_type(expr->type);
      assert(is_constant_string2t(with.update_field));
      unsigned c = struct_union_get_component_number(
                     expr->type, to_constant_string2t(with.update_field).value)
                     .value();
      uint64_t mem_bits = type_byte_size_bits(tu.members[c]).to_uint64();
      expr2tc upd = bitcast2tc(
        get_uint_type(mem_bits), typecast2tc(tu.members[c], with.update_value));
      if (mem_bits < bits)
        upd = concat2tc(
          get_uint_type(bits),
          extract2tc(
            get_uint_type(bits - mem_bits),
            with.source_value,
            bits - 1,
            mem_bits),
          upd);
      a = convert_ast(upd);
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
    while (is_typecast2t(*ptr) && !is_pointer_type(*ptr))
      ptr = &to_typecast2t(*ptr).from;

    a = convert_ast(*ptr)->project(this, 1);
    break;
  }
  case expr2t::pointer_object_id:
  {
    const pointer_object2t &obj = to_pointer_object2t(expr);
    // Potentially walk through some typecasts
    const expr2tc *ptr = &obj.ptr_obj;
    while (is_typecast2t(*ptr) && !is_pointer_type((*ptr)))
      ptr = &to_typecast2t(*ptr).from;

    a = convert_ast(*ptr)->project(this, 0);
    break;
  }
  case expr2t::pointer_capability_id:
  {
    assert(config.ansi_c.cheri);
    const pointer_capability2t &obj = to_pointer_capability2t(expr);
    // Potentially walk through some typecasts
    const expr2tc *ptr = &obj.ptr_obj;
    while (is_typecast2t(*ptr) && !is_pointer_type((*ptr)))
      ptr = &to_typecast2t(*ptr).from;

    a = convert_ast(*ptr)->project(this, 2);
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
  case expr2t::popcount_id:
  {
    a = convert_popcount(expr);
    break;
  }
  case expr2t::bswap_id:
  {
    a = convert_bswap(expr);
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
    /* Compare the representations directly.
     *
     * This also applies to pointer-typed expressions which are represented as
     * (object, offset) structs, i.e., two pointers compare equal iff both
     * members are the same.
     *
     * 'offset' is between 0 and the size of the object, both inclusively. This
     * is in line with what's allowed by C99 and what current GCC assumes
     * regarding the one-past the end pointer:
     *
     *   Two pointers compare equal if and only if both are null pointers, both
     *   are pointers to the same object (including a pointer to an object and a
     *   subobject at its beginning) or function, both are pointers to one past
     *   the last element of the same array object, or one is a pointer to one
     *   past the end of one array object and the other is a pointer to the
     *   start of a different array object that happens to immediately follow
     *   the first array object in the address space.
     *
     * It's not strictly what Clang does, though, but de-facto, C compilers do
     * perform optimizations based on provenance, i.e., "one past the end
     * pointers cannot alias another object" as soon as it *cannot* be proven
     * that they do. Sigh. For instance
     * <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61502> makes for a "fun"
     * read illuminating how reasoning works from a certain compiler's
     * writers' points of view.
     *
     * C++ has changed this one-past behavior in [expr.eq] to "unspecified"
     * <https://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1652>
     * and C might eventually follow the same path.
     *
     * CHERI-C semantics say that only addresses should be compared, but this
     * might also change in the future, see e.g.
     * <https://github.com/CTSRD-CHERI/llvm-project/issues/649>.
     *
     * TODO: As languages begin to differ in their pointer equality semantics,
     *       we could move pointer comparisons to symex in order to express
     *       them properly according to the input language.
     */

    auto eq = to_equality2t(expr);

    if (
      is_floatbv_type(eq.side_1) && is_floatbv_type(eq.side_2) && !int_encoding)
      a = fp_api->mk_smt_fpbv_eq(args[0], args[1]);
    else
      a = args[0]->eq(this, args[1]);
    if (
      ir_ieee && int_encoding && is_floatbv_type(eq.side_1) &&
      is_floatbv_type(eq.side_2))
      a = ir_ieee_api->apply_nan_cmp(a, args[0], args[1], false);
    break;
  }
  case expr2t::notequal_id:
  {
    // Handle all kinds of structs by inverted equality. The only that's really
    // going to turn up is pointers though.

    auto neq = to_notequal2t(expr);

    if (
      is_floatbv_type(neq.side_1) && is_floatbv_type(neq.side_2) &&
      !int_encoding)
      a = fp_api->mk_smt_fpbv_eq(args[0], args[1]);
    else
      a = args[0]->eq(this, args[1]);
    a = mk_not(a);
    if (
      ir_ieee && int_encoding && is_floatbv_type(neq.side_1) &&
      is_floatbv_type(neq.side_2))
      a = ir_ieee_api->apply_nan_cmp(a, args[0], args[1], true);
    break;
  }
  case expr2t::shl_id:
  {
    const shl2t &shl = to_shl2t(expr);
    if (shl.side_1->type->get_width() != shl.side_2->type->get_width())
    {
      // frontend doesn't cast the second operand up to the width of
      // the first, which SMT does not enjoy.
      expr2tc cast = typecast2tc(shl.side_1->type, shl.side_2);
      args[1] = convert_ast(cast);
    }

    if (int_encoding)
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
    if (ashr.side_1->type->get_width() != ashr.side_2->type->get_width())
    {
      // frontend doesn't cast the second operand up to the width of
      // the first, which SMT does not enjoy.
      expr2tc cast = typecast2tc(ashr.side_1->type, ashr.side_2);
      args[1] = convert_ast(cast);
    }

    if (int_encoding)
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
    const lshr2t &lshr = to_lshr2t(expr);
    if (lshr.side_1->type->get_width() != lshr.side_2->type->get_width())
    {
      // frontend doesn't cast the second operand up to the width of
      // the first, which SMT does not enjoy.
      expr2tc cast = typecast2tc(lshr.side_1->type, lshr.side_2);
      args[1] = convert_ast(cast);
    }

    if (int_encoding)
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
    if (is_unsignedbv_type(abs.value))
    {
      // No need to do anything.
      a = args[0];
    }
    else if (is_floatbv_type(abs.value) && !int_encoding)
    {
      a = fp_api->mk_smt_fpbv_abs(args[0]);
    }
    else
    {
      // Lower as `(x >= 0) ? x : -x`. The opposite-sense `(x < 0) ? -x : x`
      // is logically equivalent but bitwuzla preprocesses the `>= 0` shape
      // significantly faster. Fixes a 7x regression on
      // sv-benchmarks/c/xcsp/AllInterval-017.
      //
      // The branch-free `bvsub(bvxor(x, ashr(x, w-1)), ashr(x, w-1))`
      // form was tried (it's the canonical SMT-LIB abs encoding) and is
      // ~8x slower on AllInterval-017 under bitwuzla — the ite form
      // gives the solver a clean case-split that meshes with the
      // surrounding all-distinct + abs-difference chain, while the xor
      // form mixes the sign bit into bitvector arithmetic and seems to
      // defeat term-graph sharing in this pattern.
      expr2tc ge = greaterthanequal2tc(abs.value, gen_zero(abs.value->type));
      expr2tc neg = neg2tc(abs.value->type, abs.value);
      expr2tc ite = if2tc(abs.type, ge, abs.value, neg);

      a = convert_ast(ite);
    }
    break;
  }
  case expr2t::cmp_three_way_id:
  {
    // C++20 spaceship `a <=> b`. Lower to the equivalent ITE chain
    // producing a comparison-category struct value:
    //   side_1 <  side_2  ->  T{-1}    (less)
    //   side_1 == side_2  ->  T{ 0}    (equivalent / equal)
    //   else              ->  T{ 1}    (greater)
    // Operands are captured once via the recursive convert_ast on the
    // children — preserving the IR-level cmp_three_way2t up to here.
    const cmp_three_way2t &cw = to_cmp_three_way2t(expr);

    expr2tc lt = lessthan2tc(cw.side_1, cw.side_2);
    expr2tc eq = equality2tc(cw.side_1, cw.side_2);
    expr2tc inner = if2tc(
      cw.type, eq, make_cmp_value(cw.type, 0), make_cmp_value(cw.type, 1));
    expr2tc outer = if2tc(cw.type, lt, make_cmp_value(cw.type, -1), inner);
    a = convert_ast(outer);
    break;
  }
  case expr2t::lessthan_id:
  {
    const lessthan2t &lt = to_lessthan2t(expr);
    // Pointer relation:
    if (is_pointer_type(lt.side_1))
    {
      a = convert_ptr_cmp(lt.side_1, lt.side_2, expr);
    }
    else if (int_encoding)
    {
      a = mk_lt(args[0], args[1]);
      if (ir_ieee && is_floatbv_type(lt.side_1) && is_floatbv_type(lt.side_2))
        a = ir_ieee_api->apply_nan_cmp(a, args[0], args[1], false);
    }
    else if (is_floatbv_type(lt.side_1) && is_floatbv_type(lt.side_2))
    {
      a = fp_api->mk_smt_fpbv_lt(args[0], args[1]);
    }
    else if (is_fixedbv_type(lt.side_1) && is_fixedbv_type(lt.side_2))
    {
      a = mk_bvslt(args[0], args[1]);
    }
    else if (is_unsignedbv_type(lt.side_1) && is_unsignedbv_type(lt.side_2))
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
    if (is_pointer_type(lte.side_1))
    {
      a = convert_ptr_cmp(lte.side_1, lte.side_2, expr);
    }
    else if (int_encoding)
    {
      a = mk_le(args[0], args[1]);
      if (ir_ieee && is_floatbv_type(lte.side_1) && is_floatbv_type(lte.side_2))
        a = ir_ieee_api->apply_nan_cmp(a, args[0], args[1], false);
    }
    else if (is_floatbv_type(lte.side_1) && is_floatbv_type(lte.side_2))
    {
      a = fp_api->mk_smt_fpbv_lte(args[0], args[1]);
    }
    else if (is_fixedbv_type(lte.side_1) && is_fixedbv_type(lte.side_2))
    {
      a = mk_bvsle(args[0], args[1]);
    }
    else if (is_unsignedbv_type(lte.side_1) && is_unsignedbv_type(lte.side_2))
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
    if (is_pointer_type(gt.side_1))
    {
      a = convert_ptr_cmp(gt.side_1, gt.side_2, expr);
    }
    else if (int_encoding)
    {
      a = mk_gt(args[0], args[1]);
      if (ir_ieee && is_floatbv_type(gt.side_1) && is_floatbv_type(gt.side_2))
        a = ir_ieee_api->apply_nan_cmp(a, args[0], args[1], false);
    }
    else if (is_floatbv_type(gt.side_1) && is_floatbv_type(gt.side_2))
    {
      a = fp_api->mk_smt_fpbv_gt(args[0], args[1]);
    }
    else if (is_fixedbv_type(gt.side_1) && is_fixedbv_type(gt.side_2))
    {
      a = mk_bvsgt(args[0], args[1]);
    }
    else if (is_unsignedbv_type(gt.side_1) && is_unsignedbv_type(gt.side_2))
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
    if (is_pointer_type(gte.side_1))
    {
      a = convert_ptr_cmp(gte.side_1, gte.side_2, expr);
    }
    else if (int_encoding)
    {
      a = mk_ge(args[0], args[1]);
      if (ir_ieee && is_floatbv_type(gte.side_1) && is_floatbv_type(gte.side_2))
        a = ir_ieee_api->apply_nan_cmp(a, args[0], args[1], false);
    }
    else if (is_floatbv_type(gte.side_1) && is_floatbv_type(gte.side_2))
    {
      a = fp_api->mk_smt_fpbv_gte(args[0], args[1]);
    }
    else if (is_fixedbv_type(gte.side_1) && is_fixedbv_type(gte.side_2))
    {
      a = mk_bvsge(args[0], args[1]);
    }
    else if (is_unsignedbv_type(gte.side_1) && is_unsignedbv_type(gte.side_2))
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
    if (int_encoding)
      return convert_concat_int_mode(args[0], args[1], expr);
    else
      a = mk_concat(args[0], args[1]);
    break;
  }
  case expr2t::implies_id:
  {
    a = mk_implies(args[0], args[1]);
    break;
  }
  /* According to the SMT-lib, LIRA focuses on arithmetic operations
     involving integers and reals. Still, it does not inherently handle
     bitwise operations, as these fall under the domain of bit-vector theories.
     Here, we combine LIRA and bitwise operations (such as &, |, ^)
     in a way that aligns with arithmetic constraints.
  */
  case expr2t::bitand_id:
  {
    a = mk_bvand(args[0], args[1]);
    break;
  }
  case expr2t::bitor_id:
  {
    a = mk_bvor(args[0], args[1]);
    break;
  }
  case expr2t::bitxor_id:
  {
    a = mk_bvxor(args[0], args[1]);
    break;
  }
  case expr2t::bitnot_id:
  {
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
    if (int_encoding)
    {
      a = mk_neg(args[0]);
    }
    else if (is_floatbv_type(neg.value))
    {
      a = fp_api->mk_smt_fpbv_neg(args[0]);
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
    a = convert_bitcast(expr);
    break;
  }
  case expr2t::extract_id:
  {
    const extract2t &ex = to_extract2t(expr);
    a = convert_ast(ex.from);
    if (ex.from->type->get_width() == ex.upper - ex.lower + 1)
      return a;
    a = mk_extract(a, ex.upper, ex.lower);
    break;
  }
  case expr2t::code_comma_id:
  {
    /* 
      TODO: for some reason comma expressions survive when they are under
      * RETURN statements. They should have been taken care of at the GOTO
      * level. Remove this code once we do!

      the expression on the right side will become the value of the entire comma-separated expression.

      e.g.
        return side_1, side_2;
      equals to
        side1;
        return side_2;
    */
    const code_comma2t &cm = to_code_comma2t(expr);
    a = convert_ast(cm.side_2);
    break;
  }
  case expr2t::forall_id:
  case expr2t::exists_id:
  {
    // TODO: technically the forall could be a list of symbols
    // TODO: how to support other assertions inside it? e.g., buffer-overflow, arithmetic-overflow, etc...
    expr2tc symbol;
    expr2tc predicate;

    if (is_forall2t(expr))
    {
      symbol = to_forall2t(expr).side_1;
      predicate = to_forall2t(expr).side_2;
    }
    else
    {
      symbol = to_exists2t(expr).side_1;
      predicate = to_exists2t(expr).side_2;
    }

    // We only want expressions of typecast(address_of(symbol)) or address_of(symbol).
    {
      if (const typecast2t *tc = try_to_typecast2t(symbol);
          tc && is_address_of2t(tc->from))
        symbol = to_address_of2t(tc->from).ptr_obj;

      else if (is_address_of2t(symbol))
        symbol = to_address_of2t(symbol).ptr_obj;

      if (!is_symbol2t(symbol))
      {
        log_error(
          "Could not resolve expression into unique symbol. Please open an "
          "issue.");
        symbol->dump();
        abort();
      }
    }

    // Create a fresh symbol to use as the bound variable.  Using the
    // original SSA symbol would be wrong whenever it already has a
    // concrete value in the smt_cache (e.g. a loop counter reused as a
    // quantifier variable); a fresh name is always unassigned.
    const expr2tc bound_symbol = symbol2tc(
      symbol->type, fmt::format("__ESBMC_quantifier_{}", quantifier_counter++));

    // Inline any nested forall/exists definition so that
    // replace_name_in_body can substitute `bound_symbol` for the outer
    // variable throughout the entire (possibly nested) predicate.  Without
    // this step the original SSA symbol would remain free inside the
    // cached inner-forall formula and the outer quantifier would be vacuous.
    expr2tc expanded = predicate->clone();
    expand_quantifier_defs_in(expanded, forall_defs_);

    replace_name_in_body(symbol, bound_symbol, expanded);

    a = mk_quantifier(
      is_forall2t(expr), {convert_ast(bound_symbol)}, convert_ast(expanded));
    break;
  }
  case expr2t::uninterpreted_func_id:
  {
    const uninterpreted_func2t &uf = to_uninterpreted_func2t(expr);
    // 'args' already holds the converted arguments (the default operand loop
    // above). The solver declares one function symbol per name and applies it
    // here, so functional congruence is enforced natively.
    a = mk_smt_uninterpreted_function(
      uf.function_name.as_string(), args, convert_sort(uf.type));
    break;
  }
  default:
    log_error("Couldn't convert expression in unrecognised format\n{}", *expr);
    abort();
  }

  {
    struct smt_cache_entryt entry = {expr, a, ctx_level};
    std::lock_guard lock(smt_cache_mutex);
    smt_cache.insert(entry);
  }
  return a;
}

void smt_solver_baset::assert_expr(const expr2tc &e)
{
  assert_ast(convert_ast(e));
}

smt_sortt smt_solver_baset::convert_sort(const type2tc &type)
{
  smt_sort_cachet::const_iterator it = sort_cache.find(type);
  if (it != sort_cache.end())
  {
    return it->second;
  }

  smt_sortt result = nullptr;
  switch (type->type_id)
  {
  case type2t::bool_id:
    result = boolean_sort;
    break;

  case type2t::complex_id:
  case type2t::struct_id:
    result = tuple_api->mk_struct_sort(type);
    break;

  case type2t::code_id:
  case type2t::pointer_id:
    result = tuple_api->mk_struct_sort(pointer_struct);
    break;

  case type2t::unsignedbv_id:
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

  case type2t::vector_id:
  case type2t::array_id:
  {
    // Nested infinite arrays (e.g. Solidity nested mappings): do NOT flatten.
    // Create Array(BV64, Array(BV64, V)) with recursive convert_sort.
    // Only applicable to genuine array types — vectors are always finite,
    // and to_array_type(vector_type) throws std::bad_cast under -DNDEBUG-off
    // builds (where the type_macros' dynamic_cast is real, not static_cast).
    if (is_array_type(type))
    {
      const array_type2t &arrtype = to_array_type(type);
      if (arrtype.size_is_infinite && is_array_type(arrtype.subtype))
      {
        type2tc t = make_array_domain_type(arrtype);
        smt_sortt d = mk_int_bv_sort(t->get_width());
        smt_sortt r = convert_sort(arrtype.subtype);
        result = mk_array_sort(d, r);
        break;
      }
    }

    // Index arrays by the smallest integer required to represent its size.
    // Unless it's either infinite or dynamic in size, in which case use the
    // machine int size. Also, faff about if it's an array of arrays, extending
    // the domain.
    type2tc t = make_array_domain_type(to_array_type(flatten_array_type(type)));
    smt_sortt d = mk_int_bv_sort(t->get_width());

    // Determine the range if we have arrays of arrays.
    type2tc range = get_flattened_array_subtype(type);
    if (is_tuple_ast_type(range))
    {
      type2tc thetype = flatten_array_type(type);
      rewrite_ptrs_to_structs(thetype);
      result = tuple_api->mk_struct_sort(thetype);
      break;
    }

    // Work around QF_AUFBV demanding arrays of bitvectors.
    smt_sortt r;
    if (is_bool_type(range) && !array_api->supports_bools_in_arrays)
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
  case type2t::union_id:
  {
    result = mk_int_bv_sort(type_byte_size_bits(type).to_uint64());
    break;
  }

  case type2t::empty_id:
    // Empty type can appear during Solidity nested mapping encoding
    // when the 'with' expression generates intermediate void-typed subexpressions.
    // Return a minimal sort as placeholder — these are never directly used in
    // solver queries and the verification result is unaffected.
    result = mk_int_bv_sort(1);
    break;

  default:
    log_error(
      "Unexpected type ID {} reached SMT conversion", get_type_id(type));
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

  if (fraction < 0)
    fraction = -fraction;

  fraction = fraction * precision;

  if (fraction == 0)
    result = double2string(integer);
  else
  {
    int64_t numerator = (integer * precision + fraction);
    result = itos(numerator) + "/" + double2string(precision);
  }

  return result;
}

smt_astt smt_solver_baset::convert_terminal(const expr2tc &expr)
{
  switch (expr->expr_id)
  {
  case expr2t::constant_int_id:
  {
    const constant_int2t &theint = to_constant_int2t(expr);
    unsigned int width = expr->type->get_width();
    if (int_encoding)
      return mk_smt_int(theint.value);

    return mk_smt_bv(theint.value, width);
  }
  case expr2t::constant_fixedbv_id:
  {
    const constant_fixedbv2t &thereal = to_constant_fixedbv2t(expr);
    if (int_encoding)
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

    return mk_smt_bv(BigInt(fin), bitwidth);
  }
  case expr2t::constant_floatbv_id:
  {
    const constant_floatbv2t &thereal = to_constant_floatbv2t(expr);
    if (int_encoding)
    {
      if (thereal.value.is_zero() || thereal.value.is_NaN())
        return mk_smt_real("0");
      if (thereal.value.is_infinity())
      {
        // Encode ±∞ as ±double_inf_sentinel (one above double max_normal) for
        // all float widths. Using the double sentinel universally ensures that
        // a float INFINITY constant typecast to double (C's INFINITY macro is
        // float) produces the same value as a double IEEE_DIV(x,0) result.
        // The double sentinel exceeds both single and double max_normal, so
        // isinf/isfinite predicates work correctly for both precisions.
        // NaN handling is deferred to the IEEE corner-case phase.
        smt_astt sentinel = get_double_inf_sentinel();
        if (thereal.value.get_sign())
          return mk_sub(get_zero_real(), sentinel);
        return sentinel;
      }
      BigInt frac, exp;
      thereal.value.extract_base2(frac, exp);
      std::string result;
      if (exp >= 0)
        result = integer2string(frac * power(2, exp));
      else
        result = integer2string(frac) + "/" + integer2string(power(2, -exp));
      if (thereal.value.get_sign())
        result = "-" + result;
      return mk_smt_real(result);
    }

    unsigned int fraction_width = to_floatbv_type(thereal.type).fraction;
    unsigned int exponent_width = to_floatbv_type(thereal.type).exponent;
    if (thereal.value.is_NaN())
      return fp_api->mk_smt_fpbv_nan(
        thereal.value.get_sign(), exponent_width, fraction_width + 1);

    bool sign = thereal.value.get_sign();
    if (thereal.value.is_infinity())
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
    const symbol2t &sym = to_symbol2t(expr);

    /* The `__ESBMC_alloc` symbol is required for finalize_pointer_chain() to
     * work, see #775. We should actually ensure here, that this is the version
     * of the symbol active when smt_memspace allocates the new object.
     *
     * XXXfbrausse: How can we ensure this? */
    if (sym.thename == "c:@__ESBMC_alloc")
      current_valid_objects_sym = expr;

    if (sym.thename == "c:@__ESBMC_is_dynamic")
      cur_dynamic = expr;

    // Special case for tuple symbols
    if (is_tuple_ast_type(expr))
      return tuple_api->mk_tuple_symbol(
        sym.get_symbol_name(), convert_sort(sym.type));

    if (is_array_type(expr))
    {
      // Determine the range if we have arrays of arrays.
      type2tc range = get_flattened_array_subtype(expr->type);

      // If this is an array of structs, we have a tuple array sym.
      if (is_tuple_ast_type(range))
        return tuple_api->mk_tuple_array_symbol(expr);
    }

    // Just a normal symbol. Possibly an array symbol.
    std::string name = sym.get_symbol_name();
    smt_sortt sort = convert_sort(sym.type);

    if (is_array_type(expr))
    {
      smt_sortt subtype = convert_sort(get_flattened_array_subtype(sym.type));
      return array_api->mk_array_symbol(name, sort, subtype);
    }

    smt_astt sym_ast = mk_smt_symbol(name, sort);

    ir_ieee_api->assert_symbol_range(name, sym_ast, sym);

    return sym_ast;
  }

  default:
    log_error("Converting unrecognized terminal expr to SMT\n{}", *expr);
    abort();
  }
}

std::string smt_solver_baset::mk_fresh_name(const std::string &tag)
{
  std::string new_name = "smt_conv::" + tag;
  std::stringstream ss;
  ss << new_name << fresh_map[new_name]++;
  return ss.str();
}

smt_astt smt_solver_baset::mk_fresh(
  smt_sortt s,
  const std::string &tag,
  smt_sortt array_subtype)
{
  std::string newname = mk_fresh_name(tag);

  if (s->id == SMT_SORT_STRUCT)
    return tuple_api->mk_tuple_symbol(newname, s);

  if (s->id == SMT_SORT_ARRAY)
  {
    assert(
      array_subtype != nullptr &&
      "Must call mk_fresh for arrays with a subtype");
    return array_api->mk_array_symbol(newname, s, array_subtype);
  }

  return mk_smt_symbol(newname, s);
}

smt_astt smt_solver_baset::convert_popcount(const expr2tc &expr)
{
  expr2tc op = to_popcount2t(expr).operand;

  // repeatedly compute x = (x & bitmask) + ((x >> shift) & bitmask)
  auto const width = op->type->get_width();
  for (std::size_t shift = 1; shift < width; shift <<= 1)
  {
    // x >> shift
    expr2tc shifted_x = lshr2tc(op->type, op, from_integer(shift, op->type));

    // bitmask is a string of alternating shift-many bits starting from lsb set
    // to 1
    std::string bitstring;
    bitstring.reserve(width);
    for (std::size_t i = 0; i < width / (2 * shift); ++i)
      bitstring += std::string(shift, '0') + std::string(shift, '1');
    expr2tc bitmask =
      constant_int2tc(op->type, binary2integer(bitstring, false));

    // build the expression
    op = add2tc(
      op->type,
      bitand2tc(op->type, op, bitmask),
      bitand2tc(op->type, shifted_x, bitmask));
  }
  // the result is restricted to the result type
  op = typecast2tc(expr->type, op);

  // Try to simplify the expression before encoding it
  simplify(op);

  return convert_ast(op);
}

smt_astt smt_solver_baset::convert_bswap(const expr2tc &expr)
{
  expr2tc op = to_bswap2t(expr).value;

  const std::size_t bits_per_byte = 8;
  const std::size_t width = expr->type->get_width();

  const std::size_t bytes = width / bits_per_byte;
  if (bytes <= 1)
    return convert_ast(op);

  std::vector<expr2tc> thebytes;
  for (std::size_t byte = 0; byte < bytes; byte++)
  {
    thebytes.emplace_back(extract2tc(
      get_uint8_type(),
      op,
      (byte * bits_per_byte + (bits_per_byte - 1)),
      (byte * bits_per_byte)));
  }

  expr2tc swap = thebytes[0];
  for (std::size_t i = 1; i < thebytes.size(); i++)
    swap = concat2tc(get_uint_type((i + 1) * 8), swap, thebytes[i]);

  return convert_ast(swap);
}

smt_astt smt_solver_baset::convert_member(const expr2tc &expr)
{
  const member2t &member = to_member2t(expr);

  // Special case unions: bitcast it to bv then convert it back to the
  // requested member type
  if (is_union_type(member.source_value))
  {
    BigInt size = type_byte_size_bits(member.source_value->type);
    expr2tc to_bv =
      bitcast2tc(get_uint_type(size.to_uint64()), member.source_value);
    type2tc type = expr->type;

    // For array members, use byte_extract so that endianness is respected.
    // concat2tc(T, A, B) places A in the high bits; stitch accordingly.
    if (is_array_type(type))
    {
      if (is_multi_dimensional_array(type))
        type = flatten_array_type(type);

      const bool big_endian =
        (config.ansi_c.endianess == configt::ansi_ct::IS_BIG_ENDIAN);
      const array_type2t &arr = to_array_type(type);
      const unsigned int elem_bits = arr.subtype->get_width();
      const unsigned int elem_bytes = elem_bits / 8;
      const unsigned int num_elems = size.to_uint64() / elem_bits;
      const type2tc bytetype = get_uint8_type();

      std::vector<expr2tc> elems;
      elems.reserve(num_elems);
      for (unsigned int i = 0; i < num_elems; i++)
      {
        // Collect elem_bytes bytes for this element.
        std::vector<expr2tc> raw_bytes;
        raw_bytes.reserve(elem_bytes);
        for (unsigned int j = 0; j < elem_bytes; j++)
          raw_bytes.push_back(byte_extract2tc(
            bytetype, to_bv, gen_ulong(i * elem_bytes + j), big_endian));

        // Stitch bytes into one value.  big-endian: byte 0 at MSB (forward
        // accumulation); little-endian: byte 0 at LSB (reverse accumulation).
        expr2tc val;
        if (big_endian)
        {
          val = raw_bytes[0];
          for (unsigned int j = 1; j < elem_bytes; j++)
            val = concat2tc(
              get_uint_type(val->type->get_width() + 8), val, raw_bytes[j]);
        }
        else
        {
          val = raw_bytes[elem_bytes - 1];
          for (int j = (int)elem_bytes - 2; j >= 0; j--)
            val = concat2tc(
              get_uint_type(val->type->get_width() + 8), val, raw_bytes[j]);
        }
        elems.push_back(bitcast2tc(arr.subtype, val));
      }
      return convert_ast(constant_array2tc(type, elems));
    }

    return convert_ast(bitcast2tc(
      type,
      typecast2tc(
        get_uint_type(type_byte_size_bits(type).to_uint64()), to_bv)));
  }

  assert(
    is_struct_type(member.source_value) ||
    is_complex_type(member.source_value) ||
    is_pointer_type(member.source_value));
  unsigned int idx =
    get_member_name_field(member.source_value->type, member.member);

  smt_astt src = convert_ast(member.source_value);
  return src->project(this, idx);
}

smt_astt smt_solver_baset::round_real_to_int(smt_astt a)
{
  // SMT truncates downwards; however C truncates towards zero, which is not
  // the same. (Technically, it's also platform dependant). To get around this,
  // add one to the result in all circumstances, except where the value was
  // already an integer.
  smt_astt is_lt_zero = mk_lt(a, mk_smt_real("0"));

  // The actual conversion
  smt_astt as_int = mk_real2int(a);

  smt_astt one = mk_smt_int(BigInt(1));
  smt_astt plus_one = mk_add(one, as_int);

  // If it's an integer, just keep it's untruncated value.
  smt_astt is_int = mk_isint(a);
  smt_astt selected = mk_ite(is_int, as_int, plus_one);

  // Switch on whether it's > or < 0.
  return mk_ite(is_lt_zero, selected, as_int);
}

smt_astt smt_solver_baset::round_int_to_fp(
  smt_astt int_val,
  const floatbv_type2t &fbv_type,
  unsigned int source_width)
{
  // IEEE 754 round-to-nearest-even for integer-to-float casts under
  // --ir-ieee.  Integers with |i| < 2^S are exactly representable (S is the
  // total significand width).  For larger values, we compute the RNE-rounded
  // result exactly using SMT integer div/mod, building a cascaded ITE over
  // binade ranges [2^(S-1+k), 2^(S+k)) for k = 1..max_k.
  const unsigned int S = fbv_type.fraction + 1;

  // All source values fit within the significand: exact lift suffices.
  if (source_width <= S)
    return mk_int2real(int_val);

  // max_k: the highest binade index we need to cover.
  // For a W-bit source the maximum |i| is 2^(W-1) (signed, from INT_MIN) or
  // 2^W - 1 (unsigned), both of which sit in binade k = W - S.
  const unsigned int max_k = source_width - S;

  smt_astt zero_i = mk_smt_int(BigInt(0));
  smt_astt is_neg = mk_lt(int_val, zero_i);
  smt_astt abs_val = mk_ite(is_neg, mk_neg(int_val), int_val);

  // Base: exact conversion for values below the precision threshold.
  smt_astt result_abs = abs_val;

  smt_astt two = mk_smt_int(BigInt(2));

  // Build ITEs from k = 1 up to max_k.  Each step wraps the previous result:
  //   result_abs = ite(abs_val >= lo_k, quantized_k, result_abs)
  // When expanded, the outermost true condition dominates, so the highest
  // matching k (the correct binade) determines the final value.
  for (unsigned int k = 1; k <= max_k; k++)
  {
    BigInt lo = power(BigInt(2), BigInt(S - 1 + k)); // lower bound of binade
    BigInt ulp = power(BigInt(2), BigInt(k));        // unit of least precision
    BigInt half_u = power(BigInt(2), BigInt(k - 1)); // ulp / 2

    smt_astt lo_expr = mk_smt_int(lo);
    smt_astt ulp_expr = mk_smt_int(ulp);
    smt_astt half_expr = mk_smt_int(half_u);

    smt_astt remainder = mk_mod(abs_val, ulp_expr);
    smt_astt floor_val = mk_sub(abs_val, remainder); // round down
    smt_astt ceil_val = mk_add(floor_val, ulp_expr); // round up

    // Tie-breaking: when remainder == ulp/2, choose the even neighbour
    // (the one whose index floor_val/ulp is even).
    smt_astt quotient = mk_div(floor_val, ulp_expr);
    smt_astt floor_even = mk_eq(mk_mod(quotient, two), zero_i);
    smt_astt tie_val = mk_ite(floor_even, floor_val, ceil_val);

    smt_astt quantized = mk_ite(
      mk_eq(remainder, zero_i),
      abs_val, // exactly on a representable value
      mk_ite(
        mk_lt(remainder, half_expr),
        floor_val, // closer to floor
        mk_ite(mk_gt(remainder, half_expr), ceil_val, tie_val)));

    result_abs = mk_ite(mk_le(lo_expr, abs_val), quantized, result_abs);
  }

  smt_astt signed_result = mk_ite(is_neg, mk_neg(result_abs), result_abs);
  return mk_int2real(signed_result);
}

smt_astt smt_solver_baset::round_fixedbv_to_int(
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

smt_astt smt_solver_baset::make_bool_bit(smt_astt a)
{
  assert(
    a->sort->id == SMT_SORT_BOOL &&
    "Wrong sort fed to "
    "smt_solver_baset::make_bool_bit");
  smt_astt one =
    (int_encoding) ? mk_smt_int(BigInt(1)) : mk_smt_bv(BigInt(1), 1);
  smt_astt zero =
    (int_encoding) ? mk_smt_int(BigInt(0)) : mk_smt_bv(BigInt(0), 1);
  return mk_ite(a, one, zero);
}

smt_astt smt_solver_baset::make_bit_bool(smt_astt a)
{
  assert(
    ((!int_encoding && a->sort->id == SMT_SORT_BV) ||
     (int_encoding && a->sort->id == SMT_SORT_INT)) &&
    "Wrong sort fed to smt_solver_baset::make_bit_bool");

  smt_astt one =
    (int_encoding) ? mk_smt_int(BigInt(1)) : mk_smt_bv(BigInt(1), 1);
  return mk_eq(a, one);
}

/** Make an n-ary function application.
 *  Takes a vector of smt_ast's, and creates a single
 *  function app over all the smt_ast's.
 */
template <typename Object, typename Method>
static smt_astt
make_n_ary(const Object o, const Method m, const smt_solver_baset::ast_vec &v)
{
  assert(!v.empty());

  // Chain these.
  smt_astt result = v.front();
  for (std::size_t i = 1; i < v.size(); ++i)
    result = (o->*m)(result, v[i]);

  return result;
}

smt_astt smt_solver_baset::make_n_ary_and(const ast_vec &v)
{
  return v.empty() ? mk_smt_bool(true) // empty conjunction is true
                   : make_n_ary(this, &smt_solver_baset::mk_and, v);
}

smt_astt smt_solver_baset::make_n_ary_or(const ast_vec &v)
{
  return v.empty() ? mk_smt_bool(false) // empty disjunction is false
                   : make_n_ary(this, &smt_solver_baset::mk_or, v);
}

expr2tc
smt_solver_baset::fix_array_idx(const expr2tc &idx, const type2tc &arr_sort)
{
  if (int_encoding)
    return idx;

  smt_sortt s = convert_sort(arr_sort);
  size_t domain_width = s->get_domain_width();

  // Otherwise, we need to extract the lower bits out of this.
  return typecast2tc(
    get_uint_type(domain_width), idx, gen_zero(get_int32_type()));
}

/** Convert the size of an array to its bit width. Essential log2 with
 *  some rounding. */
static unsigned long size_to_bit_width(unsigned long sz)
{
  uint64_t domwidth = 2;
  unsigned int dombits = 1;

  // Shift domwidth up until it's either larger or equal to sz, or we risk
  // overflowing.
  while (domwidth != 0x8000000000000000ULL && domwidth < sz)
  {
    domwidth <<= 1;
    dombits++;
  }

  if (domwidth == 0x8000000000000000ULL)
    dombits = 64;

  return dombits;
}

unsigned long array_domain_width_or_word_size(const array_type2t &arr)
{
  // For constant-size arrays compute the minimal index width; for dynamic/VLA
  // or infinite arrays the size is not known statically, so fall back to the
  // machine word size which is always a valid index width.
  if (!is_nil_expr(arr.array_size) && is_constant_int2t(arr.array_size))
    return size_to_bit_width(
      to_constant_int2t(arr.array_size).value.to_uint64());
  return config.ansi_c.word_size;
}

type2tc make_array_domain_type(const array_type2t &arr)
{
  // Start special casing if this is an array of arrays.
  if (!is_array_type(arr.subtype))
  {
    // Normal array, work out what the domain sort is.
    if (config.options.get_bool_option("int-encoding"))
      return get_uint_type(config.ansi_c.int_width);

    return get_uint_type(array_domain_width_or_word_size(arr));
  }

  // Infinite arrays of arrays (e.g. nested Solidity mappings) are NOT
  // flattened — each level uses its own domain.  flatten_array_type()
  // already skips infinite arrays, so the domain must stay single-level.
  if (arr.size_is_infinite)
    return get_uint_type(array_domain_width_or_word_size(arr));

  // This is a finite array of arrays -- we're going to convert this into a
  // single array that has an extended domain. Work out that width.

  unsigned int domwidth = array_domain_width_or_word_size(arr);

  type2tc subarr = arr.subtype;
  while (is_array_type(subarr))
  {
    domwidth += array_domain_width_or_word_size(to_array_type(subarr));
    subarr = to_array_type(subarr).subtype;
  }

  return get_uint_type(domwidth);
}

expr2tc smt_solver_baset::array_domain_to_width(const type2tc &type)
{
  const unsignedbv_type2t &uint = to_unsignedbv_type(type);
  return constant_int2tc(index_type2(), BigInt::power2(uint.width));
}

static expr2tc gen_additions(const type2tc &type, std::vector<expr2tc> &exprs)
{
  // Reached end of recursion
  if (exprs.size() == 2)
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

expr2tc
smt_solver_baset::decompose_select_chain(const expr2tc &expr, expr2tc &base)
{
  const index2t *idx = &to_index2t(expr);

  // First we need to find the flatten_array_type, to cast symbols/constants
  // with different types during the addition and multiplication. They'll be
  // casted to the flattened array index type
  while (is_index2t(idx->source_value))
    idx = &to_index2t(idx->source_value);

  type2tc subtype = make_array_domain_type(
    to_array_type(flatten_array_type(idx->source_value->type)));

  // Rewrite the store chain as additions and multiplications
  idx = &to_index2t(expr);

  // Multiplications will hold of the mult2tc terms, we have to
  // add them together in the end
  std::vector<expr2tc> multiplications;
  multiplications.push_back(typecast2tc(subtype, idx->index));

  while (is_index2t(idx->source_value))
  {
    idx = &to_index2t(idx->source_value);

    type2tc t = flatten_array_type(idx->type);
    assert(is_array_type(t));

    multiplications.push_back(mul2tc(
      subtype,
      typecast2tc(subtype, to_array_type(t).array_size),
      typecast2tc(subtype, idx->index)));
  }

  if (multiplications.size() == 1)
  {
    // Single-level select into a multi-dimensional type (e.g. reading one row
    // of a 2-D array after a slice). Return the lone index directly.
    base = idx->source_value;
    expr2tc output = multiplications[0];
    simplify(output);
    return output;
  }

  // Add them together
  expr2tc output = gen_additions(subtype, multiplications);

  // Try to simplify the expression
  simplify(output);

  // Give the caller the base array object / thing. So that it can actually
  // select out of the right piece of data.
  base = idx->source_value;
  return output;
}

expr2tc smt_solver_baset::decompose_store_chain(
  const expr2tc &expr,
  expr2tc &update_val)
{
  const with2t *with = &to_with2t(expr);

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

  while (is_with2t(with->update_value) && is_array_type(with->update_value))
  {
    with = &to_with2t(with->update_value);

    type2tc t = flatten_array_type(with->update_value->type);

    multiplications.push_back(mul2tc(
      subtype,
      typecast2tc(subtype, to_array_type(t).array_size),
      typecast2tc(subtype, with->update_field)));
  }

  if (multiplications.size() == 1)
  {
    // Single-level store into a multi-dimensional type (e.g. updating one row
    // of a 2-D array after a slice). Return the lone index directly.
    update_val = with->update_value;
    expr2tc output = typecast2tc(subtype, to_with2t(expr).update_field);
    simplify(output);
    return output;
  }

  // Add them together
  expr2tc output = gen_additions(subtype, multiplications);

  // Try to simplify the expression
  simplify(output);

  // Fix base expr
  update_val = with->update_value;
  return output;
}

smt_astt smt_solver_baset::convert_array_index(const expr2tc &expr)
{
  const index2t &index = to_index2t(expr);
  expr2tc src_value = index.source_value;

  expr2tc newidx;
  // Source type might not be an array (e.g. vector); to_array_type() throws
  // std::bad_cast under -DNDEBUG-off builds. Gate the size_is_infinite probe
  // on is_array_type() before dereferencing.
  const bool src_is_infinite_array =
    is_array_type(index.source_value->type) &&
    to_array_type(index.source_value->type).size_is_infinite;
  if (is_index2t(index.source_value) && !src_is_infinite_array)
  {
    // Finite multi-dimensional arrays: flatten via decompose_select_chain.
    newidx = decompose_select_chain(expr, src_value);
  }
  else
  {
    // Single-level index, or infinite arrays (nested Solidity mappings) —
    // use direct select without flattening.
    newidx = fix_array_idx(index.index, index.source_value->type);
  }

  // Firstly, if it's a string, shortcircuit.
  if (is_constant_string2t(index.source_value))
  {
    smt_astt tmp = convert_ast(src_value);
    return tmp->select(this, newidx);
  }

  smt_astt a = convert_ast(src_value);
  a = a->select(this, newidx);

  const type2tc &arrsubtype =
    is_vector_type(index.source_value->type)
      ? to_vector_type(index.source_value->type).subtype
      : to_array_type(index.source_value->type).subtype;
  if (is_bool_type(arrsubtype) && !array_api->supports_bools_in_arrays)
    return make_bit_bool(a);

  return a;
}

smt_astt smt_solver_baset::convert_array_store(const expr2tc &expr)
{
  const with2t &with = to_with2t(expr);
  expr2tc update_val = with.update_value;
  expr2tc newidx;

  if (
    is_array_type(with.type) &&
    is_array_type(to_array_type(with.type).subtype) &&
    !to_array_type(with.type).size_is_infinite)
  {
    // Finite multi-dimensional arrays: flatten into single array with extended
    // domain via decompose_store_chain.
    newidx = decompose_store_chain(expr, update_val);
  }
  else
  {
    // Single-level arrays, or infinite arrays (including nested infinite arrays
    // used by Solidity nested mappings) — use direct index.
    newidx = fix_array_idx(with.update_field, with.type);
  }

  assert(is_array_type(expr->type));
  smt_astt src, update;
  const array_type2t &arrtype = to_array_type(expr->type);

  // Workaround for bools-in-arrays.
  if (is_bool_type(arrtype.subtype) && !array_api->supports_bools_in_arrays)
  {
    expr2tc cast = typecast2tc(get_uint_type(1), update_val);
    update = convert_ast(cast);
  }
  else
  {
    update = convert_ast(update_val);
  }

  src = convert_ast(with.source_value);
  return src->update(this, update, 0, newidx);
}

type2tc smt_solver_baset::flatten_array_type(const type2tc &type)
{
  // If vector, convert to array
  if (is_vector_type(type))
    return array_type2tc(
      to_vector_type(type).subtype, to_vector_type(type).array_size, false);

  // If this is not an array, we return an array of size 1
  if (!is_array_type(type))
    return array_type2tc(type, gen_one(int_type2()), false);

  // Don't touch these
  if (to_array_type(type).size_is_infinite)
    return type;

  // No need to handle one dimensional arrays
  if (!is_array_type(to_array_type(type).subtype))
    return type;

  type2tc subtype = get_flattened_array_subtype(type);
  assert(is_array_type(to_array_type(type).subtype));

  type2tc type_rec = type;
  expr2tc arr_size1 = to_array_type(type_rec).array_size;

  type_rec = to_array_type(type_rec).subtype;
  expr2tc arr_size2 = to_array_type(type_rec).array_size;

  if (arr_size1->type != arr_size2->type)
    arr_size1 = typecast2tc(arr_size2->type, arr_size1);

  expr2tc arr_size = mul2tc(arr_size1->type, arr_size1, arr_size2);

  while (is_array_type(to_array_type(type_rec).subtype))
  {
    arr_size = mul2tc(
      arr_size1->type,
      to_array_type(to_array_type(type_rec).subtype).array_size,
      arr_size);
    type_rec = to_array_type(type_rec).subtype;
  }
  simplify(arr_size);
  return array_type2tc(subtype, arr_size, false);
}

static expr2tc constant_index_into_array(const expr2tc &array, uint64_t idx)
{
  assert(is_array_type(array));

  const array_type2t &arr_type = to_array_type(array->type);
  const expr2tc &arr_size = arr_type.array_size;
  assert(arr_size);
  assert(is_constant_int2t(arr_size));

  if (is_constant_array_of2t(array))
  {
    assert(idx < to_constant_int2t(arr_size).value);
    return to_constant_array_of2t(array).initializer;
  }

  if (is_constant_array2t(array))
  {
    assert(idx < to_constant_int2t(arr_size).value);
    return to_constant_array2t(array).datatype_members[idx];
  }

  return index2tc(
    arr_type.subtype, array, constant_int2tc(arr_size->type, idx));
}

/**
 * Transforms an expression of array type with constant size into a
 * constant_array2t expression of flattened form, that is, the subtype is not of
 * array type.
 *
 * Works in tandem with flatten_array_type().
 *
 * TODO: also support constant_array_of2t as an optimization
 */
expr2tc smt_solver_baset::flatten_array_body(const expr2tc &expr)
{
  const array_type2t &arr_type = to_array_type(expr->type);
  bool subtype_is_array = is_array_type(arr_type.subtype);

  // innermost level, just return the array
  if (!subtype_is_array && is_constant_array2t(expr))
    return expr;

  const expr2tc &arr_size = arr_type.array_size;
  assert(is_constant_int2t(arr_size));
  const BigInt &size = to_constant_int2t(arr_size).value;
  assert(size.is_uint64());

  std::vector<expr2tc> sub_exprs;
  for (uint64_t i = 0, sz = size.to_uint64(); i < sz; i++)
  {
    expr2tc elem = constant_index_into_array(expr, i);

    if (subtype_is_array)
    {
      expr2tc flat_elem = flatten_array_body(elem);
      const auto &elems = to_constant_array2t(flat_elem).datatype_members;
      sub_exprs.insert(sub_exprs.end(), elems.begin(), elems.end());
    }
    else
      sub_exprs.push_back(elem);
  }

  return constant_array2tc(flatten_array_type(expr->type), sub_exprs);
}

type2tc smt_solver_baset::get_flattened_array_subtype(const type2tc &type)
{
  // Get the subtype of an array, ensuring that any intermediate arrays have
  // been flattened.

  // For infinite arrays of arrays (nested Solidity mappings), do NOT flatten
  // past the first level — each level uses its own SMT array sort.
  if (
    is_array_type(type) && to_array_type(type).size_is_infinite &&
    is_array_type(to_array_type(type).subtype))
    return to_array_type(type).subtype;

  type2tc type_rec = type;
  while (is_array_type(type_rec) || is_vector_type(type_rec))
  {
    type_rec = is_array_type(type_rec) ? to_array_type(type_rec).subtype
                                       : to_vector_type(type_rec).subtype;
  }

  // type_rec is now the base type.
  return type_rec;
}

void smt_solver_baset::pre_solve()
{
  // A new solve produces a fresh model; drop memoised l_get values.
  l_get_cache.clear();

  // NB: always perform tuple constraint adding first, as it covers tuple
  // arrays too, and might end up generating more ASTs to be encoded in
  // the array api class.
  tuple_api->add_tuple_constraints_for_solving();
  array_api->add_array_constraints_for_solving();
}

expr2tc smt_solver_baset::get(const expr2tc &expr)
{
  if (is_constant_number(expr))
    return expr;

  if (is_symbol2t(expr) && to_symbol2t(expr).thename == "NULL")
    return expr;

  expr2tc res = expr;

  // Special cases:
  switch (res->expr_id)
  {
  case expr2t::index_id:
  {
    // If we try to get an index from the solver, it will first
    // return the whole array and then get the index, we can
    // do better and call get_array_element directly
    index2t index = to_index2t(res);
    expr2tc src_value = index.source_value;

    expr2tc newidx;
    // Same NDEBUG-off safety guard as in convert_array_index() above.
    const bool src_is_infinite_array =
      is_array_type(index.source_value->type) &&
      to_array_type(index.source_value->type).size_is_infinite;
    if (is_index2t(index.source_value) && !src_is_infinite_array)
    {
      newidx = decompose_select_chain(expr, src_value);
    }
    else
    {
      newidx = fix_array_idx(index.index, index.source_value->type);
    }

    // if the source value is a constant, there's no need to
    // call the array api
    if (is_constant_number(src_value))
      return src_value;

    // Convert the idx, it must be an integer
    expr2tc idx = get(newidx);
    if (is_constant_int2t(idx))
    {
      // Convert the array so we can call the array api
      smt_astt array = convert_ast(src_value);

      // Retrieve the element
      if (is_tuple_array_ast_type(src_value->type))
        res = tuple_api->tuple_get_array_elem(
          array, to_constant_int2t(idx).value.to_uint64(), res->type);
      else
        res = array_api->get_array_elem(
          array,
          to_constant_int2t(idx).value.to_uint64(),
          get_flattened_array_subtype(res->type));

      // If we got a nil result, return original expression
      if (is_nil_expr(res))
        return expr;
    }

    // TODO: Give up, then what?
    break;
  }

  case expr2t::with_id:
  {
    // This will be converted
    const with2t &with = to_with2t(res);
    expr2tc update_val = with.update_value;

    if (
      is_array_type(with.type) &&
      is_array_type(to_array_type(with.type).subtype) &&
      !to_array_type(with.type).size_is_infinite)
    {
      decompose_store_chain(expr, update_val);
    }

    /* Try to construct a constant struct when we handle  
     * struct type "with" expr2tc
     *
     * Simplify the source value. If it is a constant,
     * we will replace the corresponding update value
     * i.e. S = {.x=0, .y=0}; S = S WITH [x:=1];
     * Reduced to this: S = {.x=1, .y=0}
     */
    if (is_struct_type(with.type))
    {
      expr2tc source = get(with.source_value);
      if (is_constant_struct2t(source))
      {
        std::vector<expr2tc> members;
        constant_struct2t s = to_constant_struct2t(source);
        const constant_string2t &update_name =
          to_constant_string2t(with.update_field);
        for (size_t i = 0; i < s.datatype_members.size(); i++)
        {
          irep_idt name = to_struct_type(with.type).member_names[i];
          if (update_name.value == name)
            members.push_back(update_val);
          else
            members.push_back(s.datatype_members[i]);
        }

        return get(constant_struct2tc(with.type, members));
      }
    }

    /* This function get() is only used to obtain assigned values to the RHS of
     * SSA_step assignments in order to generate counter-examples. with2t
     * expressions for these RHS are only generated during the transformations
     * performed by symex_assign(), which from the counter-example's point of
     * view behave like no-ops as the RHS of counter-example assignments should
     * only show the concretely updated value in expressions of composite type.
     * Thus, there is no need to construct the full with2t expression here,
     * since it can't sensibly be interpreted anyways due to simplification
     * during convert_ast().
     *
     * Thereby we also do not have to care about cases when src->type and the
     * constructed with2t's source's type differ, e.g., arrays of differing
     * sizes would be constructed for regression/esbmc/loop_unroll_incr_true. */
    return get(update_val);
  }

  case expr2t::address_of_id:
    return res;

  case expr2t::overflow_id:
  case expr2t::pointer_offset_id:
  case expr2t::same_object_id:
  case expr2t::symbol_id:
    return get_by_type(res);

  case expr2t::member_id:
  {
    const member2t &mem = to_member2t(res);
    expr2tc mem_src = mem.source_value;

    if (is_symbol2t(mem_src) && !is_pointer_type(expr) && !is_struct_type(expr))
    {
      return get_by_type(res);
    }
    else if (is_array_type(expr))
    {
      return get_by_type(res);
    }

    simplify(res);
    return res;
  }

  case expr2t::if_id:
  {
    // Special case for ternary if, for cases when we are updating the member
    // of a struct (SSA indexes are omitted):
    //
    // d2 == (!(SAME-OBJECT(pd, &d1)) ? (d2 WITH [a:=0]) : d2)
    //
    // i.e., update the field 'a' of 'd2', if a given condition holds,
    // otherwise, do nothing.

    // The problem of relying on the simplification code is because the type of
    // the ternary is a struct, and if the condition holds, we extract the
    // update value from the WITH expression, in this case '0', and cast it to
    // struct. So now we try to query the solver for which side is used and
    // we return it, without casting to the ternary if type.
    if2t i = to_if2t(res);

    expr2tc c = get(i.cond);
    if (is_true(c))
      return get(i.true_value);

    if (is_false(c))
      return get(i.false_value);
  }

  default:;
  }

  if (is_array_type(expr->type))
  {
    // Resolve symbolic array_size fields to the concrete values the solver
    // assigned them. Functional rewrite: build a new array_type if any size
    // changed, then rebuild res with that type. Mirrors the original two-level
    // walk (outer array + its immediate subtype if also array); preserves the
    // historic behaviour of not recursing further.
    auto resolve_size = [this](const expr2tc &s) {
      if (!is_nil_expr(s) && is_symbol2t(s))
        return get(s);
      return s;
    };

    const array_type2t &outer = to_array_type(res->type);
    expr2tc new_outer_size = resolve_size(outer.array_size);
    type2tc new_subtype = outer.subtype;
    if (is_array_type(new_subtype))
    {
      const array_type2t &inner = to_array_type(new_subtype);
      expr2tc new_inner_size = resolve_size(inner.array_size);
      if (new_inner_size != inner.array_size)
        new_subtype =
          array_type2tc(inner.subtype, new_inner_size, inner.size_is_infinite);
    }
    if (new_outer_size != outer.array_size || new_subtype != outer.subtype)
    {
      type2tc new_type =
        array_type2tc(new_subtype, new_outer_size, outer.size_is_infinite);
      res = res->with_type(new_type);
    }
  }

  // Recurse on operands
  bool have_all = true;
  bool has_null_operands = false;

  res->Foreach_operand([this, &have_all, &has_null_operands](expr2tc &e) {
    if (!e)
    {
      has_null_operands = true;
      have_all = false;
      return;
    }

    expr2tc new_e = get(e);
    if (new_e)
      e = new_e;
    else
      have_all = false;
  });

  // If we have null operands, return early to avoid crashes in simplify()
  if (has_null_operands)
    return expr;

  // Only simplify if all operands are valid
  if (have_all)
    simplify(res);

  return res;
}

expr2tc smt_solver_baset::get_by_ast(const type2tc &type, smt_astt a)
{
  switch (type->type_id)
  {
  case type2t::bool_id:
    return get_bool(a) ? gen_true_expr() : gen_false_expr();

  case type2t::unsignedbv_id:
  case type2t::signedbv_id:
  case type2t::fixedbv_id:
    return get_by_value(type, get_bv(a, is_signedbv_type(type)));

  case type2t::floatbv_id:
    if (int_encoding)
    {
      BigInt numerator, denominator;
      if (get_rational(a, numerator, denominator))
      {
        double value = convert_rational_to_double(numerator, denominator);
        unsigned int type_width = type->get_width();

        if (type_width == ieee_float_spect::single_precision().width())
        {
          // Single precision (32-bit float)
          ieee_floatt ieee_val(ieee_float_spect::single_precision());
          ieee_val.from_float(static_cast<float>(value));
          return constant_floatbv2tc(ieee_val);
        }
        else if (type_width == ieee_float_spect::double_precision().width())
        {
          // Double precision (64-bit double)
          ieee_floatt ieee_val(ieee_float_spect::double_precision());
          ieee_val.from_double(value);
          return constant_floatbv2tc(ieee_val);
        }
        else
        {
          log_error(
            "Unsupported floatbv width ({}) for rational conversion",
            type_width);
        }
      }
      return expr2tc();
    }
    return constant_floatbv2tc(fp_api->get_fpbv(a));

  case type2t::complex_id:
  case type2t::struct_id:
  case type2t::pointer_id:
    return tuple_api->tuple_get(type, a);

  case type2t::union_id:
  {
    expr2tc uint_rep =
      get_by_ast(get_uint_type(type_byte_size_bits(type).to_uint64()), a);
    std::vector<expr2tc> members;
    /* TODO: this violates the assumption in the rest of ESBMC that
     *       constant_union2t only have at most 1 member initializer.
     *       Maybe it makes sense to go for one of the largest ones instead of
     *       all members? */
    for (const type2tc &member_type : to_union_type(type).members)
    {
      expr2tc cast = bitcast2tc(
        member_type,
        typecast2tc(
          get_uint_type(type_byte_size_bits(member_type).to_uint64()),
          uint_rep));
      simplify(cast);
      members.push_back(cast);
    }
    return constant_union2tc(
      type, "" /* TODO: which field assigned last? */, members);
  }

  case type2t::array_id:
    return get_array(type, a);

  default:
    if (!options.get_bool_option("non-supported-models-as-zero"))
    {
      log_error(
        "Unimplemented type'd expression ({}) in smt get",
        fmt::underlying(type->type_id));
      abort();
    }
    else
    {
      log_warning(
        "Unimplemented type'd expression ({}) in smt get. Returning zero!",
        fmt::underlying(type->type_id));
      return gen_zero(type);
    }
  }
}

// Convert a rational number represented by two BigInts into a IEEE 754 double.
// This replaces the previous std::stod-based approach which could not cope
// with extremely large/small numerators or denominators exposed by --ir mode.
double smt_solver_baset::convert_rational_to_double(
  const BigInt &numerator,
  const BigInt &denominator)
{
  constexpr size_t INITIAL_BUFFER_SIZE = 1024;
  constexpr size_t MAX_BUFFER_SIZE = 100000;

  // Start with the legacy fixed-size buffers.
  std::vector<char> num_buffer(INITIAL_BUFFER_SIZE, '\0');
  std::vector<char> den_buffer(INITIAL_BUFFER_SIZE, '\0');

  // Populate the buffer with the decimal representation of `value`, growing the
  // buffer as needed. We keep the legacy fixed-size path to avoid extra
  // allocations on the common fast path.
  auto ensure_string = [&](const BigInt &value, std::vector<char> &buffer) {
    while (true)
    {
      // 1) Try to reuse the current buffer (may already be large).
      if (!buffer.empty())
      {
        char *result = value.as_string(buffer.data(), buffer.size(), 10);

        if (result != nullptr)
        {
          // 1a) as_string returns a pointer to the first digit; copy it forward.
          size_t len = strnlen(result, buffer.size());
          if (len > 0 && len < buffer.size())
          {
            // Include the trailing '\0' so later std::stod sees the number.
            for (size_t i = 0; i <= len; ++i)
              buffer[i] = result[i];
            return true;
          }
        }
      }

      // 2) Need a larger buffer: estimate required digits (+ sign + '\0').
      size_t required = static_cast<size_t>(value.digits(10)) + 2;
      size_t next_size = buffer.size() * 2;
      if (next_size < required)
        next_size = required;
      if (next_size == 0)
        next_size = INITIAL_BUFFER_SIZE;
      if (next_size > MAX_BUFFER_SIZE)
        return false;

      // 3) Grow buffer and retry.
      buffer.assign(next_size, '\0');
    }
  };

  // Extract decimal strings for numerator/denominator (with fallback).
  bool num_ok = ensure_string(numerator, num_buffer);
  bool den_ok = ensure_string(denominator, den_buffer);

  if (!num_ok || !den_ok)
  {
    // If conversion still fails, keep legacy behaviour: approximate sign.
    bool num_positive = !numerator.is_zero();
    bool den_positive = !denominator.is_zero();

    if (num_positive && den_positive)
    {
      log_warning(
        "BigInt as_string() failed for very small rational - returning minimal "
        "positive value");
      return DBL_MIN;
    }
    else
    {
      log_warning("BigInt as_string() failed and cannot determine sign");
      return 0.0;
    }
  }

  size_t num_len = strnlen(num_buffer.data(), num_buffer.size());
  size_t den_len = strnlen(den_buffer.data(), den_buffer.size());

  if (num_len >= num_buffer.size() - 1 || den_len >= den_buffer.size() - 1)
  {
    // Bail out if we still risk truncation.
    log_warning(
      "BigInt to string conversion may have been truncated - buffer too small");
    return 0.0;
  }

  bool numerator_negative = numerator.is_negative();
  bool denominator_negative = denominator.is_negative();

  BigInt num_abs = numerator;
  if (numerator_negative)
    num_abs.negate();

  BigInt den_abs = denominator;
  if (denominator_negative)
    den_abs.negate();

  if (den_abs.is_zero())
  {
    log_warning("Encountered rational with zero denominator during conversion");
    return 0.0;
  }

  BigInt quotient;
  BigInt remainder;
  BigInt::div(num_abs, den_abs, quotient, remainder);

  // Use ieee_floatt to map arbitrarily large integers onto the nearest double.

  ieee_floatt quotient_ieee(ieee_float_spect::double_precision());
  quotient_ieee.from_integer(quotient);
  double quotient_double = quotient_ieee.to_double();

  bool result_positive = !(numerator_negative ^ denominator_negative);

  if (std::isinf(quotient_double))
    return result_positive ? std::numeric_limits<double>::infinity()
                           : -std::numeric_limits<double>::infinity();

  // Only a finite number of fractional bits affects a double. We accumulate a
  // few extra guard bits so that rounding to nearest double later is stable.
  constexpr unsigned guard_bits = 10;
  const unsigned max_fraction_bits =
    std::numeric_limits<double>::digits + guard_bits;

  long double fraction = 0.0L;
  long double factor = 0.5L;

  for (unsigned i = 0; i < max_fraction_bits && !remainder.is_zero(); ++i)
  {
    remainder *= 2;
    if (remainder.compare(den_abs) >= 0)
    {
      remainder -= den_abs;
      fraction += factor;
    }
    factor *= 0.5L;
  }

  // Combine integer and fractional part using long double to minimize rounding
  // error before the final cast back to double.
  long double total = static_cast<long double>(quotient_double) + fraction;
  double result = static_cast<double>(total);

  if (result == 0.0 && !num_abs.is_zero())
  {
    double min_positive = std::numeric_limits<double>::denorm_min();
    result = result_positive ? min_positive : -min_positive;
    return result;
  }

  if (!result_positive)
    result = -result;

  return result;
}

expr2tc smt_solver_baset::get_by_type(const expr2tc &expr)
{
  switch (expr->type->type_id)
  {
  case type2t::bool_id:
  case type2t::unsignedbv_id:
  case type2t::signedbv_id:
  case type2t::fixedbv_id:
  case type2t::floatbv_id:
  case type2t::union_id:
    return get_by_ast(expr->type, convert_ast(expr));

  case type2t::array_id:
    return get_array(expr);

  case type2t::complex_id:
  case type2t::struct_id:
  case type2t::pointer_id:
    return tuple_api->tuple_get(expr);

  default:
    if (!options.get_bool_option("non-supported-models-as-zero"))
    {
      log_error(
        "Unimplemented type'd expression ({}) in smt get",
        fmt::underlying(expr->type->type_id));
      abort();
    }
    else if (!is_code_type(expr))
    {
      log_warning(
        "Unimplemented type'd expression ({}) in smt get. Returning zero!",
        fmt::underlying(expr->type->type_id));
      return gen_zero(expr->type);
    }
    else
    {
      log_warning(
        "Unimplemented type'd expression ({}) in smt get. Returning nil!",
        fmt::underlying(expr->type->type_id));
      return expr2tc();
    }
  }
}

expr2tc smt_solver_baset::get_array(const type2tc &type, smt_astt array)
{
  // XXX -- printing multidimensional arrays?

  // Fetch the array bounds, if it's huge then assume this is a 1024 element
  // array. Then fetch all elements and formulate a constant_array.
  size_t w = array->sort->get_domain_width();
  if (w > 10)
    w = 10;

  const type2tc flat_type = flatten_array_type(type);
  array_type2t ar = to_array_type(flat_type);

  expr2tc arr_size;
  if (type == flat_type && !ar.size_is_infinite)
    // avoid handelling the flattend multidimensional and malloc arrays(assume size is infinite)
    arr_size = to_array_type(flat_type).array_size;
  else
    arr_size = constant_int2tc(index_type2(), BigInt(1ULL << w));

  type2tc arr_type = array_type2tc(ar.subtype, arr_size, false);
  std::vector<expr2tc> fields;

  bool uses_tuple_api = is_tuple_array_ast_type(type);

  BigInt elem_size;
  if (is_constant_int2t(arr_size))
  {
    elem_size = to_constant_int2t(arr_size).value;
    assert(elem_size.is_uint64());
  }
  else
    elem_size = BigInt(1ULL << w);

  for (size_t i = 0; i < elem_size; i++)
  {
    expr2tc elem;
    if (uses_tuple_api)
      elem =
        tuple_api->tuple_get_array_elem(array, i, to_array_type(type).subtype);
    else
      elem = array_api->get_array_elem(array, i, ar.subtype);
    fields.push_back(elem);
  }

  return constant_array2tc(arr_type, fields);
}

expr2tc smt_solver_baset::get_array(const expr2tc &expr)
{
  smt_astt array = convert_ast(expr);
  return get_array(expr->type, array);
}

smt_astt smt_solver_baset::array_create(const expr2tc &expr)
{
  if (is_constant_array_of2t(expr))
    return convert_array_of_prep(expr);
  // Check size
  assert(is_constant_array2t(expr) || is_constant_vector2t(expr));
  expr2tc size = array_or_vector_size(expr->type);
  bool is_infinite = array_or_vector_size_is_infinite(expr->type);
  const auto &members = is_constant_array2t(expr)
                          ? to_constant_array2t(expr).datatype_members
                          : to_constant_vector2t(expr).datatype_members;

  // Handle constant array expressions: these don't have tuple type and so
  // don't need funky handling, but we need to create a fresh new symbol and
  // repeatedly store the desired data into it, to create an SMT array
  // representing the expression we're converting.
  std::string name = mk_fresh_name("array_create::") + ".";
  expr2tc newsym = symbol2tc(expr->type, name);

  // Guarentee nothing, this is modelling only.
  if (is_infinite)
    return convert_ast(newsym);

  if (!is_constant_int2t(size))
  {
    log_error("Non-constant sized array of type constant_array_of2t");
    abort();
  }

  const constant_int2t &thesize = to_constant_int2t(size);
  unsigned int sz = thesize.value.to_uint64();

  // Repeatedly store things into this.
  smt_astt newsym_ast = convert_ast(newsym);
  for (unsigned int i = 0; i < sz; i++)
  {
    expr2tc init = members[i];

    // Workaround for bools-in-arrays
    if (
      is_bool_type(members[i]->type) && !int_encoding &&
      !array_api->supports_bools_in_arrays)
      init = typecast2tc(unsignedbv_type2tc(1), init);

    newsym_ast = newsym_ast->update(this, convert_ast(init), i);
  }

  return newsym_ast;
}

smt_astt smt_solver_baset::convert_array_of_prep(const expr2tc &expr)
{
  const constant_array_of2t &arrof = to_constant_array_of2t(expr);
  const array_type2t &arrtype = to_array_type(arrof.type);
  expr2tc base_init;
  unsigned long array_size = 0;

  // Nested infinite arrays (e.g. Solidity nested mappings): do NOT flatten.
  // Create Array(BV64, Array(BV64, V)) where the inner initializer is itself
  // an array_of that will be recursively converted.
  if (arrtype.size_is_infinite && is_array_type(arrtype.subtype))
  {
    // Convert the inner array_of initializer directly (recursive)
    smt_astt inner = convert_ast(arrof.initializer);
    array_size = array_domain_width_or_word_size(arrtype);
    return array_api->convert_array_of(inner, array_size);
  }

  // So: we have an array_of, that we have to convert into a bunch of stores.
  // However, it might be a nested array. If that's the case, then we're
  // guaranteed to have another array_of in the initializer which we can flatten
  // to a single array of whatever's at the bottom of the array_of. Or, it's
  // a constant_array, in which case we can just copy the contents.
  if (is_array_type(arrtype.subtype))
  {
    expr2tc rec_expr = expr;

    if (is_constant_array_of2t(to_constant_array_of2t(rec_expr).initializer))
    {
      type2tc flat_type = flatten_array_type(expr->type);
      const array_type2t &arrtype2 = to_array_type(flat_type);
      array_size = array_domain_width_or_word_size(arrtype2);

      while (is_constant_array_of2t(rec_expr))
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
      for (uint64_t i = 0; i < size.to_uint64(); i++)
        new_contents.insert(
          new_contents.end(),
          constarray.datatype_members.begin(),
          constarray.datatype_members.end());

      // Create new expression, convert and return that.
      expr2tc newsize = mul2tc(
        arrtype.array_size->type,
        arrtype.array_size,
        constarray_type.array_size);
      simplify(newsize);

      type2tc new_arr_type =
        array_type2tc(constarray_type.subtype, newsize, false);
      expr2tc new_const_array =
        constant_array2tc(new_arr_type, std::move(new_contents));
      return convert_ast(new_const_array);
    }
  }
  else
  {
    base_init = arrof.initializer;
    array_size = array_domain_width_or_word_size(arrtype);
  }

  if (is_struct_type(base_init->type))
    return tuple_api->tuple_array_of(base_init, array_size);
  if (is_pointer_type(base_init->type))
    return pointer_array_of(base_init, array_size);
  else
    return array_api->convert_array_of(convert_ast(base_init), array_size);
}

smt_astt array_iface::default_convert_array_of(
  smt_astt init_val,
  unsigned long array_size,
  smt_solver_baset *ctx)
{
  // We now an initializer, and a size of array to build. So:
  // Repeatedly store things into this.
  // XXX int mode

  if (init_val->sort->id == SMT_SORT_BOOL && !supports_bools_in_arrays)
  {
    smt_astt zero = ctx->mk_smt_bv(BigInt(0), 1);
    smt_astt one = ctx->mk_smt_bv(BigInt(0), 1);
    init_val = ctx->mk_ite(init_val, one, zero);
  }

  smt_sortt domwidth = ctx->mk_int_bv_sort(array_size);
  smt_sortt arrsort = ctx->mk_array_sort(domwidth, init_val->sort);
  smt_astt newsym_ast =
    ctx->mk_fresh(arrsort, "default_array_of::", init_val->sort);

  unsigned long long sz = 1ULL << array_size;
  for (unsigned long long i = 0; i < sz; i++)
    newsym_ast = newsym_ast->update(ctx, init_val, i);

  return newsym_ast;
}

smt_astt smt_solver_baset::pointer_array_of(
  const expr2tc &init_val [[maybe_unused]],
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
  expr2tc zero_val = constant_int2tc(machine_ptr, BigInt(0));
  std::vector<expr2tc> operands;
  operands.reserve(2);
  operands.push_back(zero_val);
  operands.push_back(zero_val);

  expr2tc strct = constant_struct2tc(pointer_struct, std::move(operands));
  return tuple_api->tuple_array_of(strct, array_width);
}

smt_astt smt_solver_baset::tuple_array_create_despatch(
  const expr2tc &expr,
  smt_sortt domain)
{
  // Take a constant_array2t or an array_of, and format the data from them into
  // a form palatable to tuple_array_create.

  // Strip out any pointers
  type2tc arr_type = expr->type;
  rewrite_ptrs_to_structs(arr_type);

  if (is_constant_array_of2t(expr))
  {
    const constant_array_of2t &arr = to_constant_array_of2t(expr);
    smt_astt arg = convert_ast(arr.initializer);

    return tuple_api->tuple_array_create(arr_type, &arg, true, domain);
  }

  assert(is_constant_array2t(expr));
  const constant_array2t &arr = to_constant_array2t(expr);
  std::vector<smt_astt> args(arr.datatype_members.size());
  unsigned int i = 0;
  for (auto const &it : arr.datatype_members)
  {
    args[i] = convert_ast(it);
    i++;
  }

  return tuple_api->tuple_array_create(arr_type, args.data(), false, domain);
}

void smt_solver_baset::rewrite_ptrs_to_structs(type2tc &type)
{
  // Type may contain pointers; replace those with the structure equivalent.
  // Ideally the real solver will never see pointer types.
  // Create a delegate that recurses over all subtypes, replacing pointers
  // as we go.
  struct
  {
    const type2tc &pointer_struct;

    void operator()(type2tc &e) const
    {
      if (is_pointer_type(e))
      {
        // Replace this field of the expr with a pointer struct :O:O:O:O
        e = pointer_struct;
      }
      else
      {
        // Recurse
        e->Foreach_subtype(*this);
      }
    }
  } delegate = {pointer_struct};

  type->Foreach_subtype(delegate);
}

// Default behaviors for SMT AST's

void smt_ast::assign(smt_solver_baset *ctx, smt_astt sym) const
{
  ctx->assert_ast(eq(ctx, sym));
}

smt_astt
smt_ast::ite(smt_solver_baset *ctx, smt_astt cond, smt_astt falseop) const
{
  return ctx->mk_ite(cond, this, falseop);
}

smt_astt smt_ast::eq(smt_solver_baset *ctx, smt_astt other) const
{
  // Simple approach: this is a leaf piece of SMT, compute a basic equality.
  return ctx->mk_eq(this, other);
}

smt_astt smt_ast::update(
  smt_solver_baset *ctx,
  smt_astt value,
  unsigned int idx,
  const expr2tc &idx_expr) const
{
  // If we're having an update applied to us, then the only valid situation
  // this can occur in is if we're an array.
  assert(sort->id == SMT_SORT_ARRAY);

  // We're an array; just generate a 'with' operation.
  expr2tc index;
  if (is_nil_expr(idx_expr))
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

smt_astt smt_ast::select(smt_solver_baset *ctx, const expr2tc &idx) const
{
  assert(
    sort->id == SMT_SORT_ARRAY &&
    "Select operation applied to non-array scalar AST");

  smt_astt args[2];
  args[0] = this;
  args[1] = ctx->convert_ast(idx);
  return ctx->mk_select(args[0], args[1]);
}

smt_astt smt_ast::project(
  smt_solver_baset *ctx [[maybe_unused]],
  unsigned int idx [[maybe_unused]]) const
{
  log_error("Projecting from non-tuple based AST");
  abort();
}

std::string smt_solver_baset::dump_smt()
{
  log_error("SMT dump not implemented for {}", solver_text());
  abort();
}

void smt_solver_baset::print_model()
{
  log_error("SMT model printing not implemented for {}", solver_text());
  abort();
}

tvt smt_solver_baset::l_get(smt_astt a)
{
  // Memoise against the current model. The cache is cleared whenever the
  // model can change (pre_solve / push_ctx / pop_ctx), so a hit always
  // reflects the assignment produced by the most recent solve. Guard ASTs
  // recur across thousands of SSA steps during trace building and each
  // miss bottoms out in an O(formula) get_bool(), so this collapses
  // repeated queries to one solver call per distinct AST.
  auto it = l_get_cache.find(a);
  if (it != l_get_cache.end())
    return it->second;
  tvt res = get_bool(a) ? tvt(true) : tvt(false);
  l_get_cache.emplace(a, res);
  return res;
}

tvt smt_solver_baset::l_get(const expr2tc &expr)
{
  assert(is_bool_type(expr));
  return l_get(convert_ast(expr));
}

void smt_solver_baset::dump_expr(const expr2tc &expr)
{
  convert_ast(expr)->dump();
}

expr2tc smt_solver_baset::get_by_ast(const expr2tc &expr)
{
  return get_by_ast(expr->type, convert_ast(expr));
}

expr2tc smt_solver_baset::get_by_value(const type2tc &type, BigInt value)
{
  switch (type->type_id)
  {
  case type2t::bool_id:
    return value.is_zero() ? gen_false_expr() : gen_true_expr();

  case type2t::unsignedbv_id:
  case type2t::signedbv_id:
    return constant_int2tc(type, value);

  case type2t::fixedbv_id:
  {
    // Build the fixedbv from its spec + raw bit pattern directly, mirroring
    // fixedbvt::from_expr (spec from the type, v = the value's signed binary
    // round-trip) without staging a legacy constant_exprt / type back-migration.
    fixedbvt fbv(fixedbv_spect(to_fixedbv_type(type)));
    fbv.set_value(
      binary2integer(integer2binary(value, type->get_width()), true));
    return constant_fixedbv2tc(fbv);
  }

  case type2t::floatbv_id:
  {
    // Likewise mirror ieee_floatt::from_expr: spec from the type, then unpack
    // the raw IEEE bit pattern (the value's unsigned binary round-trip).
    ieee_floatt f(ieee_float_spect(to_floatbv_type(type)));
    f.unpack(binary2integer(integer2binary(value, type->get_width()), false));
    return constant_floatbv2tc(f);
  }

  default:;
  }

  if (options.get_bool_option("non-supported-models-as-zero"))
  {
    log_warning(
      "Can't generate one for type {}. Returning zero", get_type_id(type));
    return gen_zero(type);
  }

  log_error("Can't generate one for type {}", get_type_id(type));
  abort();
}

smt_sortt smt_solver_baset::mk_bool_sort()
{
  log_error("Chosen solver doesn't support boolean sorts");
  abort();
}

smt_sortt smt_solver_baset::mk_real_sort()
{
  log_error("Chosen solver doesn't support real sorts");
  abort();
}

smt_sortt smt_solver_baset::mk_int_sort()
{
  log_error("Chosen solver doesn't support integer sorts");
  abort();
}

smt_sortt smt_solver_baset::mk_bv_sort(std::size_t)
{
  log_error("Chosen solver doesn't support bit vector sorts");
  abort();
}

smt_sortt smt_solver_baset::mk_fbv_sort(std::size_t)
{
  log_error("Chosen solver doesn't support bit vector sorts");
  abort();
}

smt_sortt smt_solver_baset::mk_array_sort(smt_sortt, smt_sortt)
{
  log_error("Chosen solver doesn't support array sorts");
  abort();
}

smt_sortt smt_solver_baset::mk_bvfp_sort(std::size_t, std::size_t)
{
  log_error("Chosen solver doesn't support bit vector sorts");
  abort();
}

smt_sortt smt_solver_baset::mk_bvfp_rm_sort()
{
  log_error("Chosen solver doesn't support bit vector sorts");
  abort();
}

smt_astt smt_solver_baset::mk_bvredor(smt_astt op)
{
  // bvredor = bvnot(bvcomp(x,0)) ? bv1 : bv0;

  smt_astt comp = mk_eq(op, mk_smt_bv(BigInt(0), op->sort->get_data_width()));

  smt_astt ncomp = mk_not(comp);

  // If it's true, return 1. Return 0, othewise.
  return mk_ite(ncomp, mk_smt_bv(1, 1), mk_smt_bv(BigInt(0), 1));
}

smt_astt smt_solver_baset::mk_bvredand(smt_astt op)
{
  // bvredand = bvcomp(x,-1) ? bv1 : bv0;

  smt_astt comp =
    mk_eq(op, mk_smt_bv(BigInt(ULLONG_MAX), op->sort->get_data_width()));

  // If it's true, return 1. Return 0, othewise.
  return mk_ite(comp, mk_smt_bv(1, 1), mk_smt_bv(BigInt(0), 1));
}

smt_astt smt_solver_baset::mk_add(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_bvadd(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_sub(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_bvsub(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_mul(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_bvmul(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_mod(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_bvsmod(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_bvumod(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_div(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_bvsdiv(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_bvudiv(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_shl(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_bvshl(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_bvashr(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_bvlshr(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_neg(smt_astt a)
{
  (void)a;
  abort();
}

smt_astt smt_solver_baset::mk_bvneg(smt_astt a)
{
  (void)a;
  abort();
}

smt_astt smt_solver_baset::mk_bvnot(smt_astt a)
{
  (void)a;
  abort();
}

smt_astt smt_solver_baset::mk_bvxor(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_bvor(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_bvand(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_implies(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_xor(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_or(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_and(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_not(smt_astt a)
{
  (void)a;
  abort();
}

smt_astt smt_solver_baset::mk_lt(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_bvult(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_bvslt(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_gt(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  return mk_lt(b, a);
}

smt_astt smt_solver_baset::mk_bvugt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return mk_not(mk_bvule(a, b));
}

smt_astt smt_solver_baset::mk_bvsgt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return mk_not(mk_bvsle(a, b));
}

smt_astt smt_solver_baset::mk_le(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_bvule(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_bvsle(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_ge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return mk_not(mk_lt(a, b));
}

smt_astt smt_solver_baset::mk_bvuge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return mk_not(mk_bvult(a, b));
}

smt_astt smt_solver_baset::mk_bvsge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  return mk_not(mk_bvslt(a, b));
}

smt_astt smt_solver_baset::mk_eq(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_neq(smt_astt a, smt_astt b)
{
  return mk_not(mk_eq(a, b));
}

smt_astt smt_solver_baset::mk_smt_uninterpreted_function(
  const std::string &name,
  const std::vector<smt_astt> &args,
  smt_sortt rangesort)
{
  // Fallback for backends without native uninterpreted-function support: mint a
  // fresh result and Ackermannise functional congruence against every earlier
  // application of the same function. This asserts only a valid property of any
  // function (equal arguments imply an equal result), so it prunes no
  // behaviour. Solvers that expose UFs override this and never reach here.
  smt_astt result = mk_smt_symbol(
    "__esbmc_uf_ackermann::" + name + "$" +
      std::to_string(uf_ackermann_counter++),
    rangesort);

  auto &history = uf_ackermann_history[name];
  for (const auto &prev : history)
  {
    // Only relate applications of equal arity; a different arity cannot be the
    // same function signature. Argument sorts are assumed consistent across
    // applications of one name (a single C declaration), as is required for the
    // native path too.
    if (prev.args.size() != args.size())
      continue;

    // Build (a0 == p0) && (a1 == p1) && ...; an empty argument list leaves the
    // guard tautological, so two nullary applications are simply made equal.
    smt_astt args_equal = mk_smt_bool(true);
    for (std::size_t i = 0; i < args.size(); ++i)
      args_equal = mk_and(args_equal, mk_eq(args[i], prev.args[i]));

    assert_ast(mk_implies(args_equal, mk_eq(result, prev.result)));
  }

  history.push_back({args, result, ctx_level});
  return result;
}

smt_astt smt_solver_baset::mk_store(smt_astt a, smt_astt b, smt_astt c)
{
  (void)a;
  (void)b;
  (void)c;
  abort();
}

smt_astt smt_solver_baset::mk_select(smt_astt a, smt_astt b)
{
  (void)a;
  (void)b;
  abort();
}

smt_astt smt_solver_baset::mk_real2int(smt_astt a)
{
  (void)a;
  abort();
}

smt_astt smt_solver_baset::mk_int2real(smt_astt a)
{
  (void)a;
  abort();
}

smt_astt smt_solver_baset::mk_isint(smt_astt a)
{
  (void)a;
  abort();
}
