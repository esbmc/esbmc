#include <cstdint>
#include <goto-symex/slice.h>

#include <unordered_set>
#include <util/prefix.h>
static bool no_slice(const symbol2t &sym)
{
  return config.no_slice_names.count(sym.thename.as_string()) ||
         config.no_slice_ids.count(sym.get_symbol_name());
}

template <bool Add>
bool symex_slicet::get_symbols(const expr2tc &expr)
{
  bool res = false;
  // Recursively look if any of the operands has a inner symbol
  expr->foreach_operand([this, &res](const expr2tc &e) {
    if (!is_nil_expr(e))
      res |= get_symbols<Add>(e);
    return res;
  });

  if (!is_symbol2t(expr))
    return res;

  const symbol2t &s = to_symbol2t(expr);
  if constexpr (Add)
    res |= depends.insert(s.get_symbol_name()).second;
  else
    res |= no_slice(s) || depends.find(s.get_symbol_name()) != depends.end();
  return res;
}

bool symex_slicet::get_array_symbols(const expr2tc &expr)
{
  bool res = false;
  expr->foreach_operand([this, &res](const expr2tc &e) {
    if (!is_nil_expr(e))
      res |= get_array_symbols(e);
    return res;
  });

  // This should come from assertions, the idea is that
  // ASSERT(array[42] == 2) will generate: array$foo WITH 42 := 2
  // We can then just add anything that changes the index 42 as a dependency
  if (is_index2t(expr))
  {
    const index2t &w = to_index2t(expr);
    if (is_symbol2t(w.source_value))
    {
      const symbol2t &s = to_symbol2t(w.source_value);
      array_depends[s.thename.as_string()].emplace(w.index);
      return true;
    }
  }
  return false;
}

void symex_slicet::run_on_assert(symex_target_equationt::SSA_stept &SSA_step)
{
  get_array_symbols(SSA_step.guard);
  get_array_symbols(SSA_step.cond);
  get_symbols<true>(SSA_step.guard);
  get_symbols<true>(SSA_step.cond);
}

void symex_slicet::run_on_assume(symex_target_equationt::SSA_stept &SSA_step)
{
  if (!slice_assumes)
  {
    get_symbols<true>(SSA_step.guard);
    get_symbols<true>(SSA_step.cond);
    return;
  }

  if (!get_symbols<false>(SSA_step.cond))
  {
    // we don't really need it
    SSA_step.ignore = true;
    ++sliced;
    if (is_symbol2t(SSA_step.cond))
      log_debug(
        "slice",
        "slice ignoring assume symbol {}",
        to_symbol2t(SSA_step.cond).get_symbol_name());
    else
      log_debug("slice", "slice ignoring assume expression");
  }
  else
  {
    // If we need it, add the symbols to dependency
    get_symbols<true>(SSA_step.guard);
    get_symbols<true>(SSA_step.cond);
  }
}

void symex_slicet::run_on_assignment(
  symex_target_equationt::SSA_stept &SSA_step)
{
  assert(is_symbol2t(SSA_step.lhs));
  // TODO: create an option to ignore nondet symbols (test case generation)

  auto it = array_depends.find(to_symbol2t(SSA_step.lhs).thename.as_string());
  // We can't really apply magic renumbering for now. We can, convert expressions into identities
  if (it != array_depends.end() && is_with2t(SSA_step.rhs))
  {
    with2t &w = to_with2t(SSA_step.rhs);
    bool can_slice = is_constant_int2t(w.update_field);
    // TODO: support for symbolic constraints
    if (!can_slice)
      it->second.insert(w.update_field);
    for (const expr2tc &index : it->second)
    {
      if (!can_slice)
        break;

      if (!is_constant_int2t(index))
        can_slice = false;
      else
        can_slice &= to_constant_int2t(index).value !=
                     to_constant_int2t(w.update_field).value;
    }

    if (can_slice)
      SSA_step.cond = equality2tc(SSA_step.lhs, w.source_value);
  }

  if (!get_symbols<false>(SSA_step.lhs))
  {
    // Should we add nondet to the dependency list (mostly for test cases)?
    if (!slice_nondet)
    {
      auto expr = get_nondet_symbol(SSA_step.rhs);
      if (expr && is_symbol2t(expr))
      {
        auto &sym = to_symbol2t(expr);
        if (has_prefix(sym.thename.as_string(), "nondet$"))
          return;
      }
    }

    // we don't really need it
    SSA_step.ignore = true;
    ++sliced;
    log_debug(
      "slice",
      "slice ignoring assignment to symbol {}",
      to_symbol2t(SSA_step.lhs).get_symbol_name());
  }
  else
  {
    get_symbols<true>(SSA_step.guard);
    get_symbols<true>(SSA_step.rhs);

    // Remove this symbol as we won't be seeing any references to it further
    // into the history.
    depends.erase(to_symbol2t(SSA_step.lhs).get_symbol_name());
  }
}

void symex_slicet::run_on_renumber(symex_target_equationt::SSA_stept &SSA_step)
{
  assert(is_symbol2t(SSA_step.lhs));

  if (!get_symbols<false>(SSA_step.lhs))
  {
    // we don't really need it
    SSA_step.ignore = true;
    ++sliced;
    log_debug(
      "slice",
      "slice ignoring renumbering symbol {}",
      to_symbol2t(SSA_step.lhs).get_symbol_name());
  }

  // Don't collect the symbol; this insn has no effect on dependencies.
}

/**
 * Naive slicer: slice every step after the last assertion
 * @param eq symex formula to be sliced
 * @return number of steps that were ignored
 */
bool simple_slice::run(symex_target_equationt::SSA_stepst &steps)
{
  sliced = 0;
  fine_timet algorithm_start = current_time();
  // just find the last assertion
  symex_target_equationt::SSA_stepst::iterator last_assertion = steps.end();

  for (symex_target_equationt::SSA_stepst::iterator it = steps.begin();
       it != steps.end();
       it++)
    if (it->is_assert())
      last_assertion = it;

  // slice away anything after it
  symex_target_equationt::SSA_stepst::iterator s_it = last_assertion;

  if (s_it != steps.end())
    for (s_it++; s_it != steps.end(); s_it++)
    {
      s_it->ignore = true;
      ++sliced;
    }

  fine_timet algorithm_stop = current_time();
  log_status(
    "Slicing time: {}s (removed {} assignments)",
    time2string(algorithm_stop - algorithm_start),
    sliced);

  return true;
}

bool claim_slicer::run(symex_target_equationt::SSA_stepst &steps)
{
  sliced = 0;
  fine_timet algorithm_start = current_time();
  size_t counter = 1;
  for (symex_target_equationt::SSA_stepst::iterator it = steps.begin();
       it != steps.end();
       it++)
  {
    // just find the next assertion
    if (it->is_assert())
    {
      if (
        counter++ ==
        claim_to_keep) // this is the assertion that we should not skip!
      {
        it->ignore = false;
        if (!is_goto_cov)
          // obtain the guard info from the assertions
          claim_msg = from_expr(ns, "", it->source.pc->guard);
        else
          // in goto-coverage mode, the assertions are converted to assert(0）
          // the original guards are stored in comment.
          claim_msg = it->comment;
        claim_loc = it->source.pc->location.as_string();
        continue;
      }

      it->ignore = true;
      ++sliced;
    }
  }

  fine_timet algorithm_stop = current_time();
  if (show_slice_info)
    log_status(
      "Slicing for Claim {} ({}s)",
      claim_msg,
      time2string(algorithm_stop - algorithm_start));
  else
    log_debug(
      "c++",
      "Slicing for Claim {} ({}s)",
      claim_msg,
      time2string(algorithm_stop - algorithm_start));

  return true;
}
// Recursively try to extract the nondet symbol of an expression
expr2tc symex_slicet::get_nondet_symbol(const expr2tc &expr)
{
  switch (expr->expr_id)
  {
  case expr2t::symbol_id:
    return expr;

  case expr2t::with_id:
    return get_nondet_symbol(to_with2t(expr).update_value);

  case expr2t::byte_extract_id:
    return get_nondet_symbol(to_byte_extract2t(expr).source_value);

  case expr2t::typecast_id:
    return get_nondet_symbol(to_typecast2t(expr).from);

  case expr2t::bitcast_id:
    return get_nondet_symbol(to_bitcast2t(expr).from);

  case expr2t::if_id:
  {
    // TODO: I am not sure whether it is possible for both sides to have inputs
    // Might ask the solver for this
    auto side_1 = get_nondet_symbol(to_if2t(expr).true_value);
    auto side_2 = get_nondet_symbol(to_if2t(expr).false_value);
    return side_1 ? side_1 : side_2;
  }
  default:
    return expr2tc();
  }
}

void replace_symbol_on_expression(expr2tc &step, const std::unordered_map<std::string, expr2tc> symbol_map)
{
  if(!step)
    return;
   if(is_symbol2t(step))
   {
     auto it = symbol_map.find(to_symbol2t(step).get_symbol_name());
     if(it != symbol_map.end())
       step = it->second;
   }
   step->Foreach_operand([&symbol_map](expr2tc &e){ replace_symbol_on_expression(e, symbol_map);});
}

void symex_slicet::slice_id_operations(symex_target_equationt::SSA_stepst &eq) {
  std::unordered_map<std::string, expr2tc> symbol_map;

  for (auto &step : eq)
  {
    // Helper definition
    typedef goto_trace_stept::typet ssa_type;
    replace_symbol_on_expression(step.cond, symbol_map);
    replace_symbol_on_expression(step.guard, symbol_map);
    replace_symbol_on_expression(step.rhs, symbol_map);
    replace_symbol_on_expression(step.lhs, symbol_map);
    switch (step.type)
    {
    case ssa_type::ASSIGNMENT:
    {
      assert(is_equality2t(step.cond));
      equality2t &eq = to_equality2t(step.cond);
      assert(is_symbol2t(eq.side_1));
      if(is_symbol2t(eq.side_2))
      {
        symbol_map[to_symbol2t(eq.side_1).get_symbol_name()] = eq.side_2;
        step.ignore = true;
        ++sliced;
      }

      break;
    }
    default:
      break;
    }
  }
}