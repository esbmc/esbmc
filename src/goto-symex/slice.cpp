#include <goto-symex/slice.h>

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

  /* Array slicer, extract dependencies of the form arr_symbol[constant_index]
     Needs to come before the symbols as it short circuits.
     This should be safe as if some symbolic index is eventually added, it will go to the next case. So the full symbol will go to the dependency tree
   */
  if (is_index2t(expr))
  {
    const index2t &index = to_index2t(expr);
    if (is_symbol2t(index.source_value) && is_constant_number(index.index) && !to_constant_int2t(index.index).value.is_negative() )
    {
      const symbol2t &s = to_symbol2t(index.source_value);
      const constant_int2t &i = to_constant_int2t(index.index);
      if constexpr (Add)
        return indexes[s.get_symbol_name()].insert(i.as_ulong()).second;
    }
  }

  // Recursively look if any of the operands has a inner symbol
  expr->foreach_operand(
    [this, &res](const expr2tc &e)
    {
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

void symex_slicet::run_on_assert(symex_target_equationt::SSA_stept &SSA_step)
{
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

    const symbol2t &lhs = to_symbol2t(SSA_step.lhs);
    auto it = indexes.find(lhs.get_symbol_name());
    // WITH(symbol[constant_index] := constant_value)
    if (
      is_with2t(SSA_step.rhs) &&
      is_symbol2t(to_with2t(SSA_step.rhs).source_value) &&
	is_constant_int2t(to_with2t(SSA_step.rhs).update_field) &&
	!to_constant_int2t(to_with2t(SSA_step.rhs).update_field).value.is_negative())
    {
      const with2t &with = to_with2t(SSA_step.rhs);
      const symbol2t &rhs_symbol = to_symbol2t(with.source_value);
      const constant_int2t &rhs_index = to_constant_int2t(with.update_field);

      // Is lhs in the watch list?
      if (it != indexes.end())
      {
        // Found an array in the dependency list! Its guard should be added to the dependency list
        get_symbols<true>(SSA_step.guard);

        // Is this updating a watched index?
        if (it->second.count(rhs_index.as_ulong()) > 0)
        {
          // Add next array as a dependency and remove one index.
          indexes[rhs_symbol.get_symbol_name()] = it->second;
          indexes[rhs_symbol.get_symbol_name()].erase(rhs_index.as_ulong());

          // Finally, the update_value becomes a dependency as well
          get_symbols<true>(with.update_value);
        }
        else
        {
          log_debug(
            "slice",
            "slice ignoring update to array {} at index {}",
            rhs_symbol.get_symbol_name(),
            rhs_index.as_ulong());
          // Don't need the update, transform into ID and propagate dependences
          SSA_step.cond = equality2tc(SSA_step.lhs, with.source_value);
          indexes[rhs_symbol.get_symbol_name()] = it->second;
        }
        return;
      }
    }

    // We might be trying to initialize an array in a weird way
    if (it != indexes.end())
    {
      get_symbols<true>(SSA_step.guard);
      get_symbols<true>(SSA_step.rhs);
      return;
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
        claim_cstr = it->comment + " at " + claim_loc;
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
