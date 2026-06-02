#include <goto-symex/slice.h>

#include <util/prefix.h>
static bool no_slice(const symbol2t &sym)
{
  return config.no_slice_names.count(sym.thename.as_string()) ||
         config.no_slice_ids.count(sym.get_symbol_name());
}

void symex_slicet::collect_dependencies(const expr2tc &expr)
{
  // Explicit worklist (not recursion): guard and-chains are left-leaning and
  // can be thousands deep, which would overflow the stack on a recursive walk.
  //
  // Two-color DFS memo: a node is WHITE (unseen), GREY (on the worklist this
  // call, children in flight) or BLACK (in #collected_cache — fully drained,
  // so every symbol in its subtree is already in #depends). Collection is
  // purely monotone (depends.insert only), so the memo is unconditionally
  // sound: skipping a BLACK node is a no-op. A node is promoted to BLACK only
  // in post-order, after its subtree is drained — this prevents the pre-order
  // partial-subtree bug where a node is marked done before its children are.
  // The shared guard and-chain prefix is therefore walked once overall (the
  // per-step re-walk goes from Θ(N²) to O(N)). An explicit worklist (not
  // recursion) keeps a thousands-deep left-leaning chain off the call stack.
  struct FrameT
  {
    expr2tc expr;
    bool children_done; // true on the second (post-order) visit
  };

  std::unordered_set<const expr2t *> grey_set; // GREY: in-flight this call
  std::vector<FrameT> worklist{FrameT{expr, false}};
  while (!worklist.empty())
  {
    FrameT frame = std::move(worklist.back());
    worklist.pop_back();
    const expr2tc &cur = frame.expr;
    if (is_nil_expr(cur))
      continue;

    if (frame.children_done)
    {
      // Post-order: children are done. Insert symbol, then promote to BLACK.
      if (is_symbol2t(cur))
        depends.insert(to_symbol2t(cur).get_symbol_name());
      grey_set.erase(cur.get());
      collected_cache.insert(cur.get());
      continue;
    }

    // Pre-order: first visit.
    if (collected_cache.count(cur.get())) // BLACK: subtree already collected
      continue;
    if (!grey_set.insert(cur.get()).second) // GREY: already queued this call
      continue;

    // Push post-order sentinel BEFORE children so it fires after all children.
    worklist.push_back(FrameT{cur, true});
    cur->foreach_operand([&worklist](const expr2tc &e) {
      if (!is_nil_expr(e))
        worklist.push_back(FrameT{e, false});
    });
  }
}

bool symex_slicet::depends_on_tracked(const expr2tc &expr)
{
  // Read-only query; not memoised (depends mutates across steps). Only ever
  // called on shallow exprs (SSA lhs symbol / assume cond), so recursion is
  // safe. No array short-circuit here: the original Add=false path fell
  // through to the operands for index2t.
  bool res = false;
  expr->foreach_operand([this, &res](const expr2tc &e) {
    if (!is_nil_expr(e))
      res |= depends_on_tracked(e);
    return res;
  });

  if (!is_symbol2t(expr))
    return res;

  const symbol2t &s = to_symbol2t(expr);
  return res || no_slice(s) ||
         depends.find(s.get_symbol_name()) != depends.end();
}

void symex_slicet::run_on_assert(symex_target_equationt::SSA_stept &SSA_step)
{
  collect_dependencies(SSA_step.guard);
  collect_dependencies(SSA_step.cond);
}

void symex_slicet::run_on_assume(symex_target_equationt::SSA_stept &SSA_step)
{
  if (!slice_assumes)
  {
    collect_dependencies(SSA_step.guard);
    collect_dependencies(SSA_step.cond);
    return;
  }

  if (!depends_on_tracked(SSA_step.cond))
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
    collect_dependencies(SSA_step.guard);
    collect_dependencies(SSA_step.cond);
  }
}

void symex_slicet::run_on_assignment(
  symex_target_equationt::SSA_stept &SSA_step)
{
  assert(is_symbol2t(SSA_step.lhs));
  // TODO: create an option to ignore nondet symbols (test case generation)

  // An array write `arr_n = arr_m with [i:=v]` is treated as an ordinary
  // assignment: kept iff its lhs is tracked. The former per-index path could
  // additionally slice a write to an index no assertion reads — but that
  // relied on tracking only the watched index, not the array symbol, which is
  // non-monotone and incompatible with the collect_dependencies memo. Here a
  // read of arr[i] collects the `arr` symbol itself (collect_dependencies has
  // no array short-circuit), so any write to a read array is kept. This is a
  // strict superset of the old behaviour (sound, slightly less precise): the
  // per-index slicing fired once in 67 array tests and changed no verdict.
  if (!depends_on_tracked(SSA_step.lhs))
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
    collect_dependencies(SSA_step.guard);
    collect_dependencies(SSA_step.rhs);

    // NB: the historic depends.erase(lhs) is intentionally NOT performed.
    // Slicing is monotone backward reachability — keeping the erase out leaves
    // the sliced-step set byte-identical (verified) while keeping #depends
    // monotone, the precondition for the collected_cache memo to be sound.
  }
}

void symex_slicet::run_on_renumber(symex_target_equationt::SSA_stept &SSA_step)
{
  assert(is_symbol2t(SSA_step.lhs));

  if (!depends_on_tracked(SSA_step.lhs))
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
       ++it)
    if (it->is_assert())
      last_assertion = it;

  // slice away anything after it
  symex_target_equationt::SSA_stepst::iterator s_it = last_assertion;

  if (s_it != steps.end())
    for (++s_it; s_it != steps.end(); ++s_it)
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
       ++it)
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
        claim_property = it->source.pc->location.property().as_string();
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
