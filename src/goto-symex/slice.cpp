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

void symex_slicet::scan_array_uses(const expr2tc &expr)
{
  // Same explicit two-color worklist as collect_dependencies, but it writes
  // #index_reads / #array_disqualified instead of #depends, with its own memo
  // (#scanned_cache). Both outputs are insert-only, so a BLACK skip is a no-op
  // and the shared guard prefix is scanned once overall (kept O(N)).
  struct FrameT
  {
    expr2tc expr;
    bool children_done;
  };

  std::unordered_set<const expr2t *> grey_set;
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
      grey_set.erase(cur.get());
      scanned_cache.insert(cur.get());
      continue;
    }

    if (scanned_cache.count(cur.get()))
      continue;
    if (!grey_set.insert(cur.get()).second)
      continue;

    // arr_symbol[constant_index]: record the read index and do NOT treat the
    // array symbol as a wholesale use. We descend only into the index subtree
    // (it may itself contain reads); the source symbol is intentionally left
    // unvisited so it is neither disqualified nor marked BLACK here.
    if (is_index2t(cur))
    {
      const index2t &index = to_index2t(cur);
      if (
        is_symbol2t(index.source_value) && is_constant_int2t(index.index) &&
        !to_constant_int2t(index.index).value.is_negative())
      {
        index_reads[to_symbol2t(index.source_value).get_symbol_name()].insert(
          to_constant_int2t(index.index).value.to_uint64());

        worklist.push_back(FrameT{cur, true});
        worklist.push_back(FrameT{index.index, false});
        continue;
      }
    }

    // Any array-typed symbol reached other than as the source of a constant
    // index read defeats per-index reasoning for that version.
    if (is_symbol2t(cur) && is_array_type(cur->type))
      array_disqualified.insert(to_symbol2t(cur).get_symbol_name());

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
  scan_array_uses(SSA_step.guard);
  scan_array_uses(SSA_step.cond);
  collect_dependencies(SSA_step.guard);
  collect_dependencies(SSA_step.cond);
}

void symex_slicet::run_on_assume(symex_target_equationt::SSA_stept &SSA_step)
{
  if (!slice_assumes)
  {
    scan_array_uses(SSA_step.guard);
    scan_array_uses(SSA_step.cond);
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
    scan_array_uses(SSA_step.guard);
    scan_array_uses(SSA_step.cond);
    collect_dependencies(SSA_step.guard);
    collect_dependencies(SSA_step.cond);
  }
}

void symex_slicet::run_on_assignment(
  symex_target_equationt::SSA_stept &SSA_step)
{
  assert(is_symbol2t(SSA_step.lhs));
  // TODO: create an option to ignore nondet symbols (test case generation)

  // An array write `arr_n = arr_m with [const_i := v]` whose lhs is tracked is
  // still kept as a step, but the *store* is dropped (rewritten to the identity
  // `arr_n == arr_m`) when index const_i is provably never read — see the kept
  // branch below. Eligibility uses the monotone #index_reads /
  // #array_disqualified maps populated by #scan_array_uses; dropping a store to
  // an unread index is sound (a dead store) and keeps #depends monotone.
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
    scan_array_uses(SSA_step.guard);

    const symbol2t &lhs = to_symbol2t(SSA_step.lhs);

    // Dead-store elimination for `lhs = src with [const_i := v]`: if index
    // const_i of this array version is provably never read (not in the
    // per-version read-set and the version is not disqualified), the store is
    // dead. Rewrite the encoded condition to the identity `lhs == src`,
    // dropping the store AND its update value v from dependency collection. If
    // lhs[const_i] is never observed then the reads inside v are dead too, so we
    // deliberately do NOT scan v in that case. Keep SSA_step.rhs unchanged:
    // trace construction may use it to recover the source-level assigned value.
    //
    // The read-set propagation mirrors the original per-index slicer: src
    // inherits lhs's read-set; when a store is KEPT (index const_i is read) the
    // written index is omitted while propagating, since an earlier store to
    // that same index produces a value that this store overwrites. Existing
    // direct reads of src are preserved by inserting into src's current set.
    if (
      is_with2t(SSA_step.rhs) &&
      is_symbol2t(to_with2t(SSA_step.rhs).source_value) &&
      is_constant_int2t(to_with2t(SSA_step.rhs).update_field) &&
      !to_constant_int2t(to_with2t(SSA_step.rhs).update_field)
         .value.is_negative())
    {
      const with2t &with = to_with2t(SSA_step.rhs);
      const expr2tc src = with.source_value;
      const expr2tc update_value = with.update_value;
      const std::string lhs_name = lhs.get_symbol_name();
      const std::string src_name = to_symbol2t(src).get_symbol_name();
      const size_t idx = to_constant_int2t(with.update_field).value.to_uint64();

      const bool disq = array_disqualified.count(lhs_name) != 0;
      if (!disq && !index_reads[lhs_name].count(idx))
      {
        // Dead store: encode identity, but leave rhs as the original WITH for
        // counterexample reconstruction.
        log_debug(
          "slice",
          "slice dropping dead store to array {} at index {}",
          src_name,
          idx);
        SSA_step.cond = equality2tc(SSA_step.lhs, src);

        // src inherits lhs's reads (monotone — over-approximate, never erase).
        index_reads[src_name].insert(
          index_reads[lhs_name].begin(), index_reads[lhs_name].end());

        collect_dependencies(SSA_step.guard);
        collect_dependencies(src);
        return;
      }

      // Store kept: v is live, scan it for reads. src inherits lhs's reads
      // except for the overwritten index. Do not assign the set wholesale:
      // src may also have direct downstream reads already recorded.
      scan_array_uses(update_value);
      auto &src_reads = index_reads[src_name];
      for (const auto read_index : index_reads[lhs_name])
        if (read_index != idx)
          src_reads.insert(read_index);
      if (disq)
        array_disqualified.insert(src_name);
    }
    else
      scan_array_uses(SSA_step.rhs);

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
          claim_msg = id2string(it->comment);
        claim_loc = it->source.pc->location.as_string();
        claim_property = it->source.pc->location.property().as_string();
        claim_cstr = id2string(it->comment) + " at " + claim_loc;
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
