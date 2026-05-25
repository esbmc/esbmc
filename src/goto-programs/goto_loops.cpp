#include "irep2/irep2_utils.h"
#include <goto-programs/goto_loops.h>
#include <util/expr_util.h>

bool check_var_name(const expr2tc &expr)
{
  if (!is_symbol2t(expr))
    return false;

  symbol2t s = to_symbol2t(expr);
  std::string identifier = s.thename.as_string();

  std::size_t found = identifier.find("__ESBMC_");
  if (found != std::string::npos)
    return false;

  found = identifier.find("return_value___");
  if (found != std::string::npos)
    return false;

  found = identifier.find("pthread_lib");
  if (found != std::string::npos)
    return false;

  // Don't add variables that we created for k-induction
  found = identifier.find("$");
  if (found != std::string::npos)
    return false;

  if (identifier == "__func__")
    return false;

  if (identifier == "__PRETTY_FUNCTION__")
    return false;

  if (identifier == "__LINE__")
    return false;

  return true;
}

void goto_loopst::find_function_loops()
{
  for (goto_programt::instructionst::iterator it =
         goto_function.body.instructions.begin();
       it != goto_function.body.instructions.end();
       it++)
  {
    // We found a loop, let's record its instructions
    if (it->is_backwards_goto())
    {
      assert(it->targets.size() == 1);
      goto_programt::instructionst::iterator &loop_head = *it->targets.begin();
      goto_programt::instructionst::iterator &loop_exit = it;

      // This means something like:
      // A: if(g) goto A;
      // Convert it into: assume(!g);
      if (loop_head->location_number == loop_exit->location_number)
      {
        simplify(loop_head->guard);
        it->make_assumption(not2tc(loop_head->guard));
        continue;
      }
      create_function_loop(loop_head, loop_exit);
    }
  }
}

void goto_loopst::create_function_loop(
  goto_programt::instructionst::iterator loop_head,
  goto_programt::instructionst::iterator loop_exit)
{
  goto_programt::instructionst::iterator it = loop_head;

  function_loops.push_front(loopst());
  function_loopst::iterator it1 = function_loops.begin();

  // Set original iterators
  it1->set_original_loop_head(loop_head);
  it1->set_original_loop_exit(loop_exit);

  // Push the current function name to the list of functions
  std::vector<irep_idt> function_names;
  function_names.push_back(function_name);

  // Copy the loop body
  std::size_t size = 0;
  while (it != loop_exit)
  {
    // This should be done only when we're running k-induction
    // Maybe a flag on the class?
    get_modified_variables(it, it1, function_names);
    ++it;

    // Count the number of instruction
    ++size;
  }

  // Include loop_exit
  it1->set_size(size + 1);
}

void goto_loopst::collect_loop_symbols(
  const expr2tc &expr,
  loopst::loop_varst &out) const
{
  if (is_nil_expr(expr))
    return;

  expr->foreach_operand(
    [this, &out](const expr2tc &e) { collect_loop_symbols(e, out); });

  if (is_symbol2t(expr) && check_var_name(expr))
    out.insert(expr);
}

// Walk an assignment LHS, classifying each leaf symbol as modified or
// merely read. See add_loop_var for the rationale; this is the same
// distinction applied to the function-summary path.
void goto_loopst::collect_lhs_symbols(
  const expr2tc &expr,
  loopst::loop_varst &modified,
  loopst::loop_varst &unmodified) const
{
  if (is_nil_expr(expr))
    return;

  if (is_dereference2t(expr))
  {
    collect_loop_symbols(to_dereference2t(expr).value, unmodified);
    return;
  }
  if (is_index2t(expr))
  {
    const index2t &idx = to_index2t(expr);
    collect_lhs_symbols(idx.source_value, modified, unmodified);
    collect_loop_symbols(idx.index, unmodified);
    return;
  }

  expr->foreach_operand(
    [this, &modified, &unmodified](const expr2tc &e)
    { collect_lhs_symbols(e, modified, unmodified); });

  if (is_symbol2t(expr) && check_var_name(expr))
    modified.insert(expr);
}


bool goto_loopst::compute_function_summary(
  const irep_idt &fname,
  std::vector<irep_idt> &in_progress,
  function_summaryt &out)
{
  auto cached = function_summary_cache.find(fname);
  if (cached != function_summary_cache.end())
  {
    // Union the cached summary into out.
    out.modified.insert(
      cached->second.modified.begin(), cached->second.modified.end());
    out.unmodified.insert(
      cached->second.unmodified.begin(), cached->second.unmodified.end());
    return true;
  }

  auto it = goto_functions.function_map.find(fname);
  if (it == goto_functions.function_map.end())
  {
    log_error("failed to find `{}' in function_map", id2string(fname));
    abort();
  }
  if (!it->second.body_available)
  {
    function_summary_cache[fname] = function_summaryt{};
    return true;
  }

  in_progress.push_back(fname);
  bool complete = true;
  function_summaryt local;

  for (const auto &instr : it->second.body.instructions)
  {
    if (instr.is_assign())
    {
      collect_lhs_symbols(
        to_code_assign2t(instr.code).target, local.modified, local.unmodified);
    }
    else if (instr.is_function_call())
    {
      const code_function_call2t &call = to_code_function_call2t(instr.code);
      if (is_dereference2t(call.function))
        continue;

      collect_loop_symbols(call.ret, local.modified);

      const irep_idt &callee = to_symbol2t(call.function).thename;
      if (
        std::find(in_progress.begin(), in_progress.end(), callee) !=
        in_progress.end())
      {
        // Cycle cut: matches the legacy "do nothing on recursion" behaviour.
        // Mark the result incomplete so we don't cache this summary — the
        // missing recursive contributions are call-site dependent.
        complete = false;
        continue;
      }

      if (!compute_function_summary(callee, in_progress, local))
        complete = false;
    }
    else if (instr.is_goto() || instr.is_assert() || instr.is_assume())
    {
      collect_loop_symbols(instr.guard, local.unmodified);
    }
  }

  in_progress.pop_back();

  // Fold local into out regardless of completeness.
  out.modified.insert(local.modified.begin(), local.modified.end());
  out.unmodified.insert(local.unmodified.begin(), local.unmodified.end());

  if (complete)
    function_summary_cache[fname] = std::move(local);
  return complete;
}

void goto_loopst::get_modified_variables(
  goto_programt::instructionst::iterator instruction,
  function_loopst::iterator loop,
  std::vector<irep_idt> &function_names)
{
  if (instruction->is_assign())
  {
    const code_assign2t &assign = to_code_assign2t(instruction->code);
    add_loop_var(*loop, assign.target, true);
  }
  else if (instruction->is_function_call())
  {
    code_function_call2t &function_call =
      to_code_function_call2t(instruction->code);

    // Don't do function pointers
    if (is_dereference2t(function_call.function))
      return;

    // First, add its return
    add_loop_var(*loop, function_call.ret, true);

    const irep_idt &identifier = to_symbol2t(function_call.function).thename;

    // This means recursion, do nothing — matches legacy behaviour.
    if (
      std::find(function_names.begin(), function_names.end(), identifier) !=
      function_names.end())
      return;

    // Compute (or fetch the cached) summary of the callee. Two loops that
    // both call the same helper now share the helper's per-symbol set
    // instead of re-walking the helper from scratch.
    function_summaryt summary;
    compute_function_summary(identifier, function_names, summary);
    for (const auto &v : summary.modified)
      loop->add_modified_var_to_loop(v);
    for (const auto &v : summary.unmodified)
      loop->add_unmodified_var_to_loop(v);
  }
  else if (
    instruction->is_goto() || instruction->is_assert() ||
    instruction->is_assume())
  {
    add_loop_var(*loop, instruction->guard, false);
  }
}

void goto_loopst::add_loop_var(
  loopst &loop,
  const expr2tc &expr,
  bool is_modified)
{
  if (is_nil_expr(expr))
    return;

  // When walking an assign LHS, only the storage being written counts as
  // modified; sub-expressions used to *locate* that storage are reads.
  //   `*p = ...`        — pointer `p` is read, the pointee is modified
  //   `arr[i] = ...`    — index `i` is read, `arr` element is modified
  //   `s.f = ...`       — struct `s` storage is modified
  // Without this distinction `*Var_Ptr = ...` adds `Var_Ptr` to the loop's
  // modified set, which then causes the inductive step to havoc the
  // pointer at loop entry — unsound if the loop never reassigns it.
  if (is_modified && is_dereference2t(expr))
  {
    add_loop_var(loop, to_dereference2t(expr).value, false);
    return;
  }
  if (is_modified && is_index2t(expr))
  {
    const index2t &idx = to_index2t(expr);
    add_loop_var(loop, idx.source_value, true);
    add_loop_var(loop, idx.index, false);
    return;
  }

  expr->foreach_operand([this, &loop, &is_modified](const expr2tc &e) {
    add_loop_var(loop, e, is_modified);
  });

  if (is_symbol2t(expr) && check_var_name(expr))
  {
    if (is_modified)
      loop.add_modified_var_to_loop(expr);
    else
      loop.add_unmodified_var_to_loop(expr);
  }
}

void goto_loopst::dump() const
{
  for (auto &function_loop : function_loops)
    function_loop.dump();
}
