#include <goto-programs/goto_k_induction.h>
#include <algorithm>
#include <array>
#include <functional>
#include <goto-programs/goto_loops.h>
#include <goto-programs/loopst.h>
#include <goto-programs/remove_no_op.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_guard.h>
#include <pointer-analysis/andersen.h>
#include <util/config/config.h>
#include <util/lang/c_types.h>
#include <util/expr/expr_util.h>
#include <util/base/i2string.h>
#include <util/base/prefix.h>
#include <util/irep/std_expr.h>
#include <iterator>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace
{
using instructiont = goto_programt::instructiont;
// Keyed on the branch itself, so a branch visited twice contributes one
// conjunct to the emitted ASSUME. location_number would not separate two
// branches a previous loop's transformation displaced: both carry 0.
using guardst = std::unordered_map<const instructiont *, guard2tc>;

/// Cached result of expanding a forward GOTO branch during the entry-
/// condition collection. The boolean is the recursion's return value at
/// this branch (`false_branch && true_branch`, i.e. true iff both
/// subbranches reach the loop end); the guardst is the set of guards
/// that should be merged into the caller's local guardst when this
/// cache entry fires. Storing only the boolean (the legacy design)
/// silently dropped these guards on every cache hit, weakening the
/// entry-condition assume.
struct branch_cache_entryt
{
  bool reaches;
  guardst guards_to_merge;
};
using marked_branchst =
  std::unordered_map<const instructiont *, branch_cache_entryt>;

/// Walk the loop body and collect, into @p guards, the conditions under
/// which control flows from @p loop_head to @p loop_exit. Used by
/// transform_loop to derive the entry condition of the loop (the
/// conjunction of every IF-branch guard taken along a body-reaching
/// path). @p cache is a loop-scoped memoisation table keyed on the
/// IF itself; transform_loop builds a fresh one per loop. Keying on
/// location_number instead collides on every instruction a previous
/// loop's transformation spliced in: those carry 0 until
/// goto_functionst::update() runs.
bool get_entry_cond_rec(
  const goto_programt::targett &loop_head,
  const goto_programt::targett &loop_exit,
  guardst &guards,
  marked_branchst &cache)
{
  // Let's walk the loop and collect the constraints to enter the
  // loop. This might be messy because of side-effects

  // entry and exit numbers
  auto const &entry_number = loop_head->location_number;
  auto const &exit_number = loop_exit->location_number;

  // We jumped outside the loop, don't collect this constraint
  if (entry_number > exit_number)
    return true;

  goto_programt::targett tmp_head = loop_head;
  for (; tmp_head != loop_exit; tmp_head++)
  {
    auto it = cache.find(&*tmp_head);
    if (it != cache.end())
    {
      // Re-inject the guards the first visit collected here. Storing
      // only `reaches` lost these guards on every cache hit and silently
      // weakened the entry-condition assume.
      guards.insert(
        it->second.guards_to_merge.begin(), it->second.guards_to_merge.end());
      return it->second.reaches;
    }

    /* TODO: disable this for now, it will be used for termination evaluation
     * in the future.

    // Return, assume(0) and assert(0) stop the execution, so ignore these
    // branches too
    if(tmp_head->is_return())
      return true;

    if(tmp_head->is_assume() || tmp_head->is_assert())
      if(is_false(tmp_head->guard))
        return true;
    */

    if (tmp_head->is_goto() && !tmp_head->is_backwards_goto())
    {
      expr2tc g = tmp_head->guard;
      simplify(g);

      // If the guard is false, we can skip it right away
      if (is_false(g))
        continue;

      // We need to walk the branches and collect constraints that force
      // the path inside the loop and reach the end of the loop body

      auto const branch = &*tmp_head;

      // Walk the true branch
      bool true_branch = true;
      guardst true_branch_guard;
      if (!is_false(g))
      {
        true_branch_guard[branch].add(g);
        true_branch = get_entry_cond_rec(
          tmp_head->targets.front(), loop_exit, true_branch_guard, cache);
      }

      // Walk the false branch
      bool false_branch = true;
      guardst false_branch_guard;
      if (!is_true(g))
      {
        goto_programt::targett new_tmp_head = tmp_head;
        make_not(g);
        false_branch_guard[branch].add(g);
        false_branch = get_entry_cond_rec(
          ++new_tmp_head, loop_exit, false_branch_guard, cache);
      }

      // Cache: store BOTH the recursion's reach-status at this branch
      // AND the guards that should be re-injected on a later cache hit.
      branch_cache_entryt entry;
      entry.reaches = false_branch && true_branch;

      // If both sides reach the end of the loop or if neither reaches it
      // we can ignore them
      if (!(false_branch ^ true_branch))
      {
        cache[branch] = std::move(entry);
        return false_branch && true_branch;
      }

      // At least only one of the branches reach the end of the loop, so
      // collect the guards from the non-reaching side.
      if (!true_branch)
      {
        guards.insert(true_branch_guard.begin(), true_branch_guard.end());
        entry.guards_to_merge = std::move(true_branch_guard);
        cache[branch] = std::move(entry);
        return false;
      }

      if (!false_branch)
      {
        guards.insert(false_branch_guard.begin(), false_branch_guard.end());
        entry.guards_to_merge = std::move(false_branch_guard);
        cache[branch] = std::move(entry);
        return false;
      }
    }
  }

  return false;
}

/// loop_varst is an unordered_set hashed by irep2_hash, which folds in
/// irep_idt::hash() -- the string's interning sequence number, not its text.
/// Iteration order therefore depends on what else has been interned earlier in
/// the run, so an unrelated pass that interns a string permutes these havocs
/// (docs/roadmap/scope-clang-c-irep2.md §13). Order by printed form, which is
/// stable across runs; the assignments are mutually independent, so only the
/// order changes.
std::vector<expr2tc> ordered(const loopst::loop_varst &vars)
{
  std::vector<expr2tc> ordered(vars.begin(), vars.end());
  std::sort(
    ordered.begin(), ordered.end(), [](const expr2tc &a, const expr2tc &b) {
      return a->pretty() < b->pretty();
    });
  return ordered;
}

using pinst = std::vector<std::pair<expr2tc, expr2tc>>;

/// What the inductive step havocs at a loop head and on each other edge into
/// the loop.
struct havocst
{
  /// Pointers whose objects are havoced whole (see derivationst).
  std::vector<expr2tc> through;
  std::vector<expr2tc> vars;
  /// Assignments that follow the havocs and give back to a havoced pointer
  /// the objects it may point to (see pin_havoced_pointers).
  pinst pins;
};

/// The intrinsic symex runs to havoc whole every object a pointer may reach.
expr2tc havoc_object_function()
{
  return symbol2tc(
    code_type2tc(
      std::vector<type2tc>{pointer_type2tc(get_empty_type())},
      get_empty_type(),
      std::vector<irep_idt>{"ptr"},
      false),
    "c:@F@__ESBMC_havoc_object");
}

void add_havoc_assigns(
  goto_programt &dest,
  const havocst &havocs,
  const locationt &location)
{
  const type2tc void_ptr = pointer_type2tc(get_empty_type());
  const expr2tc havoc_object = havoc_object_function();
  // Ahead of the assignments, which interval analysis reads back from the
  // loop head.
  for (const expr2tc &ptr : havocs.through)
  {
    goto_programt::targett t = dest.add_instruction(FUNCTION_CALL);
    t->inductive_step_instruction = true;
    t->code = code_function_call2tc(
      expr2tc(),
      havoc_object,
      std::vector<expr2tc>{typecast2tc(void_ptr, ptr)});
    t->location = location;
  }

  const auto add = [&](const expr2tc &lhs, const expr2tc &rhs) {
    goto_programt::targett t = dest.add_instruction(ASSIGN);
    t->inductive_step_instruction = true;
    t->code = code_assign2tc(lhs, rhs);
    t->location = location;
  };
  for (auto const &lhs : havocs.vars)
    add(lhs, gen_nondet(lhs->type));
  for (auto const &[lhs, rhs] : havocs.pins)
    add(lhs, rhs);
}

void make_nondet_assign(
  goto_functiont &goto_function,
  goto_programt::targett &loop_head,
  const havocst &havocs)
{
  goto_programt dest;
  add_havoc_assigns(dest, havocs, loop_head->location);
  const size_t inserted = dest.instructions.size();
  goto_function.body.insert_swap(loop_head, dest);

  // insert_swap leaves loop_head on the first inserted instruction, so put it
  // back on the original head: exactly `inserted` forward, and never the old
  // "walk while inductive_step_instruction" heuristic, which also swallowed an
  // ASSUME a previous pass had left after the head and so retargeted the back
  // edge past the loop's exit IF.
  std::advance(loop_head, inserted);
}

bool contains_rec(const expr2tc &expr, const loopst::loop_varst &vars)
{
  // Check this node first: if it's a tracked symbol or pointee, we're done.
  if (vars.find(expr) != vars.end())
    return true;

  // Otherwise recurse into operands and stop at the first match.
  bool res = false;
  expr->foreach_operand([&vars, &res](const expr2tc &e) {
    if (res || is_nil_expr(e))
      return;
    res = contains_rec(e, vars);
  });
  return res;
}

void remove_unrelated_loop_cond(guardst &guards, const loopst &loop)
{
  auto const &loop_vars = loop.get_modified_loop_vars();
  if (!loop_vars.size())
  {
    guards.clear();
    return;
  }

  guardst::iterator g = guards.begin();
  while (g != guards.end())
  {
    expr2tc g_expr = g->second.as_expr();

    if (!contains_rec(g_expr, loop_vars))
      g = guards.erase(g);
    else
      ++g;
  }
}

void assume_loop_entry_cond_before_loop(
  goto_functiont &goto_function,
  goto_programt::targett &loop_head,
  const guardst &guards)
{
  // Combine all per-branch loop-entry guards into one ASSUME and
  // insert it via raw `insert(loop_head, ...)` immediately before the
  // loop_head IF. This runs *after* make_nondet_assign in
  // transform_loop, so the layout becomes
  //
  //   NONDET havocs               [inductive_step]   <- make_nondet_assign
  //   ASSUME(entry_cond)          [inductive_step]   <- this insert
  //   loop_head: IF !cond GOTO exit                  <- original IF
  //   body
  //   GOTO loop_head  -> back-edge targets the IF
  //   exit:
  //
  // Two properties matter for soundness/precision:
  //   1. The ASSUME comes *after* the NONDET havocs, so it constrains
  //      the just-havoced state (the actual inductive-hypothesis pin),
  //      not the pre-loop concrete values.
  //   2. The ASSUME sits *before* loop_head and is reached only via
  //      fall-through from the havocs. The back-edge
  //      (adjust_loop_head_and_exit retargets it at loop_head, the
  //      IF) skips both the havocs and the ASSUME on every iteration
  //      after the first, so the natural exit on `cond` is never
  //      blocked by a re-firing entry-cond assume.
  //
  // The legacy `insert_swap(tmp_head, ASSUME)` placement (one ASSUME
  // per branch, swapped *at* the branch instruction) was unsound
  // for the loop_head IF because insert_swap pins external jumps to
  // the iterator: back-edges then landed on the ASSUME and re-fired
  // it on every iteration, killing the natural-exit path in IS and
  // producing vacuous UNSAT proofs (e.g. SV-COMP
  // sll_of_sll_nondet_append-2). The earlier fix that combined the
  // ASSUME and placed it *after* loop_head fixed SLL but over-
  // constrained loops where the body modifies a loop-exit variable
  // (e.g. `i++` in array_3-1): the back-edge re-evaluated the
  // ASSUME on the iteration where the body's increment had pushed
  // the variable past the exit condition, again killing a natural
  // path.
  //
  // Iterate the collected guards directly instead of walking the
  // instruction range and looking each one up: after
  // make_nondet_assign's insert_swap the walk-and-lookup approach
  // missed every guard, so the combined entry condition came out empty
  // and no ASSUME was inserted, regressing post-loop assertions in
  // loop-invariants, incremental-smt, witnesses_validate and
  // esbmc-solidity (issue #4846).
  guard2tc combined;
  for (auto const &kv : guards)
  {
    expr2tc loop_cond = kv.second.as_expr();

    if (is_nil_expr(loop_cond) || is_true(loop_cond))
      continue;

    // A guard that simplifies to false would make the assume kill the
    // path even before the loop. Preserve the legacy "bail out" choice:
    // any unresolved false among the branch guards skips the whole
    // entry-cond instrumentation.
    if (is_false(loop_cond))
      return;

    combined.add(loop_cond);
  }

  expr2tc combined_expr = combined.as_expr();
  if (
    is_nil_expr(combined_expr) || is_true(combined_expr) ||
    is_false(combined_expr))
    return;

  goto_programt::instructiont instruction;
  instruction.type = ASSUME;
  instruction.guard = combined_expr;
  instruction.inductive_step_instruction = true;
  instruction.location = loop_head->location;
  goto_function.body.instructions.insert(loop_head, instruction);
}

void adjust_loop_head_and_exit(
  goto_programt::targett &loop_head,
  goto_programt::targett &loop_exit)
{
  loop_exit->targets.clear();
  loop_exit->targets.push_front(loop_head);

  goto_programt::targett _loop_exit = loop_exit;
  ++_loop_exit;

  // Zero means that the instruction was added during
  // the k-induction transformation
  if (_loop_exit->location_number == 0)
  {
    // Clear the target
    loop_head->targets.clear();

    // And set the target to be the newly inserted assume(cond)
    loop_head->targets.push_front(_loop_exit);
  }
}

/// Whether control reaches the loop's back edge from @p from without leaving
/// @p loop_range. The program-order range between the head and the back edge
/// over-approximates the loop body: a block that only jumps out of the loop
/// can sit inside the range, and havocing an edge into that block would
/// clobber the loop's variables on a path that never runs the loop.
/// @p continue_past_failed_assertions follows a path through a false ASSERT,
/// which get_successors ends; set it when claims past a violation are checked.
bool reaches_back_edge(
  const goto_programt &body,
  goto_programt::const_targett from,
  const std::unordered_set<const instructiont *> &loop_range,
  const instructiont *back_edge,
  bool continue_past_failed_assertions)
{
  std::unordered_set<const instructiont *> seen;
  std::vector<goto_programt::const_targett> work{from};
  while (!work.empty())
  {
    const goto_programt::const_targett it = work.back();
    work.pop_back();
    if (&*it == back_edge)
      return true;
    if (!loop_range.count(&*it) || !seen.insert(&*it).second)
      continue;

    goto_programt::const_targetst successors;
    body.get_successors(it, successors);
    if (
      continue_past_failed_assertions && it->is_assert() && successors.empty())
      successors.push_back(std::next(it));
    work.insert(work.end(), successors.begin(), successors.end());
  }
  return false;
}

/// The unconditional jumps that enter the loop somewhere other than its head: a
/// rotated (bottom-test) loop is entered at its test rather than at its head,
/// and a switch dispatches through trampolines that jump past the head into the
/// middle of the body. Without a havoc on those edges the inductive step
/// symexes the concrete initial state and truncates at k instead of inducting
/// (#7565). Jumps to the head are already covered: insert_swap pins those to
/// the havoc block it writes there.
///
/// Only unconditional jumps qualify. The havocs go in front of the jump
/// itself, so every execution reaching them goes on to enter the loop; a
/// conditional jump also falls through to a non-entry path, so splitting that
/// edge would need a trampoline block, and the new edges would have to be
/// revalidated against goto_loopst, adjust_loop_head_and_exit's
/// location_number == 0 heuristic and remove_unreachable. A loop whose only
/// entry is conditional therefore stays un-havoced and can still be falsely
/// proved -- one #7565 residual, pinned by
/// regression/k-induction/github_7565_conditional_entry_fail.
std::vector<goto_programt::targett> collect_entry_jumps(
  goto_programt &body,
  const loopst &loop,
  bool continue_past_failed_assertions)
{
  std::unordered_set<const instructiont *> loop_range;
  for (goto_programt::targett it = loop.get_original_loop_head();
       it != body.instructions.end();
       ++it)
  {
    loop_range.insert(&*it);
    if (it == loop.get_original_loop_exit())
      break;
  }
  const instructiont *back_edge = &*loop.get_original_loop_exit();

  std::vector<goto_programt::targett> jumps;
  for (auto it = body.instructions.begin(); it != body.instructions.end(); ++it)
  {
    if (loop_range.count(&*it) || !it->is_goto() || !is_true(it->guard))
      continue;
    for (const auto &target : it->targets)
      if (
        target != loop.get_original_loop_head() && loop_range.count(&*target) &&
        reaches_back_edge(
          body, target, loop_range, back_edge, continue_past_failed_assertions))
      {
        jumps.push_back(it);
        break;
      }
  }
  return jumps;
}

/// Havoc the loop's modified variables on each edge in @p jumps, by splicing
/// the assignments in front of the jump. insert_swap keeps them on every
/// incoming edge, and because nothing but assignments is added the control
/// flow graph is unchanged -- no block to place, no edge for loop detection
/// to mistake for a back edge, and nothing for remove_unreachable to strip.
/// Every other edge into the jump crosses the havocs too, which on an
/// irreducible CFG costs precision but never soundness: a nondet assignment
/// subsumes the concrete one it replaces.
void havoc_entry_jumps(
  goto_programt &body,
  const havocst &havocs,
  const std::vector<goto_programt::targett> &jumps)
{
  for (const goto_programt::targett &jump : jumps)
  {
    // Only the loop head is insert_swapped between collection and here, and
    // collect_entry_jumps excluded it, so the iterator still holds the GOTO.
    assert(jump->is_goto() && is_true(jump->guard));
    goto_programt dest;
    add_havoc_assigns(dest, havocs, jump->location);
    body.insert_swap(jump, dest);
  }
}

/// Per-loop k-induction transformation: havoc each loop's modified
/// variables and inject an ASSUME of the loop entry condition right
/// before the loop head.
void transform_loop(
  goto_functiont &goto_function,
  loopst &loop,
  const havocst &havocs,
  bool continue_past_failed_assertions)
{
  goto_programt::targett loop_head = loop.get_original_loop_head();
  goto_programt::targett loop_exit = loop.get_original_loop_exit();

  // Collected here, applied last: splicing a havoc block displaces the GOTO
  // onto a fresh instruction whose location_number is 0, which is what
  // adjust_loop_head_and_exit keys its loop-exit test on.
  const std::vector<goto_programt::targett> entry_jumps = collect_entry_jumps(
    goto_function.body, loop, continue_past_failed_assertions);

  // Loop-scoped cache for get_entry_cond_rec. Nested loops in the same
  // function don't reuse entries because we construct a fresh cache per
  // call.
  marked_branchst cache;
  guardst guards;
  get_entry_cond_rec(loop_head, loop_exit, guards, cache);

  // Remove loop conditions not related to the written variables
  remove_unrelated_loop_cond(guards, loop);

  // Order matters: the entry-cond ASSUME must constrain the state
  // *after* the loop vars have been havoced, so we emit the NONDET
  // havocs first and then place the ASSUME just before loop_head.
  // make_nondet_assign also rewinds loop_head back onto the original
  // loop head (post its advance-by-`inserted` step), so by the time
  // assume_loop_entry_cond_before_loop runs the iterator is correct
  // and the ASSUME ends up between the havocs and the IF.

  // Create the nondet assignments on the beginning of the loop
  make_nondet_assign(goto_function, loop_head, havocs);

  // Assume the loop entry condition before going into the loop
  assume_loop_entry_cond_before_loop(goto_function, loop_head, guards);

  // Check if the loop exit needs to be updated. We must point to the
  // assume that was inserted in the previous transformation
  adjust_loop_head_and_exit(loop_head, loop_exit);

  havoc_entry_jumps(goto_function.body, havocs, entry_jumps);
}

/// What a pointer may point to: the named objects, and whether it may also
/// reach the heap or anything at all.
struct targetst
{
  loopst::loop_varst named;
  bool heap = false;
  bool anything = false;
  bool nondet = false;
};

targetst
targets_of(andersent &points_to, const loopst &loop, const expr2tc &ptr)
{
  value_setst::valuest values;
  points_to.get_values(loop.get_original_loop_head(), ptr, values);
  targetst t;
  for (const expr2tc &v : values)
  {
    const expr2tc object =
      is_object_descriptor2t(v) ? to_object_descriptor2t(v).object : expr2tc();
    if (is_nil_expr(object))
      t.anything = true;
    else if (andersent::is_nondet_object(object))
      t.nondet = true;
    else if (is_symbol2t(object) && check_var_name(object))
      t.named.insert(object);
    else if (is_dynamic_object2t(object))
      t.heap = true;
    else
      t.anything = true;
  }
  return t;
}

bool names(const loopst::loop_varst &vars, const irep_idt &name)
{
  return std::any_of(vars.begin(), vars.end(), [&name](const expr2tc &v) {
    return is_symbol2t(v) && to_symbol2t(v).thename == name;
  });
}

void for_each_subexpr(
  const expr2tc &e,
  const std::function<void(const expr2tc &)> &f)
{
  if (is_nil_expr(e))
    return;
  f(e);
  e->foreach_operand([&f](const expr2tc &op) { for_each_subexpr(op, f); });
}

/// Whether \p code allocates.
bool allocates(const expr2tc &code)
{
  bool found = false;
  for_each_subexpr(code, [&found](const expr2tc &e) {
    if (!is_sideeffect2t(e))
      return;
    switch (to_sideeffect2t(e).kind)
    {
    case sideeffect2t::allockind::malloc:
    case sideeffect2t::allockind::realloc:
    case sideeffect2t::allockind::alloca:
    case sideeffect2t::allockind::cpp_new:
    case sideeffect2t::allockind::cpp_new_arr:
      found = true;
      break;
    default:
      break;
    }
  });
  return found;
}

/// Whether a value of \p type can hold a data address. CIL computes field
/// addresses in integers, `(unsigned long)p + off`, so a pointer-wide integer
/// can.
bool carries_address(const type2tc &type)
{
  if (is_pointer_type(type))
    return !is_code_type(to_pointer_type(type).subtype);
  if (is_bv_type(type))
    return type->get_width() >= config.ansi_c.pointer_width();
  if (is_array_type(type))
    return carries_address(to_array_type(type).subtype);
  if (is_struct_type(type) || is_union_type(type))
  {
    const std::vector<type2tc> members = struct_union_members(type);
    return std::any_of(members.begin(), members.end(), carries_address);
  }
  return is_symbol_type(type);
}

/// Where the program's loops and direct calls are, and so which instructions
/// may run inside a loop. Symex disables the step on a call through a
/// function pointer, on recursion and on a thread, so only direct calls run a
/// function.
class loop_shapest
{
public:
  explicit loop_shapest(const goto_functionst &goto_functions);

  /// Whether the instruction at \p location of \p function may run inside
  /// some loop: one of its own function, or one that calls it.
  bool in_loop(const irep_idt &function, unsigned location) const;

  /// Whether it is inside a loop of \p function itself.
  bool in_own_loop(const irep_idt &function, unsigned location) const
  {
    return shapes.at(function).in_loop(location);
  }

private:
  /// Location numbers from a loop head to its back edge.
  struct ranget
  {
    unsigned head;
    unsigned back;

    bool contains(unsigned location) const
    {
      return head <= location && location <= back;
    }
  };

  /// A body's loops and direct calls, by location number.
  struct shapet
  {
    std::vector<ranget> loops;
    std::vector<std::pair<unsigned, irep_idt>> calls;

    bool in_loop(unsigned location) const
    {
      return std::any_of(
        loops.begin(), loops.end(), [location](const ranget &l) {
          return l.contains(location);
        });
    }
  };

  using functionst = std::unordered_set<irep_idt, irep_id_hash>;

  functionst reached_from(std::vector<irep_idt> functions) const;

  std::unordered_map<irep_idt, shapet, irep_id_hash> shapes;
  /// The functions some loop may call.
  functionst called_in_loop;
};

loop_shapest::loop_shapest(const goto_functionst &goto_functions)
{
  forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available)
      continue;
    shapet &shape = shapes[it->first];
    for (const instructiont &instr : it->second.body.instructions)
    {
      const unsigned loc = instr.location_number;
      if (instr.is_goto())
        for (const auto &target : instr.targets)
          if (target->location_number <= loc)
            shape.loops.push_back({target->location_number, loc});
      if (
        instr.is_function_call() &&
        is_symbol2t(to_code_function_call2t(instr.code).function))
      {
        const irep_idt &callee =
          to_symbol2t(to_code_function_call2t(instr.code).function).thename;
        shape.calls.emplace_back(loc, callee);
      }
    }
  }

  std::vector<irep_idt> called;
  for (const auto &[function, shape] : shapes)
    for (const auto &[location, callee] : shape.calls)
      if (shape.in_loop(location))
        called.push_back(callee);
  called_in_loop = reached_from(std::move(called));
}

/// \p functions and what they call in turn, of those with a body.
loop_shapest::functionst
loop_shapest::reached_from(std::vector<irep_idt> functions) const
{
  functionst reached;
  while (!functions.empty())
  {
    const irep_idt f = functions.back();
    functions.pop_back();
    const auto it = shapes.find(f);
    if (it == shapes.end() || !reached.insert(f).second)
      continue;
    for (const auto &call : it->second.calls)
      functions.push_back(call.second);
  }
  return reached;
}

bool loop_shapest::in_loop(const irep_idt &function, unsigned location) const
{
  return called_in_loop.count(function) != 0 ||
         shapes.at(function).in_loop(location);
}

/// What a pointer the loop writes through is computed from: the pointers in
/// scope at the loop head that it is an offset of, and the named objects whose
/// address it takes.
struct rootst
{
  loopst::loop_varst pointers;
  loopst::loop_varst objects;
  /// Whether it is, in part, an object allocated in the iteration.
  bool allocated = false;

  bool empty() const
  {
    return pointers.empty() && objects.empty() && !allocated;
  }

  void merge(const rootst &other)
  {
    pointers.insert(other.pointers.begin(), other.pointers.end());
    objects.insert(other.objects.begin(), other.objects.end());
    allocated |= other.allocated;
  }
};

/// A loop's body, by location number, and the variables the inductive step's
/// havoc of which nothing reads: every path from the head or an unconditional
/// jump into the body assigns them whole before reading them, no loop nested
/// in the body assigns them, and nothing outside the body reads them. None in
/// a loop a conditional jump enters, which gets no havoc. Those hold what the
/// body assigned them, provided a call writes no variable whose address is not
/// taken.
struct iterationt
{
  unsigned head = 0;
  unsigned back = 0;
  std::unordered_set<irep_idt, irep_id_hash> reassigned;

  bool contains(unsigned location) const
  {
    return head <= location && location <= back;
  }
};

/// Traces a value back through the locals, parameters and results of called
/// functions, which every call starts afresh, to what the loop head holds. A
/// global that no loop assigns, or a local of the loop's function that none
/// of its loops assigns, whose address nothing takes is never havoced and
/// keeps one value throughout the loop, so every object a write derived from
/// it reaches is one it points to at the head. A local the loop reassigns
/// before every read is traced through its assignments in the body. An object
/// allocated on the way was allocated in the same iteration, which the step
/// executes, so it needs no havoc.
class derivationst
{
public:
  /// \p unsettled: the variables some loop of their function assigns without
  /// reassigning them before every read (see iterationt).
  derivationst(
    const goto_functionst &goto_functions,
    const loop_shapest &shapes,
    std::unordered_set<irep_idt, irep_id_hash> unsettled);

  /// Whether an integer variable or load may hold no address, by the
  /// points-to sets.
  using addresslesst = std::function<bool(const expr2tc &)>;

  /// Adds to \p out what \p value, written in a loop of \p function, derives
  /// from. Returns false when the trace cannot bound it.
  bool roots(
    const irep_idt &function,
    const std::unordered_set<irep_idt, irep_id_hash> &address_taken,
    const iterationt &iteration,
    const addresslesst &addressless,
    const expr2tc &value,
    rootst &out) const;

  /// Whether some loop may assign \p symbol, and so some inductive step
  /// havoc it.
  bool assigned_in_loop(const irep_idt &symbol) const
  {
    return looped.count(symbol) != 0;
  }

  /// Whether \p symbol is a local or parameter of a function other than
  /// \p function, which every call of that function declares afresh.
  bool local_elsewhere(const irep_idt &symbol, const irep_idt &function) const
  {
    const auto it = owner.find(symbol);
    return it != owner.end() && it->second != function;
  }

private:
  /// An assignment to a symbol: a value, the result of a call to \p callee,
  /// or neither, for a write to part of the symbol or a call through a
  /// function pointer.
  struct deft
  {
    expr2tc value;
    irep_idt callee;
    unsigned location;
  };

  struct tracedt
  {
    rootst roots;
    bool traced;
  };
  using memot = std::unordered_map<irep_idt, tracedt, irep_id_hash>;

  struct queryt
  {
    const irep_idt &function;
    const std::unordered_set<irep_idt, irep_id_hash> &address_taken;
    const iterationt &iteration;
    const addresslesst &addressless;
    rootst &out;
    /// Both by whether the trace left the iteration.
    std::array<std::unordered_set<irep_idt, irep_id_hash>, 2> &visiting;
    std::array<memot, 2> &memo;
    /// Short-circuits on a symbol still being traced, after which what the
    /// symbols on the way computed is partial and not memoised.
    unsigned &cycles;
    /// Steps left before the trace gives up and rejects, which a cycle
    /// through many symbols, defeating the memo, would otherwise make
    /// exponential.
    std::size_t &budget;
    /// Whether the trace has left the iteration for what the loop's function
    /// computed before the loop.
    bool outside;

    queryt sub(rootst &roots, bool leaves = false) const
    {
      return {
        function,
        address_taken,
        iteration,
        addressless,
        roots,
        visiting,
        memo,
        cycles,
        budget,
        outside || leaves};
    }
  };

  void
  define(const expr2tc &lhs, const deft &def, bool in_loop, bool in_own_loop);
  void record(
    const goto_functionst &goto_functions,
    const irep_idt &function,
    const instructiont &instr,
    bool in_loop,
    bool in_own_loop);

  bool trace(queryt &q, const expr2tc &value) const;
  bool trace_address(queryt &q, expr2tc object) const;
  bool trace_symbol(queryt &q, const expr2tc &symbol) const;
  bool trace_def(queryt &q, const deft &def) const;
  /// Whether an address may flow into \p value, of a type that holds none.
  bool may_hold_address(
    const queryt &q,
    const expr2tc &value,
    std::unordered_set<irep_idt, irep_id_hash> &seen) const;
  /// Traces what \p name holds with \p trace, once per query, leaving the
  /// iteration if \p leaves.
  bool trace_once(
    queryt &q,
    const irep_idt &name,
    bool leaves,
    const std::function<bool(queryt &)> &trace) const;

  std::unordered_map<irep_idt, std::vector<deft>, irep_id_hash> defs;
  /// What each function returns.
  std::unordered_map<irep_idt, std::vector<expr2tc>, irep_id_hash> returns;
  /// The function each local and parameter belongs to.
  std::unordered_map<irep_idt, irep_idt, irep_id_hash> owner;
  std::unordered_set<irep_idt, irep_id_hash> looped;
  /// Assigned inside a loop of its own function, which a caller's loop cannot
  /// do to a local: every call declares it afresh.
  std::unordered_set<irep_idt, irep_id_hash> looped_here;
  std::unordered_set<irep_idt, irep_id_hash> unsettled;
};

void derivationst::define(
  const expr2tc &lhs,
  const deft &def,
  bool in_loop,
  bool in_own_loop)
{
  if (is_nil_expr(lhs))
    return;
  if (is_if2t(lhs))
  {
    define(
      to_if2t(lhs).true_value, {{}, {}, def.location}, in_loop, in_own_loop);
    define(
      to_if2t(lhs).false_value, {{}, {}, def.location}, in_loop, in_own_loop);
    return;
  }

  expr2tc symbol = lhs;
  while (is_member2t(symbol) || is_index2t(symbol) || is_typecast2t(symbol))
    symbol = *symbol->get_sub_expr(0);
  if (!is_symbol2t(symbol))
    return;
  const irep_idt &name = to_symbol2t(symbol).thename;
  defs[name].push_back(symbol == lhs ? def : deft{{}, {}, def.location});
  if (in_loop)
    looped.insert(name);
  if (in_own_loop)
    looped_here.insert(name);
}

void derivationst::record(
  const goto_functionst &goto_functions,
  const irep_idt &function,
  const instructiont &instr,
  bool in_loop,
  bool in_own_loop)
{
  if (instr.is_decl())
  {
    const code_decl2t &decl = to_code_decl2t(instr.code);
    owner.emplace(decl.value, function);
    if (!is_nil_expr(decl.init))
      define(
        symbol2tc(decl.type, decl.value),
        {decl.init, {}, instr.location_number},
        in_loop,
        in_own_loop);
  }
  else if (instr.is_assign())
    define(
      to_code_assign2t(instr.code).target,
      {to_code_assign2t(instr.code).source, {}, instr.location_number},
      in_loop,
      in_own_loop);
  else if (instr.is_return())
    returns[function].push_back(to_code_return2t(instr.code).operand);
  else if (instr.is_function_call())
  {
    const code_function_call2t &call = to_code_function_call2t(instr.code);
    const irep_idt name =
      is_symbol2t(call.function) ? to_symbol2t(call.function).thename : "";
    define(
      call.ret, {expr2tc(), name, instr.location_number}, in_loop, in_own_loop);
    const auto callee = goto_functions.function_map.find(name);
    if (
      callee == goto_functions.function_map.end() ||
      !is_code_type(callee->second.type))
      return;
    const auto &names = to_code_type(callee->second.type).argument_names;
    for (size_t i = 0; i < std::min(names.size(), call.operands.size()); ++i)
      defs[names[i]].push_back({call.operands[i], {}, instr.location_number});
  }
}

derivationst::derivationst(
  const goto_functionst &goto_functions,
  const loop_shapest &shapes,
  std::unordered_set<irep_idt, irep_id_hash> unsettled)
  : unsettled(std::move(unsettled))
{
  forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available)
      continue;
    if (is_code_type(it->second.type))
      for (const irep_idt &arg : to_code_type(it->second.type).argument_names)
        owner.emplace(arg, it->first);
    for (const instructiont &instr : it->second.body.instructions)
      record(
        goto_functions,
        it->first,
        instr,
        shapes.in_loop(it->first, instr.location_number),
        shapes.in_own_loop(it->first, instr.location_number));
  }
}

bool derivationst::roots(
  const irep_idt &function,
  const std::unordered_set<irep_idt, irep_id_hash> &address_taken,
  const iterationt &iteration,
  const addresslesst &addressless,
  const expr2tc &value,
  rootst &out) const
{
  std::array<std::unordered_set<irep_idt, irep_id_hash>, 2> visiting;
  std::array<memot, 2> memo;
  unsigned cycles = 0;
  std::size_t budget = 100000;
  queryt q{
    function,
    address_taken,
    iteration,
    addressless,
    out,
    visiting,
    memo,
    cycles,
    budget,
    false};
  return trace(q, value);
}

/// Whether \p e holds no address but those its operands hold.
bool computes_from_operands(const expr2tc &e)
{
  switch (e->expr_id)
  {
  case expr2t::if_id:
  case expr2t::bitand_id:
  case expr2t::bitor_id:
  case expr2t::bitxor_id:
  case expr2t::bitnot_id:
  case expr2t::shl_id:
  case expr2t::ashr_id:
  case expr2t::lshr_id:
    return true;
  default:
    return is_arith_expr(e);
  }
}

bool derivationst::trace(queryt &q, const expr2tc &value) const
{
  if (q.budget == 0)
    return false;
  --q.budget;
  if (is_nil_expr(value))
    return true;
  // An array used as a pointer stands for itself.
  if (is_array_type(value->type))
    return trace_address(q, value);
  if (is_constant_int2t(value))
    return true;
  // An integer that holds no address is an offset, whatever its value.
  if (is_bv_type(value->type) && q.addressless(value))
    return true;
  if (!carries_address(value->type))
  {
    // Symex still finds the object of an address an integer truncates.
    std::unordered_set<irep_idt, irep_id_hash> seen;
    return !may_hold_address(q, value, seen);
  }
  if (is_symbol2t(value))
    return trace_symbol(q, value);
  if (is_typecast2t(value))
    return trace(q, to_typecast2t(value).from);
  if (is_address_of2t(value))
    return trace_address(q, to_address_of2t(value).ptr_obj);
  if (is_sideeffect2t(value))
  {
    // An unconstrained pointer reaches no object symex resolves. An object
    // allocated before the loop is one the step has to havoc.
    if (!allocates(value))
      return to_sideeffect2t(value).kind == sideeffect2t::allockind::nondet;
    if (q.outside)
      return false;
    q.out.allocated = true;
    return true;
  }
  if (!computes_from_operands(value))
    return false;

  // Only an offset added to one address, or taken from it, keeps to that
  // address's object: the simplifier folds A + (B - A) to B
  // (add2t::do_simplify).
  bool traced = true;
  size_t operand = 0;
  size_t carriers = 0;
  value->foreach_operand([&](const expr2tc &op) {
    rootst roots;
    queryt sub = q.sub(roots);
    traced = traced && trace(sub, op);
    if (!roots.empty())
    {
      ++carriers;
      traced = traced && (is_if2t(value) || is_add2t(value) ||
                          (is_sub2t(value) && operand == 0));
    }
    q.out.merge(roots);
    ++operand;
  });
  return traced && (is_if2t(value) || carriers <= 1);
}

/// Traces the address of \p object, or of part of it.
bool derivationst::trace_address(queryt &q, expr2tc object) const
{
  while (is_member2t(object) || is_index2t(object))
    object = *object->get_sub_expr(0);
  if (is_dereference2t(object))
    return trace(q, to_dereference2t(object).value);
  if (is_symbol2t(object) && check_var_name(object))
  {
    q.out.objects.insert(object);
    return true;
  }
  // A string literal or NULL: nothing the loop may write.
  return is_constant_string2t(object) || is_null_object2t(object);
}

bool derivationst::trace_symbol(queryt &q, const expr2tc &symbol) const
{
  const irep_idt &name = to_symbol2t(symbol).thename;
  if (name == "NULL")
    return true;
  if (q.address_taken.count(name))
    return false;

  const auto own = owner.find(name);
  const bool global = own == owner.end();
  const bool here = !global && own->second == q.function;
  const bool moves = (global ? looped : looped_here).count(name) != 0;
  if (global || here)
  {
    if (!check_var_name(symbol))
      return false;
    // An integer is no address of its own: what it was computed from is.
    if (!moves && is_pointer_type(symbol))
    {
      q.out.pointers.insert(symbol);
      return true;
    }
    if (global || (moves && !q.iteration.reassigned.count(name)))
      return false;
  }
  // The step havocs a callee's local at its loop's head, which the
  // assignments below do not show, unless nothing reads that havoc.
  else if (moves && unsettled.count(name))
    return false;

  // What every assignment to it computes; for one the body reassigns, every
  // assignment in the body, and for one of the loop's function that no loop
  // assigns, what the function computed before the loop. A local never
  // assigned holds no address symex resolves.
  const bool in_body = here && moves;
  const auto it = defs.find(name);
  return it == defs.end() ||
         trace_once(q, name, here && !moves, [&](queryt &sub) {
           return std::all_of(
             it->second.begin(), it->second.end(), [&](const deft &d) {
               return (in_body && !q.iteration.contains(d.location)) ||
                      trace_def(sub, d);
             });
         });
}

bool derivationst::trace_once(
  queryt &q,
  const irep_idt &name,
  bool leaves,
  const std::function<bool(queryt &)> &trace) const
{
  const bool outside = q.outside || leaves;
  memot &memo = q.memo[outside];
  if (const auto m = memo.find(name); m != memo.end())
  {
    q.out.merge(m->second.roots);
    return m->second.traced;
  }
  if (!q.visiting[outside].insert(name).second)
  {
    ++q.cycles;
    return true;
  }
  const unsigned cycles = q.cycles;
  rootst roots;
  queryt sub = q.sub(roots, leaves);
  const bool traced = trace(sub);
  q.visiting[outside].erase(name);
  if (q.cycles == cycles)
    memo.emplace(name, tracedt{roots, traced});
  q.out.merge(roots);
  return traced;
}

bool derivationst::may_hold_address(
  const queryt &q,
  const expr2tc &value,
  std::unordered_set<irep_idt, irep_id_hash> &seen) const
{
  if (q.budget == 0)
    return true;
  --q.budget;
  if (is_nil_expr(value) || is_constant_int2t(value))
    return false;
  if (is_bv_type(value->type) && q.addressless(value))
    return false;
  if (
    (is_pointer_type(value->type) &&
     !is_code_type(to_pointer_type(value->type).subtype)) ||
    is_array_type(value->type) || is_address_of2t(value) ||
    is_dereference2t(value))
    return true;
  // A bool is 0 or 1, never an address.
  if (is_bool_type(value->type))
    return false;
  if (is_sideeffect2t(value))
    return to_sideeffect2t(value).kind != sideeffect2t::allockind::nondet;
  const auto holds = [&](const deft &d) {
    if (!is_nil_expr(d.value))
      return may_hold_address(q, d.value, seen);
    if (d.callee.empty())
      return true;
    const auto it = returns.find(d.callee);
    if (it == returns.end())
      return has_prefix(d.callee.as_string(), "c:@F@__ESBMC") ||
             has_prefix(d.callee.as_string(), "c:@F@__builtin");
    if (!seen.insert(d.callee).second)
      return false;
    return std::any_of(
      it->second.begin(), it->second.end(), [&](const expr2tc &v) {
        return may_hold_address(q, v, seen);
      });
  };
  if (is_symbol2t(value))
  {
    const irep_idt &name = to_symbol2t(value).thename;
    if (q.address_taken.count(name))
      return true;
    if (!seen.insert(name).second)
      return false;
    const auto it = defs.find(name);
    return it != defs.end() &&
           std::any_of(it->second.begin(), it->second.end(), holds);
  }
  bool holds_one = false;
  value->foreach_operand([&](const expr2tc &op) {
    holds_one = holds_one || may_hold_address(q, op, seen);
  });
  return holds_one;
}

bool derivationst::trace_def(queryt &q, const deft &def) const
{
  if (!is_nil_expr(def.value))
    return trace(q, def.value);
  if (def.callee.empty())
    return false;
  // A function without a body returns a nondet value, unless symex runs it.
  const auto it = returns.find(def.callee);
  if (it == returns.end())
    return !has_prefix(def.callee.as_string(), "c:@F@__ESBMC") &&
           !has_prefix(def.callee.as_string(), "c:@F@__builtin");
  return trace_once(q, def.callee, false, [&](queryt &sub) {
    return std::all_of(
      it->second.begin(), it->second.end(), [&](const expr2tc &value) {
        return trace(sub, value);
      });
  });
}

/// What the storage \p lvalue designates may hold. A pointee `*p` has no node
/// of its own: it holds what the objects p points to hold.
targetst
held_by(andersent &points_to, const loopst &loop, const expr2tc &lvalue)
{
  if (!is_dereference2t(lvalue))
    return targets_of(points_to, loop, lvalue);

  value_setst::valuest objects;
  points_to.get_values(
    loop.get_original_loop_head(), to_dereference2t(lvalue).value, objects);
  targetst held;
  for (const expr2tc &v : objects)
  {
    const expr2tc object =
      is_object_descriptor2t(v) ? to_object_descriptor2t(v).object : expr2tc();
    if (is_nil_expr(object))
      held.anything = true;
    else if (!andersent::is_nondet_object(object))
    {
      const targetst t = targets_of(points_to, loop, object);
      held.named.insert(t.named.begin(), t.named.end());
      held.heap |= t.heap;
      held.anything |= t.anything;
      held.nondet |= t.nondet;
    }
  }
  return held;
}

/// Whether the variable or load \p value holds no address in any run: the
/// points-to sets give it none but the object an unconstrained pointer
/// reaches, whose address symex never resolves. An empty set counts, so this
/// relies on Andersen seeing every way symex writes memory.
bool holds_no_address(
  andersent &points_to,
  const loopst &loop,
  const expr2tc &value)
{
  if (!is_symbol2t(value) && !is_dereference2t(value))
    return false;
  const targetst t = held_by(points_to, loop, value);
  return t.named.empty() && !t.heap && !t.anything;
}

/// What \p instr reads, and the variable it assigns whole, if any. A DEAD
/// reads nothing.
void reads_and_assigns(
  const instructiont &instr,
  std::vector<irep_idt> &reads,
  irep_idt &assigned)
{
  const auto read = [&reads](const expr2tc &e) {
    for_each_subexpr(e, [&reads](const expr2tc &sub) {
      if (is_symbol2t(sub))
        reads.push_back(to_symbol2t(sub).thename);
    });
  };
  expr2tc target;
  if (instr.is_assign())
  {
    target = to_code_assign2t(instr.code).target;
    read(to_code_assign2t(instr.code).source);
  }
  else if (instr.is_function_call())
  {
    const code_function_call2t &call = to_code_function_call2t(instr.code);
    target = call.ret;
    read(call.function);
    for (const expr2tc &op : call.operands)
      read(op);
  }
  else if (instr.is_decl())
  {
    assigned = to_code_decl2t(instr.code).value;
    read(to_code_decl2t(instr.code).init);
  }
  else if (instr.type != DEAD)
    read(instr.code);
  read(instr.guard);

  if (!is_nil_expr(target) && is_symbol2t(target))
    assigned = to_symbol2t(target).thename;
  else
    read(target);
}

/// See iterationt. Call it before any loop of \p function is transformed.
iterationt iteration_of(const goto_functiont &function, const loopst &loop)
{
  iterationt iteration;
  iteration.head = loop.get_original_loop_head()->location_number;
  iteration.back = loop.get_original_loop_exit()->location_number;

  std::vector<goto_programt::const_targett> body;
  std::unordered_map<const instructiont *, size_t> index;
  for (goto_programt::const_targett it = loop.get_original_loop_head();; ++it)
  {
    index.emplace(&*it, body.size());
    body.push_back(it);
    if (it == loop.get_original_loop_exit())
      break;
  }

  // A jump into the body starts an iteration too, havoced only when it is
  // unconditional (see collect_entry_jumps). Nothing a loop reads past its
  // exit may be counted: the step can leave the loop right after the havoc.
  std::vector<size_t> work{0};
  std::unordered_set<irep_idt, irep_id_hash> excluded;
  for (const auto &instr : function.body.instructions)
  {
    if (index.count(&instr))
      continue;
    for (const auto &t : instr.targets)
      if (const auto f = index.find(&*t); f != index.end() && f->second != 0)
      {
        if (!is_true(instr.guard))
          return iteration;
        work.push_back(f->second);
      }
    std::vector<irep_idt> reads;
    irep_idt assigned;
    reads_and_assigns(instr, reads, assigned);
    excluded.insert(reads.begin(), reads.end());
  }

  std::vector<std::vector<irep_idt>> reads(body.size());
  std::vector<irep_idt> assigned(body.size());
  std::unordered_map<irep_idt, size_t, irep_id_hash> vars;
  for (size_t i = 0; i < body.size(); ++i)
  {
    reads_and_assigns(*body[i], reads[i], assigned[i]);
    if (!assigned[i].empty())
      vars.emplace(assigned[i], vars.size());
  }

  // A loop nested in the body havocs what it assigns at its own head.
  for (size_t i = 0; i < body.size(); ++i)
    for (const auto &t : body[i]->targets)
      if (
        const auto f = index.find(&*t);
        f != index.end() && f->second != 0 && f->second <= i)
        for (size_t j = f->second; j <= i; ++j)
          if (!assigned[j].empty())
            excluded.insert(assigned[j]);

  // Must-assigned since an entry, over the iteration: the back edge to the
  // head and every edge out of the loop end it.
  std::vector<std::vector<bool>> in(
    body.size(), std::vector<bool>(vars.size(), true));
  for (size_t entry : work)
    in[entry].assign(vars.size(), false);
  while (!work.empty())
  {
    const size_t i = work.back();
    work.pop_back();
    std::vector<bool> out = in[i];
    if (!assigned[i].empty())
      out[vars.at(assigned[i])] = true;

    std::vector<size_t> next;
    const instructiont &instr = *body[i];
    for (const auto &t : instr.targets)
      if (const auto f = index.find(&*t); f != index.end() && f->second != 0)
        next.push_back(f->second);
    if (!(instr.is_goto() && is_true(instr.guard)) && i + 1 < body.size())
      next.push_back(i + 1);

    for (size_t n : next)
    {
      bool changed = false;
      for (size_t v = 0; v < out.size(); ++v)
        if (in[n][v] && !out[v])
        {
          in[n][v] = false;
          changed = true;
        }
      if (changed)
        work.push_back(n);
    }
  }

  for (const auto &var : vars)
    if (!excluded.count(var.first))
      iteration.reassigned.insert(var.first);
  for (size_t i = 0; i < body.size(); ++i)
    for (const irep_idt &name : reads[i])
      if (const auto v = vars.find(name); v != vars.end() && !in[i][v->second])
        iteration.reassigned.erase(name);
  return iteration;
}

/// A loop, and what its pointer writes are resolved with.
struct loop_writest
{
  loopst &loop;
  const irep_idt &function;
  andersent &points_to;
  const derivationst &derivations;
  /// Whether symex drops a write through a pointer it cannot resolve, which
  /// tracing a pointer to what it derives from relies on.
  bool drops_unresolved;
  const std::unordered_set<irep_idt, irep_id_hash> &address_taken;
  const iterationt &iteration;
};

/// What the inductive step havocs to cover a loop's writes through pointers.
struct covert
{
  /// Named objects, havoced by name.
  loopst::loop_varst objects;
  /// Pointers whose objects are havoced whole.
  loopst::loop_varst through;
  /// Whether a write may also reach any variable whose address is taken.
  bool anything = false;
  /// Named objects written through a pointer that is loaded from memory.
  loopst::loop_varst loaded;
};

/// Adds to \p cover what a write through \p ptr may reach: the named objects
/// the points-to sets give it or, failing those, what it derives from.
bool cover_write(const loop_writest &w, const expr2tc &ptr, covert &cover)
{
  const targetst t = targets_of(w.points_to, w.loop, ptr);
  rootst roots;
  const bool derived = w.drops_unresolved &&
                       w.derivations.roots(
                         w.function,
                         w.address_taken,
                         w.iteration,
                         [&w](const expr2tc &value) {
                           return holds_no_address(w.points_to, w.loop, value);
                         },
                         ptr,
                         roots);
  // An unconstrained pointer writes no named object: symex sends such a write
  // to an invalid object. An empty set is different — no constraint reached
  // the pointer, so the analysis knows nothing about it.
  const bool named = !t.heap && !t.anything && (!t.named.empty() || t.nondet);
  if (!named && !derived)
  {
    if (messaget::state.target("k-induction", VerbosityLevel::Debug))
      log_debug(
        "k-induction",
        "cannot name what {} may point to at {}: {}",
        ptr->pretty(0),
        w.loop.get_original_loop_head()->location.as_string(),
        t.anything ? "anything"
        : t.heap   ? "heap"
                   : "nothing");
    return false;
  }

  cover.objects.insert(t.named.begin(), t.named.end());
  if (named)
  {
    if (!derived)
      cover.loaded.insert(t.named.begin(), t.named.end());
    return true;
  }
  cover.anything |= t.anything;
  cover.objects.insert(roots.objects.begin(), roots.objects.end());
  cover.through.insert(roots.pointers.begin(), roots.pointers.end());
  return true;
}

/// The step leaves a pointer loaded from an object it havocs whole unresolved
/// and drops a write through it, so no such object may hold an address the
/// loop writes through a loaded pointer.
bool havocs_stored_pointer(const loop_writest &w, const covert &cover)
{
  return std::any_of(
    cover.through.begin(), cover.through.end(), [&](const expr2tc &ptr) {
      const targetst held =
        held_by(w.points_to, w.loop, dereference2tc(get_empty_type(), ptr));
      return std::any_of(
        held.named.begin(), held.named.end(), [&](const expr2tc &object) {
          return cover.loaded.count(object) != 0;
        });
    });
}

/// A write through one pointee may move another's pointer (`*pp = r` moves
/// p): the named pointers the loop's pointees may move. One reaching anything
/// can only move a pointer whose address is taken somewhere, which \p cover
/// records.
loopst::loop_varst clobbered_by_pointees(const loop_writest &w, covert &cover)
{
  loopst::loop_varst clobbered;
  for (const expr2tc &pointee : w.loop.get_written_pointees())
  {
    const targetst t =
      targets_of(w.points_to, w.loop, to_dereference2t(pointee).value);
    clobbered.insert(t.named.begin(), t.named.end());
    cover.anything |= t.anything;
  }
  return clobbered;
}

/// The inductive step havocs only its loop's modified variables, so storage
/// the loop writes through a pointer would keep its pre-loop value and the
/// step would prove too much (#5224). Add that storage to the modified
/// variables: `*p` itself for a write inside `*p` while no loop moves `p`,
/// otherwise the named objects the whole-program points-to sets resolve the
/// written pointer to. Failing those, the pointer is traced to loop-head
/// pointers, added to \p through, whose objects are to be havoced whole.
/// Returns false, and the caller disables the inductive step, when nothing
/// covers a write.
bool havoc_written_objects(const loop_writest &w, loopst::loop_varst &through)
{
  loopst &loop = w.loop;
  if (loop.unnamed_pointer_write())
    return false;

  covert cover;
  for (const expr2tc &ptr : loop.get_written_pointers())
    if (!cover_write(w, ptr, cover))
      return false;

  const loopst::loop_varst clobbered = clobbered_by_pointees(w, cover);

  // check_var_name also filters the modified set, so a pointer it rejects may
  // be reassigned unseen. A pointer another loop assigns is havoced there,
  // and a havoc through it here would reach nothing.
  std::vector<expr2tc> pointees(
    loop.get_written_pointees().begin(), loop.get_written_pointees().end());
  const auto moves = [&](const expr2tc &pointee) {
    const expr2tc &ptr = to_dereference2t(pointee).value;
    const irep_idt &name = to_symbol2t(ptr).thename;
    return !check_var_name(ptr) || names(loop.get_modified_loop_vars(), name) ||
           w.derivations.assigned_in_loop(name) || names(cover.objects, name) ||
           names(clobbered, name) ||
           (cover.anything && w.address_taken.count(name));
  };
  for (auto it = std::find_if(pointees.begin(), pointees.end(), moves);
       it != pointees.end();
       it = std::find_if(pointees.begin(), pointees.end(), moves))
  {
    if (!cover_write(w, to_dereference2t(*it).value, cover))
      return false;
    pointees.erase(it);
  }

  if (havocs_stored_pointer(w, cover))
    return false;

  for (const expr2tc &obj : cover.objects)
    loop.add_modified_var_to_loop(obj);
  for (const expr2tc &pointee : pointees)
    loop.add_modified_var_to_loop(pointee);
  through = std::move(cover.through);
  return true;
}

/// A value of \p type with the provenance \p t allows: an address into one of
/// its named objects at any offset, NULL, or, when the pointer may be
/// unconstrained, anything. The offset is unbounded, so an integer keeps its
/// full range.
expr2tc pinned_value(const type2tc &type, const targetst &t)
{
  const type2tc bytes = pointer_type2tc(get_uint8_type());
  std::vector<expr2tc> named(t.named.begin(), t.named.end());
  std::sort(named.begin(), named.end(), [](const expr2tc &a, const expr2tc &b) {
    return a->pretty() < b->pretty();
  });
  expr2tc value = t.nondet ? gen_nondet(bytes) : gen_zero(bytes);
  for (const expr2tc &obj : named)
    value = if2tc(
      bytes,
      gen_nondet(get_bool_type()),
      add2tc(
        bytes,
        typecast2tc(bytes, address_of2tc(obj->type, obj)),
        gen_nondet(signed_size_type2())),
      value);
  return typecast2tc(type, value);
}

/// Adds to \p pins an assignment of a \p t value to each address \p lvalue
/// stores. Returns false when one sits in an array element or a union member,
/// which no single assignment reaches.
bool pin(const expr2tc &lvalue, const targetst &t, pinst &pins)
{
  const type2tc &type = lvalue->type;
  if (is_struct_type(type))
  {
    const struct_type2t &s = to_struct_type(type);
    for (size_t i = 0; i < s.members.size(); ++i)
      if (!pin(member2tc(s.members[i], lvalue, s.member_names[i]), t, pins))
        return false;
    return true;
  }
  if (!carries_address(type))
    return true;
  if (!is_pointer_type(type) && !is_bv_type(type))
    return false;
  pins.emplace_back(lvalue, pinned_value(type, t));
  return true;
}

/// Symex resolves a dereference of a havoced, hence nondet, pointer to an
/// invalid object, so a write through it would be dropped and the inductive
/// step would prove too much. Pin each havoced value that may hold a named
/// address back to the objects the points-to sets give it. One that may also
/// hold a heap or unknown address needs no pin: by inclusion it never reaches
/// a pointer the loop writes through, or havoc_written_objects would have
/// refused that pointer. A local of a function the loop calls needs none
/// either: every call declares it afresh, so the step never reads its havoced
/// value. Returns false when a value cannot be pinned.
bool pin_havoced_pointers(const loop_writest &w, pinst &pins)
{
  for (const expr2tc &var : ordered(w.loop.get_modified_loop_vars()))
  {
    if (
      is_symbol2t(var) &&
      w.derivations.local_elsewhere(to_symbol2t(var).thename, w.function))
      continue;
    const targetst t = held_by(w.points_to, w.loop, var);
    if (t.named.empty() || t.heap || t.anything)
      continue;
    if (!pin(var, t, pins))
      return false;
  }
  return true;
}

/// The functions the entry point may reach, calls through function pointers
/// included: every function named in a reachable body.
std::unordered_set<irep_idt, irep_id_hash>
reachable_functions(const goto_functionst &goto_functions)
{
  std::unordered_set<irep_idt, irep_id_hash> seen{goto_functions.main_id()};
  std::vector<irep_idt> work{goto_functions.main_id()};
  while (!work.empty())
  {
    auto it = goto_functions.function_map.find(work.back());
    work.pop_back();
    if (it == goto_functions.function_map.end() || !it->second.body_available)
      continue;
    for (const auto &instr : it->second.body.instructions)
      for (const expr2tc &e : {instr.code, instr.guard})
        for_each_subexpr(e, [&](const expr2tc &sub) {
          if (
            is_symbol2t(sub) && is_code_type(sub->type) &&
            seen.insert(to_symbol2t(sub).thename).second)
            work.push_back(to_symbol2t(sub).thename);
        });
  }
  return seen;
}

/// The variables whose address the program takes anywhere.
std::unordered_set<irep_idt, irep_id_hash>
address_taken_symbols(const goto_functionst &goto_functions)
{
  std::unordered_set<irep_idt, irep_id_hash> taken;
  forall_goto_functions (it, goto_functions)
    for (const auto &instr : it->second.body.instructions)
      for (const expr2tc &e : {instr.code, instr.guard})
        for_each_subexpr(e, [&taken](const expr2tc &sub) {
          if (!is_address_of2t(sub))
            return;
          for_each_subexpr(to_address_of2t(sub).ptr_obj, [&](const expr2tc &o) {
            if (is_symbol2t(o))
              taken.insert(to_symbol2t(o).thename);
          });
        });
  return taken;
}

/// True iff the program contains a reachable call to __VERIFIER_nondet_memory
/// in user (non-hidden) code. That intrinsic havocs a caller object with fresh
/// nondeterminism through a pointer (`*(p + i) = nondet_uchar()` over memory
/// with no nameable symbol) — the opaque "havoc_memory" analogue of the
/// per-element user nondet-array builder loops the SV-COMP "havoc_object" shape
/// uses. The inductive step cannot generalise such an input, so a reachable
/// call makes it unsound: it proves unsafe programs SAFE (the
/// SoftwareSystems-Intel-TDX-Module-ReachSafety `*_havoc_memory` tasks). The
/// model body is linked into *every* program, so gate on an actual call, not
/// on its always-present loop. See #5593 (recurrence of #5224 / #5230).
bool calls_nondet_memory(const goto_functionst &goto_functions)
{
  forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available || it->second.body.hide)
      continue;
    for (const auto &instr : it->second.body.instructions)
    {
      if (!instr.is_function_call())
        continue;
      const code_function_call2t &call = to_code_function_call2t(instr.code);
      if (
        is_symbol2t(call.function) &&
        to_symbol2t(call.function).thename == "c:@F@__VERIFIER_nondet_memory")
        return true;
    }
  }
  return false;
}
} // namespace

/// Havocs what the loop of \p w writes and transforms it for the inductive
/// step. Returns false when a write through a pointer stays uncovered.
bool havoc_loop(
  goto_functionst &goto_functions,
  goto_functiont &goto_function,
  const loop_writest &w,
  bool continue_past_failed_assertions)
{
  loopst &loop = w.loop;
  // Before the empty-modified-set skip: a loop that only writes through
  // pointers has no named modified variables until they are resolved.
  loopst::loop_varst through;
  bool covered =
    !loop.writes_through_pointer() || havoc_written_objects(w, through);
  if (loop.get_modified_loop_vars().empty() && through.empty())
    return covered;

  havocst havocs{ordered(through), ordered(loop.get_modified_loop_vars()), {}};
  if (loop.writes_through_pointer() && !pin_havoced_pointers(w, havocs.pins))
    covered = false;
  // Symex answers the call by name, but goto_loopst, for the loops still to
  // come, and later passes resolve every call through function_map.
  if (!havocs.through.empty())
  {
    const expr2tc havoc_object = havoc_object_function();
    goto_functions.function_map[to_symbol2t(havoc_object).thename].type =
      havoc_object->type;
  }
  transform_loop(goto_function, loop, havocs, continue_past_failed_assertions);
  return covered;
}

/// The locals and parameters some loop of their function havocs in the
/// inductive step and may read the havoc of.
std::unordered_set<irep_idt, irep_id_hash>
unsettled_by_loops(goto_functionst &goto_functions)
{
  std::unordered_set<irep_idt, irep_id_hash> unsettled;
  Forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available)
      continue;
    // A callee's local the loop havocs is declared afresh by every call.
    std::unordered_set<irep_idt, irep_id_hash> locals;
    if (is_code_type(it->second.type))
      for (const irep_idt &arg : to_code_type(it->second.type).argument_names)
        locals.insert(arg);
    for (const instructiont &instr : it->second.body.instructions)
      if (instr.is_decl())
        locals.insert(to_code_decl2t(instr.code).value);

    goto_loopst loops(it->first, goto_functions, it->second);
    for (const loopst &loop : loops.get_loops())
    {
      const iterationt iteration = iteration_of(it->second, loop);
      for (const expr2tc &var : loop.get_modified_loop_vars())
        if (
          is_symbol2t(var) && locals.count(to_symbol2t(var).thename) &&
          !iteration.reassigned.count(to_symbol2t(var).thename))
          unsettled.insert(to_symbol2t(var).thename);
    }
  }
  return unsettled;
}

bool goto_k_induction(
  goto_functionst &goto_functions,
  const namespacet &,
  bool continue_past_failed_assertions)
{
  // Andersen and loop_shapest key on location numbers, which inlining
  // restarts in every function.
  goto_functions.update();

  // Build the points-to sets once, up front, on the pristine program: the
  // havoc a transformed loop gains would widen every later query to TOP.
  andersent points_to;
  points_to(goto_functions);
  const auto reachable = reachable_functions(goto_functions);
  const auto address_taken = address_taken_symbols(goto_functions);
  const loop_shapest shapes(goto_functions);

  // Objects allocated in the iteration, and the callee locals derivationst
  // finds never assigned, need no havoc only while symex drops a write through
  // a pointer it cannot resolve: a pointer check claims that pointer valid, and
  // the step assumes the claims of its early iterations. A leak check would
  // also miss the objects that earlier iterations allocated.
  const bool drops_unresolved =
    config.options.get_bool_option("no-pointer-check") &&
    !config.options.get_bool_option("memory-leak-check");
  const derivationst derivations(
    goto_functions,
    shapes,
    drops_unresolved ? unsettled_by_loops(goto_functions)
                     : std::unordered_set<irep_idt, irep_id_hash>());

  // A reachable __VERIFIER_nondet_memory call havocs a caller object the
  // inductive step cannot generalise, so its unsoundness is independent of any
  // loop shape — disable the inductive step up front. See #5593.
  bool disable_inductive_step = calls_nondet_memory(goto_functions);
  Forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available)
      continue;
    // Library helpers (body.hide) write through pointers in nearly every
    // model (memcpy, string ops, ...). Letting them set the gate would
    // disable the inductive step for essentially every program. Their loops
    // are never the user property's witness, so exclude them from the
    // decision — mirrors the body.hide guard in goto_termination. So is a
    // function nothing calls.
    const bool decides =
      !it->second.body.hide && reachable.count(it->first) != 0;
    goto_loopst loops(it->first, goto_functions, it->second);
    std::vector<iterationt> iterations;
    for (const loopst &loop : loops.get_loops())
      iterations.push_back(
        drops_unresolved ? iteration_of(it->second, loop) : iterationt());
    auto iteration = iterations.cbegin();
    for (auto &loop : loops.get_loops())
      if (
        !havoc_loop(
          goto_functions,
          it->second,
          {loop,
           it->first,
           points_to,
           derivations,
           drops_unresolved,
           address_taken,
           *iteration++},
          continue_past_failed_assertions) &&
        decides)
        disable_inductive_step = true;
  }
  goto_functions.update();
  return disable_inductive_step;
}
