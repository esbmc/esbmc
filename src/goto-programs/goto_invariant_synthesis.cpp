#include <goto-programs/goto_invariant_synthesis.h>
#include <goto-programs/goto_loop_invariant.h>
#include <goto-programs/goto_loops.h>
#include <goto-programs/loopst.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/expr/expr_util.h>
#include <util/irep/std_expr.h>
#include <algorithm>
#include <map>
#include <vector>

namespace
{
/// How far back from the loop head to look for the entry assignment of a
/// counter/accumulator. The scan stops early at any control flow, so this is
/// only a guard against walking a very long straight-line prologue.
constexpr size_t kMaxEntryScanBack = 64;

bool mentions_modified_var(
  const expr2tc &expr,
  const loopst::loop_varst &modified)
{
  if (!expr)
    return false;

  if (modified.find(expr) != modified.end())
    return true;

  bool found = false;
  expr->foreach_operand([&modified, &found](const expr2tc &sub) {
    if (!found && mentions_modified_var(sub, modified))
      found = true;
  });
  return found;
}

/// The loop's entry condition. The head IF holds the *exit* condition: a while
/// loop is lowered to `IF !(cond) GOTO exit`, but a pass that simplifies the
/// guard (--interval-analysis does) leaves the equivalent `IF i > n GOTO exit`
/// with no not2t to strip. Negate and simplify, which covers both spellings.
bool guard_condition(const goto_programt::targett &head_if, expr2tc &cond)
{
  const expr2tc &g = head_if->guard;
  if (is_nil_expr(g))
    return false;

  cond = is_not2t(g) ? to_not2t(g).value : not2tc(g);
  simplify(cond);
  return true;
}

/// Split `cond` into counter and bound for the `<`/`<=` shapes this pass
/// handles, and report which one it was. Other comparisons (and decrementing
/// loops) are left to a later revision.
bool split_bound(
  const expr2tc &cond,
  expr2tc &counter,
  expr2tc &bound,
  bool &inclusive)
{
  if (is_lessthanequal2t(cond))
  {
    counter = to_lessthanequal2t(cond).side_1;
    bound = to_lessthanequal2t(cond).side_2;
    inclusive = true;
    return true;
  }
  if (is_lessthan2t(cond))
  {
    counter = to_lessthan2t(cond).side_1;
    bound = to_lessthan2t(cond).side_2;
    inclusive = false;
    return true;
  }
  return false;
}

/// True when the instruction cannot appear in a body this pass is willing to
/// summarise. Branches would make the per-iteration effect conditional, and a
/// call or return can write the counter or accumulator out of sight.
bool breaks_straight_line(const goto_programt::targett &it)
{
  return it->is_goto() || it->is_function_call() || it->is_return() ||
         it->is_throw() || it->is_catch() || it->is_atomic_begin() ||
         it->is_atomic_end();
}

/// `lhs = lhs + addend` — the only body assignment shape recognised here.
bool is_self_increment(
  const expr2tc &target,
  const expr2tc &source,
  expr2tc &addend)
{
  if (!is_add2t(source))
    return false;

  const auto &add = to_add2t(source);
  if (add.side_1 == target)
  {
    addend = add.side_2;
    return true;
  }
  if (add.side_2 == target)
  {
    addend = add.side_1;
    return true;
  }
  return false;
}

/// Gate for the symbolic-addend regime and for the `i >= i0` conjunct; see the
/// header. Also keeps pointers out of the arithmetic builders entirely, where a
/// `p = p + 1` accumulator would build mul2t over pointer types and trip
/// assert_arith_2ops_consistency.
bool is_unsigned_integer(const expr2tc &expr)
{
  return expr && is_unsignedbv_type(expr->type);
}

/// Type gate for the constant-addend regime, which admits signed counters; see
/// the header for why the sign is not what the restriction was ever about.
bool is_integer(const expr2tc &expr)
{
  return expr &&
         (is_unsignedbv_type(expr->type) || is_signedbv_type(expr->type));
}

/// Key for the per-variable write map. Every caller has already established
/// the expression is a symbol, so this is the whole identity -- and cheaper
/// than pretty(), which dumps the expression textually on every lookup.
std::string write_key(const expr2tc &var)
{
  return to_symbol2t(var).thename.as_string();
}

bool is_constant_one(const expr2tc &expr)
{
  return is_constant_int2t(expr) && to_constant_int2t(expr).value == 1;
}

/// True when a LOOP_INVARIANT already sits in the window goto_loop_invariant's
/// extractor searches -- the extractor's own constant, so the two agree by
/// construction. Both invariants would be folded into one conjunction, so
/// adding a guess next to a user-written one risks failing the user's proof.
bool has_user_invariant(
  const goto_programt::targett &head,
  const goto_programt::targett &begin)
{
  goto_programt::targett it = head;
  for (size_t steps = 0;
       it != begin && steps < goto_loop_invariantt::kMaxInvariantSearchBack;
       ++steps)
  {
    --it;
    if (it->is_loop_invariant())
      return true;
    if (breaks_straight_line(it))
      return false;
  }
  return false;
}

/// Value of `var` on entry to the loop: the nearest preceding assignment in the
/// straight-line prologue, and only when its RHS is a literal.
/// Returns false when the scan meets control flow first, so the value we would
/// report might not be the one that reaches the head.
bool entry_value(
  const goto_programt::targett &head,
  const goto_programt::targett &begin,
  const expr2tc &var,
  expr2tc &value)
{
  goto_programt::targett it = head;
  for (size_t steps = 0; it != begin && steps < kMaxEntryScanBack; ++steps)
  {
    --it;

    // Stepping over a jump target would leave the other incoming edge
    // unexamined, and the assignment we then report is only the value that
    // reaches the head along one path. `if (c) s = 5;` before the loop is
    // enough to make the reported entry value wrong on the other branch.
    if (it->is_target() || breaks_straight_line(it))
      return false;

    if (!it->is_assign())
      continue;

    const auto &assign = to_code_assign2t(it->code);

    // A write through a dereference, member or index (`*p = 5`, `q->v = 5`,
    // `w[0] = 9`) may land on `var` itself. It compares unequal, so skipping it
    // would let the scan walk past and report an older, stale constant -- the
    // same defect as a symbolic RHS, reached by a different spelling. We cannot
    // show it does not alias, so stop.
    if (!is_symbol2t(assign.target))
      return false;

    if (assign.target != var)
      continue;

    // M-4: only a literal is safe. A symbolic RHS records an *expression*, not
    // a value, and any write to one of its symbols between here and the loop
    // head silently changes what the closed form means (`s = k; k = 7;`).
    if (!is_constant_int2t(assign.source))
      return false;

    value = assign.source;
    return true;
  }
  return false;
}

/// The two-disjunct bound `(i <op> B) || i == E` is established only when the
/// counter's entry value cannot sit past the loop's exit value. Working the
/// cases through for a constant i0:
///
///   `<=`, E = B+1: establishment fails iff i0 > B and i0 != B+1. At i0 == 1
///                  the first conjunct forces B == 0, which makes E == 1 == i0,
///                  so it cannot fail; at i0 == 0 it cannot fail either.
///                  i0 >= 2 admits B <= i0 - 2 and does fail.
///   `<`,  E = B:   establishment fails iff i0 > B, which only i0 == 0 rules
///                  out.
///
/// Anything else needs the third disjunct, which only the constant-addend
/// regime can afford; see the header.
bool entry_admits_two_disjunct_bound(const expr2tc &entry, bool inclusive)
{
  if (!is_constant_int2t(entry))
    return false;

  const BigInt &v = to_constant_int2t(entry).value;
  return v == 0 || (inclusive && v == 1);
}

struct accumulatort
{
  expr2tc var;
  expr2tc addend;
  expr2tc entry;
};

struct affine_loopt
{
  expr2tc counter;
  expr2tc bound;
  expr2tc counter_entry;
  /// The loop's entry condition, as computed by guard_condition. Carried out of
  /// the recogniser rather than recomputed: a second call cannot fail once the
  /// first succeeded on the same head, so recomputing it left an unreachable
  /// error path and risked the two spellings drifting apart.
  expr2tc cond;
  bool inclusive = true;
  /// True when every accumulator's addend is a literal, so the closed form
  /// contains no symbolic multiplier. Licenses the three-disjunct bound, and
  /// with it signed counters and any literal entry value.
  bool constant_addends = true;
  std::vector<accumulatort> accumulators;
};

/// MSVC spells `assert(e)` as `(!!(e)) || (_wassert(...), 0)`, so on that
/// target a loop body that asserts holds a branch around the ASSERT rather than
/// the single one the glibc and Darwin spellings fold to in do_assert_fail().
/// The region such a branch spans writes nothing, so the per-iteration effect
/// is still exactly the assignments outside it and stepping over it is sound.
///
/// `first` indexes the branch that opens the region. On success `join` is the
/// index control rejoins at -- the caller resumes there -- and `saw_assert` is
/// set if the region asserts.
bool assertion_only_region(
  const std::vector<goto_programt::targett> &body,
  const std::map<const goto_programt::instructiont *, size_t> &index,
  size_t first,
  size_t &join,
  bool &saw_assert)
{
  join = first + 1;
  for (size_t i = first; i < join; ++i)
  {
    const goto_programt::targett it = body[i];

    if (it->is_goto())
    {
      for (const auto &target : it->targets)
      {
        const auto found = index.find(&*target);
        // A target outside the body leaves the loop (a `break`, whose iteration
        // count this pass cannot express); one at or behind `first` is a back
        // edge, so a nested loop rather than an assertion diamond.
        if (found == index.end() || found->second <= first)
          return false;
        join = std::max(join, found->second);
      }
      continue;
    }

    if (it->is_assert())
    {
      saw_assert = true;
      continue;
    }

    // Anything that can write a variable or call out of the region would make
    // the per-iteration effect conditional, which is what the region has to
    // rule out to be skippable.
    if (!it->is_skip() && !it->is_location())
      return false;
  }
  return true;
}

/// Summarise the loop body into per-variable (target, addend) pairs, and check
/// the counter advances by exactly one. Anything that is not a plain
/// self-increment, or a second write to a variable already summarised, makes
/// the per-iteration effect something this pass cannot express in closed form.
static bool summarise_body(
  goto_programt::targett head,
  goto_programt::targett exit,
  const expr2tc &counter,
  std::map<std::string, std::pair<expr2tc, expr2tc>> &writes,
  bool &body_asserts)
{
  body_asserts = false;

  std::vector<goto_programt::targett> body;
  std::map<const goto_programt::instructiont *, size_t> index;
  for (goto_programt::targett it = std::next(head); it != exit; ++it)
  {
    index.emplace(&*it, body.size());
    body.push_back(it);
  }

  for (size_t i = 0; i < body.size(); ++i)
  {
    const goto_programt::targett it = body[i];

    if (it->is_goto())
    {
      size_t join;
      if (!assertion_only_region(body, index, i, join, body_asserts))
        return false;
      i = join - 1;
      continue;
    }

    if (breaks_straight_line(it))
      return false;
    if (it->is_assert())
    {
      body_asserts = true;
      continue;
    }
    if (!it->is_assign())
    {
      // Whitelist rather than blacklist: OTHER carries `free`, `delete` and
      // `asm` (symex_other.cpp), whose effect the schema's havoc of
      // get_modified_loop_vars() cannot express -- it names variables and
      // says nothing about heap validity. Cutting a loop over a `free` proved
      // a post-loop use-after-free safe.
      if (!is_inert_scan_instruction(it))
        return false;
      continue;
    }

    const auto &assign = to_code_assign2t(it->code);
    if (!is_symbol2t(assign.target))
      return false;

    expr2tc addend;
    if (!is_self_increment(assign.target, assign.source, addend))
      return false;

    if (!writes
           .emplace(
             write_key(assign.target), std::make_pair(assign.target, addend))
           .second)
      return false;
  }

  const auto counter_write = writes.find(write_key(counter));
  return counter_write != writes.end() &&
         is_constant_one(counter_write->second.second);
}

/// Classify every modified variable other than the counter as an accumulator
/// whose per-iteration addend is loop-invariant, recording its entry value. A
/// variable we cannot classify means we have misread the loop, so reject
/// rather than emit a summary alongside it.
static bool classify_accumulators(
  goto_programt::targett head,
  goto_programt::targett begin,
  const loopst::loop_varst &modified,
  const std::map<std::string, std::pair<expr2tc, expr2tc>> &writes,
  affine_loopt &out)
{
  for (const auto &var : modified)
  {
    if (var == out.counter)
      continue;
    if (!is_symbol2t(var))
      return false;

    const auto write = writes.find(write_key(var));
    if (write == writes.end())
      return false;

    accumulatort acc;
    acc.var = var;
    acc.addend = write->second.second;
    if (!is_integer(acc.var) || !is_integer(acc.addend))
      return false;
    if (mentions_modified_var(acc.addend, modified))
      return false;
    if (!is_constant_int2t(acc.addend))
      out.constant_addends = false;
    if (!entry_value(head, begin, acc.var, acc.entry))
      return false;

    out.accumulators.push_back(acc);
  }

  // loop_varst is hashed on interned-string order, which varies between runs;
  // sort so the emitted invariants are identical across invocations.
  std::sort(
    out.accumulators.begin(),
    out.accumulators.end(),
    [](const accumulatort &a, const accumulatort &b) {
      return write_key(a.var) < write_key(b.var);
    });
  return true;
}

/// The syntactic shape of the head: guard, counter and bound, before the body
/// is read. Fills `out.cond`, `out.counter`, `out.bound` and `out.inclusive`.
bool match_counter_and_bound(
  const goto_programt::targett &head,
  const goto_programt::targett &exit,
  const loopst::loop_varst &modified,
  affine_loopt &out)
{
  if (!head->is_goto() || head == exit)
    return false;

  if (!guard_condition(head, out.cond))
    return false;
  if (!split_bound(out.cond, out.counter, out.bound, out.inclusive))
    return false;

  if (!is_symbol2t(out.counter) || modified.find(out.counter) == modified.end())
    return false;
  if (!is_integer(out.counter) || !is_integer(out.bound))
    return false;
  return !mentions_modified_var(out.bound, modified);
}

/// Whether the regime the accumulators put us in admits this counter. Which
/// bound shape is affordable decides how strict the counter has to be, so this
/// runs only once classify_accumulators has set `out.constant_addends`.
bool regime_admits_counter(
  const affine_loopt &out,
  bool body_asserts,
  const overflow_checkst &overflow)
{
  // Symbolic addend: two disjuncts only, so an unsigned counter entering at 0
  // or 1. See the header.
  if (!out.constant_addends)
  {
    if (!is_unsigned_integer(out.counter) || !is_unsigned_integer(out.bound))
      return false;
    if (!entry_admits_two_disjunct_bound(out.counter_entry, out.inclusive))
      return false;
  }

  // A signed counter cannot carry the `i >= i0` conjunct (its `i + 1` wraps at
  // the type maximum while still inside the guard), so the havoc is free to
  // pick i < i0, where `i - i0` is negative and the closed form describes a
  // state the loop never reaches. That is invisible unless something reads the
  // accumulator mid-loop -- so decline exactly when the body asserts. Measured:
  // without this, a correct signed loop with an in-loop `assert(sn <= 2 * n)`
  // reports FAILED on the user's own assertion while both invariant claims
  // pass. Unsigned counters keep the conjunct and are unaffected.
  if (body_asserts && !is_unsigned_integer(out.counter))
    return false;

  // goto_check instruments the guards this pass emits, and
  // --unsigned-overflow-check widens what it instruments to unsigned
  // arithmetic as well, leaving no integer type the closed form could be
  // emitted at without inventing claims. Decline outright.
  if (overflow.unsigned_arith)
    return false;

  if (!overflow.signed_arith)
    return true;

  // The closed form is built at each accumulator's type, not the counter's, so
  // an unsigned counter with a signed accumulator would still emit signed
  // arithmetic and draw claims on operations the user never wrote.
  if (!is_unsigned_integer(out.counter))
    return false;
  for (const auto &acc : out.accumulators)
    if (!is_unsigned_integer(acc.var))
      return false;
  return true;
}

/// Match the loop against the affine counter/accumulator shape. Every rejection
/// here costs only a missed invariant, so the tests are deliberately strict.
bool recognise_affine_loop(
  goto_functiont &goto_function,
  const loopst &loop,
  const overflow_checkst &overflow,
  goto_programt::targett &head_out,
  affine_loopt &out)
{
  goto_programt::targett head = loop.effective_loop_head();
  const goto_programt::targett exit = loop.get_original_loop_exit();
  const auto &modified = loop.get_modified_loop_vars();

  if (!match_counter_and_bound(head, exit, modified, out))
    return false;

  std::map<std::string, std::pair<expr2tc, expr2tc>> writes;
  bool body_asserts = false;
  if (!summarise_body(head, exit, out.counter, writes, body_asserts))
    return false;

  const goto_programt::targett begin = goto_function.body.instructions.begin();
  if (!entry_value(head, begin, out.counter, out.counter_entry))
    return false;

  // Classify first: whether any addend is symbolic decides which bound shape is
  // affordable, and that in turn decides how strict the counter has to be.
  if (!classify_accumulators(head, begin, modified, writes, out))
    return false;

  if (!regime_admits_counter(out, body_asserts, overflow))
    return false;

  head_out = head;
  return true;
}

/// (i <op> B) || i == E, where E is the value the counter holds once the guard
/// first fails. See the header for why this is a disjunction and not the
/// tighter `i <= B + 1`, and why it stays at exactly two disjuncts.
expr2tc build_bound_invariant(const affine_loopt &shape, const expr2tc &cond)
{
  expr2tc exit_value = shape.bound;
  if (shape.inclusive)
    exit_value = add2tc(
      shape.bound->type, shape.bound, constant_int2tc(shape.bound->type, 1));

  expr2tc inv = or2tc(
    cond,
    equality2tc(shape.counter, typecast2tc(shape.counter->type, exit_value)));

  // Third arm: makes establishment unconditional when the entry value may sit
  // past the exit value. Constant-addend regime only — see the header.
  const expr2tc entry_as_counter =
    typecast2tc(shape.counter->type, shape.counter_entry);

  if (shape.constant_addends)
  {
    inv = or2tc(inv, equality2tc(shape.counter, entry_as_counter));

    // "Either the loop was entered, or the counter is still its entry value."
    // Without this the exit admits i == E for a bound that never satisfied the
    // guard: for a signed loop that is i == n < i0, and the closed form then
    // reports an accumulator value the loop could never produce -- a false
    // alarm on the user's own assertion after the loop. Costs no arithmetic on
    // i, so unlike `i >= i0` it survives the wrap that makes that conjunct
    // unusable for a signed counter.
    const expr2tc entered =
      shape.inclusive ? expr2tc(lessthanequal2tc(entry_as_counter, shape.bound))
                      : expr2tc(lessthan2tc(entry_as_counter, shape.bound));
    inv =
      and2tc(inv, or2tc(entered, equality2tc(shape.counter, entry_as_counter)));
  }

  // The counter never goes below its entry value, and saying so matters: the
  // havoc is otherwise free to pick i < i0, where (i - i0) wraps and the
  // accumulator's closed form describes a state the loop can never reach. That
  // shows up as a false alarm on the user's own in-loop assertions, and as
  // overflow claims on arithmetic the user never wrote. Costs one comparison
  // and no extra multiplier branch; for i0 == 0 it simplifies away entirely.
  // Unsigned counters only; a signed counter's `i + 1` wraps at the type
  // maximum while still inside the guard, making this false after a legitimate
  // iteration. See the header. Pinned by the lowerbnd and entrytwo tests.
  if (is_unsignedbv_type(shape.counter->type))
    inv = and2tc(
      inv,
      greaterthanequal2tc(
        shape.counter, typecast2tc(shape.counter->type, shape.counter_entry)));

  simplify(inv);
  return inv;
}

/// s == s0 + (i - i0) * e, with the difference taken at the accumulator's type.
/// Widening the counter before subtracting rather than after is what makes this
/// exact: a narrower counter's subtraction wraps at its own width and the
/// widening does not follow it, so `int i` with a `long` accumulator failed its
/// own inductive step at i near INT_MIN. Where the counter is at least as wide
/// the two spellings agree, the product depending only on (i - i0) modulo the
/// accumulator's width.
expr2tc
build_accumulator_invariant(const affine_loopt &shape, const accumulatort &acc)
{
  const type2tc &t = acc.var->type;
  const expr2tc elapsed = sub2tc(
    t, typecast2tc(t, shape.counter), typecast2tc(t, shape.counter_entry));

  expr2tc inv = equality2tc(
    acc.var,
    add2tc(
      t,
      typecast2tc(t, acc.entry),
      mul2tc(t, elapsed, typecast2tc(t, acc.addend))));
  simplify(inv);
  return inv;
}

/// Attach the synthesised conjuncts as a LOOP_INVARIANT immediately before the
/// loop head. A plain list insert is used rather than insert_swap: the latter
/// moves the head's content down, which would leave the back-edge targeting the
/// marker instead of the guard, and the extractor in goto_loop_invariant walks
/// strictly backwards from the head and would then never see it.
void emit_invariant(
  goto_functiont &goto_function,
  const goto_programt::targett &insert_before,
  const goto_programt::targett &head,
  const affine_loopt &shape,
  const expr2tc &cond)
{
  goto_programt::instructiont inv;
  inv.type = LOOP_INVARIANT;
  inv.location = head->location;
  inv.function = head->function;
  // Claim ownership: the extractor accepts this marker only for the loop head
  // it sits immediately before. See kSynthesisedInvariantProperty.
  inv.location.property(kSynthesisedInvariantProperty);

  inv.add_loop_invariant(build_bound_invariant(shape, cond));
  for (const auto &acc : shape.accumulators)
    inv.add_loop_invariant(build_accumulator_invariant(shape, acc));

  goto_function.body.instructions.insert(insert_before, inv);
}

} // namespace

void goto_synthesise_loop_invariants(
  goto_functionst &goto_functions,
  const overflow_checkst &overflow)
{
  size_t synthesised = 0;

  Forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available || it->second.body.hide)
      continue;

    goto_loopst loops(it->first, goto_functions, it->second);
    for (auto &loop : loops.get_loops())
    {
      if (loop.get_modified_loop_vars().empty())
        continue;

      goto_programt::targett head;
      affine_loopt shape;
      if (!recognise_affine_loop(it->second, loop, overflow, head, shape))
        continue;

      // goto_loop_invariant's extractor walks backwards from the *original*
      // loop head, which --interval-analysis can leave pointing at an ASSUME
      // ahead of the guard. Anchor on that instruction, not on the effective
      // head, or the marker lands after the point the extractor searches from.
      const goto_programt::targett anchor = loop.get_original_loop_head();

      // A user-written invariant on this loop is authoritative; a synthesised
      // one would be a second LOOP_INVARIANT that the extractor folds into the
      // same conjunction, so a rejected guess would fail the user's proof.
      if (has_user_invariant(anchor, it->second.body.instructions.begin()))
        continue;

      emit_invariant(it->second, anchor, head, shape, shape.cond);
      ++synthesised;
    }
  }

  if (synthesised)
    log_status(
      "Synthesised loop invariants for {} loop{}",
      synthesised,
      synthesised == 1 ? "" : "s");

  goto_functions.update();
}
