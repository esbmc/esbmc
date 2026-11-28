/// \file
/// When the inductive step may use a claim as its induction hypothesis.
///
/// The step assumes a claim on every unwinding but the last of the loop L the
/// induction runs over, and asserts it only on the last. That is sound only
/// if the base case at the same k checked the claim on the first k iterations
/// of every instance of L. It did when
///  - L is the outermost loop around the claim in its function, and that
///    function runs once per execution (L is single-instance), and
///  - the base case cut no loop other than L at the bound, so every path
///    reached the k-th iteration of L, and ran L itself to bound k.
/// The base case records the loops it cut in "kind-base-cut-loops-running";
/// the sequential driver publishes them as "kind-base-cut-loops" = "k: n1 n2"
/// once the base case at k completes, before the step at the same k. Without
/// that record nothing converts.
/// Otherwise the claim is asserted on every unwinding (#2328, #11116).

#include <goto-symex/engine/goto_symex.h>
#include <util/base/i2string.h>
#include <sstream>

namespace
{
using loop_mapt = std::unordered_map<const void *, unsigned>;

/// Records, for each instruction inside a loop of \p body, the outermost loop
/// around it. A loop spans its head to its back edge.
void map_outermost_loops(const goto_programt &body, loop_mapt &outermost)
{
  for (auto it = body.instructions.begin(); it != body.instructions.end(); ++it)
  {
    if (!it->is_backwards_goto())
      continue;
    const unsigned loop = it->loop_number;
    // Visiting back edges in order, an enclosing loop's back edge comes after
    // its inner ones, so it overwrites them.
    for (auto i = *it->targets.begin(); i != std::next(it); ++i)
      outermost[&*i] = loop;
  }
}

using callerst = std::map<irep_idt, std::vector<irep_idt>>;

/// The direct call sites of each function, by calling function. A call inside
/// a loop runs the callee many times, so it is recorded with an empty caller.
callerst find_callers(const goto_functionst &functions, const loop_mapt &outer)
{
  callerst callers;
  for (const auto &[name, function] : functions.function_map)
    for (const auto &i : function.body.instructions)
    {
      if (!i.is_function_call())
        continue;
      const expr2tc &callee = to_code_function_call2t(i.code).function;
      if (is_symbol2t(callee))
        callers[to_symbol2t(callee).thename].push_back(
          outer.count(&i) ? irep_idt() : name);
    }
  return callers;
}

/// The loops that run as one instance per execution: the outermost loops of
/// functions entered once. A function is entered once when it is the entry
/// point, or when it has exactly one direct call site, outside every loop, in
/// a function entered once. Functions called only through a pointer or as a
/// thread have no direct call site and do not qualify.
std::set<unsigned> find_single_instance_loops(
  const goto_functionst &functions,
  const loop_mapt &outermost)
{
  const callerst callers = find_callers(functions, outermost);
  std::map<irep_idt, bool> once;
  std::function<bool(const irep_idt &)> entered_once =
    [&](const irep_idt &f) -> bool {
    if (f == "__ESBMC_main")
      return true;
    if (auto it = once.find(f); it != once.end())
      return it->second;
    once[f] = false; // recursion is not entered once
    const auto it = callers.find(f);
    const bool result = it != callers.end() && it->second.size() == 1 &&
                        !it->second.front().empty() &&
                        entered_once(it->second.front());
    return once[f] = result;
  };

  std::set<unsigned> loops;
  for (const auto &[name, function] : functions.function_map)
  {
    if (!entered_once(name))
      continue;
    for (const auto &i : function.body.instructions)
      if (
        i.is_backwards_goto() && outermost.count(&i) &&
        outermost.at(&i) == i.loop_number)
        loops.insert(i.loop_number);
  }
  return loops;
}

std::set<unsigned> parse_loop_set(const std::string &s)
{
  std::set<unsigned> loops;
  std::istringstream in(s);
  for (unsigned n; in >> n;)
    loops.insert(n);
  return loops;
}
} // namespace

void goto_symext::setup_inductive_claims()
{
  // Under --termination an UNSAT step reports non-termination, so a claim
  // turned into an assumption could only produce a false verdict.
  if (
    !inductive_step || depth_limit != 0 ||
    options.get_bool_option("k-induction-parallel") ||
    options.get_bool_option("termination"))
    return;

  std::string cut;
  if (!options.get_option("kind-base-cut-loops", cut))
    return;
  // The set is only valid for the bound the base case ran at.
  const std::string::size_type colon = cut.find(':');
  if (cut.substr(0, colon) != options.get_option("unwind"))
    return;
  base_case_cut_loops = parse_loop_set(cut.substr(colon + 1));

  auto outermost = std::make_shared<loop_mapt>();
  for (const auto &[name, function] : goto_functions.function_map)
    map_outermost_loops(function.body, *outermost);
  single_instance_loops = std::make_shared<std::set<unsigned>>(
    find_single_instance_loops(goto_functions, *outermost));
  outermost_loop = std::move(outermost);
}

void goto_symext::record_base_case_cut(unsigned loop)
{
  if (!base_case || !cut_loops.insert(loop).second)
    return;
  options.set_option(
    "kind-base-cut-loops-running",
    options.get_option("kind-base-cut-loops-running") + " " + i2string(loop));
}

bool goto_symext::claim_is_induction_hypothesis()
{
  if (!base_case_cut_loops || cur_state->source.pc->inductive_assertion)
    return false;

  // The claim's loop L is the outermost loop around it, possibly through the
  // call sites on the stack. A claim outside every loop has no hypothesis.
  unsigned loop = 0;
  if (
    auto it = outermost_loop->find(&*cur_state->source.pc);
    it != outermost_loop->end())
    loop = it->second;
  for (auto frame = cur_state->call_stack.rbegin();
       frame != cur_state->call_stack.rend();
       ++frame)
  {
    const symex_targett::sourcet &call = frame->calling_location;
    if (!call.is_set)
      continue;
    if (auto it = outermost_loop->find(&*call.pc); it != outermost_loop->end())
      loop = it->second;
  }

  if (loop == 0 || !single_instance_loops->count(loop))
    return false;
  for (unsigned cut : *base_case_cut_loops)
    if (cut != loop)
      return false;

  const BigInt bound = loop_unwind_bound(loop);
  const auto it = cur_state->loop_iterations.find(loop);
  const BigInt unwind = it == cur_state->loop_iterations.end() ? 0 : it->second;
  return bound != 0 && unwind < bound - 1;
}

bool goto_symext::claim_strengthens_inductive_step()
{
  // Where the claim cannot be the hypothesis, it is still assumed after being
  // asserted on all but the last unwinding: this strengthens the step without
  // dropping the obligation (#2328). Claims from the bidirectional search
  // are left alone.
  return inductive_step && first_loop &&
         !cur_state->source.pc->inductive_assertion &&
         cur_state->loop_iterations[first_loop] < max_unwind - 1;
}
