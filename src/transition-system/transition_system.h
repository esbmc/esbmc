#pragma once

#include <irep2/irep2.h>
#include <util/irep/location.h>
#include <util/symtab/namespace.h>

#include <iosfwd>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

/** Transition system here represents a sequential circuit. They works as:
 *  1. An initial state.
 *  2. A "next state" function that runs in a loop forever.
 *  3. For verification purposes. The circuit has invariant/pre/post conditions.
 *
 *     (init) -> (pre) -> (body) -> (post)
 *                 ^                  |
 *                 \__________________/   when back_guard holds
 *
 * For example:
 *
 *     n = nondet();
 *     assume(n > 0);
 *     x = 0;
 *     assert(n < 100);     // prefix
 *     for (;;) {           // loop head
 *       y = nondet();      // body
 *       assume(y < 10);
 *       assert(x < 4);
 *       x = (x + y) & 3;
 *     }                    // back edge
 **/
/** Collect every `symbol2t` in @p e into @p out, treating @p seen as a shared
 *  visited set so a DAG node reached by many parents is walked once. Unlike
 *  irep2_utils' get_symbols this keeps `__ESBMC_` symbols: the memory-model
 *  bookkeeping they name is a real dependency of the formula.
 */
void collect_symbols(
  const expr2tc &e,
  std::unordered_set<expr2tc, irep2_hash> &out,
  std::unordered_set<const expr2t *> &seen);

class transition_systemt
{
public:
  struct propertyt
  {
    expr2tc violated;
    std::string comment;
    locationt location;
  };

  // Prefix are the commands before the loop starts
  // defs -> assignments, assumes -> assumptions, bad -> assertions
  std::vector<expr2tc> prefix_defs, prefix_assumes;
  std::vector<propertyt> prefix_bad;

  // Inductive state at the beginning (init), before (pre) and after (post)
  // each loop. Only variables the properties depend on are kept.
  struct statet
  {
    std::string name;
    expr2tc init, pre, post;
  };
  std::vector<statet> states;

  /** Fresh nondeterministic symbol read by the step. */
  std::vector<std::pair<expr2tc, std::string>> inputs;

  // Same as the prefix but for loop body
  std::vector<expr2tc> body_defs, body_assumes;

  /** `violated` already folds in the assumptions (`!(assumes -> cond)`), so it
   * is SAT exactly when that claim can fail in this step.
   */
  std::vector<propertyt> bad;

  /** Guard under which the back edge is reached */
  expr2tc back_guard;

  /** Facts only the inductive step may assume: the loop entry condition and
   * interval-analysis bounds.
   */
  std::vector<expr2tc> invariants;

  using step_localt = std::unordered_set<expr2tc, irep2_hash>;

  /** Symbols instantiated per iteration. Setting them drops the has_local()
   * cache, which is only valid while this set is unchanged.
   */
  void set_step_local(step_localt syms);
  const step_localt &get_step_local() const
  {
    return step_local;
  }

  /** Rewrites `e` for iteration `step` */
  expr2tc at_step(const expr2tc &e, unsigned step) const;

  void dump(std::ostream &out, const namespacet &ns) const;

  /** Does `e` mention a step_local symbol, i.e. does it change between steps?
   */
  bool has_local(const expr2tc &e) const;

private:
  step_localt step_local;
  mutable std::unordered_map<const expr2t *, bool> has_local_memo;
};

/** Instantiates expressions at one step, sharing the rewrite across calls so
 * subterms common to several expressions are copied once.
 */
class ts_step_renamert
{
public:
  ts_step_renamert(const transition_systemt &ts, unsigned step);
  expr2tc operator()(const expr2tc &e);

private:
  const transition_systemt &ts;
  const std::string suffix;
  std::unordered_map<const expr2t *, expr2tc> memo;
};
