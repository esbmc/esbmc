#pragma once

#include <goto-programs/goto_functions.h>
#include <irep2/irep2.h>
#include <util/config/options.h>
#include <util/irep/location.h>
#include <util/symtab/context.h>
#include <util/symtab/namespace.h>

#include <iosfwd>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct ts_propertyt
{
  /// Holds exactly when this property is violated in the step.
  expr2tc violated;
  std::string comment;
  locationt location;
};

/// A program `prefix; for(;;) body` as a transition system over the loop's
/// live state. Expressions are SSA; symbols in `step_local` are instantiated
/// per step by at_step, everything else is shared by all steps.
class transition_systemt
{
public:
  std::vector<expr2tc> prefix_defs, prefix_assumes;
  std::vector<ts_propertyt> prefix_bad;

  /// Aligned: name, value entering the loop, state at the head, state at the
  /// back edge.
  std::vector<std::string> state_names;
  std::vector<expr2tc> state_init, state_pre, state_post;

  std::vector<std::pair<expr2tc, std::string>> inputs;

  std::vector<expr2tc> body_defs, body_assumes;
  std::vector<ts_propertyt> bad;
  expr2tc back_guard;

  /// Facts that only the inductive step may assume: the loop entry condition
  /// and interval-analysis bounds.
  std::vector<expr2tc> invariants;

  std::unordered_set<expr2tc, irep2_hash> step_local;

  expr2tc at_step(const expr2tc &e, unsigned step) const;
  void dump(std::ostream &out, const namespacet &ns) const;

  bool has_local(const expr2tc &e) const;

private:
  mutable std::unordered_map<const expr2t *, bool> has_local_memo;
};

/// Instantiates expressions at one step, sharing the rewrite across calls so
/// subterms common to several expressions are copied once.
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

/// Recognise a single unbounded loop in main (after the k-induction
/// transformation) and extract it. Returns false with `reason` set when the
/// program is not of that shape.
bool extract_transition_system(
  const goto_functionst &goto_functions,
  contextt &context,
  const optionst &options,
  bool live_filter,
  transition_systemt &ts,
  std::string &reason);
