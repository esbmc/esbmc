#ifndef GOTO_PROGRAMS_GOTO_LOOP_INVARIANT_H_
#define GOTO_PROGRAMS_GOTO_LOOP_INVARIANT_H_

#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_loops.h>
#include <util/guard.h>
#include <util/context.h>
#include <irep2/irep2_expr.h>

/// \brief Entry point: process loop invariants for all functions.
/// When use_frame_rule is true, enables the Operational Frame Rule
/// (Snapshot → Havoc → Assume) for enhanced inductive verification.
void goto_loop_invariant(
  goto_functionst &goto_functions,
  contextt &context,
  bool use_frame_rule);

class goto_loop_invariantt : public goto_loopst
{
public:
  goto_loop_invariantt(
    const irep_idt &_function_name,
    goto_functionst &_goto_functions,
    goto_functiont &_goto_function,
    contextt &_context,
    bool _use_frame_rule)
    : goto_loopst(_function_name, _goto_functions, _goto_function),
      context(_context),
      use_frame_rule(_use_frame_rule)
  {
    if (function_loops.size())
      goto_loop_invariant();
  }

protected:
  contextt &context;
  bool use_frame_rule;

  void goto_loop_invariant();

  void convert_loop_with_invariant(loopst &loop);

  // Extract loop invariants from LOOP_INVARIANT instructions
  std::vector<expr2tc> extract_loop_invariants(const loopst &loop);

  // Extract loop assigns targets from LOOP_INVARIANT instructions
  std::vector<expr2tc> extract_loop_assigns(const loopst &loop);

  // Insert ASSERT invariant before loop
  void insert_assert_before_loop(
    goto_programt::targett &loop_head,
    const std::vector<expr2tc> &invariants);

  // Insert HAVOC and ASSUME before loop condition
  void insert_havoc_and_assume_before_condition(
    goto_programt::targett &loop_head,
    const loopst &loop,
    const std::vector<expr2tc> &invariants,
    const std::vector<expr2tc> &loop_assigns);

  // Insert inductive step verification and loop termination
  void insert_inductive_step_and_termination(
    const loopst &loop,
    const std::vector<expr2tc> &invariants);
};

#endif /* GOTO_PROGRAMS_GOTO_LOOP_INVARIANT_H_ */
