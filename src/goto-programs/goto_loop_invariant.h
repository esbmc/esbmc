#ifndef GOTO_PROGRAMS_GOTO_LOOP_INVARIANT_H_
#define GOTO_PROGRAMS_GOTO_LOOP_INVARIANT_H_

#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_loops.h>
#include <util/guard.h>
#include <irep2/irep2_expr.h>

void goto_loop_invariant(goto_functionst &goto_functions);

class goto_loop_invariantt : public goto_loopst
{
public:
  goto_loop_invariantt(
    const irep_idt &_function_name,
    goto_functionst &_goto_functions,
    goto_functiont &_goto_function)
    : goto_loopst(_function_name, _goto_functions, _goto_function)
  {
    if (function_loops.size())
      goto_loop_invariant();
  }

protected:
  void goto_loop_invariant();

  void convert_loop_with_invariant(loopst &loop);

  // Extract loop invariants from LOOP_INVARIANT instructions
  std::vector<expr2tc> extract_loop_invariants(const loopst &loop);

  // Insert ASSERT invariant before loop
  void insert_assert_before_loop(
    goto_programt::targett &loop_head,
    const std::vector<expr2tc> &invariants);

  // Insert HAVOC and ASSUME after loop condition
  void insert_havoc_and_assume_after_condition(
    goto_programt::targett &loop_head,
    const loopst &loop,
    const std::vector<expr2tc> &invariants);

  // Insert inductive step verification and loop termination
  void insert_inductive_step_and_termination(
    const loopst &loop,
    const std::vector<expr2tc> &invariants);
};

#endif /* GOTO_PROGRAMS_GOTO_LOOP_INVARIANT_H_ */ 