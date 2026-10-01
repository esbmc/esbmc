#include <goto-symex/equation/ssa_step_algorithm.h>
#include <util/message/message.h>

void ssa_step_algorithm::run_on_step(symex_target_equationt::SSA_stept &step)
{
  // Helper definition
  typedef symex_targett::step_typet ssa_type;
  switch (step.type)
  {
  case ssa_type::ASSIGNMENT:
    run_on_assignment(step);
    break;
  case ssa_type::ASSUME:
    run_on_assume(step);
    break;
  case ssa_type::ASSERT:
    run_on_assert(step);
    break;
  case ssa_type::OUTPUT:
    run_on_output(step);
    break;
  case ssa_type::RENUMBER:
    run_on_renumber(step);
    break;
  case ssa_type::SKIP:
    run_on_skip(step);
    break;
  case ssa_type::BREANCHING:
    run_on_branching(step);
    break;
  default:
    log_warning("Invalid type for SSA step");
    abort();
  }
}
