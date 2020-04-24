// Rafael Sá Menezes - 03/2020

#include <cache/algorithms/ssa_step_algorithm_debug.h>

void ssa_step_algorithm_debug::run_on_assignment(
  symex_target_equationt::SSA_stept &step)
{
  std::cout << "this is an assignment\n";
  step.dump();
}
void ssa_step_algorithm_debug::run_on_assert(
  symex_target_equationt::SSA_stept &step)
{
  std::cout << "this is an assert\n";
  step.dump();
}
void ssa_step_algorithm_debug::run_on_assume(
  symex_target_equationt::SSA_stept &step)
{
  std::cout << "this is an assume\n";
  step.dump();
}
void ssa_step_algorithm_debug::run_on_output(
  symex_target_equationt::SSA_stept &step)
{
  std::cout << "this is an output\n";
  step.dump();
}
void ssa_step_algorithm_debug::run_on_skip(
  symex_target_equationt::SSA_stept &step)
{
  std::cout << "this is a skip\n";
  step.dump();
}
void ssa_step_algorithm_debug::run_on_renumber(
  symex_target_equationt::SSA_stept &step)
{
  std::cout << "this is a renumber\n";
  step.dump();
}
