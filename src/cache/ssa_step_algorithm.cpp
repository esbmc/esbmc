//
// Created by rafaelsa on 04/02/2020.
//

#include <cache/ssa_step_algorithm.h>

void ssa_step_algorithm::run()
{
  typedef goto_trace_stept::typet ssa_type;
  typedef symex_target_equationt::SSA_stept ssa_step;
  typedef std::function<void(ssa_step &)> ssa_function;

  static std::map<ssa_type, ssa_function> run_on_function;
  static bool map_initialized = false;

  if(!map_initialized)
  {
    run_on_function[ssa_type::ASSIGNMENT] = [this](ssa_step &step) {
      run_on_assignment(step);
    };
    run_on_function[ssa_type::ASSUME] = [this](ssa_step &step) {
      run_on_assume(step);
    };
    run_on_function[ssa_type::ASSERT] = [this](ssa_step &step) {
      run_on_assert(step);
    };
    run_on_function[ssa_type::OUTPUT] = [this](ssa_step &step) {
      run_on_output(step);
    };
    run_on_function[ssa_type::RENUMBER] = [this](ssa_step &step) {
      run_on_renumber(step);
    };
    run_on_function[ssa_type::SKIP] = [this](ssa_step &step) {
      run_on_skip(step);
    };

    map_initialized = true;
  }

  for(auto s : steps)
  {
    run_on_function[s.type](s);
  }
}
