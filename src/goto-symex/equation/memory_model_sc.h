#pragma once

#include <goto-symex/equation/ssa_step_algorithm.h>

#include <map>
#include <vector>

/// The sequentially consistent memory model of --partial-order (#10990).
///
/// Gives every event a clock and prepends the constraints that make the
/// clocks an SC execution: program order within a thread, a total order on
/// the writes to each object, and for each read a choice of the write it
/// reads from, which must be the latest write to that object before the read.
/// A write that updates part of an object, such as an array element, also
/// reads the object, atomically with its write.
/// A spawned thread's events follow its spawn and precede every join on it,
/// and no other thread's event falls inside an atomic section.
/// The constraints go first in the equation, so every claim depends on them.
class memory_model_sct : public ssa_step_algorithm
{
public:
  memory_model_sct() : ssa_step_algorithm(true)
  {
  }

  bool run(symex_target_equationt::SSA_stepst &steps) override;

  BigInt ignored() const override
  {
    return 0;
  }

private:
  struct eventt
  {
    const symex_target_equationt::SSA_stept *step;
    expr2tc clock;
  };

  std::vector<eventt> events;
  std::map<unsigned, std::vector<size_t>> per_thread;
  std::map<irep_idt, std::vector<size_t>> reads, writes;
  std::vector<size_t> spawns, joins;
  symex_target_equationt::SSA_stepst constraints;

  void collect(const symex_target_equationt::SSA_stepst &steps);
  void program_order();
  void write_order();
  void read_from();
  void thread_order();
  void atomic_sections();
  void isolate(size_t begin, size_t end);
  void constrain(const expr2tc &cond, const eventt &about);
  expr2tc before(size_t a, size_t b) const;
  bool same_thread(size_t a, size_t b) const;
};
