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
/// A lock is taken only if every earlier lock of its mutex was released
/// before it. An execution in which a thread waits forever for a lock is not
/// one of these, so its violations are not reported.
/// The constraints go first in the equation, so every claim depends on them.
/// Each event keeps its clock in its cond, for the trace to order steps by.
/// With \p races, a claim at the end of the equation for each pair of
/// conflicting events of different threads states that no execution places
/// them next to each other.
/// With \p deadlocks, the events whose clock is below a free stop time form a
/// prefix of the execution, and only the locks in it need a release before
/// them. A claim states that no prefix leaves main, and every other thread
/// that has events left, waiting on a held mutex or an unfinished thread.
/// Every other claim then holds only over executions that run to the end.
class memory_model_sct : public ssa_step_algorithm
{
public:
  explicit memory_model_sct(bool races = false, bool deadlocks = false)
    : ssa_step_algorithm(true), races(races), deadlocks(deadlocks)
  {
  }

  bool run(symex_target_equationt::SSA_stepst &steps) override;

  BigInt ignored() const override
  {
    return 0;
  }

  /// The data-race and deadlock claims the last run() appended.
  size_t claims_added() const override
  {
    return n_claims;
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
  std::map<irep_idt, std::vector<size_t>> locks, unlocks;
  symex_target_equationt::SSA_stepst constraints, claims;
  const bool races;
  const bool deadlocks;
  size_t n_claims = 0;

  void collect(symex_target_equationt::SSA_stepst &steps);
  void program_order();
  void write_order();
  void read_from();
  void thread_order();
  void atomic_sections();
  void mutexes();
  void data_races();
  void claim_no_race(size_t a, size_t b);
  void claim_no_deadlock();
  void complete_runs_only(symex_target_equationt::SSA_stepst &steps) const;
  expr2tc finished(unsigned thread) const;
  expr2tc waits(size_t id) const;
  expr2tc done(size_t id) const;
  void isolate(size_t begin, size_t end);
  void constrain(const expr2tc &cond, const eventt &about);
  expr2tc before(size_t a, size_t b) const;
  bool same_thread(size_t a, size_t b) const;
};
