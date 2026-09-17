#ifndef ESBMC_KIND_INVARIANTS_H
#define ESBMC_KIND_INVARIANTS_H

#include <goto-programs/goto_functions.h>
#include <util/config/options.h>
#include <util/symtab/namespace.h>
#include <map>
#include <set>
#include <vector>

/// A loop-invariant candidate for --adaptive-k-induction, tied to its loop by
/// the id stamp_loop_back_edges put on the loop's back edge. The back edge is
/// the one instruction of a loop goto_k_induction neither copies nor moves, so
/// the id names the same loop in the pristine program candidates are proved
/// on and in the transformed program that assumes them.
struct kind_candidatet
{
  unsigned loop;
  expr2tc expr;
  const char *source;
};

/// A loop-head state a forward-condition counterexample passes through.
struct loop_head_samplet
{
  struct valuet
  {
    expr2tc symbol;
    BigInt value;
    /// Whether the loop writes the variable.
    bool modified;
  };
  unsigned loop;
  std::vector<valuet> values;
};

/// Most candidates a loop's pool holds, whatever their source. Every one adds
/// an obligation to each probe round and a conjunct to the assumption the
/// others are checked under.
constexpr size_t kMaxCandidatesPerLoop = 48;

/// What proving a candidate pool established.
struct kind_proof_resultt
{
  /// Indices into the pool of candidates that hold at every reachable loop
  /// head, i.e. whose base case and inductive step both passed together.
  std::vector<size_t> proven;
  /// Whether the run that proved them also passed every assertion of the
  /// program, with each loop cut by its invariants.
  bool program_proved = false;
  /// Solver runs it took.
  size_t rounds = 0;
};

/// Guess candidates for every stamped loop the loop-invariant schema can cut
/// soundly: affine closed forms, an assertion at the exit lifted to the head,
/// the floor of each ranking measure the guard gives, the literal/relational
/// templates and, @p with_intervals, interval bounds at the head. The interval analysis is only run when asked for: it
/// shares --interval-analysis's defects on programs that flag would crash on.
std::vector<kind_candidatet> generate_kind_candidates(
  goto_functionst &goto_functions,
  const optionst &options,
  const namespacet &ns,
  bool with_intervals);

/// Attach pool[i] for every i in @p ids whose loop the schema can cut as a
/// LOOP_INVARIANT tagged with i, and `true` on every other loop it can cut, so
/// none of them is left to the unwinder with its assertions unproved. Returns
/// the ids attached: a candidate that was not has no claim to be refuted by.
std::set<size_t> emit_kind_candidates(
  goto_functionst &goto_functions,
  const std::vector<kind_candidatet> &pool,
  const std::vector<size_t> &ids);

/// Insert one inductive-step-only ASSUME(true) per stamped loop of a program
/// goto_k_induction has transformed, where the havoc'd state enters the loop
/// and where every iteration returns; returns them by loop id. Each carries
/// its loop's stamp, so the other steps, which skip it, can still tell a
/// loop-head visit.
std::map<unsigned, goto_programt::targett>
insert_kind_placeholders(goto_functionst &goto_functions);

#endif
