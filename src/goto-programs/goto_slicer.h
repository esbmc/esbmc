#ifndef ESBMC_GOTO_PROGRAMS_GOTO_SLICER_H
#define ESBMC_GOTO_PROGRAMS_GOTO_SLICER_H

#include <goto-programs/goto_functions.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

class andersent;
class optionst;

/// \file
/// GOTO-level slicer: finds the instructions, loops and calls no property
/// depends on, before symex. Flow- and context-insensitive cone over abstract
/// locations, with dereferences resolved by Andersen's analysis.

enum class slice_stept
{
  /// Also plain BMC, falsification and the forward condition. The forward
  /// condition asks whether the base case's loops are exhausted, so it must
  /// run the base case's program, not one sliced for fewer claims.
  base_case,
  inductive_step
};

/// What a slice must preserve, read from the options in effect.
struct slice_criteriont
{
  slice_stept step = slice_stept::base_case;
  bool pointer_check = true;
  bool memory_leak_check = false;
  bool no_reachable_memory_leak = false;
  bool no_assertions = false;
  /// Keep every nondet assignment: a witness or test case reports them.
  bool keep_nondets = false;
  /// Replace a loop whose body is sliced by an assumption. Off where the run
  /// reports a loop that outlives its bound, which a collapsed loop cannot.
  bool collapse_loops = true;

  static slice_criteriont
  from_options(const optionst &options, slice_stept step, bool keep_nondets);
};

struct slice_resultt
{
  unsigned instructions = 0;
  unsigned removed_instructions = 0;
  /// Kept only for a symex-time check, with only its address operands.
  unsigned check_only_instructions = 0;
  unsigned loops = 0;
  /// Loops whose body is sliced, so they reduce to an assumption.
  unsigned collapsed_loops = 0;
  unsigned calls = 0;
  unsigned removed_calls = 0;

  std::unordered_set<const goto_programt::instructiont *> removed;
  std::unordered_set<const goto_programt::instructiont *> check_only;
  /// The backward GOTO of each collapsed loop, and the condition under which
  /// the loop exits: it becomes ASSUME(condition).
  std::unordered_map<const goto_programt::instructiont *, expr2tc> collapsed;
  /// Check-only assignments whose value no one reads: their right-hand side
  /// becomes zero.
  std::unordered_set<const goto_programt::instructiont *> zeroed;

  std::string summary() const;
};

class goto_slicert
{
public:
  /// \p points_to must have been run on \p goto_functions, and must not
  /// report escapes(). Only functions reachable from the entry point are
  /// analysed.
  goto_slicert(const goto_functionst &goto_functions, andersent &points_to);
  ~goto_slicert();

  slice_resultt slice(const slice_criteriont &criterion) const;

  /// The flattened program, defined in goto_slicer.cpp.
  struct programt;

private:
  std::unique_ptr<programt> program;
};

/// Rewrites \p target, which is \p analysed or a copy_from copy of it, by
/// \p result: sliced instructions become SKIP, collapsed loops ASSUMEs.
/// Instructions are never added or removed, so location numbers still match.
void apply_slice(
  const goto_functionst &analysed,
  goto_functionst &target,
  const slice_resultt &result);

/// The program each proof step runs under --goto-slice, sliced on first use.
class goto_slice_cachet
{
public:
  /// With \p in_place, the one step this process runs slices \p program
  /// itself instead of a copy (a k-induction-parallel child).
  /// \p points_to has been run on \p program and is not needed afterwards.
  goto_slice_cachet(
    goto_functionst &program,
    const optionst &options,
    andersent &points_to,
    bool keep_nondets,
    bool in_place);
  ~goto_slice_cachet();

  /// Logs the slice of every step in \p steps.
  void report(const std::vector<slice_stept> &steps);

  goto_functionst &program_for(slice_stept step);

private:
  goto_functionst &original;
  const optionst &options;
  const bool keep_nondets;
  const bool in_place;
  std::unique_ptr<goto_slicert> slicer;
  std::map<slice_stept, slice_resultt> results;
  std::map<slice_stept, goto_functionst> programs;
  std::optional<slice_stept> sliced_in_place;

  const slice_resultt &result_for(slice_stept step);
};

#endif
