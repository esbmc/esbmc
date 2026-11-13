#ifndef ESBMC_GOTO_PROGRAMS_GOTO_SLICER_H
#define ESBMC_GOTO_PROGRAMS_GOTO_SLICER_H

#include <goto-programs/goto_functions.h>

#include <memory>
#include <string>
#include <unordered_set>

class andersent;
class optionst;

/// \file
/// GOTO-level slicer: finds the instructions, loops and calls no property
/// depends on, before symex. Flow- and context-insensitive cone over abstract
/// locations, with dereferences resolved by Andersen's analysis.

enum class slice_stept
{
  base_case, ///< also plain BMC and falsification
  forward_condition,
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
  /// The backward GOTO of each collapsed loop.
  std::unordered_set<const goto_programt::instructiont *> collapsed;

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

#endif
