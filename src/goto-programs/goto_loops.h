#ifndef GOTO_PROGRAMS_GOTO_LOOPS_H_
#define GOTO_PROGRAMS_GOTO_LOOPS_H_

#include <goto-programs/goto_functions.h>
#include <goto-programs/loopst.h>
#include <util/std_types.h>
#include <unordered_map>

/// True for symbols that name genuine user storage — excludes ESBMC
/// internals, return-value temporaries, k-induction-generated names, etc.
/// Defined in goto_loops.cpp; declared here so the k-induction pass can
/// reuse it when filtering value-set objects (issue #5230).
bool check_var_name(const expr2tc &expr);

/// True iff `expr` denotes storage reached through a pointer (a dereference,
/// or an index/member off pointer-reached storage, or a pointer-typed symbol
/// being indexed) — i.e. an array-element write the inductive step cannot
/// havoc as a named symbol. Defined in goto_loops.cpp; declared here so the
/// k-induction pass can pre-scan for such writes (issue #5230).
bool indexes_through_pointer(const expr2tc &expr);

class goto_loopst
{
protected:
  irep_idt function_name;
  goto_functionst &goto_functions;
  goto_functiont &goto_function;

  typedef std::list<loopst> function_loopst;
  function_loopst function_loops;

  /// Per-callee summary: the leaf symbols (post check_var_name filtering)
  /// that walking the callee's body would contribute to the current loop.
  /// Cached across loops in the same goto_loopst instance so two loops
  /// in the *same* outer function don't re-walk a shared helper.
  ///
  /// Scope is intentionally per-instance (one goto_loopst per analysed
  /// function), not per-program: goto_k_induction constructs a fresh
  /// instance per outer function, and promoting the cache to a static
  /// would need invalidation across the goto-program rewrites that
  /// happen between functions.
  struct function_summaryt
  {
    loopst::loop_varst modified;
    loopst::loop_varst unmodified;
    /// True iff the callee writes an array element through a pointer.
    /// Propagated to the calling loop so the inductive step is disabled
    /// (see loopst::set_modifies_pointer_array and issue #5224).
    bool modifies_pointer_array = false;
  };
  std::unordered_map<irep_idt, function_summaryt, irep_id_hash>
    function_summary_cache;

  void create_function_loop(
    goto_programt::instructionst::iterator loop_head,
    goto_programt::instructionst::iterator loop_exit);

  void get_modified_variables(
    goto_programt::instructionst::iterator instruction,
    function_loopst::iterator loop,
    std::vector<irep_idt> &function_name);

  /// Compute (or fetch the cached) summary of `fname`. `in_progress` is the
  /// stack of callees currently being expanded; if a re-entry is detected
  /// the walk is cut (matches the legacy in-place behaviour). Returns true
  /// when the resulting summary is complete (no cycle-cut along the way);
  /// only complete summaries are cached.
  bool compute_function_summary(
    const irep_idt &fname,
    std::vector<irep_idt> &in_progress,
    function_summaryt &out);

  /// Collect the leaf symbols of `expr` into `out`, applying check_var_name.
  void collect_loop_symbols(const expr2tc &expr, loopst::loop_varst &out) const;

  /// Walk an assignment LHS, classifying each leaf symbol: storage that
  /// is actually written goes to `modified`, sub-expressions used to
  /// locate that storage (pointer in `*p`, index in `arr[i]`) go to
  /// `unmodified`. Sets `modifies_pointer_array` when the write is to an
  /// array element reached through a pointer (issue #5224).
  void collect_lhs_symbols(
    const expr2tc &expr,
    loopst::loop_varst &modified,
    loopst::loop_varst &unmodified,
    bool &modifies_pointer_array) const;

  void add_modified_var(loopst &loop, const expr2tc &expr);
  void add_unmodified_var(loopst &loop, const expr2tc &expr);

  void add_loop_var(loopst &loop, const expr2tc &expr, bool is_modified);

public:
  goto_loopst(
    const irep_idt &_function_name,
    goto_functionst &_goto_functions,
    goto_functiont &_goto_function)
    : function_name(_function_name),
      goto_functions(_goto_functions),
      goto_function(_goto_function)
  {
    find_function_loops();
  }

  void find_function_loops();
  void dump() const;

  const function_loopst &get_loops() const
  {
    return function_loops;
  }

  function_loopst &get_loops()
  {
    return function_loops;
  }
};

#endif /* GOTO_PROGRAMS_GOTO_LOOPS_H_ */
