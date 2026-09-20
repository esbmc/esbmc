#ifndef GOTO_PROGRAMS_GOTO_LOOPS_H_
#define GOTO_PROGRAMS_GOTO_LOOPS_H_

#include <goto-programs/goto_functions.h>
#include <goto-programs/loopst.h>
#include <util/irep/std_types.h>
#include <unordered_map>

/// True for symbols that name genuine user storage — excludes ESBMC
/// internals, return-value temporaries, k-induction-generated names, etc.
/// Defined in goto_loops.cpp; declared here so the k-induction pass can
/// reuse it when filtering value-set objects (issue #5230).
bool check_var_name(const expr2tc &expr);

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
    /// True iff the callee writes an array element through a pointer, which
    /// havocking through the call's arguments cannot cover (#5224).
    bool modifies_pointer_array = false;
    /// True iff the callee writes through a dereference.
    /// See loopst::set_writes_through_pointer and issue #7478.
    bool writes_through_pointer = false;
    /// True iff the callee may write through a pointer it was not handed, so
    /// havocing the pointees of the call's arguments misses the write.
    bool pointer_write_unresolvable = false;
    /// The pointers the callee writes through, in its own scope, and whether
    /// it writes through one it cannot name. See loopst::add_written_pointer.
    loopst::loop_varst written_pointers;
    bool unnamed_write = false;

    void record_write(const expr2tc &lhs);
    void bind_arguments(
      const goto_functiont &callee,
      const std::vector<expr2tc> &arguments);
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

  static void
  merge_summary(const function_summaryt &from, function_summaryt &out);

  static void note_pointer_write(
    const expr2tc &target,
    function_summaryt &local,
    std::vector<expr2tc> &written_ptrs);

  /// Fold a call inside a callee into that callee's summary. Returns false
  /// when the summary is incomplete, as compute_function_summary does.
  bool summarise_call(
    const code_function_call2t &call,
    std::vector<irep_idt> &in_progress,
    function_summaryt &local,
    std::vector<expr2tc> &written_ptrs);

  static void apply_callee_summary(
    loopst &loop,
    const function_summaryt &summary,
    const code_function_call2t &call);

  /// Collect the leaf symbols of `expr` into `out`, applying check_var_name.
  void collect_loop_symbols(const expr2tc &expr, loopst::loop_varst &out) const;

  /// Walk an assignment LHS, classifying each leaf symbol: storage that
  /// is actually written goes to `modified`, sub-expressions used to
  /// locate that storage (pointer in `*p`, index in `arr[i]`) go to
  /// `unmodified`. Sets `modifies_pointer_array` when the write is to an
  /// array element reached through a pointer (issue #5224).
  void collect_lhs_symbols(const expr2tc &expr, function_summaryt &out) const;

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
