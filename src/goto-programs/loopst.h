#ifndef GOTO_PROGRAMS_LOOPST_H_
#define GOTO_PROGRAMS_LOOPST_H_

#include <goto-programs/goto_functions.h>
#include <unordered_set>

class loopst
{
public:
  loopst() : size(0)
  {
  }

  typedef std::unordered_set<expr2tc, irep2_hash> loop_varst;

  const loop_varst &get_modified_loop_vars() const;
  const loop_varst &get_unmodified_loop_vars() const;

  const goto_programt::targett get_original_loop_exit() const;
  void set_original_loop_exit(goto_programt::targett _loop_exit);

  const goto_programt::targett get_original_loop_head() const;
  void set_original_loop_head(goto_programt::targett _loop_head);

  /// Effective loop head: the first non-inert instruction at-or-after
  /// `original_loop_head`. Inert here means an instruction that does
  /// not change control flow and that a structural recogniser would
  /// otherwise reject — SKIP, LOCATION, DECL, DEAD, and ASSUME.
  ///
  /// Motivation: `--interval-analysis` (and other passes that use
  /// `insert_swap`) insert ASSUME(bounds) instructions at the back-
  /// edge target. The back-edge then lands on the ASSUME, and
  /// `get_original_loop_head()` returns that ASSUME rather than the
  /// loop's IF. Structural recognisers (eca's `recognize_eca_main_loop`,
  /// the ranking certifier's `recognize_loop`, etc.) need to find the
  /// actual control-flow instruction; calling this helper instead of
  /// `get_original_loop_head()` does the right skip in one place.
  ///
  /// Returns `original_loop_exit` if every instruction in
  /// [original_loop_head, original_loop_exit) is inert (degenerate
  /// loop with no body besides the back-edge); callers should handle
  /// that case explicitly.
  goto_programt::targett effective_loop_head() const;

  void add_modified_var_to_loop(const expr2tc &expr);
  void add_unmodified_var_to_loop(const expr2tc &expr);

  /// Record that the loop writes an array element through a pointer
  /// (e.g. `p[i] = ...`, `(*p)[i] = ...`, `p->arr[i] = ...`). The
  /// k-induction inductive step havocs only named symbols (see
  /// make_nondet_assign); it cannot havoc such pointer-reached array
  /// storage, so the inductive hypothesis is under-generalised and the
  /// inductive step becomes unsound. The strategy layer disables the
  /// inductive step when any loop reports this. Stack arrays (havoc'd as
  /// whole symbols) and single pointer/field writes (constrained by the
  /// value-set assume) are sound and therefore excluded. See issue #5224.
  void set_modifies_pointer_array()
  {
    modifies_pointer_array_ = true;
  }

  /// True iff the loop writes an array element through a pointer.
  bool modifies_pointer_array() const
  {
    return modifies_pointer_array_;
  }

  /// Record a pointer whose pointee array is written *directly* in this
  /// loop's body (e.g. `(*dest)[i] = ...` records `dest`, `p[i] = ...`
  /// records `p`). Phase 2 (issue #5230) resolves these pointers against
  /// the value-set fixpoint and havocs the referenced named objects, so
  /// the inductive step can stay enabled and sound instead of being
  /// disabled outright as in Phase 1 (#5224).
  void add_pointer_array_write_ptr(const expr2tc &ptr)
  {
    pointer_array_write_ptrs_.insert(ptr);
  }

  const loop_varst &get_pointer_array_write_ptrs() const
  {
    return pointer_array_write_ptrs_;
  }

  /// Record that the loop contains a pointer-array write that cannot be
  /// resolved at the loop head — the write happens inside a callee (the
  /// pointer is a callee parameter, not in scope at the caller's loop
  /// head) or the written pointer could not be extracted from the LHS.
  /// Phase 2 must then fall back to the Phase 1 behaviour and disable the
  /// inductive step. See issue #5230.
  void set_pointer_array_write_unresolvable()
  {
    pointer_array_write_unresolvable_ = true;
  }

  bool pointer_array_write_unresolvable() const
  {
    return pointer_array_write_unresolvable_;
  }

  void dump() const;
  void dump_loop_vars() const;
  void output_to(std::ostream &oss) const;
  void output_loop_vars_to(std::ostream &oss) const;

  void set_size(std::size_t size)
  {
    this->size = size;
  }

protected:
  loop_varst modified_loop_vars;
  loop_varst unmodified_loop_vars;

  goto_programt::targett original_loop_head;
  goto_programt::targett original_loop_exit;

  std::size_t size;
  bool modifies_pointer_array_ = false;
  bool pointer_array_write_unresolvable_ = false;
  loop_varst pointer_array_write_ptrs_;
};

#endif /* GOTO_PROGRAMS_LOOPST_H_ */
