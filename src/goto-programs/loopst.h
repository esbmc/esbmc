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

  /// Record a pointer the loop writes through, or a pointer argument of a
  /// callee that does (`(*dest)[i] = ...` records `dest`, `p[i] = ...`
  /// records `p`). The loop-invariant schema havocs through these (#7478).
  void add_pointer_array_write_ptr(const expr2tc &ptr)
  {
    pointer_array_write_ptrs_.insert(ptr);
  }

  const loop_varst &get_pointer_array_write_ptrs() const
  {
    return pointer_array_write_ptrs_;
  }

  /// Record a write that havocking through the recorded pointers cannot
  /// cover: an element a callee writes past its argument, or a write whose
  /// pointer could not be extracted. The loop-invariant schema declines.
  void set_pointer_array_write_unresolvable()
  {
    pointer_array_write_unresolvable_ = true;
  }

  bool pointer_array_write_unresolvable() const
  {
    return pointer_array_write_unresolvable_;
  }

  /// Record that the loop, or a function it calls, writes through a
  /// dereference (`*p = ...`, `p->f = ...`, `p->e[i] = ...`). The pointee is
  /// not a named symbol, so a schema that havocs named symbols cannot cover it:
  /// k-induction havocs the objects the written pointers resolve to, and the
  /// loop-invariant schema declines a loop whose guard the havoc cannot reach
  /// (issue #7478).
  void set_writes_through_pointer()
  {
    writes_through_pointer_ = true;
  }

  bool writes_through_pointer() const
  {
    return writes_through_pointer_;
  }

  /// Record a pointer the loop, or a function it calls, writes through, in the
  /// scope of the function holding the dereference. A whole-program points-to
  /// analysis resolves these to the objects the inductive step must havoc.
  void add_written_pointer(const expr2tc &ptr)
  {
    written_pointers_.insert(ptr);
  }

  const loop_varst &get_written_pointers() const
  {
    return written_pointers_;
  }

  /// Record `*p` for a write that stays inside it (`*p = ...`, `p->f = ...`).
  /// While the loop leaves `p` alone, havocking `*p` covers the write even
  /// when `p` reaches memory with no name, such as the heap.
  void add_written_pointee(const expr2tc &pointee)
  {
    written_pointees_.insert(pointee);
  }

  const loop_varst &get_written_pointees() const
  {
    return written_pointees_;
  }

  /// Record a write through a pointer that neither add_written_pointer nor
  /// add_written_pointee can name: a conditional l-value, or a call through
  /// a function pointer.
  void set_unnamed_pointer_write()
  {
    unnamed_pointer_write_ = true;
  }

  bool unnamed_pointer_write() const
  {
    return unnamed_pointer_write_;
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
  bool pointer_array_write_unresolvable_ = false;
  bool writes_through_pointer_ = false;
  loop_varst pointer_array_write_ptrs_;
  loop_varst written_pointers_;
  loop_varst written_pointees_;
  bool unnamed_pointer_write_ = false;
};

#endif /* GOTO_PROGRAMS_LOOPST_H_ */
