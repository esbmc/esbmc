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
};

#endif /* GOTO_PROGRAMS_LOOPST_H_ */
