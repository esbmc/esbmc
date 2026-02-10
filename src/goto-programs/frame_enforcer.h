/// \file frame_enforcer.h
/// \brief Operational Frame Rule enforcement for inductive verification.
///
/// This module implements the "Snapshot → Havoc → Assume(Unchanged == Snapshot)"
/// pattern that bridges the inductive gap in k-induction loop verification.
///
/// The frame rule ensures that variables NOT in the explicit assigns set
/// retain their pre-havoc values, enabling verification of properties that
/// depend on historical state (e.g., array shift patterns, sliding windows).
///
/// Key operations:
/// 1. materialize_snapshots: Capture pre-state before havoc
/// 2. enforce_frame_rule: Constrain post-havoc state for unassigned variables
/// 3. replace_old_with_snapshots: Support old() references in invariants

#ifndef GOTO_PROGRAMS_FRAME_ENFORCER_H
#define GOTO_PROGRAMS_FRAME_ENFORCER_H

#include <goto-programs/goto_program.h>
#include <util/context.h>
#include <irep2/irep2_expr.h>
#include <map>
#include <vector>

class frame_enforcert
{
public:
  /// \brief Snapshot entry: maps an original expression to its snapshot symbol
  struct snapshot_entryt
  {
    expr2tc original_expr; ///< The variable being snapshotted (e.g., "i")
    expr2tc
      snapshot_sym; ///< The snapshot symbol (e.g., "__ESBMC_frame_snap_loop_0_i")
  };

  frame_enforcert(contextt &_context);

  /// \brief Materialize snapshots for a set of variables.
  /// Generates DECL + ASSIGN instructions that capture the current value of
  /// each variable into a fresh snapshot symbol. Call this BEFORE havoc.
  ///
  /// \param vars_to_snapshot Variables to create snapshots for
  /// \param dest GOTO program to append snapshot instructions to
  /// \param loc Source location for generated instructions
  /// \param scope_prefix Unique prefix for snapshot symbol names (e.g., "loop_0")
  void materialize_snapshots(
    const std::vector<expr2tc> &vars_to_snapshot,
    goto_programt &dest,
    const locationt &loc,
    const std::string &scope_prefix);

  /// \brief Enforce frame rule: add ASSUME constraints for unassigned variables.
  /// For each snapshotted variable NOT in explicit_assigns, generates
  /// ASSUME(var == snapshot_var). Call this AFTER havoc.
  ///
  /// Paper semantics:
  /// ∀ v ∈ ModSet \ AssignsSet, assume(v_new = v_old)
  ///
  /// \param explicit_assigns Variables explicitly allowed to change
  /// \param dest GOTO program to append ASSUME instructions to
  /// \param loc Source location for generated instructions
  void enforce_frame_rule(
    const std::vector<expr2tc> &explicit_assigns,
    goto_programt &dest,
    const locationt &loc);

  /// \brief Replace old() references in an expression with snapshot variables.
  /// Walks the expression tree and replaces any symbol matching a snapshotted
  /// variable's original with the corresponding snapshot symbol.
  ///
  /// \param expr Expression potentially containing old() references
  /// \return Expression with old() replaced by snapshot symbols
  expr2tc replace_old_with_snapshots(const expr2tc &expr) const;

  /// \brief Get the active snapshots (for debugging/inspection)
  const std::vector<snapshot_entryt> &get_active_snapshots() const
  {
    return active_snapshots;
  }

private:
  contextt &context;

  /// Active snapshot entries from the most recent materialize_snapshots call
  std::vector<snapshot_entryt> active_snapshots;

  /// Counter for generating unique snapshot names
  size_t snapshot_counter;

  /// \brief Create a snapshot symbol in the symbol table.
  /// Follows the same pattern as contracts.cpp create_snapshot_variable:
  /// symbolt → context.move_symbol_to_context → symbol2tc
  ///
  /// \param original Expression to snapshot (determines type)
  /// \param prefix Scope prefix for naming
  /// \param index Unique index within this scope
  /// \return symbol2tc expression for the snapshot variable
  expr2tc create_snapshot_symbol(
    const expr2tc &original,
    const std::string &prefix,
    size_t index);
};

#endif /* GOTO_PROGRAMS_FRAME_ENFORCER_H */
