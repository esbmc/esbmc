/// \file frame_enforcer.h
/// \brief Operational Frame Rule enforcement for verification.
///
/// This module implements the "Snapshot → Havoc → Assume/Assert(Unchanged == Snapshot)"
/// pattern used in two contexts:
///
/// 1. Loop invariants (ASSUME mode): Bridges the inductive gap in k-induction
///    by assuming unassigned variables retain pre-havoc values.
///
/// 2. Function contracts (ASSERT mode): Checks assigns clause compliance by
///    asserting that variables NOT in the assigns set are unchanged after
///    the function call.
///
/// Key operations:
/// 1. materialize_snapshots: Capture pre-state before havoc/call
/// 2. enforce_frame_rule: Constrain/check post-state for unassigned variables
/// 3. replace_old_with_snapshots: Support old() references in invariants
/// 4. collect_global_variables: Gather all accessible globals from symbol table

#ifndef GOTO_PROGRAMS_FRAME_ENFORCER_H
#define GOTO_PROGRAMS_FRAME_ENFORCER_H

#include <goto-programs/goto_program.h>
#include <util/context.h>
#include <irep2/irep2_expr.h>
#include <map>
#include <vector>

/// Enforcement mode: ASSUME constrains search space (loops), ASSERT checks compliance (contracts)
enum class frame_modet
{
  ASSUME,
  ASSERT
};

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
  /// each variable into a fresh snapshot symbol. Call this BEFORE havoc/call.
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

  /// \brief Enforce frame rule with configurable mode.
  /// For each snapshotted variable NOT in explicit_assigns, generates either
  /// ASSUME(var == snapshot_var) or ASSERT(var == snapshot_var).
  ///
  /// ASSUME mode (loops): constrains search space for k-induction
  /// ASSERT mode (contracts): checks assigns clause compliance
  ///
  /// \param explicit_assigns Variables explicitly allowed to change
  /// \param dest GOTO program to append instructions to
  /// \param loc Source location for generated instructions
  /// \param mode ASSUME for loops (default), ASSERT for contracts
  void enforce_frame_rule(
    const std::vector<expr2tc> &explicit_assigns,
    goto_programt &dest,
    const locationt &loc,
    frame_modet mode = frame_modet::ASSUME);

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

  /// \brief Collect all accessible global variables from the symbol table.
  /// Returns symbol2tc expressions for all static-lifetime lvalue symbols,
  /// excluding __ESBMC_* internal symbols, functions, and types.
  ///
  /// \param context Symbol table to scan
  /// \return Vector of symbol2tc expressions for global variables
  static std::vector<expr2tc> collect_global_variables(const contextt &context);

  /// \brief Classification of assigns targets into direct and pointer categories.
  /// Used to separate structurally-matchable targets from pointer-typed targets
  /// that require aliasing disjunctions.
  struct classified_assignst
  {
    std::vector<expr2tc> direct_targets;  ///< Non-pointer targets matched structurally
    std::vector<expr2tc> pointer_targets; ///< Pointer-typed targets from *ptr pattern
  };

  /// \brief Classify assigns targets into direct and pointer categories.
  /// - Pointer-typed symbols (from Clang simplifying &(*ptr) to ptr) → pointer_targets
  /// - Dereference expressions → extract pointer operand → pointer_targets
  /// - Everything else → direct_targets
  static classified_assignst
  classify_assigns_targets(const std::vector<expr2tc> &explicit_assigns);

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
