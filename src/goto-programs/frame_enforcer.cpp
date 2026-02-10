/// \file frame_enforcer.cpp
/// \brief Implementation of the Operational Frame Rule for inductive verification.
///
/// This implements the core "Snapshot → Havoc → Assume(Unchanged == Snapshot)"
/// mechanism. See frame_enforcer.h for detailed documentation.

#include "frame_enforcer.h"
#include <util/migrate.h>
#include <util/std_expr.h>
#include <util/c_types.h>
#include <irep2/irep2_utils.h>

frame_enforcert::frame_enforcert(contextt &_context)
  : context(_context), snapshot_counter(0)
{
}

void frame_enforcert::materialize_snapshots(
  const std::vector<expr2tc> &vars_to_snapshot,
  goto_programt &dest,
  const locationt &loc,
  const std::string &scope_prefix)
{
  active_snapshots.clear();

  for (const auto &var : vars_to_snapshot)
  {
    // 1. Create snapshot symbol in symbol table
    expr2tc snap_sym =
      create_snapshot_symbol(var, scope_prefix, snapshot_counter++);

    // 2. Record mapping for later use in enforce_frame_rule and replace_old
    snapshot_entryt entry;
    entry.original_expr = var;
    entry.snapshot_sym = snap_sym;
    active_snapshots.push_back(entry);

    // 3. Generate DECL instruction for the snapshot variable
    goto_programt::targett decl_inst = dest.add_instruction(DECL);
    decl_inst->code =
      code_decl2tc(var->type, to_symbol2t(snap_sym).thename);
    decl_inst->location = loc;
    decl_inst->location.comment("frame: snapshot declaration");

    // 4. Generate ASSIGN instruction: snap_var = var
    goto_programt::targett assign_inst = dest.add_instruction(ASSIGN);
    assign_inst->code = code_assign2tc(snap_sym, var);
    assign_inst->location = loc;
    assign_inst->location.comment("frame: capture pre-state");
  }
}

void frame_enforcert::enforce_frame_rule(
  const std::vector<expr2tc> &explicit_assigns,
  goto_programt &dest,
  const locationt &loc)
{
  for (const auto &entry : active_snapshots)
  {
    const expr2tc &var = entry.original_expr;
    const expr2tc &snap = entry.snapshot_sym;

    // Check if this variable is in the explicit assigns set
    bool is_assigned = false;
    for (const auto &assigned_var : explicit_assigns)
    {
      if (var == assigned_var)
      {
        is_assigned = true;
        break;
      }
    }

    // If not in assigns set, it must retain its pre-havoc value
    // Paper: ∀ v ∈ ModSet \ AssignsSet, assume(v_new = v_old)
    if (!is_assigned)
    {
      expr2tc eq = equality2tc(var, snap);
      goto_programt::targett t = dest.add_instruction(ASSUME);
      t->guard = eq;
      t->location = loc;
      t->location.comment("frame: preserve unassigned variable");
    }
  }
}

expr2tc
frame_enforcert::replace_old_with_snapshots(const expr2tc &expr) const
{
  if (is_nil_expr(expr))
    return expr;

  // If this is a symbol, check if it matches a snapshotted variable
  if (is_symbol2t(expr))
  {
    const symbol2t &sym = to_symbol2t(expr);
    std::string sym_name = id2string(sym.thename);

    // Check for __ESBMC_old pattern (from function contracts old() support)
    if (sym_name.find("___ESBMC_old") != std::string::npos)
    {
      for (const auto &entry : active_snapshots)
      {
        if (is_symbol2t(entry.original_expr))
        {
          const symbol2t &orig_sym = to_symbol2t(entry.original_expr);
          if (sym.thename == orig_sym.thename)
          {
            return entry.snapshot_sym;
          }
        }
      }
    }
  }

  // Recursively process sub-expressions
  expr2tc result = expr->clone();
  bool modified = false;

  result->Foreach_operand([this, &modified](expr2tc &op) {
    expr2tc new_op = replace_old_with_snapshots(op);
    if (new_op != op)
    {
      op = new_op;
      modified = true;
    }
  });

  return result;
}

expr2tc frame_enforcert::create_snapshot_symbol(
  const expr2tc &original,
  const std::string &prefix,
  size_t index)
{
  // Generate unique snapshot variable name
  std::string snapshot_name =
    "__ESBMC_frame_snap_" + prefix + "_" + std::to_string(index);

  // Create symbol using IRep1 types (symbol table is IRep1-based)
  symbolt snapshot_symbol;
  snapshot_symbol.name = snapshot_name;
  snapshot_symbol.id = snapshot_name;
  snapshot_symbol.type = migrate_type_back(original->type);
  snapshot_symbol.lvalue = true;
  snapshot_symbol.static_lifetime = false;
  snapshot_symbol.file_local = false;

  // Register in symbol table
  symbolt *added = context.move_symbol_to_context(snapshot_symbol);

  // Return IRep2 symbol expression
  return symbol2tc(original->type, added->id);
}
