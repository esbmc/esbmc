/// \file frame_enforcer.cpp
/// \brief Implementation of the Operational Frame Rule for verification.
///
/// This implements the core "Snapshot → Havoc/Call → Assume/Assert(Unchanged == Snapshot)"
/// mechanism. See frame_enforcer.h for detailed documentation.

#include "frame_enforcer.h"
#include <util/irep/migrate.h>
#include <util/irep/std_expr.h>
#include <util/lang/c_types.h>
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
    decl_inst->code = code_decl2tc(var->type, to_symbol2t(snap_sym).thename);
    decl_inst->location = loc;
    decl_inst->location.comment("frame: snapshot declaration");

    // 4. Generate ASSIGN instruction: snap_var = var
    goto_programt::targett assign_inst = dest.add_instruction(ASSIGN);
    assign_inst->code = code_assign2tc(snap_sym, var);
    assign_inst->location = loc;
    assign_inst->location.comment("frame: capture pre-state");
  }
}

static const expr2tc &strip_typecasts(const expr2tc &e)
{
  const expr2tc *leaf = &e;
  while (is_typecast2t(*leaf))
    leaf = &to_typecast2t(*leaf).from;
  return *leaf;
}

// The parameter a path is rooted at, and the field of *that* parameter the path
// goes through: `o->sub->a` is rooted at `o` through `sub`. Recording that much
// lets the per-field check hold every other field of `*o` unchanged, which is
// what catches a write to `o->x`. What happens under `o->sub` is not covered --
// the pointee of a field is not a parameter, so Phase 2C has nothing to root a
// snapshot at (github_7055_assigns_multilevel_inner_knownbug).
static bool root_pointer_field(const expr2tc &e, irep_idt &ptr, irep_idt &field)
{
  if (!is_member2t(e))
    return false;

  // A cast anywhere along the path -- `((Inner *)o->sub)->a` -- must not lose
  // the root, or the target falls back to direct_targets and no obligation is
  // generated at all.
  const member2t &mem = to_member2t(e);
  const expr2tc &src = strip_typecasts(mem.source_value);
  if (!is_dereference2t(src))
    return false;

  const expr2tc &under = strip_typecasts(to_dereference2t(src).value);
  if (is_symbol2t(under))
  {
    ptr = to_symbol2t(under).thename;
    field = mem.member;
    return true;
  }

  return root_pointer_field(under, ptr, field);
}

frame_enforcert::classified_assignst frame_enforcert::classify_assigns_targets(
  const std::vector<expr2tc> &explicit_assigns)
{
  classified_assignst result;

  for (const auto &target : explicit_assigns)
  {
    if (
      is_index2t(target) && is_symbol2t(to_index2t(target).source_value) &&
      is_array_type(to_index2t(target).source_value->type))
    {
      // global[i]: record the index so the assertion can spare that element.
      // As a direct target it matched nothing, and the whole array was asserted
      // unchanged, which the named write itself falsifies (#7056).
      // Tested before the pointer case: an element of an array *of pointers*
      // is pointer-typed, and classing it as a pointer target left the whole
      // array asserted unchanged, so even a body writing only the named
      // element failed.
      const index2t &idx = to_index2t(target);
      result.array_elem_targets[to_symbol2t(idx.source_value).thename]
        .push_back(idx.index);
    }
    else if (is_pointer_type(target))
    {
      // Pointer-typed symbol: Clang simplified &(*ptr) to ptr
      result.pointer_targets.push_back(target);
    }
    else if (is_dereference2t(target))
    {
      // Explicit dereference: extract the pointer operand
      result.pointer_targets.push_back(to_dereference2t(target).value);
    }
    else if (is_member2t(target))
    {
      const member2t &mem = to_member2t(target);
      if (is_symbol2t(mem.source_value))
      {
        // global_struct.field: record for per-field global compliance checking
        irep_idt struct_name = to_symbol2t(mem.source_value).thename;
        result.struct_field_targets[struct_name].insert(mem.member);
      }
      else if (is_dereference2t(mem.source_value))
      {
        // ptr->field: member2t(dereference2t(ptr_sym), field)
        // Record for per-field pointer compliance checking.
        const dereference2t &deref = to_dereference2t(mem.source_value);
        if (is_symbol2t(deref.value))
        {
          irep_idt ptr_name = to_symbol2t(deref.value).thename;
          result.ptr_field_targets[ptr_name].insert(mem.member);
        }
        else
        {
          // A path through more than one pointer, e.g. `o->sub->a`. Record the
          // parameter it is rooted at and the field it enters, so the per-field
          // check still holds every other field of that parameter unchanged.
          // Left in direct_targets it matched no snapshot and no obligation was
          // generated at all, so a body writing outside its frame verified
          // (#7055).
          irep_idt root_ptr, root_field;
          if (root_pointer_field(target, root_ptr, root_field))
            result.ptr_field_targets[root_ptr].insert(root_field);
          else
            result.direct_targets.push_back(target);
        }
      }
      else
      {
        result.direct_targets.push_back(target);
      }
    }
    else
    {
      result.direct_targets.push_back(target);
    }
  }

  return result;
}

/// Helper: emit a single ASSUME or ASSERT instruction with the given guard.
static void emit_frame_instruction(
  goto_programt &dest,
  const locationt &loc,
  const expr2tc &guard,
  frame_modet mode,
  const std::string &var_name)
{
  goto_program_instruction_typet inst_type =
    (mode == frame_modet::ASSERT) ? ASSERT : ASSUME;
  goto_programt::targett t = dest.add_instruction(inst_type);
  t->guard = guard;
  t->location = loc;
  if (mode == frame_modet::ASSERT)
  {
    t->location.comment(
      "assigns compliance: " + var_name + " not in assigns clause");
    t->location.property("assigns compliance");
  }
  else
  {
    t->location.comment("frame: preserve unassigned variable");
  }
}

// One assertion per element costs more than linearly: 256 elements solve in
// 0.6s, 512 in 2.4s, 1000 in 16s, and 10000 does not finish inside the
// regression timeout.
static const unsigned max_elementwise_frame_extent = 256;

// The array equals its snapshot with only the named indices replaced by their
// current values. Exact, and one equality whatever the extent, so it is what
// carries an array too large to hold element by element.
//
// It does not replace the element-wise form, for three reasons measured on
// this branch: it reports the array rather than the element that broke the
// frame, Bitwuzla rejects equality over constant arrays ("not fully supported
// yet") so it cannot serve the loop rule's ASSUME mode, and an element that is
// itself an array fails on every solver -- reading an array-typed rvalue is
// the same gap that stops __ESBMC_old over an array (#7057).
static void emit_array_store_frame(
  goto_programt &dest,
  const locationt &loc,
  frame_modet mode,
  const std::vector<expr2tc> &assigned_indices,
  const expr2tc &var,
  const expr2tc &snap,
  const irep_idt &arr_name)
{
  const array_type2t &atype = to_array_type(var->type);
  expr2tc updated = snap;
  for (const expr2tc &assigned : assigned_indices)
  {
    // An index outside the array excuses nothing, which is what comparing the
    // index against each element gives for free. Fed to a store it would not:
    // the solver's index domain is only as wide as the extent, so an index
    // past the end wraps and spares some unrelated element instead.
    expr2tc in_range = and2tc(
      greaterthanequal2tc(assigned, gen_zero(assigned->type)),
      lessthan2tc(assigned, typecast2tc(assigned->type, atype.array_size)));

    updated = if2tc(
      var->type,
      in_range,
      with2tc(
        var->type, updated, assigned, index2tc(atype.subtype, var, assigned)),
      updated);
  }

  emit_frame_instruction(
    dest, loc, equality2tc(var, updated), mode, id2string(arr_name));
}

// Compare the scalar leaves, not whole rows: a row of a multi-dimensional
// array is an array-typed rvalue, which no solver reads correctly (#7057).
static void emit_leaf_equalities(
  goto_programt &dest,
  const locationt &loc,
  frame_modet mode,
  const expr2tc &var_elem,
  const expr2tc &snap_elem,
  const expr2tc &exemption,
  const std::string &label)
{
  if (is_array_type(var_elem->type))
  {
    const array_type2t &at = to_array_type(var_elem->type);
    const BigInt &n = to_constant_int2t(at.array_size).value;
    for (BigInt j = 0; j < n; j += 1)
    {
      expr2tc jc = constant_int2tc(at.array_size->type, j);
      emit_leaf_equalities(
        dest,
        loc,
        mode,
        index2tc(at.subtype, var_elem, jc),
        index2tc(at.subtype, snap_elem, jc),
        exemption,
        label + "[" + integer2string(j) + "]");
    }
    return;
  }

  expr2tc guard = equality2tc(var_elem, snap_elem);
  if (!is_nil_expr(exemption))
    guard = or2tc(guard, exemption);
  emit_frame_instruction(dest, loc, guard, mode, label);
}

// Hold every element the clause did not name unchanged, rather than the array
// as a whole -- which the named write itself falsifies (#7056). A global array
// has a constant extent, so this needs neither a witness index nor a
// quantifier: one assertion per element, each excused at the named indices.
// Returns whether \p var was of that shape and so is now dealt with; what
// neither form covers falls back to the whole-array assertion.
static bool emit_array_elem_frame(
  goto_programt &dest,
  const locationt &loc,
  frame_modet mode,
  const frame_enforcert::classified_assignst &classified,
  const expr2tc &var,
  const expr2tc &snap)
{
  if (!is_symbol2t(var) || !is_array_type(var->type))
    return false;

  const irep_idt &arr_name = to_symbol2t(var).thename;
  auto ait = classified.array_elem_targets.find(arr_name);
  if (ait == classified.array_elem_targets.end())
    return false;

  const array_type2t &atype = to_array_type(var->type);
  if (!is_constant_int2t(atype.array_size))
    return false;

  const BigInt &n = to_constant_int2t(atype.array_size).value;
  if (n > BigInt(max_elementwise_frame_extent))
  {
    if (mode != frame_modet::ASSERT || is_array_type(atype.subtype))
      return false;

    emit_array_store_frame(dest, loc, mode, ait->second, var, snap, arr_name);
    return true;
  }

  // Descending to the leaves multiplies the assertion count, so the budget is
  // measured over the leaves, and a nested dimension of unknown extent has
  // none to count.
  BigInt leaves = n;
  for (type2tc sub = atype.subtype; is_array_type(sub);
       sub = to_array_type(sub).subtype)
  {
    const array_type2t &s = to_array_type(sub);
    if (!is_constant_int2t(s.array_size))
      return false;
    leaves = leaves * to_constant_int2t(s.array_size).value;
  }
  if (leaves > BigInt(max_elementwise_frame_extent))
    return false;

  for (BigInt k = 0; k < n; k += 1)
  {
    expr2tc kc = constant_int2tc(atype.array_size->type, k);
    expr2tc exemption;

    for (const expr2tc &assigned : ait->second)
    {
      expr2tc named = equality2tc(typecast2tc(assigned->type, kc), assigned);
      exemption = is_nil_expr(exemption) ? named : or2tc(exemption, named);
    }

    emit_leaf_equalities(
      dest,
      loc,
      mode,
      index2tc(atype.subtype, var, kc),
      index2tc(atype.subtype, snap, kc),
      exemption,
      id2string(arr_name) + "[" + integer2string(k) + "]");
  }
  return true;
}

// Assert only that the struct fields NOT in the assigns clause are unchanged.
// Example: __ESBMC_assigns(global_pt.x) allows global_pt.x to change but holds
// global_pt.y, global_pt.z, ... to their pre-state.
static bool emit_struct_field_frame(
  goto_programt &dest,
  const locationt &loc,
  frame_modet mode,
  const frame_enforcert::classified_assignst &classified,
  const expr2tc &var,
  const expr2tc &snap)
{
  if (!is_symbol2t(var) || !is_struct_type(var))
    return false;

  const irep_idt &sym_name = to_symbol2t(var).thename;
  auto sit = classified.struct_field_targets.find(sym_name);
  if (sit == classified.struct_field_targets.end())
    return false;

  const struct_type2t &stype = to_struct_type(var->type);
  for (size_t i = 0; i < stype.member_names.size(); ++i)
  {
    const irep_idt &field = stype.member_names[i];
    if (sit->second.count(field))
      continue; // This field is explicitly assigned — skip

    const type2tc &ftype = stype.members[i];
    expr2tc field_guard =
      equality2tc(member2tc(ftype, var, field), member2tc(ftype, snap, field));

    emit_frame_instruction(
      dest,
      loc,
      field_guard,
      mode,
      id2string(sym_name) + "." + id2string(field));
  }
  return true;
}

// Hold the parts of \p var the clause did not name unchanged, rather than \p
// var as a whole: named struct fields, or named array elements.
// Returns whether \p var was one of those shapes and so is now dealt with.
static bool emit_partial_frame(
  goto_programt &dest,
  const locationt &loc,
  frame_modet mode,
  const frame_enforcert::classified_assignst &classified,
  const expr2tc &var,
  const expr2tc &snap)
{
  return emit_struct_field_frame(dest, loc, mode, classified, var, snap) ||
         emit_array_elem_frame(dest, loc, mode, classified, var, snap);
}

// Whether the whole-object constraint must be withheld because the clause
// named elements of \p var that no encoding above could express.
//
// Asserting the whole array is merely imprecise: it reports the write the
// clause itself permits, a false positive. Assuming it is not. An assumption
// the clause contradicts is a hypothesis stronger than the truth, and anything
// proved under it may be false -- a loop writing an element it was granted
// verified an assertion that element falsifies. Withholding the constraint
// only weakens the hypothesis, which can cost a proof but cannot invent one.
static bool withhold_whole_object_frame(
  frame_modet mode,
  const frame_enforcert::classified_assignst &classified,
  const expr2tc &var)
{
  return mode == frame_modet::ASSUME && is_symbol2t(var) &&
         is_array_type(var->type) &&
         classified.array_elem_targets.count(to_symbol2t(var).thename) != 0;
}

void frame_enforcert::enforce_frame_rule(
  const std::vector<expr2tc> &explicit_assigns,
  goto_programt &dest,
  const locationt &loc,
  frame_modet mode)
{
  // Classify assigns targets for aliasing analysis
  classified_assignst classified = classify_assigns_targets(explicit_assigns);

  // A clause names its targets in the pre-state: `__ESBMC_assigns(buf[head])`
  // grants the element `head` denoted on entry. The array is snapshotted but
  // the index used to be read back after the body, so a body that moved `head`
  // -- itself in the clause, as the ring-buffer idiom needs -- chose after the
  // fact which element it had been granted. An off-by-one write verified while
  // the correct body was reported.
  for (auto &[name, indices] : classified.array_elem_targets)
    for (expr2tc &index : indices)
      index = in_pre_state(index);

  for (const auto &entry : active_snapshots)
  {
    const expr2tc &var = entry.original_expr;
    const expr2tc &snap = entry.snapshot_sym;

    // Check if this variable is directly in the assigns set
    bool is_assigned = false;
    for (const auto &direct : classified.direct_targets)
    {
      if (var == direct)
      {
        is_assigned = true;
        break;
      }
    }

    // If directly assigned, skip — no constraint needed
    if (is_assigned)
      continue;

    if (emit_partial_frame(dest, loc, mode, classified, var, snap))
      continue;

    if (withhold_whole_object_frame(mode, classified, var))
      continue;

    // Build the base guard: var == snapshot (unchanged condition)
    expr2tc guard = equality2tc(var, snap);

    // In ASSERT mode, add aliasing disjunctions for pointer targets.
    // For each pointer p in pointer_targets whose pointed-to type matches
    // var's type, add: guard = guard || (p == &var)
    // This means: "var is unchanged OR some pointer in the assigns set aliases it"
    if (mode == frame_modet::ASSERT)
    {
      for (const auto &ptr : classified.pointer_targets)
      {
        // Check type compatibility: pointer's subtype must match var's type
        if (
          is_pointer_type(ptr) &&
          to_pointer_type(ptr->type).subtype == var->type)
        {
          // address_of2tc(subtype, obj): first arg is subtype, NOT pointer type
          expr2tc addr_of_var = address_of2tc(var->type, var);
          expr2tc alias_check = equality2tc(ptr, addr_of_var);
          guard = or2tc(guard, alias_check);
        }
      }
    }

    // Emit the (possibly disjunctive) whole-variable guard
    std::string var_name = "unknown";
    if (is_symbol2t(var))
      var_name = id2string(to_symbol2t(var).thename);
    emit_frame_instruction(dest, loc, guard, mode, var_name);
  }
}

std::vector<expr2tc>
frame_enforcert::collect_global_variables(const contextt &context)
{
  std::vector<expr2tc> globals;

  context.foreach_operand([&globals](const symbolt &s) {
    // Skip functions, types, and non-lvalue symbols
    if (s.get_type().is_code() || s.is_type || !s.lvalue)
      return;

    // Only process static lifetime variables (globals and static locals)
    if (!s.static_lifetime)
      return;

    // Skip internal ESBMC symbols
    std::string sym_name = id2string(s.name);
    if (sym_name.starts_with("__ESBMC_"))
      return;

    // Build symbol expression
    type2tc global_type = migrate_symbol_type(s);
    expr2tc sym_expr = symbol2tc(global_type, s.id);

    // Skip pointer types (consistent with loop frame rule behavior)
    if (is_pointer_type(sym_expr))
      return;

    globals.push_back(sym_expr);
  });

  return globals;
}

void frame_enforcert::patch_old_snapshot_assigns(goto_programt &prog) const
{
  if (active_snapshots.empty())
    return;

  for (auto &instr : prog.instructions)
  {
    if (!instr.is_assign())
      continue;

    const code_assign2t &assign = to_code_assign2t(instr.code);

    // Check RHS is an old_snapshot side effect
    if (!is_sideeffect2t(assign.source))
      continue;
    const sideeffect2t &effect = to_sideeffect2t(assign.source);
    if (effect.kind != sideeffect2t::allockind::old_snapshot)
      continue;

    // The operand of old_snapshot is the original variable
    const expr2tc &operand = effect.operand;

    // Find matching snapshot entry by symbol identifier
    for (const auto &entry : active_snapshots)
    {
      if (!is_symbol2t(operand) || !is_symbol2t(entry.original_expr))
        continue;
      if (
        to_symbol2t(operand).thename !=
        to_symbol2t(entry.original_expr).thename)
        continue;

      // Replace RHS with (void*)&snapshot_sym
      // address_of2tc(T, snap) produces type T*; typecast to lhs type (void*)
      expr2tc addr =
        address_of2tc(entry.snapshot_sym->type, entry.snapshot_sym);
      expr2tc patched_rhs = (addr->type != assign.target->type)
                              ? typecast2tc(assign.target->type, addr)
                              : addr;
      instr.code = code_assign2tc(assign.target, patched_rhs);
      break;
    }
  }
}

expr2tc frame_enforcert::in_pre_state(const expr2tc &expr) const
{
  if (is_nil_expr(expr))
    return expr;

  if (is_symbol2t(expr))
  {
    const irep_idt &name = to_symbol2t(expr).thename;
    for (const auto &entry : active_snapshots)
      if (
        is_symbol2t(entry.original_expr) &&
        to_symbol2t(entry.original_expr).thename == name)
        return entry.snapshot_sym;
    return expr;
  }

  expr2tc result = expr->clone();
  bool modified = false;
  result->Foreach_operand([this, &modified](expr2tc &op) {
    expr2tc replaced = in_pre_state(op);
    if (replaced != op)
    {
      op = replaced;
      modified = true;
    }
  });

  return modified ? result : expr;
}

expr2tc frame_enforcert::replace_old_with_snapshots(const expr2tc &expr) const
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

  return modified ? result : expr;
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
  set_symbol_type(snapshot_symbol, original->type);
  snapshot_symbol.lvalue = true;
  snapshot_symbol.static_lifetime = false;
  snapshot_symbol.file_local = false;

  // Register in symbol table
  symbolt *added = context.move_symbol_to_context(snapshot_symbol);

  // Return IRep2 symbol expression
  return symbol2tc(original->type, added->id);
}
