#include <goto-programs/goto_value_set_invariant.h>
#include <goto-programs/goto_loops.h>
#include <irep2/irep2_expr.h>
#include <util/expr_util.h>
#include <util/message.h>

/// Synthesise (p == &obj1 + off || p == &obj2 + off || ...) from the
/// value-set recorded at @p loop_head for @p ptr_var.
/// Returns a nil expr when the set is empty, contains an unknown offset,
/// or contains a non-descriptor element (we give up safely in those cases).
static expr2tc build_value_set_constraint(
  const expr2tc &ptr_var,
  goto_programt::const_targett loop_head,
  value_set_analysist &vsa)
{
  value_setst::valuest value_set;
  vsa.get_values(loop_head, ptr_var, value_set);

  if (value_set.empty())
    return expr2tc();

  expr2tc or_accuml;

  for (const auto &obj_expr : value_set)
  {
    // Value-set entries that are not object descriptors (e.g. unknown2t,
    // invalid2t) cannot be turned into a concrete address; skip them.
    if (!is_object_descriptor2t(obj_expr))
      continue;

    const object_descriptor2t &obj = to_object_descriptor2t(obj_expr);

    // Skip entries with unknown offset — we cannot form a precise constraint.
    if (is_unknown2t(obj.offset))
      continue;

    // dynamic_object2t represents heap memory (malloc).  address_of on a
    // dynamic object is not representable in the SMT encoding, so skip it.
    if (is_dynamic_object2t(obj.object))
      continue;

    expr2tc obj_ptr;
    if (is_null_object2t(obj.object))
    {
      type2tc nullptrtype = pointer_type2tc(ptr_var->type);
      obj_ptr = symbol2tc(nullptrtype, "NULL");
    }
    else
    {
      obj_ptr = add2tc(
        ptr_var->type,
        address_of2tc(ptr_var->type, obj.object),
        obj.offset);
    }

    // Use strict pointer equality (not SAME-OBJECT) so the invariant is
    // strong enough to imply assertions that compare pointer values with ==.
    expr2tc eq = equality2tc(ptr_var, obj_ptr);

    if (is_nil_expr(or_accuml))
      or_accuml = eq;
    else
      or_accuml = or2tc(or_accuml, eq);
  }

  return or_accuml;
}

class goto_inject_value_set_invariantst : public goto_loopst
{
public:
  goto_inject_value_set_invariantst(
    const irep_idt &_function_name,
    goto_functionst &_goto_functions,
    goto_functiont &_goto_function,
    value_set_analysist &_vsa,
    const namespacet &_ns)
    : goto_loopst(_function_name, _goto_functions, _goto_function),
      vsa(_vsa),
      ns(_ns)
  {
    if (!function_loops.empty())
      inject_invariants();
  }

private:
  value_set_analysist &vsa;
  const namespacet &ns;

  void inject_invariants()
  {
    for (auto &loop : function_loops)
      process_loop(loop);
  }

  void process_loop(loopst &loop)
  {
    goto_programt::targett loop_head = loop.get_original_loop_head();
    const auto &loop_vars = loop.get_modified_loop_vars();

    // Build a conjunction of per-pointer value-set constraints.
    expr2tc combined;
    for (const auto &var : loop_vars)
    {
      if (!is_pointer_type(var))
        continue;

      expr2tc constraint =
        build_value_set_constraint(var, loop_head, vsa);
      if (is_nil_expr(constraint))
        continue;

      combined =
        is_nil_expr(combined) ? constraint : and2tc(combined, constraint);
    }

    if (is_nil_expr(combined))
      return;

    // Inject a LOOP_INVARIANT instruction immediately before the loop head.
    // Use destructive_insert (splice) so that existing backward-GOTO targets
    // that point to loop_head are NOT redirected to the new instruction.
    // insert_swap would swap instruction content and silently move the loop
    // head target, causing goto_loop_invariant_combined to fail to find
    // the invariant when it searches backwards from loop_head.
    goto_programt dest;
    goto_programt::targett inv = dest.add_instruction(LOOP_INVARIANT);
    inv->location = loop_head->location;
    inv->location.comment("value-set auto-invariant");
    inv->add_loop_invariant(combined);

    goto_function.body.destructive_insert(loop_head, dest);

    log_status(
      "[value-set] Injected pointer invariant for loop in {}",
      id2string(function_name));
  }
};

void goto_inject_value_set_invariants(
  goto_functionst &goto_functions,
  value_set_analysist &vsa,
  const namespacet &ns)
{
  Forall_goto_functions(it, goto_functions)
    if (it->second.body_available)
      goto_inject_value_set_invariantst(
        it->first, goto_functions, it->second, vsa, ns);

  goto_functions.update();
}
