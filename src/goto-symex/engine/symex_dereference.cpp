#include <goto-symex/engine/goto_symex.h>
#include <goto-symex/scheduler/reachability_tree.h>
#include <langapi/language_util.h>
#include <pointer-analysis/dereference.h>
#include <irep2/irep2.h>
#include <util/irep/migrate.h>

void symex_dereference_statet::dereference_failure(
  const std::string &property [[maybe_unused]],
  const std::string &msg,
  const guard2tc &guard)
{
  expr2tc g = goto_symex.split_rewrite(guard.as_expr());
  goto_symex.replace_dynamic_allocation(g);
  goto_symex.simplify_python_builtins(g);
  goto_symex.claim(not2tc(g), "dereference failure: " + msg);
}

void symex_dereference_statet::dereference_assume(const guard2tc &guard)
{
  expr2tc g = goto_symex.split_rewrite(guard.as_expr());
  goto_symex.replace_dynamic_allocation(g);
  goto_symex.simplify_python_builtins(g);
  goto_symex.assume(not2tc(g));
}

bool symex_dereference_statet::has_failed_symbol(
  const expr2tc &expr,
  const symbolt *&symbol)
{
  if (is_symbol2t(expr))
  {
    // Null and invalid name lookups will fail.
    if (
      to_symbol2t(expr).thename == "NULL" ||
      to_symbol2t(expr).thename == "INVALID")
      return false;

    const symbolt &ptr_symbol =
      *goto_symex.ns.lookup(to_symbol2t(expr).thename);

    const irep_idt &failed_symbol = ptr_symbol.get_type().failed_symbol();

    if (failed_symbol == "")
      return false;

    const symbolt *s = goto_symex.ns.lookup(failed_symbol);
    if (!s)
      return false;
    symbol = s;
    return true;
  }

  return false;
}

static bool is_pure_integer(const expr2tc &e)
{
  if (!is_bv_type(e) || is_dereference2t(e))
    return false;
  bool pure = true;
  e->foreach_operand([&pure](const expr2tc &op) {
    if (pure && !is_pure_integer(op))
      pure = false;
  });
  return pure;
}

/* The value set is keyed by level1 names, so `p + i` has an unknown offset
 * even where `i` has a propagated constant value; the reference is then built
 * at POINTER_OFFSET(p) + i and folds no read (#2256). */
static void fold_propagated_integers(goto_symex_statet &state, expr2tc &e)
{
  if (is_nil_expr(e) || is_constant_expr(e))
    return;
  if (is_pure_integer(e))
  {
    expr2tc value = e;
    state.rename(value);
    simplify(value);
    if (is_constant_int2t(value))
      e = value;
    return;
  }
  e->Foreach_operand(
    [&state](expr2tc &op) { fold_propagated_integers(state, op); });
}

static bool is_scalar(const type2tc &t)
{
  return is_bv_type(t) || is_fixedbv_type(t) || is_floatbv_type(t) ||
         is_pointer_type(t);
}

/* Whether @p descriptor is NULL or an array whose elements are read whole by
 * an access of @p type. */
static bool is_element_access(const type2tc &type, const expr2tc &descriptor)
{
  if (!is_object_descriptor2t(descriptor))
    return true;
  const expr2tc &root = to_object_descriptor2t(descriptor).get_root_object();
  if (is_null_object2t(root))
    return true;
  if (!is_array_type(root))
    return false;
  const type2tc &sub = to_array_type(root->type).subtype;
  return is_scalar(sub) && sub->get_width() == type->get_width();
}

void symex_dereference_statet::points_to(
  const expr2tc &expr,
  value_setst::valuest &value_set)
{
  const expr2tc pointer = goto_symex.split_rewrite(expr);
  state.value_set.get_value_set(pointer, value_set, true);

  // Only for whole-element reads of arrays: a known offset into an aggregate,
  // or one cutting across elements, takes dereference paths that the symbolic
  // offset has kept out of reach, and the latter slows the solver.
  if (!is_pointer_type(pointer))
    return;
  const type2tc &type = to_pointer_type(pointer->type).subtype;
  if (
    !is_scalar(type) ||
    !std::all_of(value_set.begin(), value_set.end(), [&type](const expr2tc &d) {
      return is_element_access(type, d);
    }))
    return;
  expr2tc folded = pointer;
  fold_propagated_integers(state, folded);
  if (folded == pointer)
    return;
  value_set.clear();
  state.value_set.get_value_set(folded, value_set, true);
}

void symex_dereference_statet::get_value_set(
  const expr2tc &pointer,
  value_setst::valuest &value_set)
{
  // Here we obtain the set of objects via value set analysis.
  points_to(pointer, value_set);

  // add value set objects during the symbolic execution.
  if (
    goto_symex.options.get_bool_option("add-symex-value-sets") &&
    goto_symex.options.get_bool_option("inductive-step"))
  {
    if (value_set.empty())
      return;
    const expr2tc expr = goto_symex.split_rewrite(pointer);

    if (is_pointer_type(expr))
    {
      // we will accumulate the objects that the pointer points to.
      expr2tc or_accuml;
      // Append `eq` to `or_accuml`, building up a disjunction.
      auto add_disjunct = [&or_accuml](const expr2tc &eq) {
        or_accuml = or_accuml ? or2tc(or_accuml, eq) : eq;
      };

      // add each object to the resulting assume statement.
      for (auto it = value_set.begin(); it != value_set.end(); ++it)
      {
        // The value-set may contain `unknown`/`invalid` entries
        // alongside concrete object_descriptors. Previously we
        // returned at the first non-descriptor and dropped the
        // whole assume; that left the freshly-nondet pointer
        // unconstrained even when the analysis had identified
        // concrete candidates. Skip the non-descriptor entries
        // instead. This is sound: the pre-havoc value-set is a
        // sound over-approximation, but we are choosing to
        // assume that the pointer points at one of the concrete
        // candidates we *can* name in the SMT encoding — any
        // model that routed through the `unknown` sink before is
        // expressible by one of those candidates if it survived
        // the deref-time guard chain at all.
        if (!is_object_descriptor2t(*it))
          continue;

        const object_descriptor2t &obj = to_object_descriptor2t(*it);

        // Build the object pointer. For NULL we keep the legacy
        // NULL-symbol form; otherwise we point at `address_of(object)`,
        // optionally adjusted by the known offset.
        expr2tc obj_ptr;
        if (is_null_object2t(obj.object))
        {
          type2tc nullptrtype = pointer_type2tc(expr->type);
          obj_ptr = symbol2tc(nullptrtype, "NULL");
        }
        else if (is_unknown2t(obj.offset))
        {
          // The offset is unknown but we still know which object the
          // pointer could point at. Emit the weaker SAME-OBJECT(p,
          // &obj) — any offset is allowed. Previously we returned
          // here, dropping the entire disjunction (the assume became
          // vacuously true) — sound but useless. The looser disjunct
          // keeps the assume non-trivial while staying sound.
          obj_ptr = address_of2tc(expr->type, obj.object);
        }
        else
        {
          obj_ptr = add2tc(
            expr->type, address_of2tc(expr->type, obj.object), obj.offset);
        }

        add_disjunct(same_object2tc(expr, obj_ptr));
      }

      if (!or_accuml)
        return;

      // Assume the pointer points at one of the value-set objects.
      // Was `claim` previously: that only constrained the counter-
      // example search, not the inductive hypothesis. Switching to
      // `assume` tightens the IS encoding (the value-set analysis is
      // a sound over-approximation, so assuming its result is safe).
      goto_symex.assume(or_accuml);
    }
  }
}

void symex_dereference_statet::rename(expr2tc &expr)
{
  expr = goto_symex.split_rewrite(expr);
  goto_symex.cur_state->rename(expr);
}

void symex_dereference_statet::dump_internal_state(
  const std::list<struct internal_item> &data)
{
  goto_symex.internal_deref_items.insert(
    goto_symex.internal_deref_items.begin(), data.begin(), data.end());
}

bool symex_dereference_statet::is_live_variable(const expr2tc &symbol)
{
  expr2tc sym = symbol;

  // NB, symbols shouldn't hit this point with no renaming (i.e. level0),
  // this should eventually be asserted.
  if (
    to_symbol2t(sym).rlevel == symbol_renaming_level::level0 ||
    to_symbol2t(sym).rlevel == symbol_renaming_level::level1_global ||
    to_symbol2t(sym).rlevel == symbol_renaming_level::level2_global)
    return true;

  // A thread-local global is renamed per-thread to level1/level2 (not
  // level*_global), yet it has static (whole-thread) lifetime — it is not a
  // stack local that can expire. Treat any static-lifetime symbol as live;
  // otherwise the per-thread stack-frame search below (meant for locals) fails
  // to find it and wrongly reports an expired-pointer dereference.
  const symbolt *base = goto_symex.ns.lookup(to_symbol2t(sym).thename);
  if (base && base->static_lifetime)
    return true;

  // Python objects follow garbage-collected lifetime semantics: an instance
  // referenced after its defining function returns is still valid at runtime
  // (CPython heap-allocates objects and frees them only when unreachable).
  // ESBMC stack-allocates Python aggregates, so a pointer captured into a
  // returned/escaping aggregate would otherwise be flagged as a false
  // use-after-free once the defining frame is popped. Treat user-defined
  // Python class instances as having whole-program (GC) lifetime. Their SSA
  // value bindings are likewise preserved across frame teardown (see
  // pop_frame), so reads through the escaped pointer observe the real values.
  // See issue #4773.
  if (goto_symex.is_python_gc_object(base))
    return true;

  goto_symex.replace_dynamic_allocation(sym);
  goto_symex.replace_nondet(sym);
  goto_symex.dereference(sym, dereferencet::INTERNAL);

  // Symbol is renamed to at least level 1, fetch the relevant thread data
  const execution_statet &ex_state = goto_symex.art1->get_cur_state();
  const goto_symex_statet &state =
    ex_state.threads_state[to_symbol2t(sym).thread_num];

  // Level one names represent the storage for a variable, and this symbol
  // may have entered pointer tracking at any time the variable had its address
  // taken (and subsequently been propagated from there, e.g., a variable passed
  // as reference to a function). If the stack frame that the variable was in
  //  has now expired, it's an invalid pointer. Look up the stack frames
  // currently active the corresponding thread to see whether there are any
  // records for the lexical variable that have this activation record.
  for (auto it = state.call_stack.rbegin(); it != state.call_stack.rend(); it++)
  {
    // Get the last l1 renamed symbol
    auto const &name = renaming::level2t::name_record(to_symbol2t(sym));
    auto const &local_vars = it->local_variables;
    if (local_vars.count(name))
      return true;
  }

  // There were no stack frames where that variable existed and had the correct
  // level1 num: it's dead Jim.
  return false;
}

void goto_symext::dereference(
  expr2tc &expr,
  dereferencet::modet mode,
  bool block_assertions)
{
  symex_dereference_statet symex_dereference_state(*this, *cur_state);

  dereferencet dereference(ns, new_context, options, symex_dereference_state);
  if (block_assertions)
    dereference.set_block_assertions();

  // needs to be renamed to level 1
  assert(!cur_state->call_stack.empty());
  cur_state->top().level1.rename(expr);

  guard2tc guard;
  if (is_free(mode))
  {
    expr2tc tmp = expr;
    while (is_typecast2t(tmp))
      tmp = to_typecast2t(tmp).from;

    assert(is_pointer_type(tmp));
    std::list<expr2tc> dummy;
    // Dereference to byte type, because it's guaranteed to succeed.
    tmp = dereference2tc(get_uint8_type(), tmp);

    dereference.dereference_expr(tmp, guard, dereferencet::FREE);
    expr = tmp;
  }
  else
  {
    // split_read may leave a split object whole on purpose; only what a
    // dereference resolves to is rewritten here.
    const bool derefs = dereference.has_dereference(expr);
    dereference.dereference_expr(expr, guard, mode);
    if (derefs)
      expr = split_rewrite(expr);
  }
}

void goto_symext::check_read_only_write(
  const expr2tc &object,
  const expr2tc &cond)
{
  guard2tc guard;
  guard.add(cond);
  symex_dereference_statet symex_dereference_state(*this, *cur_state);
  dereferencet(ns, new_context, options, symex_dereference_state)
    .read_only_write_check(object, guard);
}

void goto_symext::collect_write_targets(
  const guard2tc &base,
  std::vector<write_targett> &targets)
{
  for (const auto &item : internal_deref_items)
  {
    expr2tc pointer_guard = item.guard;
    cur_state->rename(pointer_guard);
    guard2tc guard = base;
    guard.add(pointer_guard);
    targets.push_back({item.object, guard});
  }
}

void goto_symext::write_nondet_targets(
  const std::vector<write_targett> &targets)
{
  for (const write_targett &target : targets)
  {
    check_read_only_write(target.object, target.guard.as_expr());
    if (is_constant_string2t(target.object))
      continue;
    assert(is_symbol2t(target.object) && "Only variables can be written");
    expr2tc val = sideeffect2tc(
      target.object->type,
      expr2tc(),
      expr2tc(),
      std::vector<expr2tc>(),
      type2tc(),
      sideeffect2t::allockind::nondet);
    symex_assign(code_assign2tc(target.object, val), false, target.guard);
  }
}
