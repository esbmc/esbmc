#include <cassert>
#include <goto-programs/destructor.h>
#include <goto-programs/goto_convert_functions.h>
#include <goto-programs/goto_inline.h>
#include <goto-programs/remove_no_op.h>
#include <util/base_type.h>
#include <util/c_types.h>
#include <util/i2string.h>
#include <util/prefix.h>
#include <util/std_code.h>
#include <util/std_expr.h>
#include <util/type_byte_size.h>

goto_convert_functionst::goto_convert_functionst(
  contextt &_context,
  optionst &_options,
  goto_functionst &_functions)
  : goto_convertt(_context, _options), functions(_functions)
{
}

void goto_convert_functionst::goto_convert()
{
  // warning! hash-table iterators are not stable

  symbol_listt symbol_list;
  context.Foreach_operand_in_order([&symbol_list](symbolt &s) {
    if (!s.is_type && s.get_type().is_code())
      symbol_list.push_back(&s);
  });

  for (auto &it : symbol_list)
  {
    convert_function(*it);
  }

  functions.compute_location_numbers();
}

bool goto_convert_functionst::hide(const goto_programt &goto_program)
{
  for (const auto &instruction : goto_program.instructions)
  {
    for (const auto &label : instruction.labels)
    {
      if (label == "__ESBMC_HIDE")
        return true;
    }
  }

  return false;
}

void goto_convert_functionst::add_return(
  goto_functiont &f,
  const irep_idt &identifier,
  const locationt &location)
{
  if (!f.body.instructions.empty() && f.body.instructions.back().is_return())
    return; // not needed, we have one already

  // see if we have an unconditional goto at the end
  if (
    !f.body.instructions.empty() && f.body.instructions.back().is_goto() &&
    is_true(f.body.instructions.back().guard))
    return;

  goto_programt::targett t = f.body.add_instruction();
  t->make_return();
  t->location = location;

  type2tc ret_type = ns.follow(to_code_type(f.type).ret_type);

  // C11 §5.1.2.2.3p1: reaching the } that terminates main returns 0. Synthesize
  // the standard-mandated implicit `return 0` instead of a nondet value so that
  // main's return value is modelled correctly (e.g. when a contract's ensures
  // clause refers to __ESBMC_return_value).
  const std::string &id = id2string(identifier);
  if (id == "c:@F@main" || has_prefix(id, "c:@F@main#"))
  {
    t->code = code_return2tc(gen_zero(ret_type));
    return;
  }

  // Build a nondet side-effect of the function's return type directly on
  // the irep2 side. Followed through symbol-typed aliases as the legacy
  // path did.
  expr2tc nondet = sideeffect2tc(
    ret_type,
    expr2tc(),
    expr2tc(),
    std::vector<expr2tc>(),
    type2tc(),
    sideeffect2t::allockind::nondet);
  t->code = code_return2tc(nondet);
}

// Stamp `loc` onto every value-level sub-expression of `expr` that lacks a
// source location (used by restore_value_locations below).
static void stamp_value_locations(exprt &expr, const locationt &loc)
{
  // Read this node's OWN location (const overload, non-recursive); the mutable
  // overload would materialise an empty #location, and find_location() would
  // descend into operands and report a child's location for this node.
  const locationt &own = static_cast<const exprt &>(expr).location();
  if (own.is_nil() || own.get_file().empty())
    expr.location() = loc;

  Forall_operands (it, expr)
    stamp_value_locations(*it, loc);
}

// IREP2 value-level expressions carry no source location (only the
// structured-CF code kinds got the V.4.1/V.4.5 non-reflected `location` field).
// The clang frontends stamp every sub-expression of a statement with that
// statement's #location, but the legacy->IREP2->legacy body round-trip drops it
// from the value operands. goto_convert then generates instructions from those
// operands (e.g. the tmp/GOTO sequence a `&&`/`||` short-circuit lowers to,
// whose location is read from the operand at goto_sideeffects.cpp) with an
// empty location -- breaking any pass keyed on instruction location, e.g.
// --condition-coverage skips conditions whose file is not the source file
// (goto_coverage.cpp), reporting 0 conditions and a spurious SUCCESSFUL.
// Restore the frontend invariant by pushing each statement's location down onto
// its location-less value operands. Each nested statement governs its subtree.
static void restore_value_locations(exprt &code, const locationt &inherited)
{
  // This statement's own location (non-recursive const read); falls back to the
  // enclosing statement's when the round-trip left this node location-less.
  const locationt &own = static_cast<const exprt &>(code).location();
  const locationt &here =
    (own.is_not_nil() && !own.get_file().empty()) ? own : inherited;

  if (here.get_file().empty())
    return; // no location to propagate yet

  Forall_operands (it, code)
  {
    if (it->is_code())
      restore_value_locations(*it, here);
    else
      stamp_value_locations(*it, here);
  }
}

// W1-loc spike Phase C (esbmc/esbmc#4715): consume one IREP2 statement `code2`
// natively (design D3), appending to `dest`, and recurse into nested blocks.
// Returns false the instant an unsupported kind (or a shape whose native
// emission would not be byte-identical) appears — the caller discards the
// partial `dest`, so a failed walk never corrupts the fallback body. So far:
// the structural leaves (block/skip), the single-instruction value statements
// (assign/expression) that reduce to one ASSIGN/OTHER with nothing to lower,
// trivial-type declarations (DECL + optional side-effect-free ASSIGN + scope-exit
// DEAD, the block managing the destructor stack as convert_block does), a value
// return (RETURN + unconditional GOTO to the function's end), a
// side-effect-free `if`/`if-else` whose branches convert natively (the
// general, unfolded branch shape only — see the assert-fold guard below), a
// side-effect-free `while` whose body converts natively (`v: if(!c) goto z;
// x: P; y: goto v; z: ;`), and `break`/`continue` (an unconditional GOTO to
// the nearest enclosing loop's break/continue target, preceded by
// unwind_destructor_stack's DEAD instructions for whatever was pushed since
// that loop was entered — the inherited goto_convertt method is called
// directly, already stack-neutral by design), and a bare "foo();" call
// statement to a plain named symbol with a body and side-effect-free
// arguments (a single FUNCTION_CALL; the return-unused requirement means
// do_function_call's temp-symbol machinery is never entered, so this kind
// carries no shared-counter byte-identity risk). Each reads its own
// code_*2t fields directly (no legacy round-trip) and carries the
// statement's own location, matching goto_convertt::convert() byte-for-byte
// on this subset.
bool goto_convert_functionst::convert_native_rec(
  const expr2tc &code2,
  goto_programt &dest)
{
  if (is_code_block2t(code2))
  {
    const code_block2t &block = to_code_block2t(code2);

    // Mirror convert_block(): save the destructor stack, convert the children
    // (a code_decl2t pushes a scope-exit code_dead), then emit those destructors
    // at the block's end_location and restore the stack. A decl-free block
    // leaves the stack untouched, so the unwind and restore are both no-ops and
    // this stays byte-identical to the pre-decl handler for the block/skip/
    // assign/expression subset.
    destructor_stackt old_stack = targets.destructor_stack;

    for (const expr2tc &stmt : block.operands)
      if (!convert_native_rec(stmt, dest))
      {
        // Fallback: undo any code_dead this partial walk pushed, so the caller's
        // goto_convert_rec re-converts against a clean destructor stack.
        targets.destructor_stack = old_stack;
        return false;
      }

    // Mirror convert_block's unreachable guard: a code_return2t emits a trailing
    // unconditional goto (to the end-of-function target), after which the block's
    // destructors are dead code and convert_block skips them. Reproduce the exact
    // test so the emitted DEADs match byte-for-byte; for the block/skip/assign/
    // expression/decl subset (no trailing goto) this is always the else branch,
    // identical to the prior unconditional unwind.
    if (
      !dest.instructions.empty() && dest.instructions.back().is_goto() &&
      is_true(dest.instructions.back().guard))
    {
      // unreachable -> skip destructors, exactly as convert_block does
    }
    else
      unwind_destructor_stack(block.end_location, old_stack.size(), dest);
    targets.destructor_stack = old_stack;

    // Mirror the tail of convert(): a statement that emitted no instruction
    // leaves an empty program, which must carry a SKIP at the block location.
    // Only an (recursively) empty block reaches here; a skip always emits.
    if (dest.instructions.empty())
      dest.add_instruction(SKIP)->location = block.location;
    return true;
  }

  if (is_code_skip2t(code2))
  {
    goto_programt::targett t = dest.add_instruction(SKIP);
    t->location = to_code_skip2t(code2).location;
    // code_skip2t is only ever built with an empty type (migrate.cpp), so this
    // node equals convert_skip()'s migrate_expr(skip) result — reuse it here
    // instead of round-tripping.
    t->code = code2;
    return true;
  }

  if (is_code_assign2t(code2))
  {
    const code_assign2t &assign = to_code_assign2t(code2);
    exprt lhs = migrate_expr_back(assign.target);
    if (has_sideeffect(lhs) || lhs.id() == "if")
      return false;

    // convert_assign()'s function-call special case (goto_convert.cpp)
    // dispatches a call-valued rhs straight to do_function_call(), bypassing
    // the atomic checks the generic path below applies — a call result
    // assigned to a C11 _Atomic lhs is NOT wrapped atomically in legacy
    // either, so this branch must not add an is_atomic_symbol(lhs) guard:
    // legacy itself doesn't apply one here. Delegate to the real
    // do_function_call() (not a reimplementation) so `has_next =
    // ESBMC_range_has_next_(...)` — the statement a desugared Python `for`
    // loop's preprocessor hoists the call into (see docs/spike-v1k-w1loc.md)
    // — converts natively. Narrow slice: callee and arguments must be
    // side-effect-free, so do_function_call's own remove_sideeffects() calls
    // on them are no-ops we can skip issuing; convert_function's
    // tmp_symbol/context rollback (above) still protects the temp
    // do_function_call allocates if a later statement in this body is
    // unsupported and forces a fallback.
    if (
      is_sideeffect2t(assign.source) &&
      to_sideeffect2t(assign.source).kind ==
        sideeffect2t::allockind::function_call)
    {
      const sideeffect2t &se = to_sideeffect2t(assign.source);
      if (has_sideeffect(se.operand))
        return false;
      for (const expr2tc &arg : se.arguments)
        if (has_sideeffect(arg))
          return false;

      exprt function_legacy = migrate_expr_back(se.operand);
      exprt::operandst args_legacy;
      for (const expr2tc &arg : se.arguments)
        args_legacy.push_back(migrate_expr_back(arg));

      do_function_call(
        lhs, function_legacy, args_legacy, assign.location, dest);
      return true;
    }

    // convert_assign() collapses to a single ASSIGN instruction only when there
    // is nothing to lower: no side effect in either operand (so goto_sideeffects
    // never runs, and the value operand locations restore_value_locations would
    // stamp are dropped again by the final migrate_expr anyway), the source is
    // not a code-typed statement, and neither side touches a C11 _Atomic object
    // (which would take the convert_assign_atomic path). Decide with the exact
    // predicates convert_assign uses, on a throwaway legacy view of the two
    // operands; any richer shape falls back so flag-on stays byte-identical to
    // flag-off.
    //
    // A top-level ternary is the one non-side-effect shape remove_sideeffects
    // still enters (goto_sideeffects.cpp early-returns only on
    // `!has_sideeffect(e) && e.id() != "if"`): under --validate-violation-witness
    // it lowers `c ? a : b` to a DECL/IF/GOTO branch, which a single ASSIGN would
    // not reproduce. Mirror that exact entry condition so we fall back on it.
    exprt rhs = migrate_expr_back(assign.source);
    if (
      has_sideeffect(rhs) || rhs.id() == "if" || rhs.type().is_code() ||
      is_atomic_symbol(lhs, ns) || has_atomic_read(rhs, ns))
      return false;

    // For side-effect-free operands the instruction convert_assign emits is
    // migrate_expr(code_assignt(lhs, rhs)) located at the statement — which
    // round-trips back to `code2` itself (migrate_expr drops the operand
    // locations, so none of restore_value_locations' stamping survives in the
    // stored code). Emit it directly, no round-trip, carrying the statement's
    // own location, exactly as copy(new_assign, ASSIGN) would.
    goto_programt::targett t = dest.add_instruction(ASSIGN);
    t->code = code2;
    t->location = assign.location;
    return true;
  }

  if (is_code_expression2t(code2))
  {
    const code_expression2t &expr_stmt = to_code_expression2t(code2);

    // convert_expression() emits a single OTHER only in its plain else-branch
    // (goto_convert.cpp): a side-effect operand is lowered by
    // remove_sideeffects, a code-typed operand is re-dispatched through convert()
    // (goto_convert.cpp), and a top-level ternary is peeled off
    // unconditionally into convert_ifthenelse (goto_convert.cpp), before
    // remove_sideeffects runs. Fall back on all of those, deciding with the exact
    // predicates convert_expression uses on a throwaway legacy view of the
    // operand.
    exprt op = migrate_expr_back(expr_stmt.operand);
    if (op.is_nil() || has_sideeffect(op) || op.is_code() || op.id() == "if")
      return false;

    // convert_expression locates the OTHER at the operand location, which
    // restore_value_locations sets to the enclosing statement location. When the
    // statement carries its own located #location that already IS that location,
    // so emit code2 directly (migrate_expr drops the operand location, so the
    // stored code round-trips to code2) at the statement location. Fall back on a
    // location-less statement, whose operand the round-trip would instead stamp
    // with an inherited block location.
    if (expr_stmt.location.is_nil() || expr_stmt.location.get_file().empty())
      return false;

    goto_programt::targett t = dest.add_instruction(OTHER);
    t->code = code2;
    t->location = expr_stmt.location;
    return true;
  }

  if (is_code_decl2t(code2))
  {
    const code_decl2t &decl = to_code_decl2t(code2);

    symbolt *s = context.find_symbol(decl.value);
    if (s == nullptr)
      return false;

    // convert_decl (goto_convert.cpp) has several paths this native handler
    // does not reproduce; fall back on each so flag-on stays byte-identical:
    //  - a static-lifetime or code-typed symbol is a no-op SKIP,
    //  - an array type may be a VLA needing rewrite_vla_decl / a dynamic-size
    //    generator — exclude all arrays conservatively,
    //  - a type with a destructor pushes a second stack entry and lowers a
    //    FUNCTION_CALL at scope exit,
    //  - a temporary_object or side-effect initializer is lowered.
    // What remains is exactly convert_decl's plain path: a DECL, an optional
    // side-effect-free ASSIGN, and one scope-exit code_dead.
    if (
      s->static_lifetime || s->get_type().is_code() || s->get_type().is_array())
      return false;

    code_function_callt destructor;
    if (get_destructor(ns, s->get_type(), destructor))
      return false;

    exprt initializer = is_nil_expr(decl.init) ? static_cast<exprt>(nil_exprt())
                                               : migrate_expr_back(decl.init);
    // A top-level ternary initializer is side-effect-free yet still lowered to a
    // DECL/IF/GOTO branch by remove_sideeffects under --validate-violation-witness
    // (goto_sideeffects.cpp), which a single ASSIGN would not reproduce —
    // mirror the same guard the assign handler carries.
    if (
      initializer.is_not_nil() &&
      (has_sideeffect(initializer) || initializer.id() == "if" ||
       (initializer.id() == "sideeffect" &&
        initializer.statement() == "temporary_object")))
      return false;

    // Emit exactly as convert_decl does: copy() migrates the freshly-built
    // legacy node, so the DECL/ASSIGN instructions match byte-for-byte. Build the
    // DECL/ASSIGN symbol from the decl operand type (migrate_type_back(decl.type),
    // what convert_decl uses via new_code.op0()); the scope-exit code_dead uses
    // the symbol-table type (s->get_type()), also matching convert_decl — the two
    // sources coincide today but are kept distinct so a stale symbol-table type
    // cannot silently diverge the DECL bytes.
    const symbol_exprt var(s->id, migrate_type_back(decl.type));

    code_declt decl_code(var);
    decl_code.location() = decl.location;
    copy(decl_code, DECL, dest);

    if (initializer.is_not_nil())
    {
      code_assignt assign(var, initializer);
      assign.location() = decl.location;
      copy(assign, ASSIGN, dest);
    }

    targets.destructor_stack.push_back(
      code_deadt(symbol_exprt(s->id, s->get_type())));
    return true;
  }

  if (is_code_return2t(code2))
  {
    const code_return2t &ret = to_code_return2t(code2);

    // convert_return (goto_convert.cpp) emits, for the plain case, a RETURN
    // instruction (only when the function returns a value) followed by an
    // unconditional GOTO to the end-of-function target. Reproduce that exactly,
    // and fall back on every shape convert_return transforms, deciding with the
    // same predicates on a throwaway legacy view of the return value:
    //  - a side-effect value (remove_sideeffects lowers it into extra instrs),
    //  - a cpp-throw return value (converted as a statement, no RETURN),
    //  - a top-level ternary (remove_sideeffects lowers `c ? a : b`),
    //  - a missing value in a value-returning function (nondet replacement).
    // A void function returning a value is a C/C++ constraint violation the
    // frontend rejects, so it never reaches here; only a valueless void return
    // does, which correctly emits just the end-of-function goto below.
    // convert_return unwinds the destructor stack only when it holds a
    // destructor FUNCTION_CALL, which cannot happen here: the decl handler
    // falls back on any type with a destructor, so a native subtree's stack
    // holds only scope-exit code_dead entries, which convert_return leaves
    // alone; the enclosing block handler reproduces the (skipped) scope-exit
    // behaviour via the trailing-goto guard above.
    exprt val = is_nil_expr(ret.operand) ? static_cast<exprt>(nil_exprt())
                                         : migrate_expr_back(ret.operand);
    if (
      val.is_not_nil() &&
      (has_sideeffect(val) || val.is_code() || val.id() == "if"))
      return false;

    if (targets.has_return_value)
    {
      if (val.is_nil())
        return false; // convert_return replaces a missing value with nondet
      // The RETURN instruction convert_return emits is migrate_expr(code_returnt)
      // located at the statement; migrate_expr drops the value-operand location
      // restore_value_locations stamped, so it round-trips to code2 itself. Emit
      // it directly, exactly as the assign/expression handlers do.
      goto_programt::targett r = dest.add_instruction();
      r->make_return();
      r->code = code2;
      r->location = ret.location;
    }

    goto_programt::targett g = dest.add_instruction();
    g->make_goto(targets.return_target, gen_true_expr());
    g->location = ret.location;
    return true;
  }

  if (is_code_ifthenelse2t(code2))
  {
    const code_ifthenelse2t &ite = to_code_ifthenelse2t(code2);

    // A side-effecting guard needs remove_sideeffects (goto_convert.cpp),
    // which this kind doesn't reproduce; the condition-coverage options
    // suppress that call regardless, so fall back on those too.
    if (
      has_sideeffect(ite.cond) ||
      options.get_bool_option("condition-coverage") ||
      options.get_bool_option("condition-coverage-claims") ||
      options.get_bool_option("condition-coverage-rm") ||
      options.get_bool_option("condition-coverage-claims-rm"))
      return false;

    bool has_else = !is_nil_expr(ite.else_case);

    destructor_stackt stack_before_then = targets.destructor_stack;
    goto_programt tmp_op1;
    if (!convert_native_rec(ite.then_case, tmp_op1))
      return false;
    // A non-block branch (e.g. a bare decl) could leak a scope-exit code_dead
    // with no enclosing block to unwind it.
    if (targets.destructor_stack.size() != stack_before_then.size())
    {
      targets.destructor_stack = stack_before_then;
      return false;
    }

    goto_programt tmp_op2;
    if (has_else)
    {
      destructor_stackt stack_before_else = targets.destructor_stack;
      if (!convert_native_rec(ite.else_case, tmp_op2))
        return false;
      if (targets.destructor_stack.size() != stack_before_else.size())
      {
        targets.destructor_stack = stack_before_else;
        return false;
      }
    }

    // generate_ifthenelse (goto_convert.cpp) folds a branch that reduces
    // to a lone `assert(false)` directly into the guard instead of emitting
    // the general shape below (--validate-violation-witness disables this);
    // fall back rather than reproduce the fold.
    if (!options.get_bool_option("validate-violation-witness"))
    {
      auto is_lone_false_assert = [](const goto_programt &p) {
        return p.instructions.size() == 1 &&
               p.instructions.back().is_assert() &&
               is_false(p.instructions.back().guard) &&
               p.instructions.back().labels.empty();
      };
      if (
        is_lone_false_assert(tmp_op1) ||
        (has_else && is_lone_false_assert(tmp_op2)))
        return false;
      if (
        !has_else && tmp_op1.instructions.size() == 2 &&
        tmp_op1.instructions.front().is_assert() &&
        is_false(tmp_op1.instructions.front().guard) &&
        tmp_op1.instructions.front().labels.empty() &&
        tmp_op1.instructions.back().labels.empty())
        return false;
    }

    const locationt &location = ite.location;

    // v: if(!c) goto y/z; w: P; x: goto z; (else only) y: Q; (else only) z: ;
    goto_programt tmp_z;
    goto_programt::targett z = tmp_z.add_instruction();
    z->make_skip();
    z->location = location;

    goto_programt tmp_y;
    goto_programt::targett y;
    if (has_else)
    {
      tmp_y.swap(tmp_op2);
      y = tmp_y.instructions.begin();
    }

    goto_programt tmp_v;
    goto_programt::targett v = tmp_v.add_instruction();
    v->make_goto(has_else ? y : z, not2tc(ite.cond));
    v->location = location;

    goto_programt tmp_w;
    tmp_w.swap(tmp_op1);

    goto_programt tmp_x;
    if (has_else)
    {
      goto_programt::targett x = tmp_x.add_instruction();
      x->make_goto(z);
      x->location = tmp_w.instructions.back().location;
    }

    dest.destructive_append(tmp_v);
    dest.destructive_append(tmp_w);
    if (has_else)
    {
      dest.destructive_append(tmp_x);
      dest.destructive_append(tmp_y);
    }
    dest.destructive_append(tmp_z);
    return true;
  }

  if (is_code_while2t(code2))
  {
    const code_while2t &w = to_code_while2t(code2);
    const locationt &location = w.location;

    // convert_while saves/restores the break/continue targets around the
    // body regardless of whether the body ends up using them; do the same so
    // the code_break2t/code_continue2t arms below (which read
    // targets.break_target/break_stack_size etc.) resolve to this loop's
    // targets for anything in the body, and a nested loop's own
    // set_break/set_continue correctly shadows them for its own body.
    break_continue_targetst old_break_continue(targets);

    //    while(c) P;
    //--------------------
    // v: if(!c) goto z;
    // x: P;
    // y: goto v;          <-- continue target
    // z: ;                <-- break target
    goto_programt tmp_z;
    goto_programt::targett z = tmp_z.add_instruction();
    z->make_skip();
    z->location = location;

    goto_programt tmp_branch;
    if (has_sideeffect(w.cond))
    {
      // convert_while (goto_convert.cpp) builds this branch via the
      // shared generate_conditional_branch helper, which itself decomposes
      // &&/||/not and calls remove_sideeffects() at the leaf. Delegate to
      // that helper verbatim instead of reimplementing it, so a direct call
      // in the condition (`while (has_more()) ...`) gets identical
      // temp-symbol numbering to the legacy path (the per-function
      // tmp_symbol counter is rolled back on a later fallback — see the
      // rollback note in convert_function, above). Note this is NOT what a
      // desugared Python `for`/`while <call>` loop produces: the
      // preprocessor always rewrites those into an explicit `while True: if
      // not <call>(): break` before goto_convert ever sees them (confirmed
      // empirically — see docs/spike-v1k-w1loc.md), so this path is reached
      // by a C/C++ `while` whose condition is directly a call.
      exprt cond_legacy = migrate_expr_back(w.cond);
      generate_conditional_branch(
        gen_not(cond_legacy), z, location, tmp_branch);
    }
    else
    {
      goto_programt::targett t = tmp_branch.add_instruction();
      t->make_goto(z, not2tc(w.cond));
      t->location = location;
    }
    goto_programt::targett v = tmp_branch.instructions.begin();
    v->location = location;

    goto_programt tmp_y;
    goto_programt::targett y = tmp_y.add_instruction();

    targets.set_break(z);
    targets.set_continue(y);

    // Same defensive check as the if/else branches: a body that isn't itself
    // a code_block2t could in principle leak a scope-exit code_dead with no
    // enclosing block to unwind it.
    destructor_stackt stack_before_body = targets.destructor_stack;
    goto_programt tmp_x;
    bool body_ok = convert_native_rec(w.body, tmp_x);

    old_break_continue.restore(targets);

    if (!body_ok || targets.destructor_stack.size() != stack_before_body.size())
    {
      targets.destructor_stack = stack_before_body;
      return false;
    }

    y->make_goto(v);
    y->guard = gen_true_expr();
    y->location = location;
    // pragma_unroll_count defaults to 0 both here and on a fresh instruction
    // (see goto_program.h), so an unconditional assignment is the exact
    // equivalent of convert_while's `if (!"#pragma_unroll".empty()) ...` —
    // absent and explicit-zero are indistinguishable either way.
    y->pragma_unroll_count = w.pragma_unroll_count;

    dest.destructive_append(tmp_branch);
    dest.destructive_append(tmp_x);
    dest.destructive_append(tmp_y);
    dest.destructive_append(tmp_z);
    return true;
  }

  if (is_code_break2t(code2))
  {
    const code_break2t &b = to_code_break2t(code2);

    // A break outside a loop/switch shouldn't reach here (switch isn't a
    // supported kind), but stay defensive rather than trust the invariant.
    if (!targets.break_set)
      return false;

    // unwind_destructor_stack emits the exit DEADs into `dest` then restores
    // targets.destructor_stack to its pre-call state — a break is one exit
    // path among several, so the entries stay live for the rest of the
    // block's normal flow. The inherited goto_convertt method already does
    // this exactly; no reimplementation needed.
    unwind_destructor_stack(b.location, targets.break_stack_size, dest);

    goto_programt::targett t = dest.add_instruction();
    t->make_goto(targets.break_target);
    t->location = b.location;
    return true;
  }

  if (is_code_continue2t(code2))
  {
    const code_continue2t &c = to_code_continue2t(code2);

    if (!targets.continue_set)
      return false;

    unwind_destructor_stack(c.location, targets.continue_stack_size, dest);

    goto_programt::targett t = dest.add_instruction();
    t->make_goto(targets.continue_target);
    t->location = c.location;
    return true;
  }

  if (is_code_assert2t(code2))
  {
    const code_assert2t &a = to_code_assert2t(code2);

    // convert_assert (goto_convert.cpp) removes side effects from the
    // guard before emitting; require a side-effect-free guard for the same
    // reason as every other statement kind here. code_assert2t's guard is
    // already expr2tc, so (unlike the legacy-exprt kinds) there is no
    // migrate_expr round-trip to do.
    if (has_sideeffect(a.guard))
      return false;

    // --no-assertions: convert_assert removes side effects (a no-op here)
    // and returns without emitting an ASSERT — match that exactly, an empty
    // conversion rather than falling back.
    if (options.get_bool_option("no-assertions"))
      return true;

    goto_programt::targett t = dest.add_instruction(ASSERT);
    t->guard = a.guard;
    t->location = a.location;
    t->location.property("assertion");
    t->location.user_provided(true);
    return true;
  }

  if (is_code_assume2t(code2))
  {
    const code_assume2t &a = to_code_assume2t(code2);
    if (has_sideeffect(a.guard))
      return false;

    goto_programt::targett t = dest.add_instruction(ASSUME);
    t->guard = a.guard;
    t->location = a.location;
    return true;
  }

  if (is_code_function_call2t(code2))
  {
    const code_function_call2t &f = to_code_function_call2t(code2);

    // Narrow slice: a bare "foo();" statement (return value unused, so no
    // do_function_call temp-symbol machinery is ever entered) calling a
    // plain named symbol (not the dereference/if/typecast-callee shapes
    // do_function_call dispatches separately) with side-effect-free
    // arguments (so its remove_sideeffects preamble is a no-op). Falls back
    // on everything else, including every builtin name
    // do_function_call_symbol special-cases (assume/assert/loop_invariant/
    // etc.) — those are reached only when the callee symbol has no body,
    // the same condition this handler excludes on below.
    if (!is_nil_expr(f.ret) || !is_symbol2t(f.function))
      return false;

    for (const expr2tc &arg : f.operands)
      if (has_sideeffect(arg))
        return false;

    const symbol2t &fsym = to_symbol2t(f.function);
    symbolt *s = context.find_symbol(fsym.thename);
    if (!s || !s->get_type().is_code())
      return false;

    bool skip_body =
      options.get_bool_option("enable-unreachability-intrinsic") &&
      (s->name == "reach_error" || s->name == "__VERIFIER_error");
    if (s->get_value().is_nil() || !s->get_value().has_operands() || skip_body)
      return false;

    goto_programt::targett t = dest.add_instruction();
    t->make_function_call(code2);
    t->location = f.location;
    return true;
  }

  return false; // unsupported kind: whole body falls back to goto_convert_rec
}

bool goto_convert_functionst::try_convert_body_native(
  const expr2tc &body2,
  goto_programt &dest)
{
  goto_programt native;
  if (!convert_native_rec(body2, native))
    return false; // dest untouched; caller falls back to goto_convert_rec

  // Mirror goto_convert_rec()'s post-passes. finish_gotos only resolves *named*
  // (labelled) gotos, of which the native subset emits none — a code_return2t's
  // goto targets the end-of-function iterator directly, not a label — so it stays
  // a no-op. optimize_guarded_gotos runs identically here and in the legacy path
  // over the same instruction sequence, so the folded output matches regardless.
  finish_gotos(native);
  optimize_guarded_gotos(native);
  dest.destructive_append(native);
  return true;
}

void goto_convert_functionst::convert_function(symbolt &symbol)
{
  irep_idt identifier = symbol.id;

  // Apply a SFINAE test: discard unused C++ templates.
  // Note: can be removed probably? as the new clang-cpp-frontend should've
  // done a pretty good job at resolving template overloading
  if (
    symbol.get_value().get("#speculative_template") == "1" &&
    symbol.get_value().get("#template_in_use") != "1")
    return;

  // make tmp variables local to function
  tmp_symbol = symbol_generator(id2string(symbol.id) + "::$tmp::");

  auto it = functions.function_map.find(identifier);
  if (it == functions.function_map.end())
    functions.function_map.emplace(identifier, goto_functiont());

  goto_functiont &f = functions.function_map.at(identifier);
  f.type = migrate_symbol_type(symbol);
  f.exception_spec = exception_specificationt::from_type(symbol.get_type());
  f.body_available = symbol.get_value().is_not_nil();

  if (!f.body_available)
    return;

  if (!symbol.get_value().is_code())
  {
    log_error("got invalid code for function `{}'", id2string(identifier));
    abort();
  }

  // V.4.4 (esbmc/esbmc#4715): the function body is always lowered through the
  // IREP2 round-trip. get_value2() returns the IREP2 body directly when a
  // frontend stored it (e.g. the Python frontend post-adjust), or lazily
  // forward-migrates the legacy body for other frontends; migrate_expr_back
  // then yields the codet goto_convert_rec consumes. The pre-V.4.4 legacy
  // bypass (to_code(symbol.get_value())) and the --no-irep2-bodies escape hatch
  // are gone — this is the only body path.
  exprt roundtrip_body_storage = migrate_expr_back(symbol.get_value2());
  // Re-attach the per-statement source locations the round-trip dropped from
  // value operands, so goto_convert-generated instructions stay located.
  restore_value_locations(roundtrip_body_storage, locationt());
  const codet &code = to_code(roundtrip_body_storage);

  // The closing-brace location is the END_FUNCTION instruction's location; for
  // an empty function body it is the *only* located instruction, so losing it
  // leaves the whole function unlocated and passes keyed on instruction
  // location skip it (e.g. --branch-function-coverage stops counting the
  // function entry point). code_block2t now carries #end_location through the
  // round-trip (W1, esbmc/esbmc#4715), so read it straight off the body block.
  locationt end_location;
  if (code.get_statement() == "block")
    end_location =
      static_cast<const locationt &>(to_code_block(code).end_location());
  else
    end_location.make_nil();

  // add "end of function"
  goto_programt tmp_end_function;
  goto_programt::targett end_function = tmp_end_function.add_instruction();
  end_function->type = END_FUNCTION;
  end_function->location = end_location;

  targets = targetst();
  targets.set_return(end_function);
  // constructor/destructor return types migrate to empty_type (see
  // util/migrate.cpp), so the legacy three-way id check collapses to this.
  targets.has_return_value =
    to_code_type(f.type).ret_type->type_id != type2t::empty_id;

  // W1-loc spike Phase C (esbmc/esbmc#4715): --irep2-native-body routes the
  // body through the IREP2-native dispatcher, which consumes code_*2t directly
  // (no whole-body legacy round-trip) and inherits statement locations onto
  // value operands. Until every kind in this body is supported it returns
  // false and we fall back to goto_convert_rec on the round-tripped `code`, so
  // flag-on is byte-identical to flag-off. `code`/`end_location` above are
  // still computed from the round-trip; the native path only replaces the
  // body-instruction dispatch.
  //
  // A native attempt that reaches a side-effecting code_while2t condition or
  // a code_assign2t with a function-call rhs (below) calls the shared
  // remove_sideeffects()/do_function_call() helpers, which allocate temp
  // symbols from the per-function tmp_symbol counter and add them to
  // context. If a *later* statement in the same body is still unsupported,
  // the whole attempt is discarded and dest falls back to goto_convert_rec —
  // but tmp_symbol.counter and context aren't part of that discarded dest,
  // so without an explicit rollback the fallback's own temp numbering would
  // start from wherever the abandoned attempt left off, instead of from
  // scratch like flag-off mode. Snapshot both before the attempt and roll
  // back on failure so the fallback is byte-identical regardless of what the
  // discarded attempt touched — and regardless of which future native kind
  // ends up being the one that allocates a temp before failing.
  unsigned tmp_counter_before = tmp_symbol.counter;
  irep_idt context_mark_before = context.mark();
  if (!(options.get_bool_option("irep2-native-body") &&
        try_convert_body_native(symbol.get_value2(), f.body)))
  {
    tmp_symbol.counter = tmp_counter_before;
    context.erase_since(context_mark_before);
    goto_convert_rec(code, f.body);
  }

  // add non-det return value, if needed
  if (targets.has_return_value)
    add_return(f, identifier, end_location);

  // Wrap the body of functions name __VERIFIER_atomic_* with atomic_begin
  // and atomic_end
  if (
    !f.body.instructions.empty() &&
    has_prefix(id2string(identifier), "c:@F@__VERIFIER_atomic_"))
  {
    goto_programt::instructiont a_begin;
    a_begin.make_atomic_begin();
    a_begin.location = f.body.instructions.front().location;
    f.body.insert_swap(f.body.instructions.begin(), a_begin);

    goto_programt::targett a_end = f.body.add_instruction();
    a_end->make_atomic_end();
    a_end->location = end_location;

    Forall_goto_program_instructions (i_it, f.body)
    {
      if (i_it->is_goto() && i_it->targets.front()->is_end_function())
      {
        i_it->targets.clear();
        i_it->targets.push_back(a_end);
      }
    }
  }

  // add "end of function"
  f.body.destructive_append(tmp_end_function);

  // do function tags (they are empty at this point)
  f.update_instructions_function(identifier);

  f.body.update();

  if (config.ansi_c.cheri)
  {
    // Hide the cheri ptr compressed and decompressed traces
    const irep_idt &n = symbol.location.get_file();
    if (has_suffix(n, "cheri_compressed_cap_common.h"))
      f.body.hide = true;
  }

  if (hide(f.body))
    f.body.hide = true;
}

void goto_convert(
  contextt &context,
  optionst &options,
  goto_functionst &functions)
{
  goto_convert_functionst goto_convert_functions(context, options, functions);

  goto_convert_functions.thrash_type_symbols();
  goto_convert_functions.goto_convert();
}

void goto_convert_functionst::collect_type(
  const irept &type,
  typename_sett &deps)
{
  if (type.id() == "pointer")
    return;

  if (type.id() == "symbol")
  {
    assert(type.identifier() != "");
    deps.insert(type.identifier());
    return;
  }

  collect_expr(type, deps);
}

static bool denotes_thrashable_subtype(const irep_idt &id)
{
  return id == "type" || id == "subtype";
}

void goto_convert_functionst::collect_expr(
  const irept &expr,
  typename_sett &deps)
{
  if (expr.id() == "pointer")
    return;

  forall_irep (it, expr.get_sub())
  {
    collect_expr(*it, deps);
  }

  forall_named_irep (it, expr.get_named_sub())
  {
    if (denotes_thrashable_subtype(it->first))
      collect_type(it->second, deps);
    else
      collect_expr(it->second, deps);
  }

  forall_named_irep (it, expr.get_comments())
  {
    if (denotes_thrashable_subtype(it->first))
      collect_type(it->second, deps);
    else
      collect_expr(it->second, deps);
  }
}

// Read-only twin of rename_types: does this type subtree contain a
// `symbol`-id node (other than the recursive `sname` self-reference)
// that rename_types would rewrite? Walks with const accessors so it
// never detaches.
bool goto_convert_functionst::type_needs_rename(
  const irept &type,
  const irep_idt &sname) const
{
  if (type.id() == "pointer")
    return false;

  if (type.id() == "symbol")
    // rename_types replaces every symbol type except the self-recursive
    // sname guard. A non-sname symbol type is always rewritten.
    return type.identifier() != sname;

  return expr_needs_rename(type, sname);
}

// Read-only twin of rename_exprs.
bool goto_convert_functionst::expr_needs_rename(
  const irept &expr,
  const irep_idt &sname) const
{
  if (expr.id() == "pointer")
    return false;

  forall_irep (it, expr.get_sub())
    if (expr_needs_rename(*it, sname))
      return true;

  forall_named_irep (it, expr.get_named_sub())
  {
    if (denotes_thrashable_subtype(it->first))
    {
      if (type_needs_rename(it->second, sname))
        return true;
    }
    else if (expr_needs_rename(it->second, sname))
      return true;
  }

  forall_named_irep (it, expr.get_comments())
    if (expr_needs_rename(it->second, sname))
      return true;

  return false;
}

void goto_convert_functionst::rename_types(
  irept &type,
  const symbolt &cur_name_sym,
  const irep_idt &sname)
{
  if (type.id() == "pointer")
    return;

  // Some type symbols aren't entirely correct. This is because (in the current
  // 27_exStbFb test) some type symbols get the module name inserted into the
  // name -- so int32_t becomes main::int32_t.
  //
  // Now this makes entire sense, because int32_t could be something else in
  // some other file. However, because type symbols aren't squashed at type
  // checking time (which, you know, might make sense) we now don't know what
  // type symbol to link "int32_t" up to. So; instead we test to see whether
  // a type symbol is linked correctly, and if it isn't we look up what module
  // the current block of code came from and try to guess what type symbol it
  // should have.

  typet type2;
  if (type.id() == "symbol")
  {
    if (type.identifier() == sname)
    {
      // A recursive symbol -- the symbol we're about to link to is in fact the
      // one that initiated this chain of renames. This leads to either infinite
      // loops or segfaults, depending on the phase of the moon.
      // It should also never happen, but with C++ code it does, because methods
      // are part of the type, and methods can take a full struct/object as a
      // parameter, not just a reference/pointer. So, that's a legitimate place
      // where we have this recursive symbol dependency situation.
      // The workaround to this is to just ignore it, and hope that it doesn't
      // become a problem in the future.
      return;
    }

    if (ns.lookup(type.identifier()))
    {
      // If we can just look up the current type symbol, use that.
      type2 = ns.follow((typet &)type);
    }
    else
    {
      // Otherwise, try to guess the namespaced type symbol
      std::string ident =
        cur_name_sym.module.as_string() + type.identifier().as_string();

      // Try looking that up.
      if (ns.lookup(irep_idt(ident)))
      {
        irept tmptype = type;
        tmptype.identifier(irep_idt(ident));
        type2 = ns.follow((typet &)tmptype);
      }
      else
      {
        // And if we fail
        log_error(
          "Can't resolve type symbol {} at symbol squashing time", ident);
        abort();
      }
    }

    type = type2;
    return;
  }

  rename_exprs(type, cur_name_sym, sname);
}

void goto_convert_functionst::rename_exprs(
  irept &expr,
  const symbolt &cur_name_sym,
  const irep_idt &sname)
{
  if (expr.id() == "pointer")
    return;

  // Walk children, but only descend mutably into a child that actually
  // contains something to rename. The const probe (expr_needs_rename /
  // type_needs_rename) reads without detaching; the mutable Forall_*
  // path below detaches every node it touches (a COW deep-copy under
  // sharing). On eca-rers-style inputs the expression trees are
  // massively shared and carry few or no renamable type symbols, so
  // gating each child on the probe prunes nearly all of the detaches
  // that dominated peak memory. Each child is probed once, then walked
  // mutably end-to-end, so the probe is not re-run as we recurse.
  Forall_irep (it, expr.get_sub())
    if (expr_needs_rename(*it, sname))
      rename_exprs(*it, cur_name_sym, sname);

  Forall_named_irep (it, expr.get_named_sub())
  {
    if (denotes_thrashable_subtype(it->first))
    {
      if (type_needs_rename(it->second, sname))
        rename_types(it->second, cur_name_sym, sname);
    }
    else if (expr_needs_rename(it->second, sname))
    {
      rename_exprs(it->second, cur_name_sym, sname);
    }
  }

  Forall_named_irep (it, expr.get_comments())
    if (expr_needs_rename(it->second, sname))
      rename_exprs(it->second, cur_name_sym, sname);
}

void goto_convert_functionst::wallop_type(
  irep_idt name,
  typename_mapt &typenames,
  const irep_idt &sname)
{
  std::set<irep_idt> in_progress;
  wallop_type_impl(name, typenames, sname, in_progress);
}

// Internal implementation with cycle detection
void goto_convert_functionst::wallop_type_impl(
  irep_idt name,
  typename_mapt &typenames,
  const irep_idt &sname,
  std::set<irep_idt> &in_progress)
{
  // Check if this type exists in the typenames map
  typename_mapt::iterator it = typenames.find(name);
  if (it == typenames.end())
  {
    // Type not found in map - might be a built-in type or already processed
    return;
  }

  std::set<irep_idt> &deps = it->second;

  // If this type doesn't depend on anything, no need to rename anything.
  if (deps.size() == 0)
    return;

  // Check if we're already processing this type (cycle detection)
  if (in_progress.find(name) != in_progress.end())
  {
    // We have a cycle - just return without processing to break the cycle
    // Don't clear dependencies as the original type processing will handle that
    return;
  }

  // Mark this type as being processed
  in_progress.insert(name);

  // Create a copy of dependencies to avoid modification during iteration
  std::set<irep_idt> deps_copy = deps;

  // Iterate over our dependencies ensuring they're resolved.
  for (const auto &dep : deps_copy)
    wallop_type_impl(dep, typenames, sname, in_progress);

  // And finally perform renaming.
  symbolt *s = context.find_symbol(name);
  if (s != nullptr)
  {
    typet t = s->get_type();
    rename_types(t, *s, sname);
    s->set_type(std::move(t));
  }

  deps.clear();

  // Remove from in_progress set as we're done processing this type
  in_progress.erase(name);
}

void goto_convert_functionst::thrash_type_symbols()
{
  // This function has one purpose: remove as many type symbols as possible.
  // This is easy enough by just following each type symbol that occurs and
  // replacing it with the value of the type name. However, if we have a pointer
  // in a struct to itself, this breaks down. Therefore, don't rename types of
  // pointers; they have a type already; they're pointers.

  // Collect a list of all type names. This is required before this entire
  // thing has no types, and there's no way (in C++ converted code at least)
  // to decide what name is a type or not.
  typename_sett names;
  context.foreach_operand([this, &names](const symbolt &s) {
    collect_expr(s.get_value(), names);
    collect_type(s.get_type(), names);
  });

  // No type symbols anywhere → nothing to thrash. The Clang C/C++
  // frontends expand user types eagerly, so `names` is empty or holds
  // only a handful of (self-referential) struct/union tags; bail out
  // before the dependency computation and the whole-context rename
  // walk when there's nothing to do.
  if (names.empty())
    return;

  // Try to compute their dependencies.

  typename_mapt typenames;
  context.foreach_operand([this, &names, &typenames](const symbolt &s) {
    if (names.find(s.id) != names.end())
    {
      typename_sett list;
      collect_expr(s.get_value(), list);
      collect_type(s.get_type(), list);
      typenames[s.id] = list;
    }
  });

  for (auto &it : typenames)
    it.second.erase(it.first);

  // Now, repeatedly rename all types. When we encounter a type that contains
  // unresolved symbols, resolve it first, then include it into this type.
  // This means that we recurse to whatever depth of nested types the user
  // has. With at least a meg of stack, I doubt that's really a problem.
  std::map<irep_idt, std::set<irep_idt>>::iterator it;
  for (it = typenames.begin(); it != typenames.end(); it++)
    wallop_type(it->first, typenames, it->first);

  // And now all the types have a fixed form, rename types in all existing code.
  // Probe each symbol's type/value with the read-only checks first; only
  // copy-out / rename / copy-back when there is actually a symbol type to
  // rewrite. The copy-out itself (get_type/get_value return by value) plus
  // the mutable rename walk are what detach the shared irep trees, so
  // skipping them for symbols with nothing to rename is the bulk of the win.
  context.Foreach_operand([this](symbolt &s) {
    if (type_needs_rename(s.get_type(), s.id))
    {
      typet t = s.get_type();
      rename_types(t, s, s.id);
      s.set_type(std::move(t));
    }
    if (expr_needs_rename(s.get_value(), s.id))
    {
      exprt v = s.get_value();
      rename_exprs(v, s, s.id);
      s.set_value(std::move(v));
    }
  });
}
