#include <goto-programs/goto_atomicity_check.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/migrate.h>
#include <util/symbol_generator.h>

class goto_atomicity_checkt
{
public:
  goto_atomicity_checkt(const namespacet &_ns, contextt &_context)
    : ns(_ns), context(_context), tmp_sym("goto_atomicity::")
  {
  }

  /// Instruments all ASSIGN, ASSERT, ASSUME, GOTO, and RETURN instructions
  /// in @p program that reference global variables, inserting snapshot
  /// temporaries and atomicity assertions.
  void check(goto_programt &program);

private:
  const namespacet &ns;
  contextt &context;
  symbol_generator tmp_sym;

  /// Returns true iff the named symbol is a global (static_lifetime or
  /// dynamically allocated), excluding ESBMC internals.
  bool is_global(const irep_idt &id) const;

  /// Counts global variable reads reachable in expr (does not descend into
  /// address_of sub-expressions).
  unsigned count_globals(const expr2tc &expr) const;

  /// For each global read found in expr, creates a fresh static temp symbol,
  /// appends DECL+ASSIGN instructions to new_code, and appends the equality
  /// (tmp == global_expr) to conjuncts. Returns true if any global was found.
  bool collect_globals(
    const expr2tc &expr,
    const locationt &loc,
    goto_programt &new_code,
    std::vector<expr2tc> &conjuncts);

  /// Replicates the heuristic from break_globals2assignments(exprt&,...):
  /// returns false (skip) when the first operand of guard is a complex
  /// expression or references a pthread symbol.
  bool should_instrument_guard(const expr2tc &guard) const;

  /// Instruments an ASSIGN instruction: inserts snapshot vars + optional
  /// ATOMIC_BEGIN + ASSERT before it, and ATOMIC_END after it when the
  /// assignment target is itself a global.
  void instrument_assign(goto_programt &program, goto_programt::targett it);

  /// Instruments a guard-bearing instruction (ASSERT/ASSUME/GOTO/RETURN):
  /// inserts snapshot vars + bare ASSERT into new_code (no ATOMIC wrapping).
  void instrument_guard(
    goto_programt::targett it,
    goto_programt &new_code);
};

bool goto_atomicity_checkt::is_global(const irep_idt &id) const
{
  if (id == "__ESBMC_alloc" || id == "__ESBMC_alloc_size")
    return false;
  const symbolt *sym = ns.lookup(id);
  if (!sym)
    return false;
  return sym->static_lifetime || sym->type.is_dynamic_set();
}

unsigned goto_atomicity_checkt::count_globals(const expr2tc &expr) const
{
  if (is_nil_expr(expr) || is_address_of2t(expr))
    return 0;
  if (is_symbol2t(expr))
    return is_global(to_symbol2t(expr).thename) ? 1 : 0;
  unsigned n = 0;
  expr->foreach_operand(
    [this, &n](const expr2tc &e) { n += count_globals(e); });
  return n;
}

bool goto_atomicity_checkt::collect_globals(
  const expr2tc &expr,
  const locationt &loc,
  goto_programt &new_code,
  std::vector<expr2tc> &conjuncts)
{
  if (is_nil_expr(expr) || is_address_of2t(expr))
    return false;

  auto snapshot = [&](const expr2tc &global_expr) {
    typet old_type = migrate_type_back(global_expr->type);
    symbolt &tmp = tmp_sym.new_symbol(context, old_type, "tmp$");
    tmp.static_lifetime = true;

    type2tc sym_type = global_expr->type;
    expr2tc sym_expr = symbol2tc(sym_type, tmp.id);

    new_code.add_instruction(DECL)->code = code_decl2tc(sym_type, tmp.id);
    new_code.instructions.back().location = loc;

    goto_programt::targett asgn = new_code.add_instruction(ASSIGN);
    asgn->code = code_assign2tc(sym_expr, global_expr);
    asgn->location = loc;

    conjuncts.push_back(equality2tc(sym_expr, global_expr));
  };

  if (is_symbol2t(expr))
  {
    if (!is_global(to_symbol2t(expr).thename))
      return false;
    snapshot(expr);
    return true;
  }

  // For compound memory reads, check the base object identifier.
  // Mirrors the original break_globals2assignments_rec logic:
  //   dereference/implicit_dereference/member: use op0's identifier
  //   index: use op1's identifier (the index operand, matching original op1())
  irep_idt base_id;
  if (is_dereference2t(expr))
  {
    const expr2tc &ptr = to_dereference2t(expr).value;
    if (is_symbol2t(ptr))
      base_id = to_symbol2t(ptr).thename;
  }
  else if (is_member2t(expr))
  {
    const expr2tc &src = to_member2t(expr).source_value;
    if (is_symbol2t(src))
      base_id = to_symbol2t(src).thename;
  }
  else if (is_index2t(expr))
  {
    // Preserve original behaviour: op1() was the index, not the array
    const expr2tc &idx = to_index2t(expr).index;
    if (is_symbol2t(idx))
      base_id = to_symbol2t(idx).thename;
  }

  if (!base_id.empty() && is_global(base_id))
  {
    snapshot(expr);
    return true;
  }

  bool found = false;
  expr->foreach_operand([&](const expr2tc &e) {
    if (collect_globals(e, loc, new_code, conjuncts))
      found = true;
  });
  return found;
}

bool goto_atomicity_checkt::should_instrument_guard(
  const expr2tc &guard) const
{
  if (is_nil_expr(guard))
    return false;

  // Get the first operand of the guard expression.
  expr2tc first_op;
  guard->foreach_operand([&first_op](const expr2tc &e) {
    if (is_nil_expr(first_op) && !is_nil_expr(e))
      first_op = e;
  });

  // No operands — it is a leaf (e.g. a bare symbol); proceed.
  if (is_nil_expr(first_op))
    return true;

  // Skip guards whose first operand references a pthread symbol.
  if (is_symbol2t(first_op))
  {
    if (
      id2string(to_symbol2t(first_op).thename).find("pthread") !=
      std::string::npos)
      return false;
  }

  // Skip when the first operand is itself complex (has sub-operands).
  bool first_has_ops = false;
  first_op->foreach_operand(
    [&first_has_ops](const expr2tc &) { first_has_ops = true; });
  return !first_has_ops;
}

static expr2tc make_conjunction(const std::vector<expr2tc> &conjuncts)
{
  assert(!conjuncts.empty());
  expr2tc result = conjuncts[0];
  for (std::size_t i = 1; i < conjuncts.size(); ++i)
    result = and2tc(result, conjuncts[i]);
  return result;
}

void goto_atomicity_checkt::instrument_assign(
  goto_programt &program,
  goto_programt::targett it)
{
  assert(it->is_assign() && is_code_assign2t(it->code));
  const code_assign2t &assign = to_code_assign2t(it->code);
  const locationt &loc = it->location;

  // Skip compiler-generated temporaries.
  if (
    is_symbol2t(assign.target) &&
    id2string(to_symbol2t(assign.target).thename).find("tmp$") !=
      std::string::npos)
    return;

  unsigned lhs_globals = count_globals(assign.target);

  goto_programt new_code;
  std::vector<expr2tc> conjuncts;
  collect_globals(assign.source, loc, new_code, conjuncts);

  if (conjuncts.empty())
    return;

  bool need_atomic = lhs_globals > 0;
  if (need_atomic)
    new_code.add_instruction(ATOMIC_BEGIN)->location = loc;

  std::string comment =
    is_symbol2t(assign.target)
      ? "atomicity violation on assignment to " +
          id2string(to_symbol2t(assign.target).thename)
      : "atomicity violation";

  goto_programt::targett asrt = new_code.add_instruction(ASSERT);
  asrt->make_assertion(make_conjunction(conjuncts));
  asrt->location = loc;
  asrt->location.comment(comment);
  asrt->location.property("atomicity");

  // insert_swap preserves any jump targets that pointed at 'it'.
  // After the call, 'it' points to the first inserted instruction.
  std::size_t n = new_code.instructions.size();
  program.insert_swap(it, new_code);

  if (need_atomic)
  {
    // Advance past the newly inserted instructions to reach the original ASSIGN.
    goto_programt::targett orig = it;
    std::advance(orig, n);

    goto_programt::targett ae = program.insert(std::next(orig));
    ae->make_atomic_end();
    ae->location = loc;
  }
}

void goto_atomicity_checkt::instrument_guard(
  goto_programt::targett it,
  goto_programt &new_code)
{
  expr2tc guard_expr;
  if (it->is_assert() || it->is_assume() || it->is_goto())
    guard_expr = it->guard;
  else if (it->is_return() && is_code_return2t(it->code))
    guard_expr = to_code_return2t(it->code).operand;
  else
    return;

  if (
    is_nil_expr(guard_expr) || is_true(guard_expr) || is_false(guard_expr) ||
    !should_instrument_guard(guard_expr))
    return;

  const locationt &loc = it->location;
  std::vector<expr2tc> conjuncts;
  collect_globals(guard_expr, loc, new_code, conjuncts);

  if (conjuncts.empty())
    return;

  goto_programt::targett asrt = new_code.add_instruction(ASSERT);
  asrt->make_assertion(make_conjunction(conjuncts));
  asrt->location = loc;
  asrt->location.comment("atomicity violation");
  asrt->location.property("atomicity");
}

void goto_atomicity_checkt::check(goto_programt &program)
{
  // Pre-collect the iterators of all instructions we want to instrument.
  // This two-pass approach avoids re-processing the instructions we insert.
  std::vector<goto_programt::targett> assign_instrs, guard_instrs;

  for (auto it = program.instructions.begin();
       it != program.instructions.end();
       ++it)
  {
    if (it->is_assign())
      assign_instrs.push_back(it);
    else if (
      it->is_assert() || it->is_assume() || it->is_goto() || it->is_return())
      guard_instrs.push_back(it);
  }

  for (auto it : assign_instrs)
    instrument_assign(program, it);

  for (auto it : guard_instrs)
  {
    goto_programt new_code;
    instrument_guard(it, new_code);
    if (!new_code.instructions.empty())
      program.insert_swap(it, new_code);
  }
}

void goto_atomicity_check(
  goto_functionst &goto_functions,
  const namespacet &ns,
  contextt &context)
{
  goto_atomicity_checkt checker(ns, context);
  for (auto &kv : goto_functions.function_map)
    if (!kv.second.body.empty())
      checker.check(kv.second.body);
  goto_functions.update();
}
