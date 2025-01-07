#include <goto-programs/goto_program.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_type.h>
#include <pointer-analysis/value_set_analysis.h>
#include <util/std_code.h>
#include <util/std_expr.h>
#include <goto-programs/abstract-interpretation/gcse.h>
#include <ostream>
#include <sstream>
#include <util/prefix.h>
#include <fmt/format.h>
// TODO: Do an points-to abstract interpreter
std::shared_ptr<value_set_analysist> cse_domaint::vsa = nullptr;

void cse_domaint::transform(
  goto_programt::const_targett from,
  goto_programt::const_targett to,
  ai_baset &,
  const namespacet &)
{
  const goto_programt::instructiont &instruction = *from;
  switch (instruction.type)
  {
  case ASSIGN:
  {
    const code_assign2t &code = to_code_assign2t(instruction.code);
    make_expression_available(code.source);
    // Expressions that contain the target will need to be recomputed
    havoc_expr(code.target, to);
    // Target may be an expression as well
    // TODO: skip recursive definitions only. For example, x = x + 1 should be skipped as 'x' isn't available.
    //make_expression_available(code.target);
  }
  break;

  case GOTO:
  case ASSERT:
  case ASSUME:
    make_expression_available(instruction.guard);
    break;
  case DECL:
    havoc_symbol(to_code_decl2t(instruction.code).value);
    break;
  case DEAD:
    havoc_symbol(to_code_dead2t(instruction.code).value);
    break;
  case RETURN:
  {
    const code_return2t &cr = to_code_return2t(instruction.code);
    make_expression_available(cr.operand);
    break;
  }
  case FUNCTION_CALL:
  {
    const code_function_call2t &func =
      to_code_function_call2t(instruction.code);

    // each operand should be available now (unless someone is doing a sideeffect)
#if 0
    // Skip functions for now, the abstract interpreter is not context-aware
    // so it can't deal with function parameters properly.
    // Problematic benchmark: https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks/-/raw/main/c/reducercommutativity/sum05-2.yml
    for(const expr2tc &e : func.operands)
      make_expression_available(e);
#endif
    if (func.ret)
    {
      havoc_expr(func.ret, to);
      make_expression_available(func.ret);
    }

    break;
  }
  default:;
  }
}

void cse_domaint::output(std::ostream &out) const
{
  if (is_bottom())
  {
    out << "BOTTOM\n";
    return;
  }

  out << "{";
  for (const auto &e : available_expressions)
  {
    out << e->pretty(0) << "\n";
  }
  out << "}";
}

bool cse_domaint::merge(
  const cse_domaint &b,
  goto_programt::const_targett from,
  goto_programt::const_targett to)
{
  /* This analysis is supposed to be used for CFGs.
   * Since we do not have a CFG GOTO abstract interpreter we
   * simulate one by just passing through common instructions
   * and only doing intersections at target destinations */

  if (!(to->is_target() || from->is_function_call()) || is_bottom())
  {
    bool changed = available_expressions != b.available_expressions;
    available_expressions = b.available_expressions;
    return changed;
  }

  size_t size_before_intersection = available_expressions.size();
  for (auto it = available_expressions.begin();
       it != available_expressions.end();
       it++)
  {
    if (!b.available_expressions.count(*it))
    {
      it = available_expressions.erase(it);
      if (it == available_expressions.end())
        break;
    }
  }

  return size_before_intersection != available_expressions.size();
}

void cse_domaint::make_expression_available(const expr2tc &E)
{
  if (!E)
    return;

  // Logical expressions can be short-circuited
  // So, only the first operand will be available
  if (is_and2t(E) || is_or2t(E))
  {
    make_expression_available(
      is_and2t(E) ? to_and2t(E).side_1 : to_or2t(E).side_1);
    return;
  }

  // ESBMC requires that an overflow2t contains an operator as a sub-expression.
  // This means that we can't cache the sub-expressions into an intermediate
  // var.
  // However, we can do the GCSE in the operands of the nested operator.
  if (is_overflow2t(E))
  {
    expr2tc operand = to_overflow2t(E).operand;
    operand->Foreach_operand(
      [this](expr2tc &op) { make_expression_available(op); });
    return;
  }

  if (is_if2t(E))
  {
    make_expression_available(to_if2t(E).cond);
    return;
  }
  // Skip sideeffects
  if (is_sideeffect2t(E))
    return;

  // TODO: I hate floats
  if (is_floatbv_type(E))
    return;

  auto added = available_expressions.insert(E);
  // Did we check it already?
  if (!added.second)
    return;
  // Let's recursively make it available!
  E->foreach_operand(
    [this](const expr2tc &e) { make_expression_available(e); });

  // TODO: LHS members should always be recomputed
  if (is_with2t(E) || is_member2t(E) || is_dereference2t(E))
    available_expressions.erase(E);
}

bool cse_domaint::should_remove_expr(const expr2tc &taint, const expr2tc &E)
  const
{
  if (!taint)
    return false;

  if (taint == E)
    return true;

  if (!E) // not sure when this can happen
    return false;

  bool result = false;
  E->foreach_operand([this, &result, &taint](const expr2tc &e) {
    result |= should_remove_expr(taint, e);
  });
  return result;
}

bool cse_domaint::should_remove_expr(const irep_idt &sym, const expr2tc &E)
  const
{
  if (!E)
    return false;

  if (is_symbol2t(E) && to_symbol2t(E).thename == sym)
    return true;

  bool result = false;
  E->foreach_operand([this, &result, &sym](const expr2tc &e) {
    result |= should_remove_expr(sym, e);
  });
  return result;
}

void cse_domaint::havoc_symbol(const irep_idt &sym)
{
  std::vector<expr2tc> to_remove;
  for (auto x : available_expressions)
  {
    if (should_remove_expr(sym, x))
      to_remove.push_back(x);
  }
  for (auto x : to_remove)
    available_expressions.erase(x);
}

void cse_domaint::havoc_expr(
  const expr2tc &target,
  const goto_programt::const_targett &i_it)
{
  if (is_dereference2t(target) && vsa != nullptr)
  {
    auto state = (*vsa)[i_it];
    value_setst::valuest dest;
    state.value_set->get_reference_set(target, dest);
    for (const auto &x : dest)
    {
      if (is_object_descriptor2t(x))
        havoc_expr(to_object_descriptor2t(x).object, i_it);
      else
      {
        log_error("Unsupported descriptor: {}", *x);
      }
    }
  }
  std::vector<expr2tc> to_remove;
  for (auto x : available_expressions)
  {
    if (should_remove_expr(target, x))
      to_remove.push_back(x);
  }
  for (auto x : to_remove)
    available_expressions.erase(x);
}

bool goto_cse::runOnProgram(goto_functionst &F)
{
  // Initialization for the abstract analysis.
  const namespacet ns(context);
  log_status("{}", "[CSE] Computing Available Expressions for program");
  available_expressions(F, ns);
  log_status("{}", "[CSE] Finished computing AE for program");
  // Let's release the reference. TODO: create the "VSA aware" abstract interpreter
  cse_domaint::vsa = nullptr;
  return false;
}

expr2tc
goto_cse::obtain_max_sub_expr(const expr2tc &e, const cse_domaint &state) const
{
  if (!e)
    return expr2tc();

  // No need to add primitives

  if (is_constant(e) || is_symbol2t(e))
    return expr2tc();

  // TODO
  if (is_array_type(e->type))
    return expr2tc();

  if (state.available_expressions.count(e))
    return e;

  expr2tc result = expr2tc();
  e->foreach_operand([this, &result, &state](const expr2tc e_inner) {
    if (!result && e_inner)
      result = obtain_max_sub_expr(e_inner, state);
  });
  return result;
}

void goto_cse::replace_max_sub_expr(
  expr2tc &e,
  const std::unordered_map<expr2tc, expr2tc, irep2_hash> &expr2symbol,
  const goto_programt::const_targett &to,
  std::unordered_set<expr2tc, irep2_hash> &matched_expressions) const
{
  if (!e)
    return;

  // No need to add primitives
  if (is_constant(e) || is_symbol2t(e))
    return;

  // NOTE: See `make_expression_available`.
  if (is_overflow2t(e))
  {
    expr2tc operand = to_overflow2t(e).operand;
    operand->Foreach_operand(
      [this, &expr2symbol, &to, &matched_expressions](expr2tc &op) {
        replace_max_sub_expr(op, expr2symbol, to, matched_expressions);
      });
    return;
  }

  auto common = expr2symbol.find(e);
  if (common != expr2symbol.end())
  {
    matched_expressions.emplace(e);
    e = common->second;
    return;
  }
  e->Foreach_operand(
    [this, &expr2symbol, &to, &matched_expressions](expr2tc &e0) {
      replace_max_sub_expr(e0, expr2symbol, to, matched_expressions);
    });
}

bool goto_cse::runOnFunction(std::pair<const dstring, goto_functiont> &F)
{
  if (!F.second.body_available)
    return false;

  // 1. Let's count expressions, the idea is to go through all program statements
  //    and check if any sub-expr is already available
  std::unordered_set<expr2tc, irep2_hash> expressions_set;
  for (auto it = (F.second.body).instructions.begin();
       it != (F.second.body).instructions.end();
       ++it)
  {
    const cse_domaint &state = available_expressions[it];
    const expr2tc max_sub = obtain_max_sub_expr(it->code, state);

    if (!max_sub)
      continue;

    expressions_set.insert(max_sub);
  }

  // 2. Instrument new tmp symbols at the start of the function
  std::unordered_map<expr2tc, expr2tc, irep2_hash> expr2symbol;
  auto it = (F.second.body).instructions.begin();
  const cse_domaint &state = available_expressions[it];
  for (const expr2tc &e : expressions_set)
  {
    symbolt symbol = create_cse_symbol(e->type, it);
    symbolt *symbol_in_context = context.move_symbol_to_context(symbol);
    const expr2tc &symbol_as_expr = symbol2tc(e->type, symbol_in_context->id);

    if (state.available_expressions.count(e))
    {
      // TMP_SYMBOL = e;
      goto_programt::instructiont init;
      init.make_assignment();
      init.code = code_assign2tc(symbol_as_expr, e);
      init.location = it->location;
      init.function = it->function;
      F.second.body.insert_swap(it, init);
    }
    // TYPE TMP_SYMBOL;
    goto_programt::instructiont decl;
    decl.make_decl();
    decl.code = code_decl2tc(e->type, symbol_in_context->id);
    decl.location = it->location;
    F.second.body.insert_swap(it, decl);

    expr2symbol[e] = symbol_as_expr;
  }

  std::unordered_set<expr2tc, irep2_hash> initialized;
  // 3. Final step, let's initialize the symbols and replace the expressions!
  for (auto it = (F.second.body).instructions.begin();
       it != (F.second.body).instructions.end();
       ++it)
  {
    if (!available_expressions.target_is_mapped(it))
      continue;

    const cse_domaint &state = available_expressions[it];
    // Most symbols are early initialized:
    // X = A + B ===> tmp = A + B; X = tmp;
    // However, when changing dereferences we need to them posterior
    // a[i] = &addr; *a[i] = 42 ===> a[i] = &addr; tmp = a[i]; *tmp = 42;

    if (it->is_target())
    {
      // This might be a loop or an else statement.
      // TODO: clear only expressions that are no longer available
      initialized.clear();
    }
    std::unordered_set<expr2tc, irep2_hash> matched_pre_expressions;
    std::unordered_set<expr2tc, irep2_hash> matched_post_expressions;
    switch (it->type)
    {
    case GOTO:
    case ASSUME:
    case ASSERT:
      replace_max_sub_expr(it->guard, expr2symbol, it, matched_pre_expressions);
      break;
    case FUNCTION_CALL:
      replace_max_sub_expr(
        to_code_function_call2t(it->code).function,
        expr2symbol,
        it,
        matched_pre_expressions);
      break;
    case RETURN:
      replace_max_sub_expr(it->code, expr2symbol, it, matched_pre_expressions);
      break;
    case ASSIGN:
      replace_max_sub_expr(
        to_code_assign2t(it->code).source,
        expr2symbol,
        it,
        matched_pre_expressions);
      replace_max_sub_expr(
        to_code_assign2t(it->code).target,
        expr2symbol,
        it,
        matched_post_expressions);
      break;
    default:
      continue;
    }

    std::unordered_set<expr2tc, irep2_hash> local_initialized;
    for (auto &x : matched_pre_expressions)
    {
      if (!state.available_expressions.count(x) || !initialized.count(x))
      {
        goto_programt::instructiont instruction;
        instruction.make_assignment();
        instruction.code = code_assign2tc(expr2symbol[x], x);
        instruction.location = it->location;
        instruction.function = it->function;
        F.second.body.insert_swap(it, instruction);
        initialized.insert(x);
        local_initialized.insert(x);
      }
    }

    if (!matched_post_expressions.size())
      continue;

    // So far, only assignments are supported
    for (auto &x : matched_post_expressions)
    {
      // First time seeing the expr
      if (!state.available_expressions.count(x) || !initialized.count(x))
      {
        goto_programt::instructiont instruction;
        instruction.make_assignment();
        instruction.code = code_assign2tc(expr2symbol[x], x);
        instruction.location = it->location;
        instruction.function = it->function;

        if (!local_initialized.count(x))
        {
          F.second.body.insert_swap(it, instruction);
          initialized.insert(x);
        }
      }
    }
  }
  return true;
}

inline symbolt goto_cse::create_cse_symbol(
  const type2tc &t,
  const goto_programt::const_targett &to)
{
  symbolt symbol;
  symbol.type = migrate_type_back(t);
  symbol.id = fmt::format("{}${}", prefix, symbol_counter++);
  symbol.name = symbol.id;
  symbol.mode = "C";
  symbol.location = to->location;
  return symbol;
}
