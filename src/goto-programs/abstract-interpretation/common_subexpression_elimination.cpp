#include "goto-programs/goto_program.h"
#include "irep2/irep2_expr.h"
#include "util/std_code.h"
#include <goto-programs/abstract-interpretation/common_subexpression_elimination.h>
#include <ostream>
#include <sstream>
#include <util/prefix.h>
#include <fmt/format.h>
// TODO: Do an points-to abstract interpreter
std::unique_ptr<value_set_analysist> cse_domaint::vsa = nullptr;

void cse_domaint::transform(
  goto_programt::const_targett from,
  goto_programt::const_targett to,
  ai_baset &,
  const namespacet &)
{
  const goto_programt::instructiont &instruction = *from;
  switch(instruction.type)
  {
  case ASSIGN:
  {
    const code_assign2t &code = to_code_assign2t(instruction.code);
    make_expression_available(code.source);
    // Expressions that contain the target will need to be recomputed
    havoc_expr(code.target, to);
    // Target may be an expression as well
    // TODO: skip only recursive definitions. For example, x = x + 1 should be skipped
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
    break;
#if 0
    // TODO: https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks/-/raw/main/c/reducercommutativity/sum05-2.yml
    const code_function_call2t &func =
      to_code_function_call2t(instruction.code);
    // each operand should be available now (unless someone is doing a sideeffect)
    for(const expr2tc &e : func.operands)
      make_expression_available(e);
    if(func.ret)
    {
      havoc_expr(func.ret, to);
      make_expression_available(func.ret);
    }
#endif
  }
  default:;
  }
}

void cse_domaint::output(std::ostream &out) const
{
  if(is_bottom())
  {
    out << "BOTTOM\n";
    return;
  }

  out << "{";
  for(const auto &e : available_expressions)
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

  if(!(to->is_target() || from->is_function_call()) || is_bottom())
  {
    bool changed = available_expressions != b.available_expressions;
    available_expressions = b.available_expressions;
    return changed;
  }

  size_t size_before_intersection = available_expressions.size();
  for(auto it = available_expressions.begin();
      it != available_expressions.end();
      it++)
  {
    if(!b.available_expressions.count(*it))
    {
      it = available_expressions.erase(it);
      if(it == available_expressions.end())
        break;
    }
  }

  return size_before_intersection != available_expressions.size();
}

void cse_domaint::make_expression_available(const expr2tc &E)
{
  if(!E)
    return;

  // Skip sideeffects
  if(is_sideeffect2t(E))
    return;

  // TODO: I hate floats
  if(is_floatbv_type(E))
    return;

  // Did we check it already?
  if(available_expressions.count(E))
    return;

  // Let's recursively make it available!
  E->foreach_operand(
    [this](const expr2tc &e) { make_expression_available(e); });

  // TODO: LHS members should always be recomputed
  if(is_with2t(E) || is_member2t(E) || is_dereference2t(E))
    return;

  available_expressions.insert(E);
}

bool cse_domaint::should_remove_expr(const expr2tc &taint, const expr2tc &E)
  const
{
  if(!taint)
    return false;

  if(taint == E)
    return true;

  if(!E) // not sure when this can happen
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
  if(!E)
    return false;

  if(is_symbol2t(E) && to_symbol2t(E).thename == sym)
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
  for(auto x : available_expressions)
  {
    if(should_remove_expr(sym, x))
      to_remove.push_back(x);
  }
  for(auto x : to_remove)
    available_expressions.erase(x);
}

void cse_domaint::havoc_expr(
  const expr2tc &target,
  const goto_programt::const_targett &i_it)
{
  if(is_dereference2t(target) && vsa != nullptr)
  {
    auto state = (*vsa)[i_it];
    value_setst::valuest dest;
    state.value_set->get_reference_set(target, dest);
    for(const auto &x : dest)
    {
      if(is_object_descriptor2t(x))
        havoc_expr(to_object_descriptor2t(x).object, i_it);
      else
      {
        log_error("Unsupported descriptor: {}", *x);
      }
    }
  }
  std::vector<expr2tc> to_remove;
  for(auto x : available_expressions)
  {
    if(should_remove_expr(target, x))
      to_remove.push_back(x);
  }
  for(auto x : to_remove)
    available_expressions.erase(x);
}

bool goto_cse::runOnProgram(goto_functionst &F)
{
  // Initialization for the abstract analysis.
  const namespacet ns(context);

  try
  {
    cse_domaint::vsa = std::make_unique<value_set_analysist>(ns);
    (*cse_domaint::vsa)(F);
    log_status("{}", "[CSE] Computing Available Expressions for program");
    available_expressions(F, ns);
    log_status("{}", "[CSE] Finished computing AE for program");
    program_initialized = true;
  }
  catch(...)
  {
    program_initialized = false;
    log_error("Unable to initialize the GCSE");
  }

  return false;
}

expr2tc
goto_cse::obtain_max_sub_expr(const expr2tc &e, const cse_domaint &state) const
{
  if(!e)
    return expr2tc();

  // No need to add primitives
  if(is_constant(e) || is_symbol2t(e))
    return expr2tc();

  if(state.available_expressions.count(e))
    return e;

  expr2tc result = expr2tc();
  e->foreach_operand([this, &result, &state](const expr2tc e_inner) {
    if(!result && e_inner)
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
  if(!e)
    return;

  // No need to add primitives
  if(is_constant(e) || is_symbol2t(e))
    return;

  auto common = expr2symbol.find(e);
  if(common != expr2symbol.end())
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
  if(!program_initialized)
    return false;

  if(!F.second.body_available)
    return false;

  goto_loopst goto_loops(F.first, _goto_functions, F.second);
  // Helper function to check wether the instruction belongs in a loop
  // loops are trickier because they require the variable to be aways
  // initialized.
  // TODO: it might be worth to have a reverse index inside goto_loopst
  // TODO: we should only initialize once per loop!
  // TODO: variables that are not modified inside the loop could be skipped
  // TODO: besides available expressions, we could use the same domain in
  //       reverse to compute expressions that could be removed from loops
  auto is_in_loop = [&goto_loops](const goto_programt::targett it) {
    for(const auto &l : goto_loops.get_loops())
    {
      // There must be a better way of doing this :O
      goto_programt::targett l_begin = l.get_original_loop_head();
      const goto_programt::targett l_exit = l.get_original_loop_exit();
      if(l_begin == it)
        return true;
      while(l_begin != l_exit)
      {
        if(l_begin == it)
          return true;
        l_begin++;
      }
    }
    return false;
  };

  // 1. Let's count expressions, the idea is to go through all program statements
  //    and check if any sub-expr is already available
  std::unordered_set<expr2tc, irep2_hash> expressions_set;
  //expressions_map expressions;
  for(auto it = (F.second.body).instructions.begin();
      it != (F.second.body).instructions.end();
      ++it)
  {
    const cse_domaint &state = available_expressions[it];
    const expr2tc max_sub = obtain_max_sub_expr(it->code, state);

    if(!max_sub)
      continue;

    expressions_set.insert(max_sub);
  }

  // 2. Instrument new tmp symbols at the start of the function
  std::unordered_map<expr2tc, expr2tc, irep2_hash> expr2symbol;
  auto it = (F.second.body).instructions.begin();
  const cse_domaint &state = available_expressions[it];
  for(const expr2tc &e : expressions_set)
  {
    symbolt symbol = create_cse_symbol(e->type, it);
    symbolt *symbol_in_context = context.move_symbol_to_context(symbol);
    const expr2tc &symbol_as_expr = symbol2tc(e->type, symbol_in_context->id);

    if(state.available_expressions.count(e))
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

  // TODO: this initialized set is used to be sure that it was initialized at least
  //       once. It is a hack though.
  std::unordered_set<expr2tc, irep2_hash> initialized;
  // 3. Final step, let's initialize the symbols and replace the expressions!
  for(auto it = (F.second.body).instructions.begin();
      it != (F.second.body).instructions.end();
      ++it)
  {
    if(!available_expressions.target_is_mapped(it))
      continue;

    const cse_domaint &state = available_expressions[it];
    // Most symbols are early initialized:
    // X = A + B ===> tmp = A + B; X = tmp;
    // However, when changing dereferences we need to them posterior
    // a[i] = &addr; *a[i] = 42 ===> a[i] = &addr; tmp = a[i]; *tmp = 42;

    const expr2tc bck = it->code;
    std::unordered_set<expr2tc, irep2_hash> matched_pre_expressions;
    std::unordered_set<expr2tc, irep2_hash> matched_post_expressions;
    switch(it->type)
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

    for(auto &x : matched_pre_expressions)
    {
      if(
        !state.available_expressions.count(x) || is_in_loop(it) ||
        !initialized.count(x))
      {
        goto_programt::instructiont instruction;
        instruction.make_assignment();
        instruction.code = code_assign2tc(expr2symbol[x], x);
        instruction.location = it->location;
        instruction.function = it->function;
        F.second.body.insert_swap(it, instruction);

        initialized.insert(x);
      }
    }

    if(!matched_post_expressions.size())
      continue;

    // So far, only assignments are supported
    assert(is_code_assign2t(bck));
    auto &cpy = to_code_assign2t(bck);

    for(auto &x : matched_post_expressions)
    {
      // First time seeing the expr
      if(
        !state.available_expressions.count(x) || is_in_loop(it) ||
        !initialized.count(x))
      {
        it->make_skip();
        goto_programt::instructiont instruction;
        instruction.make_assignment();
        instruction.code = code_assign2tc(cpy.target, cpy.source);
        instruction.location = it->location;
        instruction.function = it->function;

        goto_programt::instructiont instruction2;
        instruction2.make_assignment();
        instruction2.code = code_assign2tc(expr2symbol[x], x);
        instruction2.location = it->location;
        instruction2.function = it->function;

        F.second.body.insert_swap(it, instruction2);
        F.second.body.insert_swap(it, instruction);
        initialized.insert(x);
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
