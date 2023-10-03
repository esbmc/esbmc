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
    make_expression_available(code.target);
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
    for(const expr2tc &e : func.operands)
      make_expression_available(e);
    if(func.ret)
      havoc_expr(func.ret, to);
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
 
  if(!(to->is_target() || from->is_function_call()))
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
      it = available_expressions.erase(it);
  }

  return size_before_intersection != available_expressions.size();
}

void cse_domaint::make_expression_available(const expr2tc &E)
{
  if(!E)
    return;

  // Skip nondets
  if(
    is_sideeffect2t(E) &&
    to_sideeffect2t(E).kind == sideeffect_data::allockind::nondet)
    return;

  // I hate floats
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
  // Not sure how this can happens
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
  {
    available_expressions.erase(x);
  }
}
bool goto_cse::common_expression::should_add_symbol(
  const goto_programt::const_targett &it,
  size_t threshold) const
{
  auto index = std::find(gen.begin(), gen.end(), it);
  return index == gen.end() ? false : true;
  //           : sequence_counter.at(index - gen.begin()) >= threshold;
}

expr2tc goto_cse::common_expression::get_symbol_for_target(
  const goto_programt::const_targett &it) const
{
  assert(gen.size());

  auto index = std::find_if(
    gen.begin(), gen.end(), [&it](const goto_programt::const_targett &v) {
      return v->location_number >= it->location_number;
    });

  if(index == gen.end())
  {
    // Either last symbol or nothing
    return symbol.at(gen.size() - 1);
  }

  //  index--;
  return symbol.at(index - gen.begin());
}

#include <memory>

bool goto_cse::runOnProgram(goto_functionst &F)
{
  const namespacet ns(context);
  log_debug("{}", "[CSE] Computing Points-To for program");
  // Initialization for the abstract analysis.

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
    //abort();
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

  // 1. Let's count expressions, the idea is to go through all program statements
  //    and check if any sub-expr is already available
  expressions_map expressions;
  for(auto it = (F.second.body).instructions.begin();
      it != (F.second.body).instructions.end();
      ++it)
  {
    const cse_domaint &state = available_expressions[it];

    // Are the intervals still available (TODO: this can be parallel)
    for(auto &exp : expressions)
    {
      if(exp.second.available && !state.available_expressions.count(exp.first))
	{
      exp.second.available = false;
      exp.second.kill.push_back(it);
	}
    }
    

    const expr2tc max_sub = obtain_max_sub_expr(it->code, state);
    if(max_sub == expr2tc())
      continue;

    auto E = expressions.find(max_sub);

    // Are we seeing it for the first time?/
    if(E == expressions.end())
    {
      common_expression exp;
      E = expressions.emplace(std::make_pair(max_sub, exp)).first;
    }

    E->second.available = true;
    E->second.gen.push_back(it);
    // Was the expression unavailable?
    if(!E->second.available)
    {
      E->second.sequence_counter[E->second.gen.size() - 1] = 0;
    }

    E->second.sequence_counter[E->second.gen.size() - 1] += 1;
  }

  // Early exit, if no symbols.

  // 3. Instrument new tmp symbols
  std::unordered_map<expr2tc, expr2tc, irep2_hash> expr2symbol;
  auto it = (F.second.body).instructions.begin();
  const cse_domaint &state = available_expressions[it];
  for(auto &[e, cse] : expressions)
  {
    symbolt symbol = create_cse_symbol(e->type, it);
    auto magic = context.move_symbol_to_context(symbol);
    const auto symbol_as_expr = symbol2tc(e->type, magic->id);

    goto_programt::instructiont init;
    init.make_assignment();
    init.code = code_assign2tc(symbol_as_expr, e);
    init.location = it->location;
    init.function = it->function;

    goto_programt::instructiont decl;
    decl.make_decl();
    decl.code = code_decl2tc(e->type, magic->id);
    decl.location = it->location;

    if(state.available_expressions.count(e))
      F.second.body.insert_swap(it, init);

    F.second.body.insert_swap(it, decl);

    expr2symbol[e] = symbol_as_expr;

    cse.symbol.push_back(symbol_as_expr);
  }

  // 4. Final step, let's replace the symbols!
  for(auto it = (F.second.body).instructions.begin();
      it != (F.second.body).instructions.end();
      ++it)
  {
    if(!available_expressions.target_is_mapped(it))
      continue;

    const cse_domaint &state = available_expressions[it];

    if(it->is_goto() || it->is_assume() || it->is_assert())
    {
      std::unordered_set<expr2tc, irep2_hash> matched_expressions;
      replace_max_sub_expr(it->guard, expr2symbol, it, matched_expressions);
      for(auto &x : matched_expressions)
      {
        if(!state.available_expressions.count(x))
        {
          goto_programt::instructiont instruction;
          instruction.make_assignment();
          instruction.code = code_assign2tc(expr2symbol[x], x);
          instruction.location = it->location;
          instruction.function = it->function;
          F.second.body.insert_swap(it, instruction);
        }
      }
      continue;
    }

    if(it->is_function_call())
    {
      std::unordered_set<expr2tc, irep2_hash> matched_expressions;
      code_function_call2t &function = to_code_function_call2t(it->code);
      replace_max_sub_expr(
        function.function, expr2symbol, it, matched_expressions);
      for(auto &x : matched_expressions)
      {
        if(!state.available_expressions.count(x))
        {
          goto_programt::instructiont instruction;
          instruction.make_assignment();
          instruction.code = code_assign2tc(expr2symbol[x], x);
          instruction.location = it->location;
          instruction.function = it->function;
          F.second.body.insert_swap(it, instruction);
        }
      }
    }

    if(it->is_return())
    {
      std::unordered_set<expr2tc, irep2_hash> matched_expressions;
      replace_max_sub_expr(it->code, expr2symbol, it, matched_expressions);
      for(auto &x : matched_expressions)
      {
        if(!state.available_expressions.count(x))
        {
          goto_programt::instructiont instruction;
          instruction.make_assignment();
          instruction.code = code_assign2tc(expr2symbol[x], x);
          instruction.location = it->location;
          instruction.function = it->function;
          F.second.body.insert_swap(it, instruction);
        }
      }
      continue;
    }

    if(it->is_assign())
    {
      if(is_symbol2t(to_code_assign2t(it->code).target))
      {
        // Are we dealing with our CSE symbol?
        auto name =
          to_symbol2t(to_code_assign2t(it->code).target).thename.as_string();
        if(has_prefix(name, prefix))
          continue;
      }
    }
    else
      continue;

    auto &assignment = to_code_assign2t(it->code);

    // RHS
    std::unordered_set<expr2tc, irep2_hash> matched_rhs_expressions;
    replace_max_sub_expr(
      assignment.source, expr2symbol, it, matched_rhs_expressions);

    for(auto &x : matched_rhs_expressions)
    {
      if(!state.available_expressions.count(x))
      {
        goto_programt::instructiont instruction;
        instruction.make_assignment();
        instruction.code = code_assign2tc(expr2symbol[x], x);
        instruction.location = it->location;
        instruction.function = it->function;
        F.second.body.insert_swap(it, instruction);
      }
    }

    // LHS
    auto cpy = assignment;
    std::unordered_set<expr2tc, irep2_hash> matched_lhs_expressions;
    replace_max_sub_expr(
      assignment.target, expr2symbol, it, matched_lhs_expressions);

    for(auto &x : matched_lhs_expressions)
    {
      // First time seeing the expr
      if(!state.available_expressions.count(x))
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
