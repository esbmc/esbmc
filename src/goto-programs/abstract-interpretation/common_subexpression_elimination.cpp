#include "irep2/irep2_expr.h"
#include <goto-programs/abstract-interpretation/common_subexpression_elimination.h>
#include <ostream>
#include <sstream>
#include <util/prefix.h>
#include <fmt/format.h>
#include <ranges>
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
  // We don't care about the type of instruction to merge.
  if(is_bottom())
  {
    available_expressions = b.available_expressions;
    return true;
  }

  if(!to->is_target() || !from->is_function_call())
  {
    bool changed = available_expressions != b.available_expressions;
    available_expressions = b.available_expressions;
    return changed;
  }

  size_t size_before_intersection = available_expressions.size();
  std::erase_if(available_expressions, [&b](const expr2tc &e) {
    return !b.available_expressions.count(e);
  });

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
  log_status("Removing {}", *target);
  std::vector<expr2tc> to_remove;
  for(auto x : available_expressions)
  {
    if(should_remove_expr(target, x))
      to_remove.push_back(x);
  }
  for(auto x : to_remove)
  {
    x->dump();
      available_expressions.erase(x);
    }
}
bool goto_cse::common_expression::should_add_symbol(
  const goto_programt::const_targett &it,
  size_t threshold) const
{
  auto index = std::find(gen.begin(), gen.end(), it);
  return index == gen.end()
    ? false : true;
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
    abort();
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
  const expressions_map &expr2symbol,
  const goto_programt::const_targett &to) const
{
  if(!e)
    return;

  // No need to add primitives
  if(is_constant(e) || is_symbol2t(e))
    return;

  auto common = expr2symbol.find(e);
  if(common != expr2symbol.end())
  {
    auto symbol = common->second.get_symbol_for_target(to);
    if(symbol)
    {
      e = symbol;
      return;
    }
  }
  e->Foreach_operand([this, &expr2symbol, &to](expr2tc &e0) {
    replace_max_sub_expr(e0, expr2symbol, to);
  });
}

bool goto_cse::runOnFunction(std::pair<const dstring, goto_functiont> &F)
{
  if(!program_initialized)
    return false;
  if(!F.second.body_available)
    return false;

  // 1. Let's count expressions

  // Compute all possible sequences
  expressions_map expressions;
  for(auto it = (F.second.body).instructions.begin();
      it != (F.second.body).instructions.end();
      ++it)
  {
    //if(
    //  !it->is_assign()) // Maybe we need to consider function call as well (havoc)
    //  continue;

    const cse_domaint &state = available_expressions[it];

    // Are the intervals still available (TODO: this can be parallel)
    for(auto &exp :
        expressions | std::ranges::views::filter([&state](auto &elem) {
          // Expression was available and now it isn't
          return elem.second.available &&
                 !state.available_expressions.count(elem.first);
        }))
    {
      exp.second.available = false;
      exp.second.kill.push_back(it);
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

  // Let's filter sequences lower than the threshold

  // 2. We might print some context now
  if(verbose_mode)
  {
    for(const auto &[k, v] : expressions)
    {
      log_status("Expression {}", *k);

      log_status("GEN");
      for(const auto &g : v.gen)
        log_status("\t{}", g->location.as_string());

      log_status("KILL");
      for(const auto &kill : v.kill)
        log_status("\t{}", kill->location.as_string());
    }
  }

  // Early exit, if no symbols.

  // 3. Instrument new tmp symbols
  std::unordered_map<expr2tc, expr2tc, irep2_hash> expr2symbol;
  auto it = (F.second.body).instructions.begin();
  for(auto &[e, cse] : expressions)
  {
    symbolt symbol = create_cse_symbol(e->type, it);
    auto magic = context.move_symbol_to_context(symbol);
    const auto symbol_as_expr = symbol2tc(e->type, magic->id);
    goto_programt::targett decl = F.second.body.insert(it);
      decl->make_decl();
      decl->code = code_decl2tc(e->type, magic->id);
      decl->location = it->location;

      expr2symbol[e] = symbol_as_expr;

   cse.symbol.push_back(symbol_as_expr);   
  }
  
  for(auto it = (F.second.body).instructions.begin();
      it != (F.second.body).instructions.end();
      ++it)
  {
    for(auto &[e, cse] :
        expressions | std::ranges::views::filter([this, &it](auto elem) {
          return elem.second.should_add_symbol(it, threshold);
        }))
    {
      auto itt = it;
      itt--;
      //      itt--;
      //      itt->dump();
      goto_programt::instructiont instruction;
      instruction.make_assignment();
      instruction.code = code_assign2tc(expr2symbol[e], e);
      //goto_programt::targett t = F.second.body.insert(itt);
      instruction.location = it->location;
      instruction.function = it->function;
      F.second.body.insert_swap(itt, instruction);
    }
  }

  // 4. Final step, let's replace the symbols!
  for(auto it = (F.second.body).instructions.begin();
      it != (F.second.body).instructions.end();
      ++it)
  {
    if(!it->is_assign())
      continue;

    auto assignment = to_code_assign2t(it->code);
    if(is_symbol2t(assignment.target))
    {
      // Are we dealing with our CSE symbol?
      auto name = to_symbol2t(assignment.target).thename.as_string();
      if(has_prefix(name, prefix))
        continue;
    }

    log_status("{}", it->location.as_string());
    replace_max_sub_expr(it->code, expressions, it);
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
