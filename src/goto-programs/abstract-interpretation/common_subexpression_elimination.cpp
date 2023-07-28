#include <goto-programs/abstract-interpretation/common_subexpression_elimination.h>
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
  case END_FUNCTION:
    // Nothing is available anymore
    // TODO: Only local expressions are not available!
    available_expressions.clear();
    break;
  case ASSIGN:
  {
    const code_assign2t &code = to_code_assign2t(instruction.code);
    // Expressions that contain the target will need to be recomputed
    havoc_expr(code.target, to);
    // Target may be an expression as well
    make_expression_available(code.target);
    make_expression_available(code.source);
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
    // havoc_rec(code_function_call.ret);
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

void cse_domaint::assign(
  const expr2tc &e,
  const goto_programt::const_targett &i_it)
{

}

bool cse_domaint::join(const cse_domaint &b)
{
  // TODO: Maybe not the best logic...
  //
  // For the abstract states 'A0' (before), 'A1' (after), and 'A2' (result):
  //
  // 1. If 'e' in 'A0' and 'e' the not in 'A1': then 'e' not in 'A2'
  // 2. If 'e' not in 'A0' and 'e' in 'A1': then 'e' in 'A2'

  if(is_bottom())
  {
    available_expressions = b.available_expressions;
    return true;
  }

  bool changed = false;
  // Union first
  for(auto e : b.available_expressions)
  {
    if(!available_expressions.count(e))
    {
      changed = true;
      available_expressions.insert(e);
    }
  }

  // Intersection second
  for(auto x : available_expressions)
  {
    if(!b.available_expressions.count(x))
    {
      changed = true;
      available_expressions.erase(x);
    }
  }

  return changed;
}

void cse_domaint::make_expression_available(const expr2tc &E)
{
  if(!E)
    return;

  // No need to add primitives
  if(is_constant(E) || is_symbol2t(E))
    return;

  // Did we check it already?
  if(available_expressions.count(E))
    return;

  // Let's recursively make it available!
  E->foreach_operand([this](const expr2tc &e) { make_expression_available(e); });
  available_expressions.insert(E);
}

bool cse_domaint::remove_expr(const expr2tc &taint, const expr2tc &E) const
{
  if(!taint)
    return false;

  if(taint == E)
    return true;

  bool result = false;
  E->foreach_operand([this, &result, &taint](const expr2tc &e)
                     { result |= remove_expr(taint, e); });
  return result;
}

bool cse_domaint::remove_expr(const irep_idt &sym, const expr2tc &E) const
{
  if(is_symbol2t(E) && to_symbol2t(E).thename == sym)
    return true;

  bool result = false;
  E->foreach_operand([this, &result, &sym](const expr2tc &e)
                     { result |= remove_expr(sym, e); });
  return result;
}

void cse_domaint::havoc_symbol(const irep_idt &sym)
{
  std::vector<expr2tc> to_remove;
  for(auto x : available_expressions)
  {
    if(remove_expr(sym, x))
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
    if(remove_expr(target, x))
      to_remove.push_back(x);
  }
  for(auto x : to_remove)
    available_expressions.erase(x);
}
#include <memory>
void common_subexpression_elimination(
  goto_functionst &goto_functions,
  const namespacet &ns,
  const optionst &options)
{
}

bool goto_cse::runOnProgram(goto_functionst &F)
{
  cse_domaint::vsa = std::make_unique<value_set_analysist>(ns);
  (*cse_domaint::vsa)(F);

  available_expressions(F, ns);
  return true;
}

expr2tc
goto_cse::obtain_max_sub_expr(const expr2tc &e, const cse_domaint &state) const
{
  if(state.available_expressions.count(e))
    return e;

  if(!e)
    return expr2tc();

  // No need to add primitives
  if(is_constant(e) || is_symbol2t(e))
    return expr2tc();

  expr2tc result = expr2tc();
  e->foreach_operand(
    [this, &result, &state](const expr2tc e_inner)
    {
      // already solved
      if(result != expr2tc())
        return;

      result = obtain_max_sub_expr(e_inner, state);
      if(result != expr2tc())
        return;

      result = obtain_max_sub_expr(e_inner, state);
    });
  return result;
}

void goto_cse::replace_max_sub_expr(
  expr2tc &e,
  std::unordered_map<expr2tc, expr2tc, irep2_hash> &expr2symbol) const
{
  if(!e)
    return;

  // No need to add primitives
  if(is_constant(e) || is_symbol2t(e))
    return;

  if(expr2symbol.count(e))
  {
    //auto v = ;
    e = expr2symbol[e];
    return;
  }

  e->Foreach_operand([this, &expr2symbol](expr2tc &e0)
                     { replace_max_sub_expr(e0, expr2symbol); });
}

bool goto_cse::runOnFunction(std::pair<const dstring, goto_functiont> &F)
{
  if(!F.second.body_available)
    return false;

  log_status("Checking function {}", F.first.as_string());
  if(F.first.as_string() != "c:@F@main")
    return false;

  // 1. Let's count expressions
  std::unordered_map<expr2tc, unsigned, irep2_hash> counter;
  for(auto it = (F.second.body).instructions.begin();
      it != (F.second.body).instructions.end();
      ++it)
  {
    if(!it->is_assign())
      continue;

    const cse_domaint &state = available_expressions[it];
    const expr2tc max_sub = obtain_max_sub_expr(it->code, state);
    if(max_sub == expr2tc())
      continue;

    counter[max_sub]++;
  }

  // 2. We might print some context now
  if(verbose_mode)
  {
    for(auto it = counter.begin(); it != counter.end(); it++)
    {
      if(it->second >= threshold)
        log_status(
          "Found common sub-expression ({} times): {}\n",
          it->second + 1,
          *it->first);
    }
  }

  // Early exit, if no symbols.
  bool exists = false;
  for(auto elem : counter)
  {
    if(elem.second >= threshold)
    {
      exists = true;
      break;
    }
  }

  if(!exists)
    return false;
  // 3. Instrument new tmp symbols
  std::unordered_map<expr2tc, expr2tc, irep2_hash> expr2symbol;
  for(auto it = (F.second.body).instructions.begin();
      it != (F.second.body).instructions.end();
      ++it)
  {
    auto itt = it;
    itt++;
    try
    {
      const cse_domaint &state = available_expressions[itt];
      std::unordered_set<expr2tc, irep2_hash> to_add;
      for(const auto &sub : counter)
      {
        if(sub.second < threshold)
          continue;
        if(state.available_expressions.count(sub.first))
          to_add.insert(sub.first);
      }

      for(const auto &e : to_add)
      {
        goto_programt::targett t = F.second.body.insert(it);
        symbol2tc symbol = create_cse_symbol(e->type);
        t->make_assignment();
        t->code = code_assign2tc(symbol, e);
        expr2symbol[e] = symbol;
        counter[e] = 0;
      }
    }
    catch(...)
    {
      continue;
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

    replace_max_sub_expr(it->code, expr2symbol);
  }
  return true;
}

symbol2tc goto_cse::create_cse_symbol(const type2tc &t)
{
  return symbol2tc(t, fmt::format("{}${}", prefix, symbol_counter++));
}
