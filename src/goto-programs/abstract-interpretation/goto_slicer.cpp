#include <goto-programs/abstract-interpretation/goto_slicer.h>
#include <goto-programs/loop_unroll.h>
#include <util/prefix.h> // namespace

void slicer_domaint::output(std::ostream &out) const
{
  if(is_bottom())
  {
    out << "BOTTOM\n";
    return;
  }

  out << "\t[ ";
  for(const auto &d : dependencies)
  {
    out << "\n\t\t" << d.first << ": {";
    for(const auto &s : d.second)
    {
      out << " " << s << " ";
    }
    out << d.first << " }\n\t";
  }

  out << "]";
}

void slicer_domaint::assign(const expr2tc &e)
{
  const code_assign2t &assignment = to_code_assign2t(e);

  // We do not support pointers
  if(is_index2t(assignment.target))
  {
    const index2t &array = to_index2t(assignment.target);
    if(!is_symbol2t(array.source_value))
      return;

    if(should_skip_symbol(to_symbol2t(array.source_value).get_symbol_name()))
      return;

    std::set<std::string> vars;
    symbolt::get_expr2_symbols(to_array_type(array.source_value->type).array_size, vars);
    symbolt::get_expr2_symbols(array.index, vars);    
    symbolt::get_expr2_symbols(assignment.source, vars);
    dependencies[to_symbol2t(array.source_value).get_symbol_name()] = vars;

    return;
  }

  
  // We don't care about non-symbolic LHS
  if(!is_symbol2t(assignment.target))
    return;

  if(should_skip_symbol(to_symbol2t(assignment.target).get_symbol_name()))
    return;

  // We do not support pointers
  if(is_pointer_type(assignment.target->type))
  {
    log_warning("Slicer does not support pointers: {}", *e);
    throw "TODO: points-to analysis";
  }

  std::set<std::string> vars;
  // Build-up dependencies over the RHS
  symbolt::get_expr2_symbols(assignment.source, vars);
  if (is_array_type(assignment.source->type))
  {
      symbolt::get_expr2_symbols(to_array_type(assignment.source->type).array_size, vars);
  }
  dependencies[to_symbol2t(assignment.target).get_symbol_name()] = vars;
}

bool slicer_domaint::should_skip_symbol(const std::string &symbol) const
{
  // TODO: This should prevent the atexit handled (Needed as we don't support pointers)
  if(has_prefix(symbol, "c:stdlib.c"))
    return true;

  return false;
}

void slicer_domaint::declaration(const expr2tc &e)
{
  auto A = to_code_decl2t(e);
  if(is_pointer_type(A.type))
  {
    log_warning("Slicer does not support pointers: {}", *e);
    throw "TODO: points-to analysis";
  }

  std::set<std::string> vars;
  if (is_array_type(A.type))
  {
      symbolt::get_expr2_symbols(to_array_type(A.type).array_size, vars);
  }
  
  dependencies[A.value.as_string()] = vars;

}

void slicer_domaint::transform(
  goto_programt::const_targett from,
  goto_programt::const_targett,
  ai_baset &,
  const namespacet &ns)
{
  (void)ns;

  const goto_programt::instructiont &instruction = *from;
  switch(instruction.type)
  {
  case DECL:
    declaration(instruction.code);
    break;

  case ASSIGN:
    assign(instruction.code);
    break;

  default:;
  }
}

bool slicer_domaint::join(const slicer_domaint &b)
{
  // If old domain was empty, do nothing
  if(b.is_bottom())
    return false;

  bool changed = false;
  // First, merge new elements
  for(auto &d : b.dependencies)
  {
    if(!dependencies.count(d.first))
    {
      dependencies[d.first] = d.second;
      changed = true;
    }
    else
      for(auto &x : d.second)
        changed |= dependencies[d.first].insert(x).second;
  }

  if(!changed)
    return false;

  /* Find a fixpoint
   * This will always terminate since:
   *   - The domain is a finite set
   *   - The meet is the union
   *
  */
  while(changed)
  {
    changed = false;
    for(auto &d : dependencies)
      for(auto &item : d.second)
        for(auto &item_2 : dependencies[item])
          changed |= dependencies[d.first].insert(item_2).second;
  }

  return true;
}

bool goto_slicer::runOnProgram(goto_functionst &F)
{
  log_status("Slicing program");
  // Compute all dependencies for all statements
  try
  {
    slicer(F, ns);
  }
  catch(...)
  {
    log_status("Unable to slice the program");
    slicer_failed = true;
  }
  return true;
}

bool goto_slicer::is_loop_empty(const loopst &loop) const
{
  goto_programt::targett loop_head = loop.get_original_loop_head();
  while((++loop_head)->is_skip())
    ;

  return loop_head == loop.get_original_loop_exit();
}

bool goto_slicer::is_loop_affecting_assertions(const loopst &loop)
{
  goto_programt::targett loop_head = loop.get_original_loop_head();
  goto_programt::targett loop_exit = loop.get_original_loop_exit();
  loop_exit++;
  while(loop_head != loop_exit)
  {
    std::set<std::string> deps;
    symbolt::get_expr2_symbols(loop_head->guard, deps);
    symbolt::get_expr2_symbols(loop_head->code, deps);
    for(auto d : deps)
      if(remaining_deps.count(d))
        return true;
    loop_head++;
  }

  return false;
}

bool goto_slicer::is_trivial_loop(const loopst &loop) const
{
  /**
   * Forward:
   * - Infinite loops (e.g. while(1))
   *
   * Base case:
   * - Unreachable loops (e.g while(0))
   * - Loops that will definitely finish (e.g. for(int i =0; i < 4; i++))
   *
   */
  if(forward_analysis)
  {
    log_error("No support for trivial loops in forward yet");
    abort();
  }
  // TODO: check nondet loops
  bounded_loop_unroller unroll(-1);
  return unroll.get_loop_bounds(loop) >= 0;
}

bool goto_slicer::runOnLoop(loopst &loop, goto_programt &goto_program)
{
  goto_programt::targett loop_head = loop.get_original_loop_head();
  goto_programt::targett loop_exit = loop.get_original_loop_exit();
  loop_exit++;
  while(loop_head != loop_exit)
  {
    loop_head->dump();
    if(loop_head->is_function_call() || loop_head->is_return())
    {
      symbolt::get_expr2_symbols(
        loop.get_original_loop_head()->code, remaining_deps);
      symbolt::get_expr2_symbols(
        loop.get_original_loop_exit()->code, remaining_deps);
      symbolt::get_expr2_symbols(
        loop.get_original_loop_head()->guard, remaining_deps);
      symbolt::get_expr2_symbols(
        loop.get_original_loop_exit()->guard, remaining_deps);
      break;
    }
    loop_head++;
  }
  return false;
}

bool goto_slicer::postProcessing(goto_functionst &F)
{
  Forall_goto_functions(it, F)
  {
    if(it->second.body_available)
    {
      goto_loopst goto_loops(it->first, F, it->second);
      auto function_loops = goto_loops.get_loops();
      if(function_loops.size())
      {
        goto_functiont &goto_function = it->second;
        goto_programt &goto_program = goto_function.body;

        // Foreach loop in the function
        for(auto itt = function_loops.rbegin(); itt != function_loops.rend();
            ++itt)
          sliceLoop(*itt, goto_program);
      }
    }
  }
  return true;
}

bool goto_slicer::sliceLoop(loopst &loop, goto_programt &goto_program)
{
  if(slicer_failed)
    return false;

  if(
    (!forward_analysis && !is_loop_affecting_assertions(loop) &&
     is_trivial_loop(loop)) ||
    (forward_analysis && is_loop_empty(loop)))
  {
    loop.make_skip();
    loops_sliced++;
    return true;
  }
  return false;
}

bool goto_slicer::should_skip_function(const std::string &func) const
{
  // We should skip trying to slice our intrinsic functions
  return has_prefix(func, "__ESBMC_") || has_prefix(func, "c:@F@__ESBMC_") || has_prefix(func, "c:@F@pthread_");
}

void goto_slicer::set_forward_slicer()
{
  dependency_instruction_type.clear();
  dependency_instruction_type.insert(
    {ASSUME, ASSERT, FUNCTION_CALL, RETURN, GOTO});

  sliceable_instruction_type.clear();
  sliceable_instruction_type.insert({DECL, DEAD,  ASSIGN});

  forward_analysis = true;
}

void goto_slicer::set_base_slicer()
{
  dependency_instruction_type.clear();
  dependency_instruction_type.insert(
    {ASSUME, ASSERT, FUNCTION_CALL, RETURN, GOTO});

  sliceable_instruction_type.clear();
  sliceable_instruction_type.insert({DECL, DEAD, ASSIGN});

  forward_analysis = false;
}

bool goto_slicer::runOnFunction(std::pair<const dstring, goto_functiont> &F)
{
  if(slicer_failed)
    return false;

  if (should_skip_function(F.first.as_string()))
     return false;

  log_status("Slicing {}", F.first.as_string());

  /**
   * The Slicer works in three steps:
   * 1. Generate a dependency set. All variables that "matter" for the current mode
   * 2. Slice instructions that do not affect the dependency set
   * 3. Create a dependency set for ASSUMES/ASSERTS only
   */
  if(F.second.body_available)
  {
    std::set<std::string> deps;
    // 1- Build a dependency set for function call arguments, assumes/asserts and goto guards
    for(auto it = (F.second.body).instructions.rbegin();
        it != (F.second.body).instructions.rend();
        ++it)
    {
      if(dependency_instruction_type.count(it->type))
      {
        symbolt::get_expr2_symbols(it->guard, deps);
        symbolt::get_expr2_symbols(it->code, deps);

        auto base = it;
        base++;
        auto state = slicer[base.base()].dependencies;
        for(auto &d : deps)
          for(auto &item : state[d])
            deps.insert(item);
      }
    }

    // 2- Slice things that do not affect
    for(auto it = (F.second.body).instructions.rbegin();
        it != (F.second.body).instructions.rend();
        ++it)
    {
      if(sliceable_instruction_type.count(it->type))
      {
        std::set<std::string> local_deps;
        symbolt::get_expr2_symbols(it->code, local_deps);
        if(contains_global_var(local_deps))
         continue;

        bool contains_dep = false;
        for(auto &item : local_deps)
        {
          if(deps.count(item))
          {
            contains_dep = true;
            break;
          }
        }
        if(!contains_dep)
        {
          it->make_skip();
          instructions_sliced++;
        }
      }
    }

    // 3 - Build a final dependency set for loop slicing
    for(auto it = (F.second.body).instructions.rbegin();
        it != (F.second.body).instructions.rend();
        ++it)
    {
      // TODO: This could be done at the initialization phase
      switch(it->type)
      {
      case ASSERT:
      case ASSUME:
      case FUNCTION_CALL:
      case RETURN:
      {
        symbolt::get_expr2_symbols(it->guard, remaining_deps);
        symbolt::get_expr2_symbols(it->code, remaining_deps);
        auto base = it;
        base++;
        auto state = slicer[base.base()].dependencies;
        for(auto &d : remaining_deps)
          for(auto &item : state[d])
            remaining_deps.insert(item);
        break;
      }
      default:
        continue;
      }
    }

  }
  return true;
}

bool goto_slicer::contains_global_var(std::set<std::string> symbols) const
{
  for(auto identifier : symbols)
  {
    if(identifier == "NULL")
      continue;
    const symbolt *s = ns.get_context().find_symbol(identifier);
    if(s == nullptr)
    {
      log_warning("Could not find {}", identifier);
      continue;
    }

    if(s->static_lifetime)
      return true; // this is affecting a global var somehow
  }
  return false;
}
