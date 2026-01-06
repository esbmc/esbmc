#include <cstdlib>
#include <goto-programs/contracts/contracts.h>
#include <goto-programs/remove_no_op.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/std_expr.h>
#include <util/symbol.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_utils.h>
#include <util/message.h>
#include <util/options.h>

code_contractst::code_contractst(
  goto_functionst &_goto_functions,
  contextt &_context,
  const namespacet &_ns)
  : goto_functions(_goto_functions), context(_context), ns(_ns)
{
}

bool code_contractst::is_compiler_generated(
  const std::string &function_name) const
{
  // Skip destructors (start with ~)
  if (!function_name.empty() && function_name[0] == '~')
    return true;

  // Skip functions with # in name (compiler-generated, e.g., exception destructors)
  if (function_name.find('#') != std::string::npos)
    return true;

  // Skip functions starting with __ESBMC_contracts_original_ (already processed)
  if (function_name.find("__ESBMC_contracts_original_") == 0)
    return true;

  // Skip other compiler-generated patterns if needed
  // For example, constructors/destructors in exception handling
  if (function_name.find("__cxa_") == 0)
    return true;

  return false;
}

symbolt *code_contractst::find_contract_symbol(const std::string &function_name)
{
  std::string contract_id = "contract::" + function_name;
  symbolt *sym = context.find_symbol(contract_id);
  if (sym != nullptr)
    return sym;
  contract_id = "contract::c:@F@" + function_name;
  return context.find_symbol(contract_id);
}

symbolt *code_contractst::find_function_symbol(const std::string &function_name)
{
  symbolt *sym = context.find_symbol(function_name);
  if (sym != nullptr)
    return sym;
  std::string func_id = "c:@F@" + function_name;
  return context.find_symbol(func_id);
}

void code_contractst::rename_function(
  const irep_idt &old_id,
  const irep_idt &new_id)
{
  auto it = goto_functions.function_map.find(old_id);
  if (it == goto_functions.function_map.end())
  {
    log_error("Function {} not found for renaming", old_id);
    abort();
  }

  // Copy function to new name
  goto_functiont &old_func = it->second;
  goto_functions.function_map[new_id] = old_func;
  goto_functions.function_map[new_id].update_instructions_function(new_id);

  // Update symbol table
  symbolt *old_sym = context.find_symbol(old_id);
  if (old_sym == nullptr)
  {
    log_error("Function symbol {} must exist in context", old_id);
    abort();
  }
  symbolt new_sym = *old_sym;
  new_sym.name = new_id;
  new_sym.id = new_id;
  context.add(new_sym);

  // Do NOT erase the old function yet - we'll replace it with the wrapper
}

expr2tc
code_contractst::extract_requires_from_body(const goto_programt &function_body)
{
  std::vector<expr2tc> requires_clauses;

  // Scan function body for contract::requires annotations
  forall_goto_program_instructions (it, function_body)
  {
    if (it->is_assume())
    {
      std::string comment = id2string(it->location.comment());
      if (comment == "contract::requires")
      {
        requires_clauses.push_back(it->guard);
      }
    }
  }

  // Combine all requires clauses with AND
  if (requires_clauses.empty())
    return gen_true_expr();
  if (requires_clauses.size() == 1)
    return requires_clauses[0];

  expr2tc result = requires_clauses[0];
  for (size_t i = 1; i < requires_clauses.size(); ++i)
  {
    result = and2tc(result, requires_clauses[i]);
  }
  return result;
}

expr2tc
code_contractst::extract_ensures_from_body(const goto_programt &function_body)
{
  std::vector<expr2tc> ensures_clauses;

  // Scan function body for contract::ensures annotations
  forall_goto_program_instructions (it, function_body)
  {
    if (it->is_assume())
    {
      std::string comment = id2string(it->location.comment());
      if (comment == "contract::ensures")
      {
        ensures_clauses.push_back(it->guard);
      }
    }
  }

  // Combine all ensures clauses with AND
  if (ensures_clauses.empty())
    return gen_true_expr();
  if (ensures_clauses.size() == 1)
    return ensures_clauses[0];

  expr2tc result = ensures_clauses[0];
  for (size_t i = 1; i < ensures_clauses.size(); ++i)
  {
    result = and2tc(result, ensures_clauses[i]);
  }
  return result;
}

expr2tc code_contractst::extract_requires_clause(const symbolt &contract_symbol)
{
  // Extract from contract symbol's value field
  // The value field should contain a struct with requires/ensures expressions
  if (contract_symbol.value.is_nil())
    return gen_true_expr();

  // For now, return the entire value as requires
  // TODO: Parse structured contract data if needed
  expr2tc req;
  migrate_expr(contract_symbol.value, req);
  return req;
}

expr2tc code_contractst::extract_ensures_clause(const symbolt &contract_symbol)
{
  // Extract from contract symbol's value field
  if (contract_symbol.value.is_nil())
    return gen_true_expr();

  // TODO: Implement proper separation of requires and ensures from contract symbol
  // Currently, we extract ensures clauses from function body instead
  // This function is used for contract replacement mode which is not fully implemented yet
  log_warning(
    "extract_ensures_clause from contract symbol not fully implemented");
  return gen_true_expr();
}

expr2tc
code_contractst::extract_assigns_clause(const symbolt & /* contract_symbol */)
{
  // TODO: Extract assigns clause from contract symbol
  log_warning("extract_assigns_clause is not yet implemented");
  return expr2tc();
}

std::vector<expr2tc>
code_contractst::extract_assigns_targets(const expr2tc &assigns_clause)
{
  std::vector<expr2tc> targets;
  if (is_nil_expr(assigns_clause))
    return targets;

  // TODO: Parse assigns clause to extract target list
  targets.push_back(assigns_clause);
  return targets;
}

void code_contractst::havoc_assigns_targets(
  const expr2tc &assigns_clause,
  goto_programt &dest,
  const locationt &location)
{
  std::vector<expr2tc> targets = extract_assigns_targets(assigns_clause);
  if (targets.empty())
    return;

  for (const auto &target : targets)
  {
    expr2tc rhs = gen_nondet(target->type);
    goto_programt::targett t = dest.add_instruction(ASSIGN);
    t->code = code_assign2tc(target, rhs);
    t->location = location;
    t->location.comment("contract assigns: assign non-deterministic value");
  }
}

void code_contractst::havoc_function_parameters(
  const symbolt &original_func,
  goto_programt &dest,
  const locationt &location)
{
  if (!original_func.type.is_code())
    return;

  const code_typet &code_type = to_code_type(original_func.type);
  const code_typet::argumentst &params = code_type.arguments();

  for (const auto &param : params)
  {
    // Build LHS symbol for the parameter
    type2tc param_type = migrate_type(param.type());
    expr2tc lhs = symbol2tc(param_type, param.get_identifier());

    // Do not assign nondeterministic values to pointers when value-set based
    // symex objects are enabled, to be consistent with loop invariant havoc.
    if (
      config.options.get_bool_option("add-symex-value-sets") &&
      is_pointer_type(lhs))
      continue;

    expr2tc rhs = gen_nondet(lhs->type);
    goto_programt::targett t = dest.add_instruction(ASSIGN);
    t->code = code_assign2tc(lhs, rhs);
    t->location = location;
    t->location.comment("contract havoc parameter");
  }
}

void code_contractst::havoc_static_globals(
  goto_programt &dest,
  const locationt &location)
{
  // Iterate over all symbols in context to find static lifetime globals
  ns.get_context().foreach_operand([&dest, &location](const symbolt &s) {
    // Skip functions, types, and non-lvalue symbols
    if (s.type.is_code() || s.is_type || !s.lvalue)
      return;

    // Only process static lifetime variables (globals and static locals)
    if (!s.static_lifetime)
      return;

    // Skip internal ESBMC symbols
    std::string sym_name = id2string(s.name);
    if (sym_name.find("__ESBMC_") == 0)
      return;

    // Build LHS symbol expression
    type2tc global_type = migrate_type(s.type);
    expr2tc lhs = symbol2tc(global_type, s.id);

    // Do not assign nondeterministic values to pointers when value-set based
    // symex objects are enabled, to be consistent with loop invariant havoc.
    if (
      config.options.get_bool_option("add-symex-value-sets") &&
      is_pointer_type(lhs))
      return;

    // Generate nondeterministic value and create assignment
    expr2tc rhs = gen_nondet(lhs->type);
    goto_programt::targett t = dest.add_instruction(ASSIGN);
    t->code = code_assign2tc(lhs, rhs);
    t->location = location;
    t->location.comment("contract havoc global");
  });
}

void code_contractst::enforce_contracts(const std::set<std::string> &to_enforce)
{
  for (const auto &function_name : to_enforce)
  {
    // Skip compiler-generated functions (destructors, constructors, exception handlers)
    // These functions are automatically generated by the compiler and should not have
    // user-defined contracts. Attempting to enforce contracts on them would be incorrect.
    if (is_compiler_generated(function_name))
    {
      continue;
    }

    symbolt *func_sym = find_function_symbol(function_name);
    if (func_sym == nullptr)
    {
      log_warning("Function {} not found", function_name);
      continue;
    }

    // Find the function in goto_functions
    auto func_it = goto_functions.function_map.find(func_sym->id);
    if (
      func_it == goto_functions.function_map.end() ||
      !func_it->second.body_available)
    {
      log_warning("Function body for {} not available", function_name);
      continue;
    }

    // Quick check: skip if function has no contracts (avoids expensive full scan)
    if (!has_contracts(func_it->second.body))
    {
      continue;
    }

    // Extract contract clauses from function body
    expr2tc requires_clause = extract_requires_from_body(func_it->second.body);
    expr2tc ensures_clause = extract_ensures_from_body(func_it->second.body);

    // Skip if no contracts found (should not happen after has_contracts check, but double-check)
    bool has_requires = !is_constant_bool2t(requires_clause) ||
                        !to_constant_bool2t(requires_clause).value;
    bool has_ensures = !is_constant_bool2t(ensures_clause) ||
                       !to_constant_bool2t(ensures_clause).value;

    if (!has_requires && !has_ensures)
    {
      continue;
    }

    // Rename original function
    irep_idt original_id = func_sym->id;
    std::string original_name_str =
      "__ESBMC_contracts_original_" + function_name;
    irep_idt original_name_id(original_name_str);

    rename_function(original_id, original_name_id);

    // Generate wrapper function
    goto_programt wrapper = generate_checking_wrapper(
      *func_sym, requires_clause, ensures_clause, original_name_id);

    // Create new function entry
    goto_functiont new_func;
    new_func.body = wrapper;
    if (func_sym->type.is_code())
      new_func.type = to_code_type(func_sym->type);
    new_func.body_available = true;
    new_func.update_instructions_function(original_id);

    goto_functions.function_map[original_id] = new_func;

    log_status("Enforced contract for function {}", function_name);
  }

  goto_functions.update();
}

goto_programt code_contractst::generate_checking_wrapper(
  const symbolt &original_func,
  const expr2tc &requires_clause,
  const expr2tc &ensures_clause,
  const irep_idt &original_func_id)
{
  goto_programt wrapper;
  locationt location = original_func.location;

  // Note: Here is the design, enforce_contracts mode does NOT havoc
  // parameters or globals. The wrapper is called by actual callers, so we
  // preserve the caller's argument values. Global variables are handled by
  // unified nondet_static initialization, not per-function havoc.

  // 1. Assume requires clause
  if (!is_nil_expr(requires_clause) && !is_constant_bool2t(requires_clause))
  {
    goto_programt::targett t = wrapper.add_instruction(ASSUME);
    t->guard = requires_clause;
    t->location = location;
    t->location.comment("contract requires");
  }
  else if (is_constant_bool2t(requires_clause))
  {
    const constant_bool2t &b = to_constant_bool2t(requires_clause);
    if (b.value) // Only add if not trivially true
    {
      goto_programt::targett t = wrapper.add_instruction(ASSUME);
      t->guard = requires_clause;
      t->location = location;
      t->location.comment("contract requires");
    }
  }

  // 2. Declare return value variable (if function has return type)
  expr2tc ret_val;
  type2tc ret_type;
  if (original_func.type.is_code())
  {
    const code_typet &code_type = to_code_type(original_func.type);
    ret_type = migrate_type(code_type.return_type());
    if (!is_nil_type(ret_type))
    {
      // Create and add symbol to symbol table
      irep_idt ret_val_id("__ESBMC_return_value");
      symbolt ret_val_symbol;
      ret_val_symbol.name = ret_val_id;
      ret_val_symbol.id = ret_val_id;
      ret_val_symbol.type = code_type.return_type();
      ret_val_symbol.lvalue = true;
      ret_val_symbol.static_lifetime = false;
      ret_val_symbol.location = location;
      ret_val_symbol.mode = original_func.mode;

      // Add symbol to context
      symbolt *added_symbol = context.move_symbol_to_context(ret_val_symbol);
      ret_val = symbol2tc(ret_type, added_symbol->id);

      goto_programt::targett decl_inst = wrapper.add_instruction(DECL);
      decl_inst->code = code_decl2tc(ret_type, added_symbol->id);
      decl_inst->location = location;
      decl_inst->location.comment("contract return value");
    }
  }

  // 3. Call original function
  if (original_func.type.is_code())
  {
    const code_typet &code_type = to_code_type(original_func.type);
    // Convert function type to irep2
    type2tc func_type = migrate_type(original_func.type);

    // Build parameter list
    std::vector<expr2tc> arguments;
    const code_typet::argumentst &params = code_type.arguments();
    for (const auto &param : params)
    {
      // Create symbol reference for each parameter
      type2tc param_type = migrate_type(param.type());
      expr2tc param_symbol = symbol2tc(param_type, param.get_identifier());
      arguments.push_back(param_symbol);
    }

    // Create function call
    expr2tc func_symbol = symbol2tc(func_type, original_func_id);
    expr2tc call_expr = code_function_call2tc(ret_val, func_symbol, arguments);

    goto_programt::targett call_inst = wrapper.add_instruction(FUNCTION_CALL);
    call_inst->code = call_expr;
    call_inst->location = location;
    call_inst->location.comment("contract call original function");
  }

  // 4. Assert ensures clause (replace __ESBMC_return_value with actual return value)
  if (!is_nil_expr(ensures_clause) && !is_constant_bool2t(ensures_clause))
  {
    // Replace __ESBMC_return_value with actual return value variable
    expr2tc ensures_guard = ensures_clause;
    if (!is_nil_expr(ret_val))
    {
      // Create a replacement function that replaces __ESBMC_return_value symbols
      ensures_guard = replace_return_value_in_expr(ensures_clause, ret_val);
    }

    goto_programt::targett t = wrapper.add_instruction(ASSERT);
    t->guard = ensures_guard;
    t->location = location;
    t->location.comment("contract ensures");
    t->location.property("contract ensures");
  }
  else if (is_constant_bool2t(ensures_clause))
  {
    const constant_bool2t &b = to_constant_bool2t(ensures_clause);
    if (b.value) // Only add if not trivially true
    {
      goto_programt::targett t = wrapper.add_instruction(ASSERT);
      t->guard = ensures_clause;
      t->location = location;
      t->location.comment("contract ensures");
      t->location.property("contract ensures");
    }
  }

  // 5. Return the value (if function has return type)
  if (!is_nil_expr(ret_val))
  {
    goto_programt::targett ret_inst = wrapper.add_instruction(RETURN);
    ret_inst->code = code_return2tc(ret_val);
    ret_inst->location = location;
    ret_inst->location.comment("contract return");
  }

  goto_programt::targett end_func = wrapper.add_instruction(END_FUNCTION);
  end_func->location = location;
  return wrapper;
}

expr2tc code_contractst::replace_return_value_in_expr(
  const expr2tc &expr,
  const expr2tc &ret_val)
{
  if (is_nil_expr(expr))
    return expr;

  // If this is a symbol with name __ESBMC_return_value, replace it
  if (is_symbol2t(expr))
  {
    const symbol2t &sym = to_symbol2t(expr);
    std::string sym_name = id2string(sym.get_symbol_name());

    // Check if symbol name contains __ESBMC_return_value (may have prefix)
    if (sym_name.find("__ESBMC_return_value") != std::string::npos)
    {
      return ret_val;
    }
  }

  // Recursively replace in all operands
  expr2tc new_expr = expr;
  new_expr->Foreach_operand([this, &ret_val](expr2tc &op) {
    op = replace_return_value_in_expr(op, ret_val);
  });

  return new_expr;
}

bool code_contractst::has_contracts(const goto_programt &function_body) const
{
  // Quick check: scan for contract markers without extracting full clauses
  forall_goto_program_instructions (it, function_body)
  {
    if (it->is_assume())
    {
      std::string comment = id2string(it->location.comment());
      if (comment == "contract::requires" || comment == "contract::ensures")
      {
        return true;
      }
    }
  }
  return false;
}

void code_contractst::replace_calls(const std::set<std::string> &to_replace)
{
  Forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available)
      continue;

    std::vector<goto_programt::targett> calls_to_replace;
    std::vector<symbolt *> contracts_to_use;

    // Find calls to replace
    Forall_goto_program_instructions (i_it, it->second.body)
    {
      if (i_it->is_function_call() && is_code_function_call2t(i_it->code))
      {
        const code_function_call2t &call = to_code_function_call2t(i_it->code);
        if (is_symbol2t(call.function))
        {
          const symbol2t &func_sym = to_symbol2t(call.function);
          std::string called_func = id2string(func_sym.get_symbol_name());

          // Skip compiler-generated functions
          if (is_compiler_generated(called_func))
          {
            continue;
          }

          for (const auto &replace_name : to_replace)
          {
            if (called_func.find(replace_name) != std::string::npos)
            {
              symbolt *contract_sym = find_contract_symbol(called_func);
              if (contract_sym != nullptr)
              {
                calls_to_replace.push_back(i_it);
                contracts_to_use.push_back(contract_sym);
                break;
              }
            }
          }
        }
      }
    }

    // Replace calls
    for (size_t i = 0; i < calls_to_replace.size(); ++i)
    {
      generate_replacement_at_call(
        *contracts_to_use[i], calls_to_replace[i], it->second.body);
    }
  }

  goto_functions.update();
}

void code_contractst::generate_replacement_at_call(
  const symbolt &contract_symbol,
  goto_programt::targett call_instruction,
  goto_programt &function_body)
{
  expr2tc requires_clause = extract_requires_clause(contract_symbol);
  expr2tc ensures_clause = extract_ensures_clause(contract_symbol);
  expr2tc assigns_clause = extract_assigns_clause(contract_symbol);

  goto_programt replacement;
  locationt call_location = call_instruction->location;

  // 1. Assert requires clause
  if (!is_nil_expr(requires_clause))
  {
    bool skip = false;
    if (is_constant_bool2t(requires_clause))
    {
      const constant_bool2t &b = to_constant_bool2t(requires_clause);
      skip = b.value;
    }
    if (!skip)
    {
      goto_programt::targett t = replacement.add_instruction(ASSERT);
      t->guard = requires_clause;
      t->location = call_location;
      t->location.comment("contract requires");
    }
  }

  // 2. Havoc assigns targets
  if (!is_nil_expr(assigns_clause))
  {
    havoc_assigns_targets(assigns_clause, replacement, call_location);
  }

  // 3. Assume ensures clause
  if (!is_nil_expr(ensures_clause))
  {
    bool skip = false;
    if (is_constant_bool2t(ensures_clause))
    {
      const constant_bool2t &b = to_constant_bool2t(ensures_clause);
      skip = b.value;
    }
    if (!skip)
    {
      goto_programt::targett t = replacement.add_instruction(ASSUME);
      t->guard = ensures_clause;
      t->location = call_location;
      t->location.comment("contract ensures");
    }
  }

  // Replace call with replacement code
  function_body.insert_swap(call_instruction, replacement);
  call_instruction->make_skip();
}

void code_contractst::build_contract_symbols()
{
  // TODO: Implement contract symbol building if needed
  // This would scan goto programs and create contract symbols
}
