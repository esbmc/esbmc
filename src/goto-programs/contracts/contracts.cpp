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

/// Check if function name is __ESBMC_is_fresh (handles Clang USR format)
static bool is_fresh_function(const std::string &funcname)
{
  return funcname == "c:@F@is_fresh" || funcname == "__ESBMC_is_fresh" ||
         funcname.find("c:@F@__ESBMC_is_fresh") == 0 ||
         funcname.find("__ESBMC_is_fresh") == 0;
}

/// Check if is_fresh call is in ensures clause by examining next instruction
static bool
is_fresh_in_ensures(goto_programt::const_targett it, const goto_programt &body)
{
  auto next_it = it;
  ++next_it;
  return next_it != body.instructions.end() && next_it->is_assume() &&
         id2string(next_it->location.comment()) == "contract::ensures";
}

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

    // Save the original function body BEFORE renaming
    // Make a copy to avoid issues with iterator invalidation
    goto_programt original_body_copy = func_it->second.body;

    // Extract contract clauses from function body
    expr2tc requires_clause = extract_requires_from_body(original_body_copy);
    expr2tc ensures_clause = extract_ensures_from_body(original_body_copy);

    // Skip if no contracts found (should not happen after has_contracts check, but double-check)
    // A contract exists if it's not a constant bool, or if it's a constant false
    // (gen_true_expr() is returned when no contract is found)
    bool has_requires = !is_constant_bool2t(requires_clause) ||
                        (is_constant_bool2t(requires_clause) &&
                         !to_constant_bool2t(requires_clause).value);
    bool has_ensures = !is_constant_bool2t(ensures_clause) ||
                       (is_constant_bool2t(ensures_clause) &&
                        !to_constant_bool2t(ensures_clause).value);

    if (!has_requires && !has_ensures)
    {
      continue;
    }

    // Remove ensures ASSUME from renamed function if it uses __ESBMC_old or __ESBMC_is_fresh
    // (these require special handling in the wrapper)
    bool needs_ensures_removal = false;
    forall_goto_program_instructions (it, original_body_copy)
    {
      if (it->is_assign() && is_code_assign2t(it->code))
      {
        const code_assign2t &assign = to_code_assign2t(it->code);
        if (
          is_sideeffect2t(assign.source) &&
          to_sideeffect2t(assign.source).kind == sideeffect2t::old_snapshot)
        {
          needs_ensures_removal = true;
          break;
        }
      }

      if (it->is_function_call() && is_code_function_call2t(it->code))
      {
        const code_function_call2t &call = to_code_function_call2t(it->code);
        if (
          is_symbol2t(call.function) &&
          is_fresh_function(to_symbol2t(call.function).thename.as_string()))
        {
          needs_ensures_removal = true;
          break;
        }
      }
    }

    // Rename original function
    irep_idt original_id = func_sym->id;
    std::string original_name_str =
      "__ESBMC_contracts_original_" + function_name;
    irep_idt original_name_id(original_name_str);

    rename_function(original_id, original_name_id);

    // Remove ensures ASSUME from renamed function (would force postconditions to be true)
    if (needs_ensures_removal)
    {
      auto &renamed_func = goto_functions.function_map[original_name_id];
      goto_programt &renamed_body = renamed_func.body;

      for (auto it = renamed_body.instructions.begin();
           it != renamed_body.instructions.end();)
      {
        bool should_remove = false;

        // Remove contract::ensures assumptions
        if (it->is_assume())
        {
          std::string comment = id2string(it->location.comment());
          if (comment == "contract::ensures")
          {
            should_remove = true;
          }
        }

        if (should_remove)
        {
          it = renamed_body.instructions.erase(it);
        }
        else
        {
          ++it;
        }
      }
    }

    // Extract is_fresh mappings from function body for ensures clause replacement
    std::vector<code_contractst::is_fresh_mapping_t> is_fresh_mappings =
      extract_is_fresh_mappings_from_body(original_body_copy);

    // Generate wrapper function, passing the original body
    goto_programt wrapper = generate_checking_wrapper(
      *func_sym,
      requires_clause,
      ensures_clause,
      original_name_id,
      original_body_copy,
      is_fresh_mappings);

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
  const irep_idt &original_func_id,
  const goto_programt &original_body,
  const std::vector<is_fresh_mapping_t> &is_fresh_mappings)
{
  goto_programt wrapper;
  locationt location = original_func.location;

  // Note: Here is the design, enforce_contracts mode does NOT havoc
  // parameters or globals. The wrapper is called by actual callers, so we
  // preserve the caller's argument values. Global variables are handled by
  // unified nondet_static initialization, not per-function havoc.

  // 0. Extract and create snapshots for __ESBMC_old() expressions
  // Note: __ESBMC_old() calls are converted to assignments in the function body
  // We need to find these assignments and extract the old_snapshot sideeffects
  std::vector<old_snapshot_t> old_snapshots;

  // Scan the original function body to find old_snapshot assignments
  {
    const goto_programt &body = original_body;

    // Scan for assignments from old_snapshot sideeffects
    forall_goto_program_instructions (it, body)
    {
      if (it->is_assign() && is_code_assign2t(it->code))
      {
        const code_assign2t &assign = to_code_assign2t(it->code);
        if (is_sideeffect2t(assign.source))
        {
          const sideeffect2t &se = to_sideeffect2t(assign.source);
          if (se.kind == sideeffect2t::old_snapshot)
          {
            // Found an old_snapshot assignment!
            // The LHS is the temporary variable, the operand is the original expression
            old_snapshots.push_back({se.operand, assign.target});
          }
        }
      }
    }
  }

  // Now we need to generate snapshot assignments in the wrapper BEFORE calling the original function
  // We'll update old_snapshots to contain new wrapper snapshot variables
  for (size_t i = 0; i < old_snapshots.size(); ++i)
  {
    expr2tc original_expr = old_snapshots[i].original_expr;
    expr2tc old_temp_var =
      old_snapshots[i].snapshot_var; // The temp var from function body

    // Create a NEW snapshot variable for the wrapper
    expr2tc new_snapshot_var = create_snapshot_variable(
      original_expr, id2string(original_func.name) + "_wrapper", i);

    // Generate snapshot declaration
    goto_programt::targett decl_inst = wrapper.add_instruction(DECL);
    decl_inst->code =
      code_decl2tc(original_expr->type, to_symbol2t(new_snapshot_var).thename);
    decl_inst->location = location;
    decl_inst->location.comment("__ESBMC_old snapshot declaration");

    // Generate snapshot assignment: new_snapshot_var = original_expr
    goto_programt::targett assign_inst = wrapper.add_instruction(ASSIGN);
    assign_inst->code = code_assign2tc(new_snapshot_var, original_expr);
    assign_inst->location = location;
    assign_inst->location.comment("__ESBMC_old snapshot assignment");

    // Store both old and new variables in the snapshot structure
    // We'll keep the old temp var as original_expr for matching,
    // and new snapshot var as snapshot_var for replacement
    old_snapshots[i].original_expr = old_temp_var;    // What to find
    old_snapshots[i].snapshot_var = new_snapshot_var; // What to replace with
  }

  // 1. Process __ESBMC_is_fresh in requires: allocate memory before function call
  //    (ensures clauses handle is_fresh separately via replace_is_fresh_in_ensures_expr)
  struct is_fresh_info
  {
    expr2tc ptr_arg;
    expr2tc size_expr;
  };
  std::vector<is_fresh_info> is_fresh_calls;

  forall_goto_program_instructions (it, original_body)
  {
    if (it->is_function_call() && is_code_function_call2t(it->code))
    {
      const code_function_call2t &call = to_code_function_call2t(it->code);
      if (is_symbol2t(call.function))
      {
        std::string funcname = to_symbol2t(call.function).thename.as_string();
        if (
          is_fresh_function(funcname) &&
          !is_fresh_in_ensures(it, original_body) && call.operands.size() >= 2)
        {
          is_fresh_info info;
          info.ptr_arg = call.operands[0]->clone();
          info.size_expr = call.operands[1]->clone();
          is_fresh_calls.push_back(info);
        }
      }
    }
  }

  // Allocate memory for requires is_fresh calls
  for (const auto &info : is_fresh_calls)
  {
    assert(
      is_pointer_type(info.ptr_arg->type) && "ptr_arg must be pointer type");
    type2tc target_ptr_type = to_pointer_type(info.ptr_arg->type).subtype;
    if (is_empty_type(target_ptr_type))
      target_ptr_type = pointer_type2tc(get_empty_type());

    expr2tc ptr_var = dereference2tc(target_ptr_type, info.ptr_arg);
    type2tc char_type = get_uint8_type();
    expr2tc malloc_expr = sideeffect2tc(
      target_ptr_type,
      expr2tc(),
      info.size_expr,
      std::vector<expr2tc>(),
      char_type,
      sideeffect2t::malloc);

    goto_programt::targett assign_inst = wrapper.add_instruction(ASSIGN);
    assign_inst->code = code_assign2tc(ptr_var, malloc_expr);
    assign_inst->location = location;
    assign_inst->location.comment("__ESBMC_is_fresh memory allocation");
  }

  // 2. Assume requires clause (after memory allocation for is_fresh)
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

  // 4. Assert ensures clause (replace __ESBMC_return_value and __ESBMC_old)
  if (!is_nil_expr(ensures_clause) && !is_constant_bool2t(ensures_clause))
  {
    // Replace __ESBMC_return_value with actual return value variable
    expr2tc ensures_guard = ensures_clause;
    if (!is_nil_expr(ret_val))
    {
      // Create a replacement function that replaces __ESBMC_return_value symbols
      ensures_guard = replace_return_value_in_expr(ensures_clause, ret_val);
    }

    if (!old_snapshots.empty())
      ensures_guard = replace_old_in_expr(ensures_guard, old_snapshots);

    // Replace is_fresh temp vars with verification: valid_object(ptr) && is_dynamic[ptr]
    if (!is_fresh_mappings.empty())
      ensures_guard =
        replace_is_fresh_in_ensures_expr(ensures_guard, is_fresh_mappings);

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

// ========== __ESBMC_is_fresh support for ensures implementation ==========

std::vector<code_contractst::is_fresh_mapping_t>
code_contractst::extract_is_fresh_mappings_from_body(
  const goto_programt &function_body) const
{
  std::vector<code_contractst::is_fresh_mapping_t> mappings;

  forall_goto_program_instructions (it, function_body)
  {
    if (it->is_function_call() && is_code_function_call2t(it->code))
    {
      const code_function_call2t &call = to_code_function_call2t(it->code);
      if (
        is_symbol2t(call.function) &&
        is_fresh_function(to_symbol2t(call.function).thename.as_string()) &&
        call.operands.size() >= 2 && !is_nil_expr(call.ret) &&
        is_symbol2t(call.ret))
      {
        code_contractst::is_fresh_mapping_t mapping;
        mapping.temp_var_name = to_symbol2t(call.ret).thename;

        expr2tc ptr_arg = call.operands[0];
        if (is_pointer_type(ptr_arg->type))
        {
          type2tc target_ptr_type = to_pointer_type(ptr_arg->type).subtype;
          if (is_empty_type(target_ptr_type))
            target_ptr_type = pointer_type2tc(get_empty_type());
          mapping.ptr_expr = dereference2tc(target_ptr_type, ptr_arg);
          mappings.push_back(mapping);
        }
      }
    }
  }

  return mappings;
}

expr2tc code_contractst::replace_is_fresh_in_ensures_expr(
  const expr2tc &expr,
  const std::vector<is_fresh_mapping_t> &mappings) const
{
  if (is_nil_expr(expr))
    return expr;

  if (is_symbol2t(expr))
  {
    const symbol2t &sym = to_symbol2t(expr);
    for (const auto &mapping : mappings)
    {
      if (sym.thename == mapping.temp_var_name)
      {
        // Replace with: valid_object(ptr) && is_dynamic[POINTER_OBJECT(ptr)]
        expr2tc valid_obj = valid_object2tc(mapping.ptr_expr);
        expr2tc ptr_obj = pointer_object2tc(pointer_type2(), mapping.ptr_expr);

        const symbolt *dyn_sym = ns.lookup("c:@__ESBMC_is_dynamic");
        if (dyn_sym == nullptr)
        {
          log_error("__ESBMC_is_dynamic symbol not found");
          abort();
        }
        type2tc dyn_arr_type = array_type2tc(get_bool_type(), expr2tc(), true);
        expr2tc dyn_arr = symbol2tc(dyn_arr_type, dyn_sym->id);
        expr2tc is_dynamic = index2tc(get_bool_type(), dyn_arr, ptr_obj);

        return and2tc(valid_obj, is_dynamic);
      }
    }
  }

  expr2tc new_expr = expr;
  new_expr->Foreach_operand([this, &mappings](expr2tc &op) {
    op = replace_is_fresh_in_ensures_expr(op, mappings);
  });

  return new_expr;
}

// ========== __ESBMC_old support implementation ==========

bool code_contractst::is_old_call(const expr2tc &expr) const
{
  if (is_nil_expr(expr))
    return false;

  // Check if this is a sideeffect with kind old_snapshot
  if (is_sideeffect2t(expr))
  {
    const sideeffect2t &se = to_sideeffect2t(expr);
    return se.kind == sideeffect2t::old_snapshot;
  }

  return false;
}

expr2tc code_contractst::create_snapshot_variable(
  const expr2tc &expr,
  const std::string &func_name,
  size_t index)
{
  // Generate unique snapshot variable name
  std::string snapshot_name =
    "__ESBMC_old_snapshot_" + func_name + "_" + std::to_string(index);

  // Create symbol and add to symbol table
  // Note: symbolt uses IRep1 (typet) while we work with IRep2 (type2tc).
  // This is ESBMC's architecture: Symbol Table is IRep1-based for global state,
  // while modern code (GOTO programs, contracts) uses IRep2 for local logic.
  // Migration is needed at the boundary when adding symbols to context.
  // If ESBMC migrates Symbol Table to IRep2, update this to use expr->type directly.
  symbolt snapshot_symbol;
  snapshot_symbol.name = snapshot_name;
  snapshot_symbol.id = snapshot_name;
  snapshot_symbol.type =
    migrate_type_back(expr->type); // IRep2 → IRep1 conversion
  snapshot_symbol.lvalue = true;
  snapshot_symbol.static_lifetime = false;
  snapshot_symbol.file_local = false;

  // Add to context (symbol table)
  symbolt *added = context.move_symbol_to_context(snapshot_symbol);

  // Return symbol expression (IRep2)
  return symbol2tc(expr->type, added->id);
}

expr2tc code_contractst::replace_old_in_expr(
  const expr2tc &expr,
  const std::vector<old_snapshot_t> &snapshots) const
{
  if (is_nil_expr(expr))
    return expr;

  // Check if this is a symbol that matches one of the old temp variables
  if (is_symbol2t(expr))
  {
    const symbol2t &sym = to_symbol2t(expr);
    std::string sym_name = id2string(sym.thename);

    // Only process symbols that are related to __ESBMC_old
    // These temp variables have names containing "___ESBMC_old"
    // This prevents accidentally replacing __ESBMC_return_value or other symbols
    if (sym_name.find("___ESBMC_old") != std::string::npos)
    {
      for (const auto &snapshot : snapshots)
      {
        if (is_symbol2t(snapshot.original_expr))
        {
          const symbol2t &snap_sym = to_symbol2t(snapshot.original_expr);
          if (sym.thename == snap_sym.thename)
          {
            return snapshot.snapshot_var;
          }
        }
      }
    }
  }

  // Check if this is an old_snapshot sideeffect (for compatibility)
  if (is_old_call(expr))
  {
    // Get the expression inside old()
    const sideeffect2t &se = to_sideeffect2t(expr);
    expr2tc original_expr = se.operand;

    // Find matching snapshot
    for (const auto &snapshot : snapshots)
    {
      if (snapshot.original_expr == original_expr)
      {
        return snapshot.snapshot_var;
      }
    }

    log_error("Cannot find snapshot for __ESBMC_old expression");
    abort();
  }

  // Recursively replace in all operands
  expr2tc new_expr = expr->clone();
  new_expr->Foreach_operand([this, &snapshots](expr2tc &op) {
    op = replace_old_in_expr(op, snapshots);
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
  // TODO: Function contract replacement mode (--replace-call-with-contract) is not fully implemented
  // The current implementation does not properly handle:
  // 1. Contract symbol extraction and storage
  // 2. __ESBMC_old() snapshot creation at call sites
  // 3. __ESBMC_return_value replacement in ensures clauses
  //
  // This feature requires significant additional work to be production-ready.
  // For now, only --enforce-contract mode is supported.
  log_error(
    "ERROR: --replace-call-with-contract mode is not yet fully implemented.\n"
    "\n"
    "Current status:\n"
    "  ✓ --enforce-contract mode: FULLY SUPPORTED\n"
    "  ✗ --replace-call-with-contract mode: NOT IMPLEMENTED\n"
    "\n"
    "The replace mode requires:\n"
    "  - Contract symbol management\n"
    "  - __ESBMC_old() snapshot handling at call sites\n"
    "  - __ESBMC_return_value replacement\n"
    "  - Proper assigns clause havoc logic\n"
    "\n"
    "Please use --enforce-contract mode instead.\n"
    "\n"
    "TODO: Complete implementation of replace_calls() and "
    "generate_replacement_at_call()");
  abort();

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
