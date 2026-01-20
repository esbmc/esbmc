#include <cstdlib>
#include <goto-programs/contracts/contracts.h>
#include <goto-programs/remove_no_op.h>
#include <util/base_type.h>
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
static bool is_fresh_in_ensures(
  goto_programt::const_targett it,
  const goto_programt &body)
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

    // CRITICAL: Always remove ensures ASSUME from renamed function
    // The ensures clause is checked in the wrapper function, not in the original function body.
    // Leaving ensures ASSUME in the original function would:
    // 1. Make the postcondition a precondition (assume before execution)
    // 2. Cause dereference failures for struct return values (accessing __ESBMC_return_value)
    // Therefore, we ALWAYS remove all contract::ensures assumptions.
    bool needs_ensures_removal = true;

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

    // TODO: Fix __ESBMC_old type issue
    // Problem: Clang frontend declares __ESBMC_old as `int __ESBMC_old(int)`, which causes
    // implicit type conversion. For example, __ESBMC_old(global_value) where global_value
    // is double becomes __ESBMC_old((int)global_value).
    // Need to apply similar type resolution approach as __ESBMC_return_value.

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
  struct is_fresh_info {
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
        if (is_fresh_function(funcname) && !is_fresh_in_ensures(it, original_body) &&
            call.operands.size() >= 2)
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
    assert(is_pointer_type(info.ptr_arg->type) && "ptr_arg must be pointer type");
    type2tc target_ptr_type = to_pointer_type(info.ptr_arg->type).subtype;
    if (is_empty_type(target_ptr_type))
      target_ptr_type = pointer_type2tc(get_empty_type());
    
    expr2tc ptr_var = dereference2tc(target_ptr_type, info.ptr_arg);
    type2tc char_type = get_uint8_type();
    expr2tc malloc_expr = sideeffect2tc(
      target_ptr_type, expr2tc(), info.size_expr, std::vector<expr2tc>(),
      char_type, sideeffect2t::malloc);
    
    goto_programt::targett assign_inst = wrapper.add_instruction(ASSIGN);
    assign_inst->code = code_assign2tc(ptr_var, malloc_expr);
    assign_inst->location = location;
    assign_inst->location.comment("__ESBMC_is_fresh memory allocation");
  }

  // Lambda function to add contract clause instruction (ASSERT or ASSUME)
  // Used for both requires (ASSUME) and ensures (ASSERT) clauses in enforce mode
  auto add_contract_clause = [&wrapper, &location](
                               const expr2tc &clause,
                               const goto_program_instruction_typet inst_type,
                               const std::string &comment) {
    if (is_nil_expr(clause))
      return;
    
    bool should_add = false;
    if (is_constant_bool2t(clause))
  {
      const constant_bool2t &b = to_constant_bool2t(clause);
      // For ASSERT: only add if false (violation)
      // For ASSUME: only add if true (trivially true clauses are skipped)
      if (inst_type == ASSERT)
        should_add = !b.value;
      else // ASSUME
        should_add = b.value;
    }
    else
    {
      should_add = true;
    }
    
    if (should_add)
    {
      goto_programt::targett t = wrapper.add_instruction(inst_type);
      t->guard = clause;
      t->location = location;
      t->location.comment(comment);
    }
  };

  // 2. Assume requires clause (after memory allocation for is_fresh)
  add_contract_clause(requires_clause, ASSUME, "contract requires");

  // 2. Declare return value variable (if function has return type)
  expr2tc ret_val;
  type2tc ret_type;
  if (original_func.type.is_code())
  {
    const code_typet &code_type = to_code_type(original_func.type);
    typet return_type_irep1 = code_type.return_type();
    log_debug(
      "contracts",
      "generate_checking_wrapper: original return_type (irep1) id={}, identifier={}",
      return_type_irep1.id().as_string(),
      return_type_irep1.id() == "symbol" ? return_type_irep1.identifier().as_string() : "N/A");
    
    // Resolve symbol_type to concrete type using ns.follow()
    // This is critical: value set analysis cannot handle symbol_type
    if (return_type_irep1.id() == "symbol")
    {
      log_debug(
        "contracts",
        "generate_checking_wrapper: resolving symbol_type {}",
        return_type_irep1.identifier().as_string());
      return_type_irep1 = ns.follow(return_type_irep1);
      log_debug(
        "contracts",
        "generate_checking_wrapper: resolved to type id={}",
        return_type_irep1.id().as_string());
    }
    
    ret_type = migrate_type(return_type_irep1);
    log_debug(
      "contracts",
      "generate_checking_wrapper: ret_type (irep2) type_id={}, is_symbol_type={}",
      ret_type ? get_type_id(*ret_type) : "nil",
      ret_type && is_symbol_type(ret_type));
    
    // Also resolve symbol_type2t in irep2 if needed
    if (is_symbol_type(ret_type))
    {
      log_debug(
        "contracts",
        "generate_checking_wrapper: ret_type is symbol_type2t, resolving...");
      ret_type = ns.follow(ret_type);
      log_debug(
        "contracts",
        "generate_checking_wrapper: resolved ret_type type_id={}",
        ret_type ? get_type_id(*ret_type) : "nil");
    }
    
    if (!is_nil_type(ret_type))
    {
      // Create and add symbol to symbol table
      irep_idt ret_val_id("__ESBMC_return_value");
      symbolt ret_val_symbol;
      ret_val_symbol.name = ret_val_id;
      ret_val_symbol.id = ret_val_id;
      ret_val_symbol.type = return_type_irep1;
      ret_val_symbol.lvalue = true;
      ret_val_symbol.static_lifetime = false;
      ret_val_symbol.location = location;
      ret_val_symbol.mode = original_func.mode;

      log_debug(
        "contracts",
        "generate_checking_wrapper: creating return_value symbol with type id={}, is_symbol={}",
        ret_val_symbol.type.id().as_string(),
        ret_val_symbol.type.id() == "symbol");

      // Add symbol to context
      symbolt *added_symbol = context.move_symbol_to_context(ret_val_symbol);
      ret_val = symbol2tc(ret_type, added_symbol->id);
      
      log_debug(
        "contracts",
        "generate_checking_wrapper: created ret_val symbol, type_id={}, is_symbol_type={}",
        ret_val->type ? get_type_id(*ret_val->type) : "nil",
        ret_val->type && is_symbol_type(ret_val->type));

      goto_programt::targett decl_inst = wrapper.add_instruction(DECL);
      decl_inst->code = code_decl2tc(ret_type, added_symbol->id);
      decl_inst->location = location;
      decl_inst->location.comment("contract return value");
      
      log_debug(
        "contracts",
        "generate_checking_wrapper: created DECL instruction, type_id={}, is_symbol_type={}",
        ret_type ? get_type_id(*ret_type) : "nil",
        ret_type && is_symbol_type(ret_type));
      
      // Note: We don't initialize return_value here for struct/union types.
      // The function call will assign the complete struct/union value to return_value,
      // which will completely overwrite any member-level initialization.
      // Initializing members individually would be redundant and can cause issues
      // in symbolic execution when the function call overwrites the entire struct.
      // 
      // This aligns with the behavior in mark_decl_as_non_det.cpp which skips
      // initialization for return_value$ prefixed variables.
      log_debug(
        "contracts",
        "generate_checking_wrapper: skipping return_value initialization for struct/union type (will be assigned by function call)");
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
  // Process ensures clause: replace return_value, old(), and is_fresh
  expr2tc ensures_guard = ensures_clause;
  if (!is_nil_expr(ensures_clause))
  {
    log_debug(
      "contracts",
      "generate_checking_wrapper: processing ensures clause, ret_val type_id={}, is_symbol_type={}",
      ret_val && ret_val->type ? get_type_id(*ret_val->type) : "nil",
      ret_val && ret_val->type && is_symbol_type(ret_val->type));
    
    // Replace __ESBMC_return_value with actual return value variable
    if (!is_nil_expr(ret_val))
    {
      log_debug(
        "contracts",
        "generate_checking_wrapper: calling replace_return_value_in_expr");
      ensures_guard = replace_return_value_in_expr(ensures_guard, ret_val);
      log_debug(
        "contracts",
        "generate_checking_wrapper: replace_return_value_in_expr completed, result type_id={}",
        ensures_guard ? get_type_id(*ensures_guard->type) : "nil");
    }

    // Replace __ESBMC_old() expressions
    if (!old_snapshots.empty())
      ensures_guard = replace_old_in_expr(ensures_guard, old_snapshots);
    
    // Replace is_fresh temp vars with verification: valid_object(ptr) && is_dynamic[ptr]
    if (!is_fresh_mappings.empty())
      ensures_guard = replace_is_fresh_in_ensures_expr(ensures_guard, is_fresh_mappings);
  }
  
  // Extract struct member accesses to temporary variables before ASSERT
  // This avoids symbolic execution issues with accessing members from 'with' expressions
  if (!is_nil_expr(ensures_guard) && !is_nil_expr(ret_val))
  {
    log_debug(
      "contracts",
      "Before extract_struct_members_to_temps: ret_val type_id={}, is_struct={}, is_union={}",
      ret_val->type ? get_type_id(*ret_val->type) : "nil",
      ret_val->type && is_struct_type(ret_val->type),
      ret_val->type && is_union_type(ret_val->type));
    
    if (is_struct_type(ret_val->type) || is_union_type(ret_val->type))
    {
      ensures_guard = extract_struct_members_to_temps(
        ensures_guard, ret_val, wrapper, location);
    }
  }
  
  // Fix type mismatches in comparison expressions involving return values
  // This removes incorrect casts and adds correct casts for constants
  if (!is_nil_expr(ensures_guard) && !is_nil_expr(ret_val))
  {
    ensures_guard = fix_comparison_types(ensures_guard, ret_val);
  }
  
  // Add ASSERT instruction for ensures clause with property
  if (!is_nil_expr(ensures_guard))
  {
    bool should_add = false;
    if (is_constant_bool2t(ensures_guard))
  {
      const constant_bool2t &b = to_constant_bool2t(ensures_guard);
      should_add = !b.value; // Only assert if false (violation)
    }
    else
    {
      should_add = true;
    }
    
    if (should_add)
    {
      goto_programt::targett t = wrapper.add_instruction(ASSERT);
      t->guard = ensures_guard;
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

  // Handle address_of(index(symbol(__ESBMC_return_value))) pattern
  // This is how __ESBMC_return_value appears in GOTO programs when declared as char[]
  if (is_address_of2t(expr))
  {
    const address_of2t &addr = to_address_of2t(expr);
    expr2tc addr_source = addr.ptr_obj;
    
    if (is_index2t(addr_source))
    {
      const index2t &index = to_index2t(addr_source);
      expr2tc index_source = index.source_value;
      
      if (is_symbol2t(index_source))
      {
        const symbol2t &sym = to_symbol2t(index_source);
        std::string sym_name = id2string(sym.get_symbol_name());
        
        if (
          sym_name.find("__ESBMC_return_value") != std::string::npos ||
          sym_name == "return_value")
        {
          // Replace &__ESBMC_return_value[0] with ret_val
          // The original expr is address_of, so its type is pointer
          // But ret_val is the actual return value (not a pointer)
          // We should return ret_val directly, as it has the correct type
          return ret_val;
        }
      }
    }
  }

  // If this is a symbol with name __ESBMC_return_value, replace it
  if (is_symbol2t(expr))
  {
    const symbol2t &sym = to_symbol2t(expr);
    std::string sym_name = id2string(sym.get_symbol_name());

    // Check if symbol name contains __ESBMC_return_value (may have prefix)
    // Also check for "return_value" which is how it appears in GOTO programs
    if (
      sym_name.find("__ESBMC_return_value") != std::string::npos ||
      sym_name == "return_value")
    {
      return ret_val;
    }
  }

  // Handle type casting pattern: ((Type*)(&__ESBMC_return_value))->member
  // Pattern: member(dereference(typecast(address_of(index(symbol(__ESBMC_return_value))))))
  // or: member(dereference(typecast(address_of(symbol(__ESBMC_return_value)))))
  if (is_member2t(expr))
  {
    const member2t &member = to_member2t(expr);
    expr2tc source = member.source_value;
    
    // Check if source is a dereference (for -> operator)
    expr2tc deref_source = source;
    if (is_dereference2t(source))
    {
      const dereference2t &deref = to_dereference2t(source);
      deref_source = deref.value;
    }

    // Check if deref_source (or source if no dereference) is a typecast
    if (is_typecast2t(deref_source))
    {
      const typecast2t &cast = to_typecast2t(deref_source);
      expr2tc cast_source = cast.from;

      // Check if cast source is address_of
      if (is_address_of2t(cast_source))
      {
        const address_of2t &addr = to_address_of2t(cast_source);
        expr2tc addr_source = addr.ptr_obj;

        // Check if address_of source is __ESBMC_return_value symbol (direct or via index)
        expr2tc final_symbol = addr_source;
        if (is_index2t(addr_source))
        {
          // Handle case: ((Type*)(&__ESBMC_return_value[0]))->member
          const index2t &index = to_index2t(addr_source);
          final_symbol = index.source_value;
        }

        if (is_symbol2t(final_symbol))
        {
          const symbol2t &sym = to_symbol2t(final_symbol);
          std::string sym_name = id2string(sym.get_symbol_name());

          if (
            sym_name.find("__ESBMC_return_value") != std::string::npos ||
            sym_name == "return_value")
          {
            // Replace the entire pattern: ((Type*)(&__ESBMC_return_value))->member
            // with: ret_val.member (direct member access)
            // ret_val is already the struct value, not a pointer
            // But we need to check if ret_val is actually a struct/union type
            if (is_struct_type(ret_val->type) || is_union_type(ret_val->type))
            {
              return member2tc(member.type, ret_val, member.member);
            }
            else
            {
              // If ret_val is not a struct/union, we can't create member access
              // This shouldn't happen for struct return types, but handle it gracefully
              log_warning(
                "contracts",
                "Cannot create member access: ret_val type is not struct/union (type={})",
                get_type_id(*ret_val->type));
              // Continue with recursive replacement
            }
          }
        }
      }
    }
  }

  // Recursively replace in all operands
  expr2tc new_expr = expr;
  new_expr->Foreach_operand([this, &ret_val](expr2tc &op) {
    op = replace_return_value_in_expr(op, ret_val);
  });

  // After replacing __ESBMC_return_value, check if we can simplify typecasts
  // If a typecast was added to match __ESBMC_return_value's pointer type,
  // but __ESBMC_return_value is now replaced with ret_val (non-pointer),
  // we may be able to remove the typecast
  if (is_typecast2t(new_expr))
  {
    const typecast2t &cast = to_typecast2t(new_expr);
    
    // If the cast source type matches the cast target type, remove the typecast
    if (cast.from->type == cast.type)
    {
      return cast.from;
    }
    
    // If cast target is pointer but cast source is not, and ret_val is not a pointer,
    // this typecast was likely added to match __ESBMC_return_value's pointer type
    // Since we've replaced __ESBMC_return_value with ret_val, we can try to remove it
    if (
      is_pointer_type(cast.type) && !is_pointer_type(cast.from->type) &&
      ret_val && !is_pointer_type(ret_val->type))
    {
      return cast.from;
    }
  }

  return new_expr;
}

expr2tc code_contractst::replace_symbol_in_expr(
  const expr2tc &expr,
  const expr2tc &old_symbol,
  const expr2tc &new_expr) const
{
  if (is_nil_expr(expr))
    return expr;

  // If this is the symbol we want to replace, return the new expression
  if (is_symbol2t(expr) && is_symbol2t(old_symbol))
  {
    const symbol2t &sym = to_symbol2t(expr);
    const symbol2t &old_sym = to_symbol2t(old_symbol);
    
    // Compare symbol names
    if (sym.thename == old_sym.thename)
    {
      return new_expr;
    }
  }

  // Recursively replace in all operands
  expr2tc result = expr->clone();
  result->Foreach_operand([this, &old_symbol, &new_expr](expr2tc &op) {
    op = replace_symbol_in_expr(op, old_symbol, new_expr);
  });

  return result;
}

expr2tc code_contractst::extract_struct_members_to_temps(
  const expr2tc &expr,
  const expr2tc &ret_val,
  goto_programt &wrapper,
  const locationt &location)
{
  if (is_nil_expr(expr) || is_nil_expr(ret_val) || !is_symbol2t(ret_val))
    return expr;
  
  const symbol2t &ret_sym = to_symbol2t(ret_val);
  
  // Map from member name to temporary variable
  std::map<irep_idt, expr2tc> member_to_temp;
  
  // Recursive function to collect and replace member accesses
  std::function<expr2tc(const expr2tc&)> process_expr = [&](const expr2tc &e) -> expr2tc {
    if (is_nil_expr(e))
      return e;
    
    // Check if this is a member access to ret_val
    if (is_member2t(e))
    {
      const member2t &member = to_member2t(e);
      
      // Check if the source is ret_val
      bool is_ret_val_member = false;
      if (is_symbol2t(member.source_value))
      {
        const symbol2t &src_sym = to_symbol2t(member.source_value);
        is_ret_val_member = (ret_sym.thename == src_sym.thename);
      }
      
      if (is_ret_val_member)
      {
        // Check if we already created a temp for this member
        auto it = member_to_temp.find(member.member);
        if (it != member_to_temp.end())
        {
          return it->second;
        }
        
        // Create temporary variable for this member
        std::string temp_name = id2string(ret_sym.thename) + 
                                "$member$" + id2string(member.member);
        irep_idt temp_id(temp_name);
        
        // Create temporary variable symbol
        symbolt temp_symbol;
        temp_symbol.name = temp_id;
        temp_symbol.id = temp_id;
        temp_symbol.type = migrate_type_back(member.type);
        temp_symbol.lvalue = true;
        temp_symbol.static_lifetime = false;
        temp_symbol.location = location;
        temp_symbol.mode = "C";
        
        // Add to context
        symbolt *added_symbol = context.move_symbol_to_context(temp_symbol);
        expr2tc temp_var = symbol2tc(member.type, added_symbol->id);
        
        // Add DECL instruction
        goto_programt::targett decl_inst = wrapper.add_instruction(DECL);
        decl_inst->code = code_decl2tc(member.type, added_symbol->id);
        decl_inst->location = location;
        decl_inst->location.comment("temp for struct member");
        
        // Add ASSIGN instruction: temp = ret_val.member
        goto_programt::targett assign_inst = wrapper.add_instruction(ASSIGN);
        assign_inst->code = code_assign2tc(temp_var, e->clone());
        assign_inst->location = location;
        assign_inst->location.comment("extract struct member");
        
        member_to_temp[member.member] = temp_var;
        
        log_debug(
          "contracts",
          "extract_struct_members_to_temps: created temp {} for member {}",
          temp_name,
          id2string(member.member));
        
        return temp_var;
      }
    }
    
    // Recursively process operands
    expr2tc result = e->clone();
    result->Foreach_operand([&](expr2tc &op) {
      op = process_expr(op);
    });
    
    return result;
  };
  
  expr2tc result = process_expr(expr);
  
  log_debug(
    "contracts",
    "extract_struct_members_to_temps: extracted {} members",
    member_to_temp.size());
  
  return result;
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
      if (is_symbol2t(call.function) &&
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

// ========== Type fixing for return value comparisons ==========

bool code_contractst::is_return_value_symbol(const symbol2t &sym) const
{
  std::string name = id2string(sym.thename);
  
  // Match various return value patterns:
  // - "return_value"
  // - "__ESBMC_return_value"  
  // - "return_value$..." (with suffix)
  // - "tag-return_value$..." (with tag prefix)
  if (name == "return_value" || name == "__ESBMC_return_value")
    return true;
  
  if (name.find("return_value") != std::string::npos)
    return true;
  
  return false;
}

expr2tc code_contractst::remove_incorrect_casts(
  const expr2tc &expr,
  const expr2tc &ret_val) const
{
  if (is_nil_expr(expr) || is_nil_expr(ret_val))
    return expr;
  
  // If this is a typecast on a return_value symbol, check if it's incorrect
  if (is_typecast2t(expr))
  {
    const typecast2t &cast = to_typecast2t(expr);
    
    // Check if we're casting a return_value symbol
    if (is_symbol2t(cast.from))
    {
      const symbol2t &sym = to_symbol2t(cast.from);
      
      if (is_return_value_symbol(sym))
      {
        // Compare the cast target type with ret_val's type
        // If they don't match, the cast is incorrect and should be removed
        if (!base_type_eq(cast.type, ret_val->type, ns))
        {
          log_debug(
            "contracts",
            "Removing incorrect cast from {} to {} (ret_val type is {})",
            get_type_id(*cast.from->type),
            get_type_id(*cast.type),
            get_type_id(*ret_val->type));
          
          // Return the original symbol without the cast
          return cast.from;
        }
      }
    }
  }
  
  // Recursively process operands
  expr2tc new_expr = expr->clone();
  new_expr->Foreach_operand([this, &ret_val](expr2tc &op) {
    op = remove_incorrect_casts(op, ret_val);
  });
  
  return new_expr;
}

expr2tc code_contractst::fix_comparison_types(
  const expr2tc &expr,
  const expr2tc &ret_val) const
{
  if (is_nil_expr(expr) || is_nil_expr(ret_val))
    return expr;
  
  // Step 1: Remove incorrect casts on return_value
  expr2tc cleaned_expr = remove_incorrect_casts(expr, ret_val);
  
  // Step 2: Fix comparison operators
  if (is_comp_expr(cleaned_expr))
  {
    expr2tc new_expr = cleaned_expr->clone();
    
    // Get the two sides of the comparison
    expr2tc *side1 = nullptr;
    expr2tc *side2 = nullptr;
    
    if (is_lessthan2t(new_expr))
    {
      lessthan2t &rel = to_lessthan2t(new_expr);
      side1 = &rel.side_1;
      side2 = &rel.side_2;
    }
    else if (is_lessthanequal2t(new_expr))
    {
      lessthanequal2t &rel = to_lessthanequal2t(new_expr);
      side1 = &rel.side_1;
      side2 = &rel.side_2;
    }
    else if (is_greaterthan2t(new_expr))
    {
      greaterthan2t &rel = to_greaterthan2t(new_expr);
      side1 = &rel.side_1;
      side2 = &rel.side_2;
    }
    else if (is_greaterthanequal2t(new_expr))
    {
      greaterthanequal2t &rel = to_greaterthanequal2t(new_expr);
      side1 = &rel.side_1;
      side2 = &rel.side_2;
    }
    else if (is_equality2t(new_expr))
    {
      equality2t &rel = to_equality2t(new_expr);
      side1 = &rel.side_1;
      side2 = &rel.side_2;
    }
    else if (is_notequal2t(new_expr))
    {
      notequal2t &rel = to_notequal2t(new_expr);
      side1 = &rel.side_1;
      side2 = &rel.side_2;
    }
    
    if (side1 && side2)
    {
      // Check if one side is return_value and the other is a constant
      bool side1_is_retval = is_symbol2t(*side1) && 
                              is_return_value_symbol(to_symbol2t(*side1));
      bool side2_is_retval = is_symbol2t(*side2) && 
                              is_return_value_symbol(to_symbol2t(*side2));
      
      // Case 1: return_value compared with integer constant, but return_value is pointer
      if (is_pointer_type(ret_val->type))
      {
        if (side1_is_retval && is_constant_int2t(*side2))
        {
          const constant_int2t &c = to_constant_int2t(*side2);
          if (c.value.is_zero())
          {
            // Replace 0 with NULL pointer of correct type
            *side2 = gen_zero(ret_val->type);
            log_debug("contracts", "Fixed pointer comparison: replaced 0 with NULL");
          }
        }
        else if (side2_is_retval && is_constant_int2t(*side1))
        {
          const constant_int2t &c = to_constant_int2t(*side1);
          if (c.value.is_zero())
          {
            *side1 = gen_zero(ret_val->type);
            log_debug("contracts", "Fixed pointer comparison: replaced 0 with NULL");
          }
        }
      }
      // Case 2: return_value is float/double, constant needs cast
      else if (is_fractional_type(ret_val->type))
      {
        if (side1_is_retval && is_constant_int2t(*side2))
        {
          // Cast integer constant to double/float
          *side2 = typecast2tc(ret_val->type, *side2);
          log_debug(
            "contracts",
            "Fixed fractional comparison: cast constant to {}",
            get_type_id(*ret_val->type));
        }
        else if (side2_is_retval && is_constant_int2t(*side1))
        {
          *side1 = typecast2tc(ret_val->type, *side1);
          log_debug(
            "contracts",
            "Fixed fractional comparison: cast constant to {}",
            get_type_id(*ret_val->type));
        }
      }
    }
    
    return new_expr;
  }
  
  // Recursively process other expressions
  expr2tc new_expr = cleaned_expr->clone();
  new_expr->Foreach_operand([this, &ret_val](expr2tc &op) {
    op = fix_comparison_types(op, ret_val);
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
  log_status(
    "Replacing calls with contracts for {} function(s)", to_replace.size());

  // Build a map of function names to their symbols, bodies, and IDs for quick lookup
  // Key: function name (e.g., "increment")
  // Value: (symbol pointer, body pointer, function ID in goto_functions)
  std::map<std::string, std::tuple<symbolt *, goto_programt *, irep_idt>> function_map;

  // Collect all functions that might be called
  Forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available)
      continue;

    symbolt *func_sym = find_function_symbol(id2string(it->first));
    if (func_sym != nullptr)
    {
      // Use the goto_functions key (it->first) as the map key, since that's what
      // get_symbol_name() returns in function calls
      std::string func_key = id2string(it->first);
      function_map[func_key] = {func_sym, &it->second.body, it->first};
      log_debug("contracts", "Added function to map: {} (name: {}, id: {})", func_key, id2string(func_sym->name), id2string(it->first));
    }
  }

  // Track functions that have been replaced (to delete their definitions)
  // Use function ID from goto_functions (it->first) for correct matching
  std::set<irep_idt> functions_to_delete;

  // Process each function and replace calls
  Forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available)
      continue;

    std::vector<goto_programt::targett> calls_to_replace;
    std::vector<std::tuple<symbolt *, goto_programt *, irep_idt>> function_info;

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
            continue;

          // Check if this function should be replaced
          for (const auto &replace_name : to_replace)
          {
            if (called_func.find(replace_name) != std::string::npos)
            {
              log_debug("contracts", "Found potential call to replace: {}", called_func);
              auto map_it = function_map.find(called_func);
              if (map_it != function_map.end())
              {
                // Check if function has contracts
                goto_programt *func_body = std::get<1>(map_it->second);
                bool has_contract = has_contracts(*func_body);
                log_debug("contracts", "Function {} has contracts: {}", called_func, has_contract);
                if (has_contract)
                {
                  log_debug("contracts", "Adding call to replacement list: {}", called_func);
                calls_to_replace.push_back(i_it);
                  function_info.push_back(map_it->second);
                  // Track this function for deletion (use the ID from goto_functions)
                  irep_idt func_id = std::get<2>(map_it->second);
                  functions_to_delete.insert(func_id);
                break;
                }
              }
              else
              {
                log_debug("contracts", "Function {} not found in function_map", called_func);
              }
            }
          }
        }
      }
    }

    // Replace calls
    log_debug("contracts", "Found {} calls to replace in function {}", calls_to_replace.size(), id2string(it->first));
    for (size_t i = 0; i < calls_to_replace.size(); ++i)
    {
      log_debug("contracts", "Replacing call {} of {}", i + 1, calls_to_replace.size());
      symbolt *func_sym = std::get<0>(function_info[i]);
      goto_programt *func_body = std::get<1>(function_info[i]);
      generate_replacement_at_call(
        *func_sym,
        *func_body,
        calls_to_replace[i],
        it->second.body);
    }
  }

  // Delete function definitions that have been replaced
  // In replace_calls mode, function calls are replaced with contracts,
  // so the function definitions are no longer needed
  // We completely remove them from function_map so they don't appear in --goto-functions-only
  for (const auto &func_id : functions_to_delete)
  {
    auto func_it = goto_functions.function_map.find(func_id);
    if (func_it != goto_functions.function_map.end())
    {
      // Completely remove the function from function_map
      // This ensures it won't appear in --goto-functions-only output
      goto_functions.function_map.erase(func_it);
      log_status("Removed function {} (replaced with contract)", func_id);
    }
    else
    {
      log_warning(
        "Function ID {} not found in function_map for deletion", func_id);
    }
  }

  goto_functions.update();
}

void code_contractst::generate_replacement_at_call(
  const symbolt &function_symbol,
  const goto_programt &function_body,
  goto_programt::targett call_instruction,
  goto_programt &caller_body)
{
  // Extract contracts from function body (similar to enforce_contracts)
  expr2tc requires_clause = extract_requires_from_body(function_body);
  expr2tc ensures_clause = extract_ensures_from_body(function_body);
  
  // Debug: log extracted clauses
  log_debug(
    "contracts",
    "generate_replacement_at_call: extracted requires clause (nil={})",
    is_nil_expr(requires_clause));
  log_debug(
    "contracts",
    "generate_replacement_at_call: extracted ensures clause (nil={})",
    is_nil_expr(ensures_clause));

  // TODO Phase 2.4: Extract assigns clause from function body
  expr2tc assigns_clause = expr2tc(); // Not implemented yet

  goto_programt replacement;
  locationt call_location = call_instruction->location;

  // Extract return value and arguments from call instruction
  expr2tc ret_val;
  std::vector<expr2tc> actual_args;
  if (is_code_function_call2t(call_instruction->code))
  {
    const code_function_call2t &call =
      to_code_function_call2t(call_instruction->code);
    ret_val = call.ret;
    actual_args = call.operands;
  }

  // Replace function parameters with actual arguments in contract clauses
  if (function_symbol.type.is_code())
  {
    const code_typet &code_type = to_code_type(function_symbol.type);
    const code_typet::argumentst &params = code_type.arguments();
    
    // Build parameter-to-argument mapping
    for (size_t i = 0; i < params.size() && i < actual_args.size(); ++i)
  {
      irep_idt param_id = params[i].get_identifier();
      expr2tc param_expr = symbol2tc(
        migrate_type(params[i].type()), param_id);
      
      // Replace parameter symbol with actual argument in requires/ensures
      requires_clause = replace_symbol_in_expr(requires_clause, param_expr, actual_args[i]);
      ensures_clause = replace_symbol_in_expr(ensures_clause, param_expr, actual_args[i]);
      
      // Debug: log parameter replacement
      log_debug(
        "contracts",
        "Parameter replacement: {} (arg nil={})",
        id2string(param_id),
        is_nil_expr(actual_args[i]));
    }
  }
  
  // Debug: log clauses after parameter replacement
  log_debug(
    "contracts",
    "After parameter replacement: requires nil={}, ensures nil={}",
    is_nil_expr(requires_clause),
    is_nil_expr(ensures_clause));

  // Lambda function to add contract clause instruction (ASSERT or ASSUME)
  // Used for both requires (ASSERT) and ensures (ASSUME) clauses
  auto add_contract_clause = [&replacement, &call_location](
                               const expr2tc &clause,
                               const goto_program_instruction_typet inst_type,
                               const std::string &comment,
                               const std::string &property = "") {
    if (is_nil_expr(clause))
      return;
    
    bool should_add = false;
    if (is_constant_bool2t(clause))
    {
      const constant_bool2t &b = to_constant_bool2t(clause);
      // For ASSERT: always add (unless trivially true, which we can optimize)
      // For ASSUME: only add if true (skip trivially false assumptions)
      if (inst_type == ASSERT)
        should_add = true;  // Always assert, even if true (verifier will optimize)
      else // ASSUME
        should_add = b.value;  // Only assume if true (skip false assumptions)
    }
    else
    {
      should_add = true;  // Non-constant expressions should always be added
    }
    
    if (should_add)
    {
      goto_programt::targett t = replacement.add_instruction(inst_type);
      t->guard = clause;
      t->location = call_location;
      t->location.comment(comment);
      if (!property.empty())
        t->location.property(property);
    }
  };

  // 1. Assert requires clause (check precondition at call site)
  add_contract_clause(
    requires_clause, ASSERT, "contract requires", "contract requires");

  // 2. Havoc all potentially modified locations
  // In replace_calls mode, we must havoc everything the function might modify,
  // otherwise the effects cannot propagate from the removed function body.

  // 2.1. Havoc assigns targets (if assigns clause exists)
  // TODO Phase 2.4: Extract assigns clause from function body
  if (!is_nil_expr(assigns_clause))
  {
    havoc_assigns_targets(assigns_clause, replacement, call_location);
  }

  // 2.2. Havoc static lifetime global variables
  // Functions may modify global state, so we must havoc globals
  // Note: This is conservative but necessary for soundness
  havoc_static_globals(replacement, call_location);
  
  // 2.3. Havoc memory locations modified through pointer parameters
  // TODO: Analyze pointer parameters and havoc dereferenced locations
  // For now, we rely on assigns clause (when implemented) and global havoc
  // This is a conservative over-approximation

  // 3. Replace __ESBMC_return_value in ensures
  expr2tc ensures_guard = ensures_clause;
  if (!is_nil_expr(ret_val) && !is_nil_expr(ensures_clause))
    {
    log_debug(
      "contracts",
      "generate_replacement_at_call: replacing __ESBMC_return_value with ret_val (type={})",
      ret_val ? get_type_id(*ret_val->type) : "nil");
    ensures_guard = replace_return_value_in_expr(ensures_clause, ret_val);
    log_debug(
      "contracts",
      "generate_replacement_at_call: replaced __ESBMC_return_value, result type={}",
      ensures_guard ? get_type_id(*ensures_guard->type) : "nil");
  }

  // TODO Phase 2.3: Replace __ESBMC_old() in ensures

  // 4. Assume ensures clause (assume postcondition at call site)
  add_contract_clause(ensures_guard, ASSUME, "contract ensures");

  // Replace call with replacement code
  // Insert replacement code before the call instruction
  // destructive_insert inserts BEFORE the target (unlike insert_swap which inserts AFTER)
  
  // Debug: log replacement code generation
  size_t replacement_size = replacement.instructions.size();
  log_debug("contracts", "Replacement code generated: {} instructions", replacement_size);
  
  if (!replacement.instructions.empty())
  {
    // Debug: log what we're inserting
    log_debug("contracts", "Inserting {} instructions before call instruction", replacement_size);
    
    caller_body.destructive_insert(call_instruction, replacement);
    
    // Debug: verify insertion
    log_debug("contracts", "Call instruction after insertion: type={}", (int)call_instruction->type);
  }
  else
  {
    log_warning(
      "contracts",
      "No replacement code generated for function {}",
      id2string(function_symbol.name));
  }
  
  // Mark the original call as SKIP
  call_instruction->make_skip();
  
  // Debug: verify skip
  log_debug("contracts", "Call instruction marked as SKIP: type={}", (int)call_instruction->type);
}

