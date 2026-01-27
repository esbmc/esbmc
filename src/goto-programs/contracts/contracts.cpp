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

// Helper structure to track assignments to a variable
struct var_assignment_info
{
  expr2tc value;                         // The assigned value
  goto_programt::const_targett location; // Where the assignment is
  expr2tc condition; // Condition under which this assignment happens (if any)
};

// Helper function to find all assignments to a variable that can reach a given point
static std::vector<var_assignment_info> find_all_assignments(
  const irep_idt &var_name,
  const goto_programt &function_body,
  const goto_programt::const_targett &target_location)
{
  std::vector<var_assignment_info> assignments;

  // Search backwards from target_location to find all assignments
  // We stop at the beginning of function body or a point where the variable is declared
  goto_programt::const_targett search_it = target_location;
  while (search_it != function_body.instructions.begin())
  {
    --search_it;

    // Check if this is a DECL for our variable - stop searching
    if (search_it->is_decl() && is_code_decl2t(search_it->code))
    {
      const code_decl2t &decl = to_code_decl2t(search_it->code);
      if (decl.value == var_name)
        break;
    }

    // Check for assignment to our variable
    if (search_it->is_assign() && is_code_assign2t(search_it->code))
    {
      const code_assign2t &assign = to_code_assign2t(search_it->code);
      if (is_symbol2t(assign.target))
      {
        const symbol2t &target_sym = to_symbol2t(assign.target);
        if (target_sym.thename == var_name)
        {
          var_assignment_info info;
          info.value = assign.source;
          info.location = search_it;
          info.condition = expr2tc(); // Will be filled later if needed
          assignments.push_back(info);
        }
      }
    }
  }

  return assignments;
}

/// Helper: check if an assignment value is a trivial boolean constant (0/1/false/true)
static bool is_trivial_bool_constant(const expr2tc &value, bool &const_val)
{
  if (is_constant_bool2t(value))
  {
    const_val = to_constant_bool2t(value).value;
    return true;
  }
  if (is_constant_int2t(value))
  {
    auto v = to_constant_int2t(value).value;
    if (v == 0 || v == 1)
    {
      const_val = (v == 1);
      return true;
    }
  }
  return false;
}

/// Try to reconstruct a short-circuit boolean expression from the GOTO program.
///
/// Clang compiles `a || b` as:
///   IF !(a) GOTO fallback     // if left is false, compute right
///   tmp = 1                   // left is true, short-circuit: result = true
///   GOTO end
///   fallback: tmp = b         // compute right side
///   end: USE tmp
///
/// And `a && b` as:
///   IF (a) GOTO compute_b     // if left is true, compute right
///   tmp = 0                   // left is false, short-circuit: result = false
///   GOTO end
///   compute_b: tmp = b        // compute right side
///   end: USE tmp
///
/// find_all_assignments returns assignments in reverse order (last first).
/// For `a || b`: [b, 1, NONDET]  (b first because it's closest to USE)
/// For `a && b`: [b, 0, NONDET]  (b first because it's closest to USE)
///
/// We detect these patterns by:
/// 1. Filtering out the NONDET initializer
/// 2. Looking for exactly one trivial constant (0 or 1) and one non-trivial expression
/// 3. Finding the GOTO instruction between the two assignments to extract the condition
/// 4. Reconstructing: if const=1 → cond || expr; if const=0 → cond && expr
static expr2tc try_reconstruct_short_circuit(
  const std::vector<var_assignment_info> &assignments,
  const goto_programt &function_body)
{
  // Filter out NONDET initializers and collect real assignments
  struct real_assign_t
  {
    expr2tc value;
    goto_programt::const_targett location;
    bool is_trivial;
    bool trivial_val; // only valid if is_trivial
  };
  std::vector<real_assign_t> real_assigns;

  for (const auto &a : assignments)
  {
    // Skip NONDET assignments (Clang initializer)
    if (is_sideeffect2t(a.value))
    {
      const sideeffect2t &se = to_sideeffect2t(a.value);
      if (se.kind == sideeffect2t::nondet)
        continue;
    }

    real_assign_t ra;
    ra.value = a.value;
    ra.location = a.location;
    ra.is_trivial = is_trivial_bool_constant(a.value, ra.trivial_val);
    real_assigns.push_back(ra);
  }

  // We need exactly 2 real assignments: one trivial constant and one expression
  if (real_assigns.size() != 2)
    return expr2tc(); // cannot reconstruct

  const real_assign_t *const_assign = nullptr;
  const real_assign_t *expr_assign = nullptr;

  for (const auto &ra : real_assigns)
  {
    if (ra.is_trivial)
      const_assign = &ra;
    else
      expr_assign = &ra;
  }

  if (!const_assign || !expr_assign)
    return expr2tc(); // both trivial or both non-trivial

  // Find the conditional GOTO that controls the short-circuit branching.
  //
  // Clang generates two layouts for short-circuit evaluation:
  //
  // OR pattern (`a || b`):
  //   IF !(a) THEN GOTO fallback    // guard = !(a)
  //   ASSIGN tmp = 1                // const (short-circuit true)
  //   GOTO end
  //   fallback: ASSIGN tmp = b      // expr
  //   end: USE tmp
  //
  // AND pattern (`a && b`):
  //   IF !(a) THEN GOTO fallback    // guard = !(a)
  //   ... (compute b) ...
  //   ASSIGN tmp = b                // expr
  //   GOTO end
  //   fallback: ASSIGN tmp = 0      // const (short-circuit false)
  //   end: USE tmp
  //
  // The conditional GOTO is always before the EARLIER assignment (in program order).
  // In find_all_assignments' reverse order, the earlier assignment has the higher index.
  // So real_assigns[1] is earlier than real_assigns[0] (reverse order).

  // Determine which assignment is earlier in the program.
  // real_assigns are in reverse order: [0] is late (closer to USE), [1] is early.
  const real_assign_t *earlier_assign = &real_assigns[1];

  // Search backwards from the earlier assignment to find the conditional GOTO.
  // We may need to skip past other instructions (including assignments and
  // declarations of OTHER variables that are part of the computation).
  expr2tc goto_condition;
  goto_programt::const_targett search = earlier_assign->location;
  // Limit search to avoid scanning the entire function
  int search_limit = 200;
  while (search != function_body.instructions.begin() && search_limit-- > 0)
  {
    --search;
    if (search->is_goto() && !is_true(search->guard))
    {
      goto_condition = search->guard;
      break;
    }
  }

  if (is_nil_expr(goto_condition))
    return expr2tc(); // could not find the conditional GOTO

  // Reconstruct the boolean expression.
  // The GOTO guard is the condition under which execution JUMPS (skips the fall-through path).
  // The fall-through path is the earlier assignment.
  //
  // For OR: IF !(a) GOTO fallback → guard=!(a), fall-through=const(1)
  //   Fall-through condition (NOT taken) = !(guard) = a
  //   a is true → result = 1 (const). So: result = a || b
  //
  // For AND: IF !(a) GOTO fallback → guard=!(a), fall-through=expr(b)
  //   Fall-through condition = !(guard) = a
  //   a is true → result = b (expr). So: result = a && b
  //
  // In both cases, the "left condition" is the negation of the GOTO guard.
  // - If the earlier (fall-through) assignment is the CONST → it's OR
  // - If the earlier (fall-through) assignment is the EXPR → it's AND

  expr2tc left_cond;
  if (is_not2t(goto_condition))
    left_cond = to_not2t(goto_condition).value;
  else
    left_cond = not2tc(goto_condition);

  if (earlier_assign == const_assign)
  {
    // Fall-through is const → short-circuit pattern
    if (const_assign->trivial_val)
    {
      // const = true on fall-through → OR: left_cond || expr
      return or2tc(left_cond, expr_assign->value);
    }
    else
    {
      // const = false on fall-through → this is unusual; treat as !left_cond && expr
      // (i.e., when left_cond is true, result is false → !left_cond && expr)
      return and2tc(not2tc(left_cond), expr_assign->value);
    }
  }
  else
  {
    // Fall-through is expr → the GOTO skips to const
    if (const_assign->trivial_val)
    {
      // Unusual case: GOTO leads to const=true, fall-through is expr
      // guard true → GOTO to const(true) → result = true
      // guard false → fall-through to expr → result = expr
      // Result: !left_cond || expr (where left_cond = !guard)
      return or2tc(not2tc(left_cond), expr_assign->value);
    }
    else
    {
      // GOTO leads to const=false, fall-through is expr
      // guard true → GOTO → result = false
      // guard false → fall-through → result = expr
      // result = !guard && expr = left_cond && expr
      return and2tc(left_cond, expr_assign->value);
    }
  }
}

// Helper function to inline temporary variables generated by Clang for short-circuit evaluation
// When ensures contains complex expressions like (a && b) || (c && d) with __ESBMC_old calls,
// Clang generates control flow with temporary variables (tmp$1, tmp$2, etc).
// This function recursively inlines these temporaries to get back the original expression.
//
// IMPORTANT: When a temporary variable has multiple assignments (conditional assignments),
// we need to handle this carefully. If one assignment is a constant (like 0 or 1) and the
// other is a more complex expression, we take the complex expression as the actual value.
static expr2tc inline_temporary_variables(
  const expr2tc &expr,
  const goto_programt &function_body,
  const goto_programt::const_targett &assume_location)
{
  if (is_nil_expr(expr))
    return expr;

  // If this is a symbol that looks like a Clang temporary (tmp$...), try to inline it
  if (is_symbol2t(expr))
  {
    const symbol2t &sym = to_symbol2t(expr);
    std::string sym_name = id2string(sym.thename);

    // Check if this is a Clang-generated temporary variable
    // These have names like "c:@F@foo::$tmp::tmp$6" or similar patterns with "$tmp"
    // Note: We DON'T inline return_value$___ESBMC_old$X because those need to be
    // matched against snapshots by replace_old_in_expr
    if (
      sym_name.find("$tmp") != std::string::npos &&
      sym_name.find("___ESBMC_old") == std::string::npos)
    {
      // Find all assignments to this variable
      std::vector<var_assignment_info> assignments =
        find_all_assignments(sym.thename, function_body, assume_location);

      if (assignments.empty())
      {
        // No assignment found, return as-is
        log_warning(
          "Could not find definition for temporary variable: {}", sym_name);
        return expr;
      }

      // If there's only one assignment, use it directly
      if (assignments.size() == 1)
      {
        const auto &assign = assignments[0];

        // Special case: if RHS is an old_snapshot sideeffect, DON'T inline further
        if (
          is_sideeffect2t(assign.value) &&
          to_sideeffect2t(assign.value).kind == sideeffect2t::old_snapshot)
        {
          return expr;
        }

        // Recursively inline the RHS
        return inline_temporary_variables(
          assign.value, function_body, assign.location);
      }

      // Multiple assignments - this happens with short-circuit evaluation.
      // Clang compiles `a || b` and `a && b` using conditional control flow
      // with temporary variables. We need to reconstruct the original boolean
      // expression instead of just picking one branch.
      //
      // First, try to reconstruct the short-circuit pattern (OR/AND):
      expr2tc reconstructed =
        try_reconstruct_short_circuit(assignments, function_body);

      if (!is_nil_expr(reconstructed))
      {
        // Successfully reconstructed the boolean expression.
        // Recursively inline any temporaries in the reconstructed expression.
        return inline_temporary_variables(
          reconstructed, function_body, assume_location);
      }

      // Fallback: could not reconstruct short-circuit pattern.
      // Use the old heuristic: pick the first non-trivial assignment.
      expr2tc best_value;
      goto_programt::const_targett best_location =
        function_body.instructions.end();

      for (const auto &assign : assignments)
      {
        // Skip trivial constant assignments (0, 1, false, true)
        bool is_trivial_constant = false;
        if (is_constant_bool2t(assign.value))
        {
          is_trivial_constant = true;
        }
        else if (is_constant_int2t(assign.value))
        {
          auto val = to_constant_int2t(assign.value).value;
          if (val == 0 || val == 1)
            is_trivial_constant = true;
        }

        if (!is_trivial_constant)
        {
          // Found a non-trivial assignment - use this one
          // Special case: if RHS is an old_snapshot sideeffect, DON'T inline further
          if (
            is_sideeffect2t(assign.value) &&
            to_sideeffect2t(assign.value).kind == sideeffect2t::old_snapshot)
          {
            return expr;
          }

          best_value = assign.value;
          best_location = assign.location;
          break;
        }
      }

      if (is_nil_expr(best_value))
      {
        // All assignments are trivial constants - just use the first one
        best_value = assignments[0].value;
        best_location = assignments[0].location;
      }

      // Recursively inline the best value
      return inline_temporary_variables(
        best_value, function_body, best_location);
    }
  }

  // For all other expression types, recursively process operands
  expr2tc result = expr->clone();
  result->Foreach_operand([&](expr2tc &op) {
    op = inline_temporary_variables(op, function_body, assume_location);
  });

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
        // Inline any Clang-generated temporary variables to get the full expression
        // This handles cases where Clang generates control flow for short-circuit evaluation
        // The inline_temporary_variables function is smart enough to handle conditional
        // assignments by preferring non-trivial expressions over constant 0/1 values.
        expr2tc inlined_guard =
          inline_temporary_variables(it->guard, function_body, it);
        ensures_clauses.push_back(inlined_guard);
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

// Helper function to check if function has an explicit empty assigns clause
// This distinguishes __ESBMC_assigns(0) from no assigns clause at all
static bool has_empty_assigns_marker(const goto_programt &function_body)
{
  forall_goto_program_instructions (it, function_body)
  {
    if (it->is_assert())
    {
      std::string comment = id2string(it->location.comment());
      if (comment == "contract::assigns_empty")
      {
        return true;
      }
    }
  }
  return false;
}

// Helper function to unwrap array-to-pointer decay in assigns targets
// In C, when an array is passed to a function, it decays to &arr[0].
// This function detects this pattern and returns the original array.
static expr2tc unwrap_array_decay(const expr2tc &expr)
{
  // Pattern: address_of(index(array, 0))
  if (is_address_of2t(expr))
  {
    const address_of2t &addr = to_address_of2t(expr);
    if (is_index2t(addr.ptr_obj))
    {
      const index2t &idx = to_index2t(addr.ptr_obj);
      // Check if index is 0
      if (is_constant_int2t(idx.index))
      {
        const constant_int2t &idx_val = to_constant_int2t(idx.index);
        if (idx_val.value == 0)
        {
          // Check if source is an array type
          if (is_array_type(idx.source_value->type))
          {
            // Return the original array
            return idx.source_value;
          }
        }
      }
    }
  }

  return expr;
}

std::vector<expr2tc>
code_contractst::extract_assigns_from_body(const goto_programt &function_body)
{
  std::vector<expr2tc> assigns_targets;

  log_debug(
    "contracts",
    "extract_assigns_from_body: scanning {} instructions",
    function_body.instructions.size());

  // Scan function body for assigns_target sideeffect assignments
  // These were created by __ESBMC_assigns() in builtin_functions.cpp
  forall_goto_program_instructions (it, function_body)
  {
    if (it->is_assign())
    {
      const code_assign2t &assign = to_code_assign2t(it->code);

      // Check if RHS is a sideeffect with assigns_target
      if (
        is_sideeffect2t(assign.source) &&
        to_sideeffect2t(assign.source).kind == sideeffect2t::assigns_target)
      {
        const sideeffect2t &se = to_sideeffect2t(assign.source);
        expr2tc target_expr = se.operand;

        // Unwrap array-to-pointer decay: &arr[0] -> arr
        // This happens when an array is passed to __ESBMC_assigns()
        target_expr = unwrap_array_decay(target_expr);

        log_debug("contracts", "  Found assigns target expression");
        assigns_targets.push_back(target_expr);
      }
    }
  }

  log_debug(
    "contracts",
    "extract_assigns_from_body: found {} assigns targets",
    assigns_targets.size());
  return assigns_targets;
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

void code_contractst::enforce_contracts(
  const std::set<std::string> &to_enforce,
  bool assume_nonnull_valid)
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

    // Rename original function
    irep_idt original_id = func_sym->id;
    std::string original_name_str =
      "__ESBMC_contracts_original_" + function_name;
    irep_idt original_name_id(original_name_str);

    rename_function(original_id, original_name_id);

    // Remove ensures ASSUME from renamed function (would force postconditions to be true)
    // We need to properly update GOTO targets before removing instructions
    {
      auto &renamed_func = goto_functions.function_map[original_name_id];
      goto_programt &renamed_body = renamed_func.body;

      // Collect all ensures instructions to remove
      std::set<goto_programt::targett> instructions_to_remove;
      for (auto it = renamed_body.instructions.begin();
           it != renamed_body.instructions.end();
           ++it)
      {
        if (it->is_assume())
        {
          std::string comment = id2string(it->location.comment());
          if (comment == "contract::ensures")
          {
            instructions_to_remove.insert(it);
          }
        }
      }

      // Build a map of instructions to remove -> their replacement target
      // The replacement target must NOT be an instruction that will also be removed
      typedef std::map<goto_programt::targett, goto_programt::targett>
        targets_mapt;
      targets_mapt targets_to_update;

      for (auto it : instructions_to_remove)
      {
        // Find the next instruction that will NOT be removed
        auto next_it = std::next(it);
        while (next_it != renamed_body.instructions.end() &&
               instructions_to_remove.count(next_it) > 0)
        {
          next_it = std::next(next_it);
        }
        targets_to_update[it] = next_it;
      }

      // Update all GOTO targets that point to removed instructions
      for (auto &inst : renamed_body.instructions)
      {
        if (inst.is_goto() || inst.is_catch())
        {
          for (auto &target : inst.targets)
          {
            auto map_it = targets_to_update.find(target);
            if (map_it != targets_to_update.end())
            {
              target = map_it->second;
            }
          }
        }
      }

      // Remove the instructions
      for (auto it : instructions_to_remove)
      {
        renamed_body.instructions.erase(it);
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
      is_fresh_mappings,
      assume_nonnull_valid);

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
  const std::vector<is_fresh_mapping_t> &is_fresh_mappings,
  bool assume_nonnull_valid)
{
  goto_programt wrapper;
  locationt location = original_func.location;

  // Note: Here is the design, enforce_contracts mode does NOT havoc
  // parameters or globals. The wrapper is called by actual callers, so we
  // preserve the caller's argument values. Global variables are handled by
  // unified nondet_static initialization, not per-function havoc.

  // 0. Add pointer validity assumptions FIRST (if --assume-nonnull-valid is set)
  // This MUST come before old_snapshot materialization because old snapshots may
  // dereference pointers (e.g., __ESBMC_old(*x)), and we need to assume pointers
  // are valid before accessing them.
  if (assume_nonnull_valid)
  {
    add_pointer_validity_assumptions(wrapper, original_func, location);
  }

  // 1. Extract and create snapshots for __ESBMC_old() expressions
  // Note: __ESBMC_old() calls are converted to assignments in the function body
  // We need to find these assignments and extract the old_snapshot sideeffects
  std::vector<old_snapshot_t> old_snapshots =
    collect_old_snapshots_from_body(original_body);

  // Materialize snapshots in wrapper (creates DECL and ASSIGN instructions)
  // This comes AFTER pointer validity assumptions so we can safely dereference pointers
  materialize_old_snapshots_at_wrapper(
    old_snapshots, wrapper, id2string(original_func.name), location);

  // 2. Process __ESBMC_is_fresh in requires: allocate memory before function call
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

  // 3. Assume requires clause (after memory allocation for is_fresh)
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
      "generate_checking_wrapper: original return_type (irep1) id={}, "
      "identifier={}",
      return_type_irep1.id().as_string(),
      return_type_irep1.id() == "symbol"
        ? return_type_irep1.identifier().as_string()
        : "N/A");

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
      "generate_checking_wrapper: ret_type (irep2) type_id={}, "
      "is_symbol_type={}",
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
        "generate_checking_wrapper: creating return_value symbol with type "
        "id={}, is_symbol={}",
        ret_val_symbol.type.id().as_string(),
        ret_val_symbol.type.id() == "symbol");

      // Add symbol to context
      symbolt *added_symbol = context.move_symbol_to_context(ret_val_symbol);
      ret_val = symbol2tc(ret_type, added_symbol->id);

      log_debug(
        "contracts",
        "generate_checking_wrapper: created ret_val symbol, type_id={}, "
        "is_symbol_type={}",
        ret_val->type ? get_type_id(*ret_val->type) : "nil",
        ret_val->type && is_symbol_type(ret_val->type));

      goto_programt::targett decl_inst = wrapper.add_instruction(DECL);
      decl_inst->code = code_decl2tc(ret_type, added_symbol->id);
      decl_inst->location = location;
      decl_inst->location.comment("contract return value");

      log_debug(
        "contracts",
        "generate_checking_wrapper: created DECL instruction, type_id={}, "
        "is_symbol_type={}",
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
        "generate_checking_wrapper: skipping return_value initialization for "
        "struct/union type (will be assigned by function call)");
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
      "generate_checking_wrapper: processing ensures clause, ret_val "
      "type_id={}, is_symbol_type={}",
      ret_val && ret_val->type ? get_type_id(*ret_val->type) : "nil",
      ret_val && ret_val->type && is_symbol_type(ret_val->type));

    // Replace __ESBMC_old() expressions
    if (!old_snapshots.empty())
      ensures_guard = replace_old_in_expr(ensures_guard, old_snapshots);

    // Replace is_fresh temp vars with verification: valid_object(ptr) && is_dynamic[ptr]
    if (!is_fresh_mappings.empty())
      ensures_guard =
        replace_is_fresh_in_ensures_expr(ensures_guard, is_fresh_mappings);
  }

  // Extract struct member accesses to temporary variables before ASSERT
  // This avoids symbolic execution issues with accessing members from 'with' expressions
  if (!is_nil_expr(ensures_guard) && !is_nil_expr(ret_val))
  {
    log_debug(
      "contracts",
      "Before extract_struct_members_to_temps: ret_val type_id={}, "
      "is_struct={}, is_union={}",
      ret_val->type ? get_type_id(*ret_val->type) : "nil",
      ret_val->type && is_struct_type(ret_val->type),
      ret_val->type && is_union_type(ret_val->type));

    if (is_struct_type(ret_val->type) || is_union_type(ret_val->type))
    {
      ensures_guard = extract_struct_members_to_temps(
        ensures_guard, ret_val, wrapper, location);
    }
  }

  // Normalize ensures guard: replace return_value, fix types, normalize floating-point
  // This unified helper applies all return_value-related transformations
  ensures_guard =
    normalize_ensures_guard_for_return_value(ensures_guard, ret_val);

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
  const expr2tc &ret_val) const
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
                "Cannot create member access: ret_val type is not struct/union "
                "(type={})",
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
  std::function<expr2tc(const expr2tc &)> process_expr =
    [&](const expr2tc &e) -> expr2tc {
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
        std::string temp_name =
          id2string(ret_sym.thename) + "$member$" + id2string(member.member);
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
    result->Foreach_operand([&](expr2tc &op) { op = process_expr(op); });

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
  size_t index) const
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

// ========== Old snapshot collection and materialization helpers ==========

std::vector<code_contractst::old_snapshot_t>
code_contractst::collect_old_snapshots_from_body(
  const goto_programt &function_body) const
{
  std::vector<code_contractst::old_snapshot_t> old_snapshots;

  // Track seen expressions to deduplicate
  // Map: original_expr -> {first_temp_var, all_temp_vars}
  struct expr_info
  {
    expr2tc original_expr;
    std::vector<expr2tc> temp_vars;
  };
  std::vector<expr_info> unique_exprs;

  // Scan for assignments from old_snapshot sideeffects
  forall_goto_program_instructions (it, function_body)
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
          // The operand is the original expression, the target is the temp variable
          expr2tc original_expr = se.operand;
          expr2tc temp_var = assign.target;

          // Check if we've seen this expression before
          auto it = std::find_if(
            unique_exprs.begin(),
            unique_exprs.end(),
            [&original_expr](const expr_info &info) {
              return info.original_expr == original_expr;
            });

          if (it != unique_exprs.end())
          {
            // Same expression - add this temp var to the list
            it->temp_vars.push_back(temp_var);
          }
          else
          {
            // New expression - create a new entry
            unique_exprs.push_back({original_expr, {temp_var}});
          }
        }
      }
    }
  }

  // Create one snapshot entry per unique expression, BUT create entries for ALL temp vars
  // This ensures that all temp vars get mapped to the same wrapper snapshot later
  for (const auto &info : unique_exprs)
  {
    // Add an entry for EACH temp var that references this expression
    // They all have the same original_expr, so they'll all get mapped to the same wrapper snapshot
    for (const auto &temp_var : info.temp_vars)
    {
      old_snapshots.push_back({info.original_expr, temp_var});
    }

    // Log if there are multiple temp vars for the same expression
    if (info.temp_vars.size() > 1)
    {
      log_debug(
        "contracts",
        "Found {} temp variables for the same __ESBMC_old expression - all "
        "will map to one snapshot",
        info.temp_vars.size());
    }
  }

  return old_snapshots;
}

void code_contractst::materialize_old_snapshots_at_wrapper(
  std::vector<code_contractst::old_snapshot_t> &old_snapshots,
  goto_programt &wrapper,
  const std::string &func_name,
  const locationt &location) const
{
  // Generate snapshot assignments in the wrapper BEFORE calling the original function
  // We'll update old_snapshots to contain new wrapper snapshot variables

  // Map to track: original_expr -> wrapper_snapshot_var
  // This ensures we only create ONE wrapper snapshot per unique expression
  std::map<std::string, expr2tc> expr_to_wrapper_snapshot;

  size_t unique_snapshot_count = 0;

  for (size_t i = 0; i < old_snapshots.size(); ++i)
  {
    expr2tc original_expr = old_snapshots[i].original_expr;
    expr2tc old_temp_var =
      old_snapshots[i].snapshot_var; // The temp var from function body

    // Create a unique key for this expression (using pointer address as simple hash)
    std::ostringstream key_stream;
    key_stream << original_expr.get();
    std::string expr_key = key_stream.str();

    expr2tc new_snapshot_var;

    // Check if we've already created a wrapper snapshot for this expression
    auto it = expr_to_wrapper_snapshot.find(expr_key);
    if (it != expr_to_wrapper_snapshot.end())
    {
      // Reuse existing wrapper snapshot
      new_snapshot_var = it->second;
      log_debug(
        "contracts",
        "Reusing wrapper snapshot for duplicate __ESBMC_old expression");
    }
    else
    {
      // Create a NEW snapshot variable for the wrapper
      new_snapshot_var = create_snapshot_variable(
        original_expr, func_name + "_wrapper", unique_snapshot_count++);

      // Generate snapshot declaration
      goto_programt::targett decl_inst = wrapper.add_instruction(DECL);
      decl_inst->code = code_decl2tc(
        original_expr->type, to_symbol2t(new_snapshot_var).thename);
      decl_inst->location = location;
      decl_inst->location.comment("__ESBMC_old snapshot declaration");

      // Generate snapshot assignment: new_snapshot_var = original_expr
      goto_programt::targett assign_inst = wrapper.add_instruction(ASSIGN);
      assign_inst->code = code_assign2tc(new_snapshot_var, original_expr);
      assign_inst->location = location;
      assign_inst->location.comment("__ESBMC_old snapshot assignment");

      // Remember this mapping
      expr_to_wrapper_snapshot[expr_key] = new_snapshot_var;
    }

    // Store both old and new variables in the snapshot structure
    // We'll keep the old temp var as original_expr for matching,
    // and new snapshot var as snapshot_var for replacement
    old_snapshots[i].original_expr = old_temp_var;    // What to find
    old_snapshots[i].snapshot_var = new_snapshot_var; // What to replace with
  }
}

std::vector<code_contractst::old_snapshot_t>
code_contractst::materialize_old_snapshots_at_callsite(
  const std::vector<code_contractst::old_snapshot_t> &old_snapshots,
  const symbolt &function_symbol,
  const std::vector<expr2tc> &actual_args,
  goto_programt &replacement,
  const locationt &call_location) const
{
  std::vector<old_snapshot_t> callsite_snapshots;

  // For each old() in the original body, create a call-site snapshot:
  //   - Evaluate the original expression with actual arguments
  //   - Store it in a fresh snapshot variable before havoc
  //   - Remember mapping from the original temp variable to the snapshot
  for (size_t i = 0; i < old_snapshots.size(); ++i)
  {
    expr2tc original_expr = old_snapshots[i].original_expr;
    expr2tc temp_var = old_snapshots[i].snapshot_var; // temp var from body

    // Apply the same parameter substitution used for requires/ensures
    if (function_symbol.type.is_code())
    {
      const code_typet &code_type = to_code_type(function_symbol.type);
      const code_typet::argumentst &params = code_type.arguments();

      for (size_t j = 0; j < params.size() && j < actual_args.size(); ++j)
      {
        irep_idt param_id = params[j].get_identifier();
        expr2tc param_expr =
          symbol2tc(migrate_type(params[j].type()), param_id);
        original_expr =
          replace_symbol_in_expr(original_expr, param_expr, actual_args[j]);
      }
    }

    // Create a NEW snapshot variable for the call site
    expr2tc snapshot_var = create_snapshot_variable(
      original_expr, id2string(function_symbol.name) + "_call", i);

    // Generate snapshot declaration at call site
    goto_programt::targett decl_inst = replacement.add_instruction(DECL);
    decl_inst->code =
      code_decl2tc(original_expr->type, to_symbol2t(snapshot_var).thename);
    decl_inst->location = call_location;
    decl_inst->location.comment("__ESBMC_old call-site snapshot declaration");

    // Generate snapshot assignment: snapshot_var = original_expr
    goto_programt::targett assign_inst = replacement.add_instruction(ASSIGN);
    assign_inst->code = code_assign2tc(snapshot_var, original_expr);
    assign_inst->location = call_location;
    assign_inst->location.comment("__ESBMC_old call-site snapshot assignment");

    // Store mapping: temp var from original body -> call-site snapshot var.
    code_contractst::old_snapshot_t snap_entry;
    snap_entry.original_expr = temp_var;    // what to find in ensures
    snap_entry.snapshot_var = snapshot_var; // what to replace with
    callsite_snapshots.push_back(snap_entry);
  }

  if (!callsite_snapshots.empty())
  {
    log_debug(
      "contracts",
      "materialize_old_snapshots_at_callsite: created {} __ESBMC_old call-site "
      "snapshot(s) for function {}",
      callsite_snapshots.size(),
      id2string(function_symbol.name));
  }

  return callsite_snapshots;
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

  // NON-RECURSIVE: Only process direct typecast on return_value symbol
  // This avoids infinite recursion and circular references
  if (is_typecast2t(expr))
  {
    const typecast2t &cast = to_typecast2t(expr);

    // Check if we're casting a return_value symbol (directly, not nested)
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

  // No recursion: return original expression unchanged
  // The caller (fix_comparison_types) will handle nested structures explicitly
  return expr;
}

expr2tc code_contractst::fix_comparison_types(
  const expr2tc &expr,
  const expr2tc &ret_val) const
{
  if (is_nil_expr(expr) || is_nil_expr(ret_val))
    return expr;

  // NON-RECURSIVE APPROACH: Use explicit stack-based traversal to avoid infinite loops
  // We only need to fix comparison expressions, so we traverse the tree explicitly
  // and only process comparison nodes and their direct children

  // Step 1: Handle top-level comparison expressions
  if (is_comp_expr(expr))
  {
    expr2tc new_expr = expr->clone();

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
      // Step 1a: Remove incorrect casts on direct return_value symbols
      *side1 = remove_incorrect_casts(*side1, ret_val);
      *side2 = remove_incorrect_casts(*side2, ret_val);

      // Step 1b: Handle nested typecasts wrapping add/sub expressions
      // Example: (double)(old_snapshot + (signed int)return_value)
      // Only process one level: typecast -> add/sub -> operands
      for (expr2tc *side_ptr : {side1, side2})
      {
        if (is_typecast2t(*side_ptr))
        {
          const typecast2t &cast = to_typecast2t(*side_ptr);
          expr2tc inner = cast.from;

          // If inner is add/sub, fix its operands (one level only)
          if (is_add2t(inner))
          {
            const add2t &add = to_add2t(inner);
            expr2tc fixed_op1 = remove_incorrect_casts(add.side_1, ret_val);
            expr2tc fixed_op2 = remove_incorrect_casts(add.side_2, ret_val);

            // Only recreate if something changed
            if (fixed_op1 != add.side_1 || fixed_op2 != add.side_2)
            {
              // For floating-point types, use IEEE addition instead of regular addition
              // This matches how floating-point operations are compiled in the actual code
              if (is_fractional_type(cast.type))
              {
                // Use IEEE floating-point addition with default rounding mode
                expr2tc rounding_mode =
                  symbol2tc(get_int32_type(), "c:@__ESBMC_rounding_mode");
                expr2tc new_add =
                  ieee_add2tc(cast.type, fixed_op1, fixed_op2, rounding_mode);
                *side_ptr =
                  new_add; // No need for outer typecast, ieee_add already has correct type
              }
              else
              {
                // For non-floating-point types, use regular addition
                type2tc add_type = inner->type;
                if (
                  fixed_op1->type == fixed_op2->type &&
                  fixed_op1->type == cast.type)
                {
                  add_type = cast.type;
                }
                expr2tc new_add = add2tc(add_type, fixed_op1, fixed_op2);
                *side_ptr = typecast2tc(cast.type, new_add);
              }
            }
          }
          else if (is_sub2t(inner))
          {
            const sub2t &sub = to_sub2t(inner);
            expr2tc fixed_op1 = remove_incorrect_casts(sub.side_1, ret_val);
            expr2tc fixed_op2 = remove_incorrect_casts(sub.side_2, ret_val);

            if (fixed_op1 != sub.side_1 || fixed_op2 != sub.side_2)
            {
              type2tc sub_type = inner->type;
              expr2tc new_sub = sub2tc(sub_type, fixed_op1, fixed_op2);
              *side_ptr = typecast2tc(cast.type, new_sub);
            }
          }
          else
          {
            // Simple typecast - fix inner expression
            expr2tc fixed = remove_incorrect_casts(inner, ret_val);
            if (fixed != inner)
            {
              *side_ptr = typecast2tc(cast.type, fixed);
            }
          }
        }
      }

      // Step 1c: Check if one side is return_value and fix type mismatches
      bool side1_is_retval =
        is_symbol2t(*side1) && is_return_value_symbol(to_symbol2t(*side1));
      bool side2_is_retval =
        is_symbol2t(*side2) && is_return_value_symbol(to_symbol2t(*side2));

      // Case 1: return_value compared with integer constant, but return_value is pointer
      if (is_pointer_type(ret_val->type))
      {
        if (side1_is_retval && is_constant_int2t(*side2))
        {
          const constant_int2t &c = to_constant_int2t(*side2);
          if (c.value.is_zero())
          {
            *side2 = gen_zero(ret_val->type);
            log_debug(
              "contracts", "Fixed pointer comparison: replaced 0 with NULL");
          }
        }
        else if (side2_is_retval && is_constant_int2t(*side1))
        {
          const constant_int2t &c = to_constant_int2t(*side1);
          if (c.value.is_zero())
          {
            *side1 = gen_zero(ret_val->type);
            log_debug(
              "contracts", "Fixed pointer comparison: replaced 0 with NULL");
          }
        }
      }
      // Case 2: return_value is float/double, constant needs cast
      else if (is_fractional_type(ret_val->type))
      {
        if (side1_is_retval && is_constant_int2t(*side2))
        {
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

  // Step 2: Handle logical operators (AND, OR) that may contain comparisons
  // Only process one level: if this is AND/OR, process its direct operands
  if (is_and2t(expr) || is_or2t(expr))
  {
    expr2tc new_expr = expr->clone();
    bool changed = false;

    // Process each operand (but only one level deep)
    new_expr->Foreach_operand([this, &ret_val, &changed](expr2tc &op) {
      if (is_comp_expr(op))
      {
        expr2tc fixed = fix_comparison_types(op, ret_val);
        if (fixed != op)
        {
          op = fixed;
          changed = true;
        }
      }
    });

    return changed ? new_expr : expr;
  }

  // Step 3: For all other expressions, return unchanged
  // We don't recursively process arbitrary expression trees to avoid infinite loops
  return expr;
}

expr2tc code_contractst::normalize_fp_add_in_ensures(const expr2tc &expr) const
{
  if (is_nil_expr(expr))
    return expr;

  // NON-RECURSIVE: Only process floating-point add2t expressions
  // Convert regular floating-point addition to IEEE_ADD to match implementation semantics

  if (is_add2t(expr))
  {
    const add2t &add = to_add2t(expr);

    // Only convert if this is a floating-point type
    if (is_fractional_type(add.type))
    {
      // Use default rounding mode symbol (same as implementation)
      expr2tc rounding_mode =
        symbol2tc(get_int32_type(), "c:@__ESBMC_rounding_mode");

      // Convert to IEEE floating-point addition
      expr2tc new_expr =
        ieee_add2tc(add.type, add.side_1, add.side_2, rounding_mode);

      log_debug(
        "contracts",
        "Normalized floating-point addition to IEEE_ADD in ensures clause");

      return new_expr;
    }
  }

  // For non-add expressions or non-floating-point types, process operands
  // but only one level deep to avoid recursion issues
  if (is_and2t(expr) || is_or2t(expr))
  {
    expr2tc new_expr = expr->clone();
    bool changed = false;

    new_expr->Foreach_operand([this, &changed](expr2tc &op) {
      expr2tc normalized = normalize_fp_add_in_ensures(op);
      if (normalized != op)
      {
        op = normalized;
        changed = true;
      }
    });

    return changed ? new_expr : expr;
  }

  // For comparison expressions, process both sides
  if (is_comp_expr(expr))
  {
    expr2tc new_expr = expr->clone();
    bool changed = false;

    if (is_equality2t(new_expr))
    {
      equality2t &rel = to_equality2t(new_expr);
      expr2tc norm1 = normalize_fp_add_in_ensures(rel.side_1);
      expr2tc norm2 = normalize_fp_add_in_ensures(rel.side_2);
      if (norm1 != rel.side_1 || norm2 != rel.side_2)
      {
        rel.side_1 = norm1;
        rel.side_2 = norm2;
        changed = true;
      }
    }
    else if (is_notequal2t(new_expr))
    {
      notequal2t &rel = to_notequal2t(new_expr);
      expr2tc norm1 = normalize_fp_add_in_ensures(rel.side_1);
      expr2tc norm2 = normalize_fp_add_in_ensures(rel.side_2);
      if (norm1 != rel.side_1 || norm2 != rel.side_2)
      {
        rel.side_1 = norm1;
        rel.side_2 = norm2;
        changed = true;
      }
    }
    else if (is_lessthan2t(new_expr))
    {
      lessthan2t &rel = to_lessthan2t(new_expr);
      expr2tc norm1 = normalize_fp_add_in_ensures(rel.side_1);
      expr2tc norm2 = normalize_fp_add_in_ensures(rel.side_2);
      if (norm1 != rel.side_1 || norm2 != rel.side_2)
      {
        rel.side_1 = norm1;
        rel.side_2 = norm2;
        changed = true;
      }
    }
    else if (is_lessthanequal2t(new_expr))
    {
      lessthanequal2t &rel = to_lessthanequal2t(new_expr);
      expr2tc norm1 = normalize_fp_add_in_ensures(rel.side_1);
      expr2tc norm2 = normalize_fp_add_in_ensures(rel.side_2);
      if (norm1 != rel.side_1 || norm2 != rel.side_2)
      {
        rel.side_1 = norm1;
        rel.side_2 = norm2;
        changed = true;
      }
    }
    else if (is_greaterthan2t(new_expr))
    {
      greaterthan2t &rel = to_greaterthan2t(new_expr);
      expr2tc norm1 = normalize_fp_add_in_ensures(rel.side_1);
      expr2tc norm2 = normalize_fp_add_in_ensures(rel.side_2);
      if (norm1 != rel.side_1 || norm2 != rel.side_2)
      {
        rel.side_1 = norm1;
        rel.side_2 = norm2;
        changed = true;
      }
    }
    else if (is_greaterthanequal2t(new_expr))
    {
      greaterthanequal2t &rel = to_greaterthanequal2t(new_expr);
      expr2tc norm1 = normalize_fp_add_in_ensures(rel.side_1);
      expr2tc norm2 = normalize_fp_add_in_ensures(rel.side_2);
      if (norm1 != rel.side_1 || norm2 != rel.side_2)
      {
        rel.side_1 = norm1;
        rel.side_2 = norm2;
        changed = true;
      }
    }

    return changed ? new_expr : expr;
  }

  // For typecast expressions, process the inner expression
  if (is_typecast2t(expr))
  {
    const typecast2t &cast = to_typecast2t(expr);
    expr2tc normalized = normalize_fp_add_in_ensures(cast.from);
    if (normalized != cast.from)
    {
      return typecast2tc(cast.type, normalized);
    }
  }

  // For all other expressions, return unchanged
  return expr;
}

// ========== Unified ensures guard normalization ==========

expr2tc code_contractst::normalize_ensures_guard_for_return_value(
  const expr2tc &ensures_clause,
  const expr2tc &ret_val) const
{
  if (is_nil_expr(ensures_clause))
    return ensures_clause;

  expr2tc ensures_guard = ensures_clause;

  // Step 1: Replace __ESBMC_return_value with actual ret_val symbol
  if (!is_nil_expr(ret_val))
  {
    ensures_guard = replace_return_value_in_expr(ensures_guard, ret_val);
  }

  // Step 2: Fix type mismatches in comparison expressions involving return values
  // This removes incorrect casts and adds correct casts for constants
  if (!is_nil_expr(ret_val))
  {
    ensures_guard = fix_comparison_types(ensures_guard, ret_val);
  }

  // Step 3: Normalize floating-point addition to use IEEE semantics (matching implementation)
  // This ensures contracts use IEEE_ADD instead of regular + for floating-point operations
  ensures_guard = normalize_fp_add_in_ensures(ensures_guard);

  return ensures_guard;
}

bool code_contractst::has_contracts(const goto_programt &function_body) const
{
  // Quick check: scan for contract markers without extracting full clauses
  forall_goto_program_instructions (it, function_body)
  {
    // Check ASSUME instructions for requires/ensures/assigns
    if (it->is_assume())
    {
      std::string comment = id2string(it->location.comment());
      if (
        comment == "contract::requires" || comment == "contract::ensures" ||
        comment == "contract::assigns")
      {
        return true;
      }
    }
  }
  return false;
}

/// Helper: check if a function name matches any pattern in to_replace
static bool matches_replace_pattern(
  const std::string &func_name,
  const std::set<std::string> &to_replace)
{
  for (const auto &pattern : to_replace)
  {
    if (pattern == "*" || func_name.find(pattern) != std::string::npos)
      return true;
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
  std::map<std::string, std::tuple<symbolt *, goto_programt *, irep_idt>>
    function_map;

  // Collect all functions that have contracts and match to_replace patterns
  // Also build a set of function keys that are candidates for replacement
  std::set<std::string>
    replaceable_funcs; // function keys that have contracts and match

  Forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available)
      continue;

    symbolt *func_sym = find_function_symbol(id2string(it->first));
    if (func_sym != nullptr)
    {
      std::string func_key = id2string(it->first);
      function_map[func_key] = {func_sym, &it->second.body, it->first};
      log_debug(
        "contracts",
        "Added function to map: {} (name: {}, id: {})",
        func_key,
        id2string(func_sym->name),
        id2string(it->first));

      // Check if this function is a candidate for replacement
      if (
        matches_replace_pattern(func_key, to_replace) &&
        has_contracts(it->second.body))
      {
        replaceable_funcs.insert(func_key);
      }
    }
  }

  // --- Hierarchical replacement: build call graph among replaceable functions ---
  // A function is a "leaf" if it does NOT call any other replaceable function.
  // A function is a "parent" if it calls at least one other replaceable function.
  //
  // Strategy:
  //   - Leaf functions: replace their call sites with contract (havoc + assume)
  //   - Parent functions: keep their function body, but replace internal sub-calls
  //     with contracts. The parent function remains available for normal inlining.
  //
  // This preserves the precision of the parent function's control flow while
  // using sub-function contracts for modular reasoning.

  std::set<std::string>
    parent_funcs; // functions that call other replaceable functions

  for (const auto &func_key : replaceable_funcs)
  {
    auto map_it = function_map.find(func_key);
    if (map_it == function_map.end())
      continue;
    goto_programt *func_body = std::get<1>(map_it->second);

    // Scan function body for calls to other replaceable functions
    forall_goto_program_instructions (i_it, *func_body)
    {
      if (i_it->is_function_call() && is_code_function_call2t(i_it->code))
      {
        const code_function_call2t &call = to_code_function_call2t(i_it->code);
        if (is_symbol2t(call.function))
        {
          std::string called_name =
            id2string(to_symbol2t(call.function).get_symbol_name());
          if (is_compiler_generated(called_name))
            continue;
          if (replaceable_funcs.count(called_name) && called_name != func_key)
          {
            parent_funcs.insert(func_key);
            log_debug(
              "contracts",
              "Function {} calls replaceable function {} -> marking as parent",
              func_key,
              called_name);
            break;
          }
        }
      }
    }
  }

  // Leaf functions = replaceable but not parent
  std::set<std::string> leaf_funcs;
  for (const auto &f : replaceable_funcs)
  {
    if (!parent_funcs.count(f))
      leaf_funcs.insert(f);
  }

  log_status(
    "Hierarchical replacement: {} leaf function(s), {} parent function(s)",
    leaf_funcs.size(),
    parent_funcs.size());
  for (const auto &f : leaf_funcs)
    log_debug("contracts", "  Leaf: {}", f);
  for (const auto &f : parent_funcs)
    log_debug("contracts", "  Parent (kept, sub-calls replaced): {}", f);

  // Track functions that have been fully replaced (to delete their definitions)
  std::set<irep_idt> functions_to_delete;

  // Process each function and replace calls
  Forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available)
      continue;

    std::string current_func = id2string(it->first);

    std::vector<goto_programt::targett> calls_to_replace;
    std::vector<std::tuple<symbolt *, goto_programt *, irep_idt>> func_info_vec;

    // Find calls to replace in this function's body
    Forall_goto_program_instructions (i_it, it->second.body)
    {
      if (i_it->is_function_call() && is_code_function_call2t(i_it->code))
      {
        const code_function_call2t &call = to_code_function_call2t(i_it->code);
        if (is_symbol2t(call.function))
        {
          const symbol2t &func_sym = to_symbol2t(call.function);
          std::string called_func = id2string(func_sym.get_symbol_name());

          if (is_compiler_generated(called_func))
            continue;

          // Determine if we should replace this call:
          // 1. If current function is a parent: only replace calls to LEAF functions
          //    (replace sub-calls but keep the parent's own control flow)
          // 2. If current function is NOT a parent (e.g., main):
          //    replace calls to LEAF functions with contract
          //    (parent functions are NOT replaced - they'll be inlined normally)
          bool should_replace = false;

          if (leaf_funcs.count(called_func))
          {
            // Always replace calls to leaf functions
            should_replace = true;
          }
          // Note: calls to parent functions from main/other contexts are NOT replaced.
          // Parent functions keep their body and will be inlined by ESBMC's normal
          // function call handling. Their internal sub-calls have already been
          // replaced with contracts above.

          if (should_replace)
          {
            auto map_it = function_map.find(called_func);
            if (map_it != function_map.end())
            {
              goto_programt *func_body = std::get<1>(map_it->second);
              if (has_contracts(*func_body))
              {
                log_debug(
                  "contracts",
                  "In {}: replacing call to leaf function {}",
                  current_func,
                  called_func);
                calls_to_replace.push_back(i_it);
                func_info_vec.push_back(map_it->second);
                // Only leaf functions get deleted
                irep_idt func_id = std::get<2>(map_it->second);
                functions_to_delete.insert(func_id);
              }
            }
          }
        }
      }
    }

    // Replace calls
    log_debug(
      "contracts",
      "Found {} calls to replace in function {}",
      calls_to_replace.size(),
      current_func);
    for (size_t i = 0; i < calls_to_replace.size(); ++i)
    {
      log_debug(
        "contracts", "Replacing call {} of {}", i + 1, calls_to_replace.size());
      symbolt *func_sym = std::get<0>(func_info_vec[i]);
      goto_programt *func_body = std::get<1>(func_info_vec[i]);
      generate_replacement_at_call(
        *func_sym, *func_body, calls_to_replace[i], it->second.body);
    }
  }

  // Delete only LEAF function definitions (they've been fully replaced with contracts).
  // Parent functions are kept - they'll be inlined normally with their sub-calls
  // already replaced by contracts.
  for (const auto &func_id : functions_to_delete)
  {
    auto func_it = goto_functions.function_map.find(func_id);
    if (func_it != goto_functions.function_map.end())
    {
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
  std::vector<expr2tc> assigns_target_exprs =
    extract_assigns_from_body(function_body);

  // Debug: log extracted clauses
  log_debug(
    "contracts",
    "generate_replacement_at_call: extracted requires clause (nil={})",
    is_nil_expr(requires_clause));
  log_debug(
    "contracts",
    "generate_replacement_at_call: extracted ensures clause (nil={})",
    is_nil_expr(ensures_clause));
  log_debug(
    "contracts",
    "generate_replacement_at_call: extracted {} assigns target expressions",
    assigns_target_exprs.size());

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
      expr2tc param_expr = symbol2tc(migrate_type(params[i].type()), param_id);

      // Replace parameter symbol with actual argument in requires/ensures
      requires_clause =
        replace_symbol_in_expr(requires_clause, param_expr, actual_args[i]);
      ensures_clause =
        replace_symbol_in_expr(ensures_clause, param_expr, actual_args[i]);

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
    "After parameter replacement: requires nil={}, ensures nil={}, function={}",
    is_nil_expr(requires_clause),
    is_nil_expr(ensures_clause),
    id2string(function_symbol.name));

  // 1.b Create call-site snapshots for __ESBMC_old() expressions (if any)
  // This mirrors the snapshot creation in generate_checking_wrapper, but
  // moves the snapshots to the call site instead of a wrapper function.
  std::vector<old_snapshot_t> body_snapshots =
    collect_old_snapshots_from_body(function_body);
  std::vector<old_snapshot_t> callsite_snapshots =
    materialize_old_snapshots_at_callsite(
      body_snapshots, function_symbol, actual_args, replacement, call_location);

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

      // For ASSERT: only add if false (violation). A true constant would be
      // a no-op assertion, so we skip it to avoid cluttering the GOTO program.
      // For ASSUME: only add if true (skip trivially false assumptions).
      if (inst_type == ASSERT)
        should_add = !b.value;
      else // ASSUME
        should_add = b.value;
    }
    else
    {
      // Non-constant expressions should always be added
      should_add = true;
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

  // 0. Extract call site condition and enhance requires clause
  // This preserves the control flow context (e.g., if conditions) in which
  // the function is called, preventing over-approximation.
  // Convert targett to const_targett for extract_call_site_condition
  goto_programt::const_targett const_call_instruction = call_instruction;
  expr2tc call_site_condition =
    extract_call_site_condition(const_call_instruction, caller_body);
  if (!is_true(call_site_condition))
  {
    // Merge call site condition with requires clause
    if (is_nil_expr(requires_clause) || is_true(requires_clause))
    {
      // If requires is nil or true, use call site condition as requires
      requires_clause = call_site_condition;
      log_debug(
        "contracts",
        "Enhanced requires with call site condition (requires was nil/true)");
    }
    else
    {
      // Combine requires and call site condition with AND
      requires_clause = and2tc(requires_clause, call_site_condition);
      log_debug(
        "contracts",
        "Enhanced requires with call site condition (combined with AND)");
    }
  }

  // 1. Assert requires clause (check precondition at call site)
  add_contract_clause(
    requires_clause, ASSERT, "contract requires", "contract requires");

  // 2. Havoc all potentially modified locations
  // In replace_calls mode, we must havoc everything the function might modify,
  // otherwise the effects cannot propagate from the removed function body.

  bool has_empty_assigns = has_empty_assigns_marker(function_body);

  if (!assigns_target_exprs.empty())
  {
    // 2.1. Precise havoc: Only havoc expressions in assigns clause
    // This implements the key feature for eliminating false counterexamples
    // Now assigns targets are expression trees that need parameter substitution
    for (const expr2tc &target_expr : assigns_target_exprs)
    {
      // Substitute function parameters with actual call arguments
      expr2tc instantiated_target = target_expr;

      if (function_symbol.type.is_code())
      {
        const code_typet &code_type = to_code_type(function_symbol.type);
        const code_typet::argumentst &params = code_type.arguments();

        for (size_t i = 0; i < params.size() && i < actual_args.size(); ++i)
        {
          const code_typet::argumentt &param = params[i];
          irep_idt param_id = param.get_identifier();

          if (!param_id.empty() && !is_nil_expr(actual_args[i]))
          {
            type2tc param_type = migrate_type(param.type());
            expr2tc param_symbol = symbol2tc(param_type, param_id);
            instantiated_target = replace_symbol_in_expr(
              instantiated_target, param_symbol, actual_args[i]);
          }
        }
      }

      // Skip pointer havoc in value-set mode (consistent with loop invariant)
      if (
        config.options.get_bool_option("add-symex-value-sets") &&
        is_pointer_type(instantiated_target))
        continue;

      // Generate nondeterministic value and create assignment
      // Special handling for array types: use ARRAY_OF(NONDET) construction
      expr2tc rhs;
      if (is_array_type(instantiated_target->type))
      {
        // For arrays, generate ARRAY_OF(nondet_element)
        const array_type2t &arr_type = to_array_type(instantiated_target->type);
        expr2tc nondet_elem = gen_nondet(arr_type.subtype);
        rhs = constant_array_of2tc(instantiated_target->type, nondet_elem);
      }
      else
      {
        rhs = gen_nondet(instantiated_target->type);
      }

      goto_programt::targett t = replacement.add_instruction(ASSIGN);
      t->code = code_assign2tc(instantiated_target, rhs);
      t->location = call_location;
      t->location.comment("contract havoc assigns");

      log_debug("contracts", "Havoc'd assigns target expression");
    }

    log_debug(
      "contracts",
      "Precise havoc: havoc'd {} expressions from assigns clause",
      assigns_target_exprs.size());
  }
  else if (has_empty_assigns)
  {
    // 2.2. Explicit empty assigns: __ESBMC_assigns(0) was used
    // This means the function is pure (no side effects), so don't havoc anything
    log_debug(
      "contracts",
      "Empty assigns: function is pure (no side effects), no havoc");
  }
  else
  {
    // 2.3. Conservative havoc: No assigns clause, so havoc all globals
    // This is the old behavior - safe but may introduce false positives
    havoc_static_globals(replacement, call_location);

    log_debug(
      "contracts",
      "Conservative havoc: no assigns clause, havoc'd all static globals");
  }

  // 2.3. Havoc memory locations modified through pointer parameters
  // TODO: Analyze pointer parameters and havoc dereferenced locations
  // For now, we rely on assigns clause and conservative global havoc
  // This is a conservative over-approximation

  // 3. Normalize ensures guard: replace return_value, fix types, normalize floating-point
  expr2tc ensures_guard =
    normalize_ensures_guard_for_return_value(ensures_clause, ret_val);

  // 3.b Replace __ESBMC_old() occurrences in ensures using call-site snapshots
  if (!callsite_snapshots.empty() && !is_nil_expr(ensures_guard))
  {
    log_debug(
      "contracts",
      "generate_replacement_at_call: replacing __ESBMC_old expressions in "
      "ensures (before type={})",
      get_type_id(*ensures_guard->type));
    ensures_guard = replace_old_in_expr(ensures_guard, callsite_snapshots);
    log_debug(
      "contracts",
      "generate_replacement_at_call: replaced __ESBMC_old expressions in "
      "ensures (after type={})",
      ensures_guard ? get_type_id(*ensures_guard->type) : "nil");
  }

  // 4. Assume ensures clause (assume postcondition at call site)
  add_contract_clause(
    ensures_guard, ASSUME, "contract ensures", "contract ensures");

  // Replace call with replacement code
  // Insert replacement code before the call instruction
  // destructive_insert inserts BEFORE the target (unlike insert_swap which inserts AFTER)

  // Debug: log replacement code generation
  size_t replacement_size = replacement.instructions.size();
  log_debug(
    "contracts",
    "Replacement code generated: {} instructions",
    replacement_size);

  if (!replacement.instructions.empty())
  {
    // Debug: log what we're inserting
    log_debug(
      "contracts",
      "Inserting {} instructions before call instruction",
      replacement_size);

    caller_body.destructive_insert(call_instruction, replacement);

    // Debug: verify insertion
    log_debug(
      "contracts",
      "Call instruction after insertion: type={}",
      (int)call_instruction->type);
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
  log_debug(
    "contracts",
    "Call instruction marked as SKIP: type={}",
    (int)call_instruction->type);
}

expr2tc code_contractst::extract_call_site_condition(
  goto_programt::const_targett call_instruction,
  const goto_programt & /* caller_body */) const
{
  // Extract the guard condition from the call instruction
  // In goto programs, each instruction has a guard field that represents
  // the condition under which the instruction executes.
  // If the call is inside an if statement, the guard will be the if condition.
  expr2tc call_guard = call_instruction->guard;

  // If guard is true (unconditional), return true expression
  if (is_true(call_guard))
    return gen_true_expr();

  // Otherwise, return the guard condition
  log_debug("contracts", "Extracted call site condition (not true)");
  return call_guard;
}

// ========== Pointer validity assumptions support ==========

void code_contractst::add_pointer_validity_assumptions(
  goto_programt &wrapper,
  const symbolt &func,
  const locationt &location)
{
  if (!func.type.is_code())
    return;

  const code_typet &code_type = to_code_type(func.type);
  const code_typet::argumentst &params = code_type.arguments();

  for (const auto &param : params)
  {
    // Construct symbol p
    type2tc param_type = migrate_type(param.type());
    expr2tc p = symbol2tc(param_type, param.get_identifier());

    // Check if this is a pointer type
    if (!is_pointer_type(p))
      continue;

    // Construct "p is valid pointer" assumption
    // We simply assume: p points to a valid object (valid_object(p))
    // valid_object includes alignment checks internally
    expr2tc validity_check = valid_object2tc(p);

    // Add ASSUME instruction
    auto t = wrapper.add_instruction(ASSUME);
    t->guard = validity_check;
    t->location = location;
    t->location.comment("assume non-null parameter is valid");

    log_debug(
      "contracts",
      "add_pointer_validity_assumptions: added validity assumption for "
      "parameter {}",
      id2string(param.get_identifier()));
  }
}
