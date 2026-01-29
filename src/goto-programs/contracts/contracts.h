/*
 * This function is used to check function contracts.
 * 
 * Verification Strategy: Abstraction and Hierarchical Verification
 * 
 * Function contracts enable a manual abstraction approach to assist over-approximation
 * verification logic. By splitting verification into system-level and function-level,
 * we should be able to reduce verification complexity. 
 * 
 * 1. Function-level verification (enforce_contracts):
 *    - Verify each function independently against its contract
 *    - Use contract as specification: assume requires -> execute function -> assert ensures
 *    - This provides over-approximation: if function satisfies contract, it's correct
 *    - Complexity: O(n) where n is function size, not system size
 * 
 * 2. System-level verification (replace_calls):
 *    - Replace function calls with contract semantics
 *    - Use contract as abstraction: assert requires -> havoc assigns -> assume ensures
 *    - This avoids exploring function body, reducing state space
 *    - Complexity: O(m) where m is call sites, not function implementations
 * 
 * Benefits:
 * - Modularity: Verify functions separately from system
 * - Scalability: System verification doesn't need to explore function internals
 * - Reusability: Once function is verified, contract can be used in any context
 * - Over-approximation: Contract provides safe abstraction (may have false positives)
 * 
 * This approach transforms a complex system verification problem into:
 * - Multiple simpler function verification problems
 * - One system verification problem using abstracted functions
 * 
 * It takes phases:
 * 1. Extract contract clauses (requires, ensures, assigns) from contract symbol
 * 2. For contract checking: rename original function, generate wrapper with assume requires -> call -> assert ensures
 * 3. For contract replacement: replace calls with assert requires -> havoc assigns -> assume ensures
 */

#ifndef ESBMC_CONTRACTS_H
#define ESBMC_CONTRACTS_H

#include <goto-programs/goto_functions.h>
#include <util/context.h>
#include <util/namespace.h>
#include <set>
#include <string>

/// \brief Basic contract handling class
/// Provides contract checking and replacement functionality at goto function level
///
/// Function contracts are specifications that describe the behavior of functions.
/// They consist of:
/// - Preconditions (requires): conditions that must hold when the function is called
/// - Postconditions (ensures): conditions that must hold when the function returns
/// - Assigns clauses: memory locations that the function may modify
///
/// Contracts enable modular verification by allowing functions to be verified
/// independently and then used as abstractions in system-level verification.
class code_contractst
{
public:
  // ========== __ESBMC_is_fresh support for ensures ==========

  /// \brief Structure to store is_fresh mapping information
  struct is_fresh_mapping_t
  {
    irep_idt
      temp_var_name; ///< Temporary variable name (e.g., return_value$___ESBMC_is_fresh$1)
    expr2tc ptr_expr; ///< Pointer expression (dereferenced from &ptr)
  };

  code_contractst(
    goto_functionst &goto_functions,
    contextt &context,
    const namespacet &ns);

  /// \brief Enforce function contracts
  /// Renames function F to __ESBMC_contracts_original_F and generates a new wrapper function F
  /// Wrapper function: assume requires -> call original function -> assert ensures
  /// \param to_enforce Set of function names to enforce contracts for
  /// \param assume_nonnull_valid If true, assume non-null pointer parameters are valid objects
  void enforce_contracts(
    const std::set<std::string> &to_enforce,
    bool assume_nonnull_valid = false);

  /// \brief Replace function calls with contracts
  /// Replaces function calls with contract semantics:
  ///   1. Assert requires clause (check precondition)
  ///   2. Havoc all potentially modified locations:
  ///      - Assigns clause targets (if specified)
  ///      - Static lifetime global variables (conservative)
  ///      - Memory locations through pointer parameters (TODO)
  ///   3. Assume ensures clause (assume postcondition)
  ///
  /// CRITICAL: We must havoc everything the function might modify,
  /// otherwise the effects cannot propagate from the removed function body.
  /// \param to_replace Set of function names to replace with contracts
  /// \param remove_functions If true, delete function definitions after replacement (default: true)
  void replace_calls(const std::set<std::string> &to_replace, bool remove_functions = true);

  /// \brief Automatically apply conservative havoc to functions with #pragma contract
  /// This method:
  ///   1. Identifies functions marked with #pragma contract annotation
  ///   2. Injects default conservative contracts (require(true), ensure(true)) if needed
  ///   3. Applies replace_calls to replace call sites with contract semantics
  ///   4. Applies enforce_contracts to verify each function independently
  ///
  /// The conservative havoc approach provides over-approximation:
  /// - May produce false counterexamples (safe but imprecise)
  /// - Useful for modular verification without explicit contract specifications
  void auto_havoc_pragma_contracts();

  /// \brief Quick check if function has any contracts
  /// \param function_body Function goto program
  /// \return True if function has any contract clauses
  bool has_contracts(const goto_programt &function_body) const;

  /// \brief Insert enforce-phase verification calls into main using NONDET branches
  /// This creates two verification paths in main:
  ///   - Path 1: Call wrappers with nondet args (verify enforce phase)
  ///   - Path 2: Execute original main (verify replace phase)
  /// \param functions_with_pragma Set of function names with pragma contracts
  void insert_enforce_verification_calls(
    const std::set<std::string> &functions_with_pragma);

  /// \brief Check if function has #pragma contract annotation
  /// \param function_symbol Function symbol to check
  /// \return True if function has #pragma contract marker
  bool has_pragma_contract(const symbolt &function_symbol) const;

  // ========== Contract Comment String Constants ==========
  // Centralized contract marker strings to avoid duplication and typos
  struct contract_comments {
    static constexpr const char* REQUIRES = "contract::requires";
    static constexpr const char* REQUIRES_ENFORCE = "contract requires (enforce)";
    static constexpr const char* REQUIRES_REPLACE = "contract requires (replace)";
    static constexpr const char* ENSURES = "contract::ensures";
    static constexpr const char* ENSURES_ENFORCE = "contract ensures (enforce)";
    static constexpr const char* ENSURES_REPLACE = "contract ensures (replace)";
    static constexpr const char* ASSIGNS = "contract::assigns";
    static constexpr const char* ASSIGNS_EMPTY = "contract::assigns_empty";
  };

  /// \brief Check if comment string matches any requires marker
  static bool is_requires_comment(const std::string &comment);
  
  /// \brief Check if comment string matches any ensures marker
  static bool is_ensures_comment(const std::string &comment);
  
  /// \brief Check if comment string matches any assigns marker
  static bool is_assigns_comment(const std::string &comment);
  
  /// \brief Check if comment string matches any contract marker
  static bool is_contract_comment(const std::string &comment);

private:
  goto_functionst &goto_functions;
  contextt &context;
  const namespacet &ns;

  /// \brief Check if a function is compiler-generated and should be skipped
  /// \param function_name Function name or ID
  /// \return True if function should be skipped (destructor, constructor, etc.)
  bool is_compiler_generated(const std::string &function_name) const;

  /// \brief Find function symbol
  /// \param function_name Function name (can be full ID or simple name)
  /// \return Pointer to function symbol, or nullptr if not found
  symbolt *find_function_symbol(const std::string &function_name);

  /// \brief Rename function
  /// \param old_id Original function ID
  /// \param new_id New function ID
  void rename_function(const irep_idt &old_id, const irep_idt &new_id);

  /// \brief Generate checking mode wrapper function body
  /// \param original_func Original function symbol
  /// \param requires_clause Requires expression
  /// \param ensures_clause Ensures expression
  /// \param original_func_id ID of the renamed original function
  /// \param original_body Original function body (before renaming)
  /// \param is_fresh_mappings Mappings for is_fresh temp variables in ensures
  /// \param assume_nonnull_valid If true, assume non-null pointer parameters are valid objects
  /// \return Generated wrapper function body
  goto_programt generate_checking_wrapper(
    const symbolt &original_func,
    const expr2tc &requires_clause,
    const expr2tc &ensures_clause,
    const irep_idt &original_func_id,
    const goto_programt &original_body,
    const std::vector<is_fresh_mapping_t> &is_fresh_mappings,
    bool assume_nonnull_valid = false);

  /// \brief Generate replacement code at function call site
  /// \param function_symbol Function symbol being called
  /// \param function_body Function body (to extract contracts from)
  /// \param call_instruction Function call instruction
  /// \param caller_body Function body containing the call
  void generate_replacement_at_call(
    const symbolt &function_symbol,
    const goto_programt &function_body,
    goto_programt::targett call_instruction,
    goto_programt &caller_body);

  /// \brief Extract call site condition from goto program
  /// Extracts the guard condition that must hold for the call instruction to execute.
  /// This includes conditions from if statements, loops, and other control flow constructs.
  /// \param call_instruction Function call instruction
  /// \param caller_body Function body containing the call
  /// \return Guard condition expression, or true_exprt() if unconditional
  expr2tc extract_call_site_condition(
    goto_programt::const_targett call_instruction,
    const goto_programt &caller_body) const;

  /// \brief Extract requires clause from contract symbol
  /// \param contract_symbol Contract symbol
  /// \return Requires expression, or true_exprt() if not present
  expr2tc extract_requires_clause(const symbolt &contract_symbol);

  /// \brief Extract ensures clause from contract symbol
  /// \param contract_symbol Contract symbol
  /// \return Ensures expression, or true_exprt() if not present
  expr2tc extract_ensures_clause(const symbolt &contract_symbol);

  /// \brief Extract requires clauses from function body
  /// \param function_body Function goto program
  /// \return Requires expression (conjunction of all requires), or true_exprt() if none
  expr2tc extract_requires_from_body(const goto_programt &function_body);

  /// \brief Extract ensures clauses from function body
  /// \param function_body Function goto program
  /// \return Ensures expression (conjunction of all ensures), or true_exprt() if none
  expr2tc extract_ensures_from_body(const goto_programt &function_body);

  /// \brief Extract assigns clause from function body
  /// \param function_body Function goto program
  /// \return Vector of assign target expressions from __ESBMC_assigns()
  std::vector<expr2tc>
  extract_assigns_from_body(const goto_programt &function_body);

  /// \brief Extract assigns clause from contract symbol
  /// \param contract_symbol Contract symbol
  /// \return Assigns expression, or nil_exprt() if not present
  expr2tc extract_assigns_clause(const symbolt &contract_symbol);

  /// \brief Replace __ESBMC_return_value symbols in expression with actual return value
  /// \param expr Expression to replace symbols in
  /// \param ret_val Actual return value expression
  /// \return Expression with __ESBMC_return_value replaced
  expr2tc replace_return_value_in_expr(
    const expr2tc &expr,
    const expr2tc &ret_val) const;

  /// \brief Extract struct/union member accesses to temporary variables
  /// For struct return values, accessing members directly (ret_val.x) can cause
  /// symbolic execution issues when ret_val's value is a 'with' expression.
  /// This function extracts member accesses to temporary variables to avoid dereference failures.
  /// \param expr Expression containing member accesses
  /// \param ret_val Return value symbol (must be struct/union type)
  /// \param wrapper GOTO program to add temporary variable declarations and assignments
  /// \param location Source location for generated instructions
  /// \return Expression with member accesses replaced by temporary variables
  expr2tc extract_struct_members_to_temps(
    const expr2tc &expr,
    const expr2tc &ret_val,
    goto_programt &wrapper,
    const locationt &location);

  /// \brief Replace a symbol in expression with another expression
  /// \param expr Expression to replace symbols in
  /// \param old_symbol Symbol to replace
  /// \param new_expr Expression to replace with
  /// \return Expression with old_symbol replaced by new_expr
  expr2tc replace_symbol_in_expr(
    const expr2tc &expr,
    const expr2tc &old_symbol,
    const expr2tc &new_expr) const;

  // ========== __ESBMC_old support ==========

  /// \brief Structure to store old() snapshot information
  struct old_snapshot_t
  {
    expr2tc original_expr; ///< Expression inside __ESBMC_old()
    expr2tc snapshot_var;  ///< Snapshot variable symbol
  };

  /// \brief Check if expression is an __ESBMC_old() call
  /// \param expr Expression to check
  /// \return True if expr is a sideeffect with kind old_snapshot
  bool is_old_call(const expr2tc &expr) const;

  /// \brief Create a snapshot variable for an old() expression
  /// \param expr Expression to snapshot
  /// \param func_name Function name (for unique naming)
  /// \param index Index of this snapshot (for unique naming)
  /// \return Symbol expression for the snapshot variable
  expr2tc create_snapshot_variable(
    const expr2tc &expr,
    const std::string &func_name,
    size_t index) const;

  /// \brief Replace __ESBMC_old() calls with snapshot variables
  /// \param expr Expression containing old() calls
  /// \param snapshots Vector of snapshot information
  /// \return Expression with old() calls replaced by snapshot variables
  expr2tc replace_old_in_expr(
    const expr2tc &expr,
    const std::vector<old_snapshot_t> &snapshots) const;

  /// \brief Collect old_snapshot assignments from function body
  /// \param function_body GOTO program to scan for old_snapshot sideeffects
  /// \return Vector of old_snapshot_t structures (original_expr, temp_var)
  std::vector<old_snapshot_t>
  collect_old_snapshots_from_body(const goto_programt &function_body) const;

  /// \brief Materialize old snapshots in wrapper function (enforce-contract mode)
  /// Creates DECL and ASSIGN instructions for snapshot variables before function call
  /// \param old_snapshots Vector of snapshots to materialize (modified in-place)
  /// \param wrapper GOTO program to add snapshot instructions to
  /// \param func_name Function name for unique variable naming
  /// \param location Source location for generated instructions
  void materialize_old_snapshots_at_wrapper(
    std::vector<old_snapshot_t> &old_snapshots,
    goto_programt &wrapper,
    const std::string &func_name,
    const locationt &location) const;

  /// \brief Materialize old snapshots at call site (replace-call mode)
  /// Creates DECL and ASSIGN instructions for snapshot variables at call location
  /// \param old_snapshots Vector of snapshots from function body
  /// \param function_symbol Function symbol for parameter substitution
  /// \param actual_args Actual arguments at call site
  /// \param replacement GOTO program to add snapshot instructions to
  /// \param call_location Source location for generated instructions
  /// \return Vector of call-site snapshots (with parameter substitution applied)
  std::vector<old_snapshot_t> materialize_old_snapshots_at_callsite(
    const std::vector<old_snapshot_t> &old_snapshots,
    const symbolt &function_symbol,
    const std::vector<expr2tc> &actual_args,
    goto_programt &replacement,
    const locationt &call_location) const;

  // ========== Type fixing for return value comparisons ==========

  /// \brief Check if a symbol represents a return value variable
  /// \param sym Symbol to check
  /// \return True if symbol is a return value variable (matches patterns like "return_value", "__ESBMC_return_value", etc.)
  bool is_return_value_symbol(const symbol2t &sym) const;

  /// \brief Remove incorrect typecasts on return value symbols
  /// \param expr Expression to process
  /// \param ret_val Return value symbol with correct type
  /// \return Expression with incorrect casts removed
  expr2tc
  remove_incorrect_casts(const expr2tc &expr, const expr2tc &ret_val) const;

  /// \brief Fix type mismatches in comparison expressions involving return values
  /// \param expr Expression to fix (typically an ensures guard)
  /// \param ret_val Return value symbol with correct type
  /// \return Expression with corrected type casts
  expr2tc
  fix_comparison_types(const expr2tc &expr, const expr2tc &ret_val) const;

  /// \brief Normalize floating-point addition in contract expressions to use IEEE semantics
  /// This ensures contracts use IEEE_ADD (matching implementation) instead of regular +
  /// \param expr Expression to normalize (typically an ensures guard)
  /// \return Expression with floating-point add2t replaced by ieee_add2t
  expr2tc normalize_fp_add_in_ensures(const expr2tc &expr) const;

  /// \brief Normalize ensures guard expression for return value handling
  /// This is a unified helper that applies all return_value-related transformations:
  /// 1. Replaces __ESBMC_return_value with actual ret_val symbol
  /// 2. Fixes type mismatches in comparisons (removes incorrect casts, adds correct casts)
  /// 3. Normalizes floating-point operations to use IEEE semantics
  /// \param ensures_clause Original ensures clause expression
  /// \param ret_val Return value symbol (may be nil if function returns void)
  /// \return Normalized ensures guard ready for ASSERT/ASSUME
  expr2tc normalize_ensures_guard_for_return_value(
    const expr2tc &ensures_clause,
    const expr2tc &ret_val) const;

  // ========== __ESBMC_is_fresh support for ensures ==========

  /// \brief Extract is_fresh mappings from function body
  /// \param function_body Function goto program
  /// \return Vector of is_fresh mappings (temp var name -> pointer expr)
  std::vector<is_fresh_mapping_t>
  extract_is_fresh_mappings_from_body(const goto_programt &function_body) const;

  /// \brief Replace is_fresh temporary variables in ensures with verification expressions
  /// \param expr Expression containing is_fresh temp variables
  /// \param mappings Vector of is_fresh mappings
  /// \return Expression with is_fresh temp variables replaced by verification expressions
  expr2tc replace_is_fresh_in_ensures_expr(
    const expr2tc &expr,
    const std::vector<is_fresh_mapping_t> &mappings) const;

  /// \brief Havoc assigns targets (similar to loop invariant approach)
  /// \param assigns_clause Assigns clause expression
  /// \param dest Destination goto program
  /// \param location Location information
  void havoc_assigns_targets(
    const expr2tc &assigns_clause,
    goto_programt &dest,
    const locationt &location);

  /// \brief Extract target variable list from assigns clause
  /// \param assigns_clause Assigns clause expression
  /// \return List of target variable expressions
  std::vector<expr2tc> extract_assigns_targets(const expr2tc &assigns_clause);

  /// \brief Havoc function parameters before checking the contract
  /// \param original_func Original function symbol
  /// \param dest Destination goto program (wrapper body)
  /// \param location Location information
  void havoc_function_parameters(
    const symbolt &original_func,
    goto_programt &dest,
    const locationt &location);

  /// \brief Havoc static lifetime global variables before checking the contract
  /// \param dest Destination goto program (wrapper body)
  /// \param location Location information
  void havoc_static_globals(goto_programt &dest, const locationt &location);

  /// \brief Add pointer validity assumptions for non-null pointer parameters
  /// Used with --assume-nonnull-valid flag in enforce-contract mode
  /// \param wrapper Destination goto program (wrapper body)
  /// \param func Function symbol
  /// \param location Location information
  void add_pointer_validity_assumptions(
    goto_programt &wrapper,
    const symbolt &func,
    const locationt &location);
};

#endif // ESBMC_CONTRACTS_H
