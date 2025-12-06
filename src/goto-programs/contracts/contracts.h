#ifndef ESBMC_CONTRACTS_H
#define ESBMC_CONTRACTS_H

#include <goto-programs/goto_functions.h>
#include <util/context.h>
#include <util/namespace.h>
#include <set>
#include <string>

/// \brief Basic contract handling class
/// Provides contract checking and replacement functionality at goto function level
class code_contractst
{
public:
  code_contractst(
    goto_functionst &goto_functions,
    contextt &context,
    const namespacet &ns);

  /// \brief Enforce function contracts
  /// Renames function F to __ESBMC_contracts_original_F and generates a new wrapper function F
  /// Wrapper function: assume requires -> call original function -> assert ensures
  /// \param to_enforce Set of function names to enforce contracts for
  void enforce_contracts(const std::set<std::string> &to_enforce);

  /// \brief Replace function calls with contracts
  /// Replaces function calls with: assert requires -> havoc assigns targets -> assume ensures
  /// \param to_replace Set of function names to replace with contracts
  void replace_calls(const std::set<std::string> &to_replace);

  /// \brief Scan all functions and create contract symbols
  /// Scans goto programs for contract annotations and creates contract symbols
  void build_contract_symbols();

  /// \brief Quick check if function has any contracts
  /// \param function_body Function goto program
  /// \return True if function has any contract clauses
  bool has_contracts(const goto_programt &function_body) const;

private:
  goto_functionst &goto_functions;
  contextt &context;
  const namespacet &ns;

  /// \brief Check if a function is compiler-generated and should be skipped
  /// \param function_name Function name or ID
  /// \return True if function should be skipped (destructor, constructor, etc.)
  bool is_compiler_generated(const std::string &function_name) const;

  /// \brief Find contract symbol (with contract:: prefix)
  /// \param function_name Function name
  /// \return Pointer to contract symbol, or nullptr if not found
  symbolt *find_contract_symbol(const std::string &function_name);

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
  /// \return Generated wrapper function body
  goto_programt generate_checking_wrapper(
    const symbolt &original_func,
    const expr2tc &requires_clause,
    const expr2tc &ensures_clause,
    const irep_idt &original_func_id);

  /// \brief Generate replacement code at function call site
  /// \param contract_symbol Contract symbol
  /// \param call_instruction Function call instruction
  /// \param function_body Function body containing the call
  void generate_replacement_at_call(
    const symbolt &contract_symbol,
    goto_programt::targett call_instruction,
    goto_programt &function_body);

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

  /// \brief Extract assigns clause from contract symbol
  /// \param contract_symbol Contract symbol
  /// \return Assigns expression, or nil_exprt() if not present
  expr2tc extract_assigns_clause(const symbolt &contract_symbol);

  /// \brief Replace __ESBMC_return_value symbols in expression with actual return value
  /// \param expr Expression to replace symbols in
  /// \param ret_val Actual return value expression
  /// \return Expression with __ESBMC_return_value replaced
  expr2tc
  replace_return_value_in_expr(const expr2tc &expr, const expr2tc &ret_val);

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
};

#endif // ESBMC_CONTRACTS_H
