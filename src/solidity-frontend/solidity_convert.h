#ifndef SOLIDITY_FRONTEND_SOLIDITY_CONVERT_H_
#define SOLIDITY_FRONTEND_SOLIDITY_CONVERT_H_

#include <memory>
#include <iostream>
#include <stack>
#include <map>
#include <util/context.h>
#include <util/namespace.h>
#include <util/std_types.h>
#include <nlohmann/json.hpp>
#include <solidity-frontend/solidity_grammar.h>
#include <solidity-frontend/pattern_check.h>

class solidity_convertert
{
public:
  solidity_convertert(
    contextt &_context,
    nlohmann::json &_ast_json,
    const std::string &_sol_func,
    const messaget &msg);
  virtual ~solidity_convertert() = default;

  bool convert();

protected:
  contextt &context;
  namespacet ns;
  nlohmann::json &ast_json;    // json for Solidity AST. Use vector for multiple contracts
  const std::string &sol_func; // Solidity function to be verified
  const messaget &msg;
  std::string absolute_path;
  int global_scope_id; // scope id of "ContractDefinition"

  unsigned int current_scope_var_num;
  const nlohmann::json *current_functionDecl;
  const nlohmann::json *current_forStmt;
  // Use current level of BinOp type as the "anchor" type for numerical literal conversion:
  // In order to remove the unnecessary implicit IntegralCast. We need type of current level of BinaryOperator.
  // All numeric literals will be implicitly converted to this type. Pop it when finishing the current level of BinaryOperator.
  // TODO: find a better way to deal with implicit type casting if it's not able to cope with compelx rules
  std::stack<const nlohmann::json *> current_BinOp_type;
  std::string current_functionName;

  bool convert_ast_nodes(const nlohmann::json &contract_def);

  // conversion functions
  // get decl in rule contract-body-element
  bool get_decl(const nlohmann::json &ast_node, exprt &new_expr);
  // get decl in rule variable-declaration-statement, e.g. function local declaration
  bool get_var_decl_stmt(const nlohmann::json &ast_node, exprt &new_expr);
  bool get_var_decl(const nlohmann::json &ast_node, exprt &new_expr);
  bool get_function_definition(const nlohmann::json &ast_node, exprt &new_expr);
  bool get_function_params(const nlohmann::json &pd, exprt &param);
  bool get_block(const nlohmann::json &expr, exprt &new_expr); // For Solidity's mutually inclusive: rule block and rule statement
  bool get_statement(const nlohmann::json &block, exprt &new_expr);
  bool get_expr(const nlohmann::json &expr, exprt &new_expr);
  bool get_binary_operator_expr(const nlohmann::json &expr, exprt &new_expr);
  bool get_unary_operator_expr(const nlohmann::json &expr, exprt &new_expr);
  bool get_cast_expr(const nlohmann::json &cast_expr, exprt &new_expr);
  bool get_var_decl_ref(const nlohmann::json &decl, exprt &new_expr);
  bool get_func_decl_ref(const nlohmann::json &decl, exprt &new_expr);
  bool get_decl_ref_builtin(const nlohmann::json &decl, exprt &new_expr);
  bool get_type_description(const nlohmann::json &type_name, typet &new_type);
  bool get_func_decl_ref_type(const nlohmann::json &decl, typet &new_type);
  bool get_array_to_pointer_type(const nlohmann::json &decl, typet &new_type);
  bool get_elementary_type_name(const nlohmann::json &type_name, typet &new_type);
  bool get_parameter_list(const nlohmann::json &type_name, typet &new_type);
  void get_state_var_decl_name(const nlohmann::json &ast_node, std::string &name, std::string &id);
  void get_var_decl_name(const nlohmann::json &ast_node, std::string &name, std::string &id);
  void get_function_definition_name(const nlohmann::json &ast_node, std::string &name, std::string &id);
  void get_location_from_decl(const nlohmann::json &ast_node, locationt &location);
  void get_start_location_from_stmt(const nlohmann::json &stmt_node, locationt &location);
  symbolt *move_symbol_to_context(symbolt &symbol);

  // auxiliary functions
  std::string get_modulename_from_path(std::string path);
  std::string get_filename_from_path(std::string path);
  const nlohmann::json& find_decl_ref(int ref_decl_id);
  void convert_expression_to_code(exprt &expr);
  bool check_intrinsic_function(const nlohmann::json &ast_node);
  nlohmann::json make_implicit_cast_expr(const nlohmann::json& sub_expr, std::string cast_type);
  nlohmann::json make_pointee_type(const nlohmann::json& sub_expr);
  nlohmann::json make_callexpr_return_type(const nlohmann::json& type_descrpt);
  nlohmann::json make_array_elementary_type(const nlohmann::json& type_descrpt);
  nlohmann::json make_array_to_pointer_type(const nlohmann::json& type_descrpt);
  std::string get_array_size(const nlohmann::json& type_descrpt);

  void get_default_symbol(
    symbolt &symbol,
    std::string module_name,
    typet type,
    std::string name,
    std::string id,
    locationt location);

  // literal conversion functions
  bool convert_integer_literal(
    const nlohmann::json &integer_literal,
    std::string the_value, exprt &dest);

  // debug functions
  void print_json(const nlohmann::json &json_in);
  void print_json_element(const nlohmann::json &json_in, const unsigned index,
    const std::string &key, const std::string& json_name);
  void print_json_array_element(const nlohmann::json &json_in,
      const std::string& node_type, const unsigned index);
  void print_json_stmt_element(const nlohmann::json &json_in,
      const std::string& node_type, const unsigned index);
};

#endif /* SOLIDITY_FRONTEND_SOLIDITY_CONVERT_H_ */
