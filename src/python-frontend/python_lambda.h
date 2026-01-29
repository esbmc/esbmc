#ifndef ESBMC_PYTHON_LAMBDA_H
#define ESBMC_PYTHON_LAMBDA_H

#include <python-frontend/python_converter.h>
#include <util/expr.h>
#include <util/std_code.h>
#include <nlohmann/json.hpp>

class python_converter;
class type_handler;

class python_lambda
{
public:
  python_lambda(
    python_converter &converter,
    contextt &context,
    type_handler &type_handler);

  // Main method to create lambda expression from AST
  exprt get_lambda_expr(const nlohmann::json &element);

  // Check if a variable assignment involves a lambda
  bool is_lambda_assignment(const nlohmann::json &ast_node) const;

  // Handle lambda assignment type adjustments
  void handle_lambda_assignment(symbolt *lhs_symbol, exprt &lhs, exprt &rhs);

private:
  python_converter &converter_;
  contextt &context_;
  type_handler &type_handler_;

  // Counter for generating unique lambda names
  static int lambda_counter_;

  // Helper methods
  void process_lambda_parameters(
    const nlohmann::json &args_node,
    code_typet &lambda_type,
    const std::string &lambda_id,
    const locationt &location);

  exprt process_lambda_body(
    const nlohmann::json &body_node,
    const locationt &location);

  typet infer_lambda_return_type(const nlohmann::json &body_node);

  std::string generate_unique_lambda_name();

  symbolt create_symbol(
    const std::string &id,
    const std::string &name,
    const typet &type,
    const locationt &location,
    const std::string &module_name,
    bool file_local,
    bool is_parameter = false);
};

#endif // ESBMC_PYTHON_LAMBDA_H
