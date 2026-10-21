#ifndef ESBMC_PYTHON_LAMBDA_H
#define ESBMC_PYTHON_LAMBDA_H

#include <python-frontend/python_converter.h>
#include <util/irep/expr.h>
#include <util/irep/std_code.h>
#include <nlohmann/json.hpp>
#include <optional>

class python_converter;
class type_handler;

class python_lambda
{
public:
  python_lambda(
    python_converter &converter,
    contextt &context,
    type_handler &type_handler);

  // Main method to create lambda expression from AST. Parameters take
  // `argument_types` when given, those of the lambda's one call.
  exprt get_lambda_expr(
    const nlohmann::json &element,
    const std::vector<std::optional<typet>> &argument_types);

  // Check if a variable assignment involves a lambda
  bool is_lambda_assignment(const nlohmann::json &ast_node) const;

  // Handle lambda assignment type adjustments
  void handle_lambda_assignment(symbolt *lhs_symbol, exprt &lhs, exprt &rhs);

  // The parameter types a copy of the lambda needs for @p call, which passes
  // an int where @p signature, taken from calls that disagree, has a float
  // (#8941). Names resolve in @p scope. Empty when @p signature serves it.
  std::vector<std::optional<typet>> narrowed_parameter_types(
    const nlohmann::json &call,
    const nlohmann::json &scope,
    const code_typet &signature) const;

  // Infer return type from a lambda body node
  typet infer_lambda_return_type(const nlohmann::json &body_node);

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
    const std::string &param_scope_id,
    const locationt &location,
    const nlohmann::json &body_node,
    const std::vector<std::optional<typet>> &call_site_types);

  // Per-parameter type taken from `argument_types` when given, else from a
  // call through the name this lambda is bound to; an entry is empty when no
  // such call pins that argument down (#7328).
  std::vector<std::optional<typet>> call_site_argument_types(
    const nlohmann::json &element,
    const std::vector<std::optional<typet>> &argument_types) const;

  exprt process_lambda_body(
    const nlohmann::json &body_node,
    const locationt &location);

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

/// The one name target a plain `x = v` or `x: T = v` statement binds, or null.
const nlohmann::json *binding_target(const nlohmann::json &stmt);

/// Every construct under @p node that binds @p name, in any nested scope too.
size_t count_bindings(const nlohmann::json &node, const std::string &name);

#endif // ESBMC_PYTHON_LAMBDA_H