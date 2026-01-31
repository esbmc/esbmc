#include <python-frontend/python_lambda.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/type_handler.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/std_code.h>

// Initialize static counter
int python_lambda::lambda_counter_ = 0;

python_lambda::python_lambda(
  python_converter &converter,
  contextt &context,
  type_handler &type_handler)
  : converter_(converter), context_(context), type_handler_(type_handler)
{
}

std::string python_lambda::generate_unique_lambda_name()
{
  return "lam" + std::to_string(++lambda_counter_);
}

bool python_lambda::is_lambda_assignment(const nlohmann::json &ast_node) const
{
  return ast_node.contains("value") && ast_node["value"].contains("_type") &&
         ast_node["value"]["_type"] == "Lambda";
}

void python_lambda::handle_lambda_assignment(
  symbolt *lhs_symbol,
  exprt &lhs,
  exprt &rhs)
{
  if (!lhs_symbol || !rhs.is_symbol())
    return;

  const symbolt *lambda_func_symbol = context_.find_symbol(rhs.identifier());

  if (!lambda_func_symbol || !lambda_func_symbol->type.is_code())
  {
    throw std::runtime_error("Lambda function symbol does not have code type");
  }

  // Create function pointer type
  typet func_ptr_type = gen_pointer_type(lambda_func_symbol->type);
  lhs_symbol->type = func_ptr_type;
  lhs.type() = func_ptr_type;

  // Convert lambda symbol to address
  rhs = address_of_exprt(rhs);
}

static bool is_param_used_as_string(
  const nlohmann::json &body_node,
  const std::string &param_name)
{
  if (!body_node.contains("_type"))
    return false;

  std::string body_type = body_node["_type"].get<std::string>();

  // Check if param is in string concatenation: param + "string" or "string" + param
  if (
    body_type == "BinOp" && body_node.contains("op") &&
    body_node["op"].contains("_type") && body_node["op"]["_type"] == "Add")
  {
    auto is_string_literal = [](const nlohmann::json &node) {
      return node.contains("_type") && node["_type"] == "Constant" &&
             node.contains("value") && node["value"].is_string();
    };

    auto is_param = [&](const nlohmann::json &node) {
      return node.contains("_type") && node["_type"] == "Name" &&
             node.contains("id") && node["id"] == param_name;
    };

    if (body_node.contains("left") && body_node.contains("right"))
    {
      if (
        (is_param(body_node["left"]) &&
         is_string_literal(body_node["right"])) ||
        (is_string_literal(body_node["left"]) && is_param(body_node["right"])))
        return true;
    }
  }

  // Check IfExp branches recursively
  if (body_type == "IfExp")
  {
    if (
      body_node.contains("body") &&
      is_param_used_as_string(body_node["body"], param_name))
      return true;
    if (
      body_node.contains("orelse") &&
      is_param_used_as_string(body_node["orelse"], param_name))
      return true;
  }

  return false;
}

typet python_lambda::infer_lambda_return_type(
  [[maybe_unused]] const nlohmann::json &body_node)
{
  // Check if body is a string operation
  if (body_node.contains("_type"))
  {
    std::string body_type = body_node["_type"].get<std::string>();

    // String concatenation (BinOp with Add and string constant)
    if (
      body_type == "BinOp" && body_node.contains("op") &&
      body_node["op"].contains("_type") && body_node["op"]["_type"] == "Add")
    {
      // Check if the right operand is a string constant
      if (
        body_node.contains("right") && body_node["right"].contains("_type") &&
        body_node["right"]["_type"] == "Constant" &&
        body_node["right"].contains("value") &&
        body_node["right"]["value"].is_string())
      {
        return gen_pointer_type(signed_char_type());
      }
    }

    // Handle IfExp (ternary expressions)
    if (body_type == "IfExp")
    {
      // Recursively check if any branch contains a string literal
      std::function<bool(const nlohmann::json &)> has_string_literal =
        [&](const nlohmann::json &node) -> bool {
        if (!node.contains("_type"))
          return false;

        std::string node_type = node["_type"].get<std::string>();

        // Direct string constant
        if (
          node_type == "Constant" && node.contains("value") &&
          node["value"].is_string())
          return true;

        // Nested IfExp - check recursively
        if (node_type == "IfExp")
        {
          return (node.contains("body") && has_string_literal(node["body"])) ||
                 (node.contains("orelse") &&
                  has_string_literal(node["orelse"]));
        }

        return false;
      };

      // If any branch has a string literal, return string pointer type
      if (
        (body_node.contains("body") && has_string_literal(body_node["body"])) ||
        (body_node.contains("orelse") &&
         has_string_literal(body_node["orelse"])))
      {
        return gen_pointer_type(signed_char_type());
      }
    }
  }

  // Default to double for numeric expressions
  return double_type();
}

symbolt python_lambda::create_symbol(
  const std::string &id,
  const std::string &name,
  const typet &type,
  const locationt &location,
  const std::string &module_name,
  bool file_local,
  bool is_parameter)
{
  symbolt symbol;
  symbol.id = id;
  symbol.name = name;
  symbol.type = type;
  symbol.location = location;
  symbol.mode = "Python";
  symbol.module = module_name;
  symbol.lvalue = true;
  symbol.is_parameter = is_parameter;
  symbol.file_local = file_local;
  symbol.static_lifetime = false;
  symbol.is_extern = false;

  return symbol;
}

void python_lambda::process_lambda_parameters(
  const nlohmann::json &args_node,
  code_typet &lambda_type,
  [[maybe_unused]] const std::string &lambda_id,
  const std::string &param_scope_id,
  const locationt &location,
  const nlohmann::json &body_node)
{
  if (!args_node.contains("args") || !args_node["args"].is_array())
    return;

  std::string module_name = location.get_file().as_string();

  for (const auto &arg : args_node["args"])
  {
    std::string arg_name = arg["arg"].get<std::string>();

    // Determine parameter type from annotation or infer from usage
    typet param_type = double_type();

    // Check for type annotation
    if (arg.contains("annotation") && !arg["annotation"].is_null())
    {
      std::string annotation = arg["annotation"].get<std::string>();
      if (annotation == "str")
        param_type = gen_pointer_type(signed_char_type());
    }
    // Infer from usage in lambda body
    else if (is_param_used_as_string(body_node, arg_name))
    {
      param_type = gen_pointer_type(signed_char_type());
    }

    // Create function argument
    code_typet::argumentt argument;
    argument.type() = param_type;
    argument.cmt_base_name(arg_name);

    // Use param_scope_id for parameter symbols to enable closure access
    std::string param_id = param_scope_id + "@" + arg_name;
    argument.cmt_identifier(param_id);
    argument.location() = location;
    lambda_type.arguments().push_back(argument);

    // Create parameter symbol
    symbolt param_symbol = create_symbol(
      param_id,
      arg_name,
      param_type,
      location,
      module_name,
      true, // file_local
      true  // is_parameter
    );

    context_.add(param_symbol);
  }
}

exprt python_lambda::process_lambda_body(
  const nlohmann::json &body_node,
  const locationt &location)
{
  // Get the body expression through the converter
  exprt body_expr = converter_.get_expr(body_node);

  // Create return statement
  code_returnt return_stmt;
  return_stmt.return_value() = body_expr;
  return_stmt.location() = location;

  // Wrap in a block
  code_blockt lambda_block;
  lambda_block.copy_to_operands(return_stmt);

  return lambda_block;
}

exprt python_lambda::get_lambda_expr(const nlohmann::json &element)
{
  // Generate unique lambda name
  std::string lambda_name = generate_unique_lambda_name();

  locationt location = converter_.get_location_from_decl(element);
  std::string module_name = location.get_file().as_string();

  // Save the original function context
  std::string old_func = converter_.get_current_func_name();

  // Determine if we're in a lambda (function name starts with "lam")
  bool in_lambda = (old_func.find("lam") == 0);

  // Determine the scope for parameters: use first lambda's scope for all nested lambdas
  std::string param_scope;
  if (in_lambda)
  {
    // Nested lambda: use parent lambda's scope for all parameters
    param_scope = old_func;
  }
  else
  {
    // Top-level lambda: use this lambda's name as the scope
    param_scope = lambda_name;
    converter_.set_current_func_name(lambda_name);
  }

  // Create function type with inferred return type
  code_typet lambda_type;
  typet return_type = double_type();

  if (element.contains("body"))
  {
    return_type = infer_lambda_return_type(element["body"]);
    converter_.set_current_element_type(return_type);
  }

  lambda_type.return_type() = return_type;

  // Lambda function symbol is always top-level: py:module@F@lambda_name
  std::string lambda_id = "py:" + module_name + "@F@" + lambda_name;

  // Parameters are created in param_scope (shared for nested lambdas)
  std::string param_scope_id = "py:" + module_name + "@F@" + param_scope;

  // Process lambda parameters: pass body for type inference
  if (element.contains("args"))
    process_lambda_parameters(
      element["args"],
      lambda_type,
      lambda_id,
      param_scope_id,
      location,
      element.contains("body") ? element["body"] : nlohmann::json());

  // Create lambda function symbol
  symbolt lambda_symbol = create_symbol(
    lambda_id,
    lambda_name,
    lambda_type,
    location,
    module_name,
    false, // file_local
    false  // is_parameter
  );

  symbolt *added_symbol = context_.move_symbol_to_context(lambda_symbol);
  assert(added_symbol);

  // Process lambda body
  if (element.contains("body"))
  {
    exprt lambda_body = process_lambda_body(element["body"], location);
    added_symbol->value = lambda_body;
  }

  // Restore context only if we changed it (top-level lambda only)
  if (!in_lambda)
    converter_.set_current_func_name(old_func);

  return symbol_expr(*added_symbol);
}