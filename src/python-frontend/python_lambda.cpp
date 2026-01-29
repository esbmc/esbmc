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
  : converter_(converter),
    context_(context),
    type_handler_(type_handler)
{
}

std::string python_lambda::generate_unique_lambda_name()
{
  return "lam" + std::to_string(++lambda_counter_);
}

bool python_lambda::is_lambda_assignment(const nlohmann::json &ast_node) const
{
  return ast_node.contains("value") && 
         ast_node["value"].contains("_type") &&
         ast_node["value"]["_type"] == "Lambda";
}

void python_lambda::handle_lambda_assignment(
  symbolt *lhs_symbol,
  exprt &lhs,
  exprt &rhs)
{
  if (!lhs_symbol || !rhs.is_symbol())
    return;

  const symbolt *lambda_func_symbol =
    context_.find_symbol(rhs.identifier());

  if (!lambda_func_symbol || !lambda_func_symbol->type.is_code())
  {
    throw std::runtime_error(
      "Lambda function symbol does not have code type");
  }

  // Create function pointer type
  typet func_ptr_type = gen_pointer_type(lambda_func_symbol->type);
  lhs_symbol->type = func_ptr_type;
  lhs.type() = func_ptr_type;

  // Convert lambda symbol to address
  rhs = address_of_exprt(rhs);
}

typet python_lambda::infer_lambda_return_type(
  const nlohmann::json &body_node)
{
  // TODO: Implement more sophisticated type inference
  // For now, default to double for numeric expressions
  return double_type();
}

void python_lambda::process_lambda_parameters(
  const nlohmann::json &args_node,
  code_typet &lambda_type,
  const std::string &lambda_id,
  const locationt &location)
{
  if (!args_node.contains("args") || !args_node["args"].is_array())
    return;

  std::string module_name = location.get_file().as_string();

  for (const auto &arg : args_node["args"])
  {
    std::string arg_name = arg["arg"].get<std::string>();

    // Determine parameter type
    // TODO: Try to infer from usage or annotation
    typet param_type = double_type();

    // Create function argument
    code_typet::argumentt argument;
    argument.type() = param_type;
    argument.cmt_base_name(arg_name);

    std::string param_id = lambda_id + "@" + arg_name;
    argument.cmt_identifier(param_id);
    argument.location() = location;
    lambda_type.arguments().push_back(argument);

    // Create parameter symbol
    symbolt param_symbol;
    param_symbol.id = param_id;
    param_symbol.name = arg_name;
    param_symbol.type = param_type;
    param_symbol.location = location;
    param_symbol.mode = "Python";
    param_symbol.module = module_name;
    param_symbol.lvalue = true;
    param_symbol.is_parameter = true;
    param_symbol.file_local = true;
    param_symbol.static_lifetime = false;
    param_symbol.is_extern = false;

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

  // Save and set lambda context
  std::string old_func_name = converter_.get_current_func_name();
  converter_.set_current_func_name(lambda_name);

  // Create function type with inferred return type
  code_typet lambda_type;
  typet return_type = double_type();

  if (element.contains("body"))
  {
    return_type = infer_lambda_return_type(element["body"]);
    converter_.set_current_element_type(return_type);
  }

  lambda_type.return_type() = return_type;

  // Build lambda identifier
  std::string lambda_id = "py:" + module_name + "@F@" + lambda_name;

  // Process lambda parameters
  if (element.contains("args"))
    process_lambda_parameters(element["args"], lambda_type, lambda_id, location);

  // Create lambda function symbol
  symbolt lambda_symbol;
  lambda_symbol.id = lambda_id;
  lambda_symbol.name = lambda_name;
  lambda_symbol.type = lambda_type;
  lambda_symbol.location = location;
  lambda_symbol.mode = "Python";
  lambda_symbol.module = module_name;
  lambda_symbol.lvalue = true;
  lambda_symbol.is_extern = false;
  lambda_symbol.file_local = false;
  lambda_symbol.static_lifetime = false;

  symbolt *added_symbol = context_.move_symbol_to_context(lambda_symbol);
  assert(added_symbol);

  // Process lambda body
  if (element.contains("body"))
  {
    exprt lambda_body = process_lambda_body(element["body"], location);
    added_symbol->value = lambda_body;
  }

  // Restore context
  converter_.set_current_func_name(old_func_name);

  return symbol_expr(*added_symbol);
}
