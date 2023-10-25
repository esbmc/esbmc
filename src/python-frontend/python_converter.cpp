#include <python-frontend/python_converter.h>
#include <python_frontend_types.h>
#include <util/std_code.h>
#include <util/c_types.h>
#include <util/arith_tools.h>
#include <util/expr_util.h>
#include <util/message.h>

#include <fstream>
#include <unordered_map>

static const std::unordered_map<std::string, std::string> operator_map = {
  {"Add", "+"},
  {"Sub", "-"},
  {"Mult", "*"},
  {"Div", "/"},
  {"Mod", "mod"},
  {"BitOr", "bitor"},
  {"BitAnd", "bitand"},
  {"BitXor", "bitxor"},
  {"Invert", "bitnot"},
  {"LShift", "shl"},
  {"RShift", "ashr"},
  {"USub", "unary-"},
  {"Eq", "="},
  {"Lt", "<"},
  {"LtE", "<="},
  {"Gt", ">"},
  {"GtE", ">="},
};

static const std::unordered_map<std::string, StatementType> statement_map = {
  {"AnnAssign", StatementType::VARIABLE_ASSIGN},
  {"Assign", StatementType::VARIABLE_ASSIGN},
  {"FunctionDef", StatementType::FUNC_DEFINITION},
  {"If", StatementType::IF_STATEMENT},
  {"AugAssign", StatementType::COMPOUND_ASSIGN},
  {"While", StatementType::WHILE_STATEMENT},
  {"Expr", StatementType::EXPR},
  {"Return", StatementType::RETURN},
};

static StatementType get_statement_type(const nlohmann::json &element)
{
  auto it = statement_map.find(element["_type"]);
  if(it != statement_map.end())
  {
    return it->second;
  }
  return StatementType::UNKNOWN;
}

// Convert Python/AST to irep2 operations
static std::string get_op(const std::string &op)
{
  auto it = operator_map.find(op);
  if(it != operator_map.end())
  {
    return it->second;
  }
  return std::string();
}

// Convert Python/AST types to irep2 types
static typet get_typet(const std::string &ast_type)
{
  if(ast_type == "float")
    return float_type();
  if(ast_type == "int")
    return int_type();
  if(ast_type == "bool")
    return bool_type();
  return empty_typet();
}

static symbolt create_symbol(
  const std::string &module,
  const std::string &name,
  const std::string &id,
  const locationt &location,
  const typet &type)
{
  symbolt symbol;
  symbol.mode = "Python";
  symbol.module = module;
  symbol.location = location;
  symbol.type = type;
  symbol.name = name;
  symbol.id = id;
  return symbol;
}

static ExpressionType get_expression_type(const nlohmann::json &element)
{
  auto type = element["_type"];
  if(type == "UnaryOp")
  {
    return ExpressionType::UNARY_OPERATION;
  }
  if(type == "BinOp" || type == "Compare")
  {
    return ExpressionType::BINARY_OPERATION;
  }
  if(type == "Constant")
  {
    return ExpressionType::LITERAL;
  }
  if(type == "Name")
  {
    return ExpressionType::VARIABLE_REF;
  }
  if(type == "Call")
  {
    return ExpressionType::FUNC_CALL;
  }
  return ExpressionType::UNKNOWN;
}

exprt python_converter::get_binary_operator_expr(const nlohmann::json &element)
{
  std::string op;

  if(element.contains("op"))
    op = element["op"]["_type"].get<std::string>();
  else if(element.contains("ops"))
    op = element["ops"][0]["_type"].get<std::string>();

  assert(!op.empty());

  exprt bin_expr(get_op(op), current_element_type);

  exprt lhs;
  if(element.contains("left"))
    lhs = get_expr(element["left"]);
  else if(element.contains("target"))
    lhs = get_expr(element["target"]);

  exprt rhs;
  if(element.contains("right"))
    rhs = get_expr(element["right"]);
  else if(element.contains("comparators"))
    rhs = get_expr(element["comparators"][0]);
  else if(element.contains("value"))
    rhs = get_expr(element["value"]);

  bin_expr.copy_to_operands(lhs, rhs);
  return bin_expr;
}

exprt python_converter::get_unary_operator_expr(const nlohmann::json &element)
{
  exprt unary_expr(
    get_op(element["op"]["_type"].get<std::string>()), current_element_type);

  // get subexpr
  exprt unary_sub = get_expr(element["operand"]);
  unary_expr.operands().push_back(unary_sub);

  return unary_expr;
}

const nlohmann::json python_converter::find_var_decl(const std::string &id)
{
  for(auto &element : ast_json["body"])
  {
    if((element["_type"] == "AnnAssign") && (element["target"]["id"] == id))
      return element;
  }
  return nlohmann::json();
}

locationt
python_converter::get_location_from_decl(const nlohmann::json &ast_node)
{
  locationt location;
  location.set_line(ast_node["lineno"].get<int>());
  location.set_file(python_filename.c_str());
  return location;
}

exprt python_converter::get_expr(const nlohmann::json &element)
{
  exprt expr;
  ExpressionType type = get_expression_type(element);

  switch(type)
  {
  case ExpressionType::UNARY_OPERATION:
  {
    expr = get_unary_operator_expr(element);
    break;
  }
  case ExpressionType::BINARY_OPERATION:
  {
    expr = get_binary_operator_expr(element);
    break;
  }
  case ExpressionType::LITERAL:
  {
    auto value = element["value"];
    if(element["value"].is_number_integer())
    {
      expr = from_integer(value.get<int>(), int_type());
    }
    else if(element["value"].is_boolean())
    {
      expr = gen_boolean(value.get<bool>());
    }
    break;
  }
  case ExpressionType::VARIABLE_REF:
  {
    // Find the variable declaration in the AST JSON
    std::string var_name = element["id"].get<std::string>();
    std::string symbol_id = "py:" + python_filename + "@" + "F" + "@" +
                            current_func_name + "@" + var_name;
    symbolt *symbol = context.find_symbol(symbol_id);
    if(symbol != nullptr)
    {
      expr = symbol_expr(*symbol);
    }
    else
    {
      log_error("Symbol not found\n");
      abort();
    }

    break;
  }
  case ExpressionType::FUNC_CALL:
  {
    if(element.contains("func") && element["_type"] == "Call")
    {
      std::string func_name = element["func"]["id"];
      std::string symbol_id = "py:" + python_filename + "@F@" + func_name;
      symbolt *func_symbol = context.find_symbol(symbol_id.c_str());

      assert(func_symbol);

      code_function_callt call;
      call.location() = func_symbol->location;
      call.function() = symbol_expr(*func_symbol);

      for(const auto &arg_node : element["args"])
      {
        call.arguments().push_back(get_expr(arg_node));
      }

      return call;
    }
    else
    {
      log_error("Invalid function call");
      abort();
    }
    break;
  }
  default:
  {
    log_error("Unimplemented type in rule expression");
    abort();
  }
  }

  return expr;
}

void python_converter::get_var_assign(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  if(ast_node.contains("annotation"))
  {
    // Get type from current annotation node
    current_element_type =
      get_typet(ast_node["annotation"]["id"].get<std::string>());
  }
  else
  {
    // Get type from declaration node
    std::string var_name = ast_node["targets"][0]["id"].get<std::string>();
    nlohmann::json ref = find_var_decl(var_name);
    assert(!ref.empty());
    current_element_type =
      get_typet(ref["annotation"]["id"].get<std::string>());
  }

  exprt lhs;

  nlohmann::json value = ast_node["value"];
  exprt rhs = get_expr(value);

  locationt location_begin;

  if(ast_node["_type"] == "AnnAssign")
  {
    // Id and name
    std::string name, id;
    auto target = ast_node["target"];
    if(!target.is_null() && target["_type"] == "Name")
    {
      name = target["id"];
      id = "py:" + python_filename + "@F@" + current_func_name + "@" + name;
    }

    // Location
    location_begin = get_location_from_decl(ast_node["target"]);

    // Debug module name
    std::string module_name = location_begin.get_file().as_string();

    // Create/init symbol
    symbolt symbol = create_symbol(
      module_name, name, id, location_begin, current_element_type);
    symbol.lvalue = true;
    symbol.static_lifetime = false;
    symbol.file_local = true;
    symbol.is_extern = false;
    symbol.value = rhs;

    lhs = symbol_expr(symbol);

    context.add(symbol);
  }
  else if(ast_node["_type"] == "Assign")
  {
    std::string name = ast_node["targets"][0]["id"].get<std::string>();
    std::string symbol_id =
      "py:" + python_filename + "@F@" + current_func_name + "@" + name;
    symbolt *symbol = context.find_symbol(symbol_id);
    assert(symbol);
    lhs = symbol_expr(*symbol);
  }

  /* If the right-hand side (rhs) of the assignment is a function call, such as: x : int = func()
   * we need to adjust the left-hand side (lhs) of the function call to refer to the lhs of the current assignment.
   */
  if(rhs.is_code() && rhs.get("statement") == "function_call")
  {
    // op0() references the left-hand side (lhs) of the function call
    rhs.op0() = lhs;
    target_block.copy_to_operands(rhs);
    return;
  }

  code_assignt code_assign(lhs, rhs);
  code_assign.location() = location_begin;
  target_block.copy_to_operands(code_assign);
}

void python_converter::get_compound_assign(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  // Get type from declaration node
  std::string var_name = ast_node["target"]["id"].get<std::string>();
  nlohmann::json ref = find_var_decl(var_name);
  assert(!ref.empty());
  current_element_type = get_typet(ref["annotation"]["id"].get<std::string>());

  exprt lhs = get_expr(ast_node["target"]);
  exprt rhs = get_binary_operator_expr(ast_node);

  code_assignt code_assign(lhs, rhs);
  code_assign.location() = get_location_from_decl(ast_node);

  target_block.copy_to_operands(code_assign);
}

static codet convert_expression_to_code(exprt &expr)
{
  if(expr.is_code())
    return static_cast<codet &>(expr);

  codet code("expression");
  code.location() = expr.location();
  code.move_to_operands(expr);

  return code;
}

void python_converter::get_conditional_stms(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  current_element_type = bool_type();

  // Extract condition from AST
  exprt cond = get_expr(ast_node["test"]);
  cond.location() = get_location_from_decl(ast_node["test"]);

  // Extract 'then' block from AST
  exprt then = get_block(ast_node["body"]);
  locationt location = get_location_from_decl(ast_node);
  then.location() = location;

  // Create if or while code
  codet code;
  auto type = ast_node["_type"];
  if(type == "If")
    code.set_statement("ifthenelse");
  else if(type == "While")
    code.set_statement("while");

  // Append "then" block
  code.copy_to_operands(cond, convert_expression_to_code(then));

  if(ast_node.contains("orelse") && !ast_node["orelse"].empty())
  {
    // Append 'else' block to the statement
    exprt else_expr = get_block(ast_node["orelse"]);
    code.copy_to_operands(convert_expression_to_code(else_expr));
  }

  target_block.copy_to_operands(code);
}

void python_converter::get_function_definition(
  const nlohmann::json &function_node)
{
  // Function return type
  code_typet type;
  nlohmann::json return_node = function_node["returns"];
  if(return_node.contains("id"))
  {
    type.return_type() = get_typet(return_node["id"].get<std::string>());
  }
  else if(return_node.contains("value") && return_node["value"].is_null())
  {
    type.return_type() = empty_typet();
  }
  else
  {
    log_error("Return type undefined\n");
    abort();
  }

  // Function location
  locationt location = get_location_from_decl(function_node);

  // Symbol identification
  current_func_name = function_node["name"].get<std::string>();
  std::string id = "py:" + python_filename + "@F@" + current_func_name;
  std::string module_name =
    python_filename.substr(0, python_filename.find_last_of("."));

  // Iterate over function arguments
  for(const nlohmann::json &element : function_node["args"]["args"])
  {
    // Argument type
    typet arg_type = get_typet(element["annotation"]["id"].get<std::string>());
    code_typet::argumentt arg;
    arg.type() = arg_type;

    // Argument name
    std::string arg_name = element["arg"].get<std::string>();
    arg.cmt_base_name(arg_name);

    // Argument id
    std::string arg_id = "py:" + python_filename + "@" + "F" + "@" +
                         current_func_name + "@" + arg_name;
    arg.cmt_identifier(arg_id);

    // Location
    locationt location = get_location_from_decl(element);
    arg.location() = location;

    // Push arg
    type.arguments().push_back(arg);

    // Create and add symbol to context
    symbolt param_symbol = create_symbol(
      location.get_file().as_string(), arg_name, arg_id, location, arg_type);
    param_symbol.lvalue = true;
    param_symbol.is_parameter = true;
    param_symbol.file_local = true;
    param_symbol.static_lifetime = false;
    param_symbol.is_extern = false;
    context.add(param_symbol);
  }

  // Function body
  exprt function_body = get_block(function_node["body"]);

  // Create symbol
  symbolt symbol =
    create_symbol(module_name, current_func_name, id, location, type);
  symbol.lvalue = true;
  symbol.is_extern = false;
  symbol.file_local = false;
  symbol.value = function_body;

  context.add(symbol);
}

void python_converter::get_return_statements(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  code_returnt return_code;
  return_code.return_value() = get_expr(ast_node["value"]);
  return_code.location() = get_location_from_decl(ast_node);
  target_block.copy_to_operands(return_code);
}

exprt python_converter::get_block(const nlohmann::json &ast_block)
{
  code_blockt block;

  // Iterate through the statements of the block
  for(auto &element : ast_block)
  {
    StatementType type = get_statement_type(element);

    switch(type)
    {
    case StatementType::VARIABLE_ASSIGN:
    {
      // Add an assignment to the block
      get_var_assign(element, block);
      break;
    }
    case StatementType::IF_STATEMENT:
    case StatementType::WHILE_STATEMENT:
    {
      get_conditional_stms(element, block);
      break;
    }
    case StatementType::COMPOUND_ASSIGN:
    {
      get_compound_assign(element, block);
      break;
    }
    case StatementType::FUNC_DEFINITION:
    {
      get_function_definition(element);
      break;
    }
    case StatementType::RETURN:
    {
      get_return_statements(element, block);
      break;
    }
    case StatementType::EXPR:
    {
      // Function calls are handled here
      exprt expr = get_expr(element["value"]);
      block.move_to_operands(expr);
      break;
    }
    case StatementType::UNKNOWN:
    default:
      log_error(
        "Unsupported statement: {}", element["_type"].get<std::string>());
      abort();
    }
  }

  return block;
}

bool python_converter::convert()
{
  python_filename = ast_json["filename"].get<std::string>();

  // Read all statements
  exprt block_expr = get_block(ast_json["body"]);

  // Get main function code
  codet main_code = convert_expression_to_code(block_expr);

  // Create and populate "main" symbol
  symbolt main_symbol;

  code_typet main_type;
  main_type.return_type() = empty_typet();

  main_symbol.id = "__ESBMC_main";
  main_symbol.name = "__ESBMC_main";
  main_symbol.type.swap(main_type);
  main_symbol.value.swap(main_code);
  main_symbol.lvalue = true;
  main_symbol.is_extern = false;
  main_symbol.file_local = false;

  if(context.move(main_symbol))
  {
    log_error("main already defined by another language module");
    return true;
  }

  return false;
}
