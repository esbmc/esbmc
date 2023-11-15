#include <python-frontend/python_converter.h>
#include <python_frontend_types.h>
#include <util/std_code.h>
#include <util/c_types.h>
#include <util/arith_tools.h>
#include <util/expr_util.h>
#include <util/message.h>

#include <fstream>
#include <regex>
#include <unordered_map>

static const std::unordered_map<std::string, std::string> operator_map = {
  {"Add", "+"},          {"Sub", "-"},         {"Mult", "*"},
  {"Div", "/"},          {"Mod", "mod"},       {"BitOr", "bitor"},
  {"BitAnd", "bitand"},  {"BitXor", "bitxor"}, {"Invert", "bitnot"},
  {"LShift", "shl"},     {"RShift", "ashr"},   {"USub", "unary-"},
  {"Eq", "="},           {"Lt", "<"},          {"LtE", "<="},
  {"NotEq", "notequal"}, {"Gt", ">"},          {"GtE", ">="},
  {"And", "and"},        {"Or", "or"},         {"Not", "not"},
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
  {"Assert", StatementType::ASSERT},
};

static bool is_relational_op(const std::string &op)
{
  return (
    op == "Eq" || op == "Lt" || op == "LtE" || op == "NotEq" || op == "Gt" ||
    op == "GtE" || op == "And" || op == "Or");
}

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
  if(type == "BoolOp")
  {
    return ExpressionType::LOGICAL_OPERATION;
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
  if(type == "IfExp")
  {
    return ExpressionType::IF_EXPR;
  }
  return ExpressionType::UNKNOWN;
}

exprt python_converter::get_logical_operator_expr(const nlohmann::json &element)
{
  std::string op(element["op"]["_type"].get<std::string>());
  exprt logical_expr(get_op(op), bool_type());

  // Iterate over operands of logical operations (and/or)
  for(const auto &operand : element["values"])
  {
    exprt operand_expr = get_expr(operand);
    logical_expr.copy_to_operands(operand_expr);
  }

  return logical_expr;
}

exprt python_converter::get_binary_operator_expr(const nlohmann::json &element)
{
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

  auto to_side_effect_call = [](exprt &expr) {
    side_effect_expr_function_callt side_effect;
    code_function_callt &code = static_cast<code_function_callt &>(expr);
    side_effect.function() = code.function();
    side_effect.location() = code.location();
    side_effect.type() = code.type();
    side_effect.arguments() = code.arguments();
    expr = side_effect;
  };

  // Function calls in expressions like "fib(n-1) + fib(n-2)" need to be converted to side effects
  if(lhs.is_function_call())
    to_side_effect_call(lhs);
  if(rhs.is_function_call())
    to_side_effect_call(rhs);

  std::string op;

  if(element.contains("op"))
    op = element["op"]["_type"].get<std::string>();
  else if(element.contains("ops"))
    op = element["ops"][0]["_type"].get<std::string>();

  assert(!op.empty());
  assert(lhs.type() == rhs.type());

  typet type = (is_relational_op(op)) ? bool_type() : lhs.type();
  exprt bin_expr(get_op(op), type);

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

exprt python_converter::get_function_call(const nlohmann::json &element)
{
  if(element.contains("func") && element["_type"] == "Call")
  {
    std::string func_name = element["func"]["id"];

    // nondet_X() functions restricted to basic types supported in Python
    std::regex pattern(
      R"(nondet_(int|char|bool|float)|__VERIFIER_nondet_(int|char|bool|float))");

    if(std::regex_match(func_name, pattern))
    {
      // Function name pattern: nondet_(type). e.g: nondet_bool(), nondet_int()
      size_t underscore_pos = func_name.rfind("_");
      std::string type = func_name.substr(underscore_pos + 1);
      exprt rhs = exprt("sideeffect", get_typet(type));
      rhs.statement("nondet");
      return rhs;
    }

    locationt location = get_location_from_decl(element);
    std::string symbol_id = "py:" + python_filename + "@F@" + func_name;

    // __ESBMC_assume
    if(func_name == "__ESBMC_assume" || func_name == "__VERIFIER_assume")
    {
      symbol_id = func_name;
      if(context.find_symbol(symbol_id.c_str()) == nullptr)
      {
        // Create/init symbol
        symbolt symbol;
        symbol.mode = "C";
        symbol.module = python_filename;
        symbol.location = location;
        symbol.type = code_typet();
        symbol.name = func_name;
        symbol.id = symbol_id;

        context.add(symbol);
      }
    }

    const symbolt *func_symbol = context.find_symbol(symbol_id.c_str());
    if(func_symbol == nullptr)
    {
      log_error("Undefined function: {}", func_name.c_str());
      abort();
    }

    code_function_callt call;
    call.location() = location;
    call.function() = symbol_expr(*func_symbol);
    const typet &return_type = to_code_type(func_symbol->type).return_type();
    call.type() = return_type;

    for(const auto &arg_node : element["args"])
    {
      call.arguments().push_back(get_expr(arg_node));
    }

    return call;
  }

  log_error("Invalid function call");
  abort();
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
  case ExpressionType::LOGICAL_OPERATION:
  {
    expr = get_logical_operator_expr(element);
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
    // Find the variable declaration in the current function
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
    expr = get_function_call(element);
    break;
  }
  // Ternary operator
  case ExpressionType::IF_EXPR:
  {
    expr = get_conditional_stm(element);
    break;
  }
  default:
  {
    log_error(
      "Unsupported expression type: {}", element["_type"].get<std::string>());
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
  if(rhs.is_function_call())
  {
    // op0() refers to the left-hand side (lhs) of the function call
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

exprt python_converter::get_conditional_stm(const nlohmann::json &ast_node)
{
  // Copy current type
  typet t = current_element_type;
  // Change to boolean before extracting condition
  current_element_type = bool_type();

  // Extract condition from AST
  exprt cond = get_expr(ast_node["test"]);
  cond.location() = get_location_from_decl(ast_node["test"]);

  // Recover type
  current_element_type = t;
  // Extract 'then' block from AST
  exprt then;
  if(ast_node["body"].is_array())
    then = get_block(ast_node["body"]);
  else
    then = get_expr(ast_node["body"]);

  locationt location = get_location_from_decl(ast_node);
  then.location() = location;

  // Extract 'else' block from AST
  exprt else_expr;
  if(ast_node.contains("orelse") && !ast_node["orelse"].empty())
  {
    // Append 'else' block to the statement
    if(ast_node["orelse"].is_array())
    {
      else_expr = get_block(ast_node["orelse"]);
    }
    else
    {
      else_expr = get_expr(ast_node["orelse"]);
    }
  }

  auto type = ast_node["_type"];

  // ternary operator
  if(type == "IfExp")
  {
    exprt if_expr("if", current_element_type);
    if_expr.copy_to_operands(cond, then, else_expr);
    return if_expr;
  }

  // Create if or while code
  codet code;
  if(type == "If")
    code.set_statement("ifthenelse");
  else if(type == "While")
    code.set_statement("while");

  // Append "then" block
  code.copy_to_operands(cond, then);
  if(!else_expr.id_string().empty())
  {
    code.copy_to_operands(else_expr);
  }

  return code;
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

  current_element_type = type.return_type();

  // Copy caller function name
  const std::string caller_func_name = current_func_name;

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

  // Create symbol
  symbolt symbol =
    create_symbol(module_name, current_func_name, id, location, type);
  symbol.lvalue = true;
  symbol.is_extern = false;
  symbol.file_local = false;

  symbolt *added_symbol = context.move_symbol_to_context(symbol);

  // Function body
  exprt function_body = get_block(function_node["body"]);
  added_symbol->value = function_body;

  // Restore caller function name
  current_func_name = caller_func_name;
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
      exprt cond = get_conditional_stm(element);
      block.copy_to_operands(cond);
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
    case StatementType::ASSERT:
    {
      current_element_type = bool_type();
      exprt test = get_expr(element["test"]);
      code_assertt assert_code;
      assert_code.assertion() = test;
      block.move_to_operands(assert_code);
      break;
    }
    case StatementType::EXPR:
    {
      // Function calls are handled here
      exprt empty;
      exprt expr = get_expr(element["value"]);
      if(expr != empty)
      {
        block.move_to_operands(expr);
      }
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

  exprt block_expr;

  // Handle --function option
  const std::string function = config.options.get_option("function");
  if(!function.empty())
  {
    /* If the user passes --function, we add only a call to the
     * respective function in __ESBMC_main instead of entire Python program
     */

    nlohmann::json function_node;
    // Find function node in AST
    for(const auto &element : ast_json["body"])
    {
      if(element["_type"] == "FunctionDef" && element["name"] == function)
      {
        function_node = element;
        break;
      }
    }

    if(function_node.empty())
    {
      log_error("Function \"{}\" not found\n", function);
      return true;
    }

    // Convert a single function
    get_function_definition(function_node);

    // Get function symbol
    std::string symbol_id = "py:" + python_filename + "@F@" + function;
    symbolt *symbol = context.find_symbol(symbol_id);
    if(!symbol)
    {
      log_error("Symbol \"{}\" not found\n", symbol_id.c_str());
      return true;
    }

    // Create function call
    code_function_callt call;
    call.location() = symbol->location;
    call.function() = symbol_expr(*symbol);

    const code_typet::argumentst &arguments =
      to_code_type(symbol->type).arguments();
    call.arguments().resize(
      arguments.size(), static_cast<const exprt &>(get_nil_irep()));

    block_expr = call;
  }
  else
  {
    // Convert all statements
    block_expr = get_block(ast_json["body"]);
  }

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
