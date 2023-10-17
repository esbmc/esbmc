#include "python-frontend/python_converter.h"
#include "util/std_code.h"
#include "util/c_types.h"
#include "util/arith_tools.h"
#include "util/expr_util.h"
#include "util/message.h"

#include <fstream>
#include <unordered_map>

enum class ExpressionType
{
  UNARY_OPERATION,
  BINARY_OPERATION,
  LITERAL,
  VARIABLE_REF,
  UNKNOWN,
};

enum class StatementType
{
  VARIABLE_ASSIGN, // Simple assignments like x = 1
  COMPOUND_ASSIGN, // Compound assignments (+=, -=, etc)
  FUNC_DEFINITION,
  IF_STATEMENT,
  UNKNOWN,
};

static const std::unordered_map<std::string, std::string> operator_map = {
  {"Add", "+"},
  {"Sub", "-"},
  {"Mult", "*"},
  {"Div", "/"},
  {"BitOr", "bitor"},
  {"BitAnd", "bitand"},
  {"BitXor", "bitxor"},
  {"Invert", "bitnot"},
  {"LShift", "shl"},
  {"RShift", "ashr"},
  {"USub", "unary-"},
  {"Eq", "="}};

static const std::unordered_map<std::string, StatementType> statement_map = {
  {"AnnAssign", StatementType::VARIABLE_ASSIGN},
  {"FunctionDef", StatementType::FUNC_DEFINITION},
  {"If", StatementType::IF_STATEMENT},
  {"AugAssign", StatementType::COMPOUND_ASSIGN},
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

  exprt lhs; // = get_expr(element["left"]);
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
  location.set_file(ast_json["filename"].get<std::string>());
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
    nlohmann::json ref = find_var_decl(var_name);
    assert(!ref.empty());

    typet type = get_typet(ref["annotation"]["id"].get<std::string>());

    symbolt *symbol = context.find_symbol(std::string("py:@") + var_name);
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
  // Get variable type
  current_element_type =
    get_typet(ast_node["annotation"]["id"].get<std::string>());

  // Id and name
  std::string name, id;
  auto target = ast_node["target"];
  if(!target.is_null() && target["_type"] == "Name")
  {
    name = target["id"];
    id = "py:@" + name;
  }

  // Variable location
  locationt location_begin = get_location_from_decl(ast_node["target"]);

  // Debug module name
  std::string module_name = location_begin.get_file().as_string();

  // Create/init symbol
  symbolt symbol =
    create_symbol(module_name, name, id, location_begin, current_element_type);
  symbol.lvalue = true;
  symbol.static_lifetime = false;
  symbol.file_local = true;
  symbol.is_extern = false;

  // Assign a value
  nlohmann::json value = ast_node["value"];
  exprt val = get_expr(value);
  symbol.value = val;

  code_assignt code_assign(symbol_expr(symbol), symbol.value);
  code_assign.location() = location_begin;
  target_block.copy_to_operands(code_assign);

  context.add(symbol);
}

void python_converter::get_compound_assign(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  exprt lhs = get_expr(ast_node["target"]);
  exprt rhs = get_binary_operator_expr(ast_node);

  code_assignt code_assign(lhs, rhs);
  code_assign.location() = get_location_from_decl(ast_node);

  target_block.copy_to_operands(code_assign);
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
    case StatementType::FUNC_DEFINITION:
    case StatementType::IF_STATEMENT:
    case StatementType::UNKNOWN:
    default:
      log_error("error");
      abort();
    }
  }

  return block;
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

void python_converter::get_if_statement(
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

  // Create if code and append "then" block
  codet code_if("ifthenelse");
  code_if.copy_to_operands(cond, convert_expression_to_code(then));

  // Append 'else' block to the statement
  if(ast_node.contains("orelse") && !ast_node["orelse"].empty())
  {
    exprt else_expr = get_block(ast_node["orelse"]);
    code_if.copy_to_operands(convert_expression_to_code(else_expr));
  }

  target_block.copy_to_operands(code_if);
}

bool python_converter::convert()
{
  codet main_code = code_blockt();
  main_code.make_block();
  current_function_name = "globalscope";

  for(auto &element : ast_json["body"])
  {
    StatementType type = get_statement_type(element);

    // Variable assignments
    if(type == StatementType::VARIABLE_ASSIGN)
    {
      get_var_assign(element, main_code);
    }
    // If statements
    else if(type == StatementType::IF_STATEMENT)
    {
      get_if_statement(element, main_code);
    }
    else if(type == StatementType::COMPOUND_ASSIGN)
    {
      get_compound_assign(element, main_code);
    }
  }

  // add "main"
  symbolt main_symbol;

  code_typet main_type;
  main_type.return_type() = empty_typet();

  main_symbol.id = "__ESBMC_main";
  main_symbol.name = "__ESBMC_main";
  main_symbol.type.swap(main_type);
  main_symbol.value.swap(main_code);

  if(context.move(main_symbol))
  {
    log_error("main already defined by another language module");
    return true;
  }

  return false;
}
