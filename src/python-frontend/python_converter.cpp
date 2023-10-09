#include "python-frontend/python_converter.h"
#include "util/std_code.h"
#include "util/c_types.h"
#include "util/arith_tools.h"
#include "util/expr_util.h"
#include "util/message.h"

#include <fstream>
#include <unordered_map>

using json = nlohmann::json;

const char *json_filename = "/tmp/ast.json";

std::unordered_map<std::string, std::string> operator_map = {
  {"Add", "+"},
  {"Sub", "-"},
  {"Mult", "*"},
  {"Div", "/"},
  {"BitOr", "bitor"},
  {"BitAnd", "bitand"},
  {"BitXor", "bitxor"},
  {"Invert", "bitnot"},
  {"LShift", "shl"},
  {"RShift", "lsr"},
  {"USub", "unary-"}};

enum class StatementType
{
  VARIABLE_ASSIGN,
  FUNC_DEFINITION,
  UNKNOWN,
};

enum class ExpressionType
{
  UNARY_OPERATION,
  BINARY_OPERATION,
  LITERAL,
  UNKNOWN,
};

StatementType get_statement_type(const json &element)
{
  if(element["_type"] == "AnnAssign")
  {
    return StatementType::VARIABLE_ASSIGN;
  }
  else if(element["_type"] == "FunctionDef")
  {
    return StatementType::FUNC_DEFINITION;
  }
  return StatementType::UNKNOWN;
}

// Convert Python/AST to irep2 operations
std::string get_op(const std::string &op)
{
  auto it = operator_map.find(op);
  if(it != operator_map.end())
  {
    return it->second;
  }
  return std::string();
}

// Convert Python/AST types to irep2 types
typet get_typet(const std::string &ast_type)
{
  if(ast_type == "float")
    return float_type();
  if(ast_type == "int")
    return int_type();
  return empty_typet();
}

symbolt create_symbol(
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

locationt get_location_from_decl(const nlohmann::json &ast_node)
{
  locationt location;
  location.set_line(ast_node["target"]["lineno"].get<int>());
  // TODO: Modify ast.py to include Python filename in the generated json
  location.set_file("program.py"); // FIXME: This should be read from JSON
  return location;
}

ExpressionType get_expression_type(const nlohmann::json &element)
{
  auto type = element["_type"];
  if(type == "UnaryOp")
  {
    return ExpressionType::UNARY_OPERATION;
  }
  if(type == "BinOp")
  {
    return ExpressionType::BINARY_OPERATION;
  }
  if(type == "Constant")
  {
    return ExpressionType::LITERAL;
  }
  return ExpressionType::UNKNOWN;
}

exprt python_converter::get_binary_operator_expr(const nlohmann::json &element)
{
  exprt bin_expr(
    get_op(element["op"]["_type"].get<std::string>()), current_element_type);

  exprt lhs = get_expr(element["left"]);
  exprt rhs = get_expr(element["right"]);

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
    break;
  }
  default:
  {
    assert(!"Unimplemented type in rule expression");
  }
  }

  return expr;
}

symbolt python_converter::get_var_decl(const nlohmann::json &ast_node)
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
    id = "c:@" + name;
  }

  // Variable location
  locationt location_begin = get_location_from_decl(ast_node);

  // Debug module name
  // FIXME: This should be read from JSON
  std::string module_name = "program.py";

  // Create/init symbol
  symbolt symbol =
    create_symbol(module_name, name, id, location_begin, current_element_type);
  symbol.lvalue = true;
  symbol.static_lifetime = false;
  symbol.file_local = true;
  symbol.is_extern = false;

  // Assign a value
  json value = ast_node["value"];
  exprt val = get_expr(value);
  symbol.value = val;

  return symbol;
}

bool python_converter::convert()
{
  codet main_code = code_blockt();
  main_code.make_block();

  std::ifstream f(json_filename);
  json ast = json::parse(f);

  for(auto &element : ast["body"])
  {
    StatementType type = get_statement_type(element);

    // Variable assignments
    if(type == StatementType::VARIABLE_ASSIGN)
    {
      symbolt symbol = get_var_decl(element);
      main_code.copy_to_operands(
        code_assignt(symbol_expr(symbol), symbol.value));
      context.add(symbol);
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
