#include <python-frontend/python_converter.h>
#include <python-frontend/json_utils.h>
#include <python_frontend_types.h>
#include <util/std_code.h>
#include <util/c_types.h>
#include <util/arith_tools.h>
#include <util/expr_util.h>
#include <util/message.h>

#include <fstream>
#include <regex>
#include <unordered_map>

using namespace json_utils;

static const std::unordered_map<std::string, std::string> operator_map = {
  {"Add", "+"},         {"Sub", "-"},          {"Mult", "*"},
  {"Div", "/"},         {"Mod", "mod"},        {"BitOr", "bitor"},
  {"FloorDiv", "/"},    {"BitAnd", "bitand"},  {"BitXor", "bitxor"},
  {"Invert", "bitnot"}, {"LShift", "shl"},     {"RShift", "ashr"},
  {"USub", "unary-"},   {"Eq", "="},           {"Lt", "<"},
  {"LtE", "<="},        {"NotEq", "notequal"}, {"Gt", ">"},
  {"GtE", ">="},        {"And", "and"},        {"Or", "or"},
  {"Not", "not"},
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
  {"ClassDef", StatementType::CLASS_DEFINITION},
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
  return (it != statement_map.end()) ? it->second : StatementType::UNKNOWN;
}

// Convert Python/AST to irep2 operations
static std::string get_op(const std::string &op)
{
  auto it = operator_map.find(op);
  if (it != operator_map.end())
  {
    return it->second;
  }
  return std::string();
}

// Convert Python/AST types to irep2 types
typet python_converter::get_typet(const std::string &ast_type)
{
  if (ast_type == "float")
    return float_type();
  if (ast_type == "int")
    /* FIXME: We need to map 'int' to another irep type that provides unlimited precision
	https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex */
    return int_type();
  if (ast_type == "uint64")
    return long_long_uint_type();
  if (ast_type == "bool")
    return bool_type();
  if (is_class(ast_type, ast_json["body"]))
    return symbol_typet("tag-" + ast_type);
  return empty_typet();
}

typet python_converter::get_typet(const nlohmann::json &elem)
{
  if (elem.is_number_integer() || elem.is_number_unsigned())
    return int_type();
  else if (elem.is_boolean())
    return bool_type();
  else if (elem.is_number_float())
    return float_type();

  log_error("Invalid type\n");
  abort();
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
  if (type == "UnaryOp")
  {
    return ExpressionType::UNARY_OPERATION;
  }
  if (type == "BinOp" || type == "Compare")
  {
    return ExpressionType::BINARY_OPERATION;
  }
  if (type == "BoolOp")
  {
    return ExpressionType::LOGICAL_OPERATION;
  }
  if (type == "Constant")
  {
    return ExpressionType::LITERAL;
  }
  if (type == "Name" || type == "Attribute")
  {
    return ExpressionType::VARIABLE_REF;
  }
  if (type == "Call")
  {
    return ExpressionType::FUNC_CALL;
  }
  if (type == "IfExp")
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
  for (const auto &operand : element["values"])
  {
    exprt operand_expr = get_expr(operand);
    logical_expr.copy_to_operands(operand_expr);
  }

  return logical_expr;
}

void python_converter::adjust_statement_types(exprt &lhs, exprt &rhs) const
{
  typet &lhs_type = lhs.type();
  typet &rhs_type = rhs.type();

  auto update_symbol = [&](exprt &expr) {
    std::string id = create_symbol_id() + "@" + expr.name().c_str();
    symbolt *s = context.find_symbol(id);
    if (s != nullptr)
    {
      s->type = expr.type();
      s->value.type() = expr.type();

      if (
        s->value.is_constant() ||
        (s->value.is_signedbv() || s->value.is_unsignedbv()))
      {
        exprt new_value =
          from_integer(std::stoi(s->value.value().c_str()), expr.type());
        s->value = new_value;
      }
    }
  };

  if (lhs_type.width() != rhs_type.width())
  {
    int lhs_type_width = std::stoi(lhs_type.width().c_str());
    int rhs_type_width = std::stoi(rhs_type.width().c_str());

    if (lhs_type_width > rhs_type_width)
    {
      // Update rhs symbol value to match with new type
      rhs_type = lhs_type;
      update_symbol(rhs);
    }
    else
    {
      // Update lhs symbol value to match with new type
      lhs_type = rhs_type;
      update_symbol(lhs);
    }
  }
}

std::string python_converter::create_symbol_id() const
{
  std::stringstream symbol_id;
  symbol_id << "py:" << python_filename;

  if (!current_class_name.empty())
    symbol_id << "@C@" << current_class_name;

  if (!current_func_name.empty())
    symbol_id << "@F@" << current_func_name;

  return symbol_id.str();
}

exprt python_converter::get_binary_operator_expr(const nlohmann::json &element)
{
  exprt lhs;
  if (element.contains("left"))
    lhs = get_expr(element["left"]);
  else if (element.contains("target"))
    lhs = get_expr(element["target"]);

  exprt rhs;
  if (element.contains("right"))
    rhs = get_expr(element["right"]);
  else if (element.contains("comparators"))
    rhs = get_expr(element["comparators"][0]);
  else if (element.contains("value"))
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
  if (lhs.is_function_call())
    to_side_effect_call(lhs);
  if (rhs.is_function_call())
    to_side_effect_call(rhs);

  std::string op;

  if (element.contains("op"))
    op = element["op"]["_type"].get<std::string>();
  else if (element.contains("ops"))
    op = element["ops"][0]["_type"].get<std::string>();

  assert(!op.empty());

  adjust_statement_types(lhs, rhs);

  assert(lhs.type() == rhs.type());

  typet type = (is_relational_op(op)) ? bool_type() : lhs.type();
  exprt bin_expr(get_op(op), type);
  bin_expr.copy_to_operands(lhs, rhs);

  // floor division (//) operation corresponds to an int division with floor rounding
  // So we need to emulate this behaviour here:
  // int result = (num/div) - (num%div != 0 && ((num < 0) ^ (den<0)) ? 1 : 0)
  // e.g.: -5//2 equals to -3, and 5//2 equals to 2
  if (op == "FloorDiv")
  {
    typet div_type = bin_expr.type();
    // remainder = num%den;
    exprt remainder("mod", div_type);
    remainder.copy_to_operands(lhs, rhs);

    // Get num signal
    exprt is_num_neg("<", bool_type());
    is_num_neg.copy_to_operands(lhs, gen_zero(div_type));
    // Get den signal
    exprt is_den_neg("<", bool_type());
    is_den_neg.copy_to_operands(rhs, gen_zero(div_type));

    // remainder != 0
    exprt pos_remainder("notequal", bool_type());
    pos_remainder.copy_to_operands(remainder, gen_zero(div_type));

    // diff_signals = is_num_neg ^ is_den_neg;
    exprt diff_signals("bitxor", bool_type());
    diff_signals.copy_to_operands(is_num_neg, is_den_neg);

    exprt cond("and", bool_type());
    cond.copy_to_operands(pos_remainder, diff_signals);
    exprt if_expr("if", div_type);
    if_expr.copy_to_operands(cond, gen_one(div_type), gen_zero(div_type));

    // floor_div = (lhs / rhs) - (1 if (lhs % rhs != 0) and (lhs < 0) ^ (rhs < 0) else 0)
    exprt floor_div("-", div_type);
    floor_div.copy_to_operands(bin_expr, if_expr); //bin_expr contains lhs/rhs

    return floor_div;
  }

  return bin_expr;
}

exprt python_converter::get_unary_operator_expr(const nlohmann::json &element)
{
  typet type = current_element_type;
  if (element["operand"].contains("value"))
    type = get_typet(element["operand"]["value"]);

  exprt unary_expr(get_op(element["op"]["_type"].get<std::string>()), type);

  // get subexpr
  exprt unary_sub = get_expr(element["operand"]);
  unary_expr.operands().push_back(unary_sub);

  return unary_expr;
}

const nlohmann::json python_converter::find_var_decl(
  const std::string &var_name,
  const nlohmann::json &json)
{
  for (auto &element : json["body"])
  {
    if (
      (element["_type"] == "AnnAssign") &&
      (element["target"]["id"] == var_name))
      return element;
  }
  return nlohmann::json();
}

locationt
python_converter::get_location_from_decl(const nlohmann::json &ast_node)
{
  locationt location;
  location.set_line(ast_node["lineno"].get<int>());
  location.set_column(ast_node["col_offset"].get<int>());
  location.set_file(python_filename.c_str());
  location.set_function(current_func_name);
  return location;
}

exprt python_converter::get_function_call(const nlohmann::json &element)
{
  if (element.contains("func") && element["_type"] == "Call")
  {
    std::string func_name = element["func"]["id"];

    // nondet_X() functions restricted to basic types supported in Python
    std::regex pattern(
      R"(nondet_(int|char|bool|float)|__VERIFIER_nondet_(int|char|bool|float))");

    if (std::regex_match(func_name, pattern))
    {
      // Function name pattern: nondet_(type). e.g: nondet_bool(), nondet_int()
      size_t underscore_pos = func_name.rfind("_");
      std::string type = func_name.substr(underscore_pos + 1);
      exprt rhs = exprt("sideeffect", get_typet(type));
      rhs.statement("nondet");
      return rhs;
    }

    locationt location = get_location_from_decl(element);
    std::string symbol_id = create_symbol_id();
    if (symbol_id.find("@F@") == symbol_id.npos)
      symbol_id += std::string("@F@") + func_name;

    // __ESBMC_assume
    if (func_name == "__ESBMC_assume" || func_name == "__VERIFIER_assume")
    {
      symbol_id = func_name;
      if (context.find_symbol(symbol_id.c_str()) == nullptr)
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

    bool is_ctor_call = is_constructor_call(element);

    if (is_ctor_call)
    {
      // Insert class name in the symbol id
      std::size_t pos = symbol_id.rfind("@F@");
      if (pos != std::string::npos)
        symbol_id.insert(pos, "@C@" + func_name);
    }

    const symbolt *func_symbol = context.find_symbol(symbol_id.c_str());
    if (func_symbol == nullptr)
    {
      log_error("Undefined function: {}", func_name.c_str());
      abort();
    }

    code_function_callt call;
    call.location() = location;
    call.function() = symbol_expr(*func_symbol);
    const typet &return_type = to_code_type(func_symbol->type).return_type();
    call.type() = return_type;

    if (is_ctor_call)
      call.arguments().push_back(
        gen_address_of(*ref_instance)); // Add self as first parameter

    for (const auto &arg_node : element["args"])
      call.arguments().push_back(get_expr(arg_node));

    return call;
  }

  log_error("Invalid function call");
  abort();
}

exprt python_converter::get_expr(const nlohmann::json &element)
{
  exprt expr;
  ExpressionType type = get_expression_type(element);

  switch (type)
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
    if (element["value"].is_number_integer())
    {
      expr = from_integer(value.get<int>(), int_type());
    }
    else if (element["value"].is_boolean())
    {
      expr = gen_boolean(value.get<bool>());
    }
    break;
  }
  case ExpressionType::VARIABLE_REF:
  {
    std::string var_name;
    if (element["_type"] == "Name")
      var_name = element["id"].get<std::string>();
    else if (element["_type"] == "Attribute")
      var_name = element["value"]["id"].get<std::string>();

    assert(!var_name.empty());

    std::string symbol_id = create_symbol_id() + std::string("@") + var_name;
    symbolt *symbol = context.find_symbol(symbol_id);
    if (!symbol)
    {
      log_error("Symbol not found: {}\n", symbol_id.c_str());
      abort();
    }
    expr = symbol_expr(*symbol);

    if (element["_type"] == "Attribute")
    {
      const std::string &attr_name = element["attr"].get<std::string>();

      // Get object type name from symbol. e.g.: tag-MyClass
      std::string obj_type_name;
      const typet &symbol_type =
        (symbol->type.is_pointer()) ? symbol->type.subtype() : symbol->type;
      for (const auto &it : symbol_type.get_named_sub())
      {
        if (it.first == "identifier")
          obj_type_name = it.second.id_string();
      }

      // Get class definition from symbols table
      symbolt *class_symbol = context.find_symbol(obj_type_name);
      if (!class_symbol)
      {
        log_error("Class not found: {}\n", obj_type_name);
        abort();
      }

      // Get attribute type from class definition
      struct_typet &class_type =
        static_cast<struct_typet &>(class_symbol->type);
      assert(class_type.has_component(attr_name));
      const typet &attr_type = class_type.get_component(attr_name).type();

      expr = member_exprt(
        symbol_exprt(symbol->id, symbol->type), attr_name, attr_type);
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

bool python_converter::is_constructor_call(const nlohmann::json &json)
{
  if (!json.contains("_type") || json["_type"] != "Call")
    return false;

  const std::string &func_name = json["func"]["id"];

  /* f:Foo = Foo()
   * The statement is a constructor call if the function call on the
   * rhs corresponds to the name of a class. */

  bool is_ctor_call = false;
  context.foreach_operand([&](const symbolt &s) {
    if (s.type.id() == "struct" && s.name == func_name)
    {
      is_ctor_call = true;
      return;
    }
  });
  return is_ctor_call;
}

void python_converter::get_var_assign(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  if (ast_node.contains("annotation"))
  {
    // Get type from current annotation node
    current_element_type =
      get_typet(ast_node["annotation"]["id"].get<std::string>());
  }
  else
  {
    // Get type from declaration node
    std::string var_name = ast_node["targets"][0]["id"].get<std::string>();
    // Get variable from current function
    nlohmann::json ref;
    for (const auto &elem : ast_json["body"])
    {
      if (elem["_type"] == "FunctionDef" && elem["name"] == current_func_name)
        ref = find_var_decl(var_name, elem);
    }

    // Get variable from global scope
    if (ref.empty())
      ref = find_var_decl(var_name, ast_json);

    assert(!ref.empty());
    current_element_type =
      get_typet(ref["annotation"]["id"].get<std::string>());
  }

  exprt lhs;
  symbolt *lhs_symbol = nullptr;
  locationt location_begin;

  if (ast_node["_type"] == "AnnAssign")
  {
    // Id and name
    std::string name, id;
    auto target = ast_node["target"];
    if (!target.is_null())
    {
      if (target["_type"] == "Name")
        name = target["id"];
      else if (target["_type"] == "Attribute")
        name = target["attr"];

      id = create_symbol_id() + "@" + name;
    }

    assert(!name.empty() && !id.empty());

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

    lhs = symbol_expr(symbol);

    if (target["_type"] == "Attribute")
    {
      // lhs is an attribute and needs to be added as member of the referred object
      // 1. Retrieve created object from symbol table
      std::string obj_id =
        create_symbol_id() + "@" + target["value"]["id"].get<std::string>();
      symbolt *obj_symbol = context.find_symbol(obj_id);
      if (!obj_symbol)
        abort();

      // 2. Insert member in the object
      member_exprt member(
        symbol_exprt(obj_symbol->id, obj_symbol->type), lhs.name(), lhs.type());

      // 3. lhs holds 'obj.member'
      lhs.swap(member);
    }
    lhs.location() = location_begin;

    lhs_symbol = context.move_symbol_to_context(symbol);
  }
  else if (ast_node["_type"] == "Assign")
  {
    std::string name = ast_node["targets"][0]["id"].get<std::string>();
    std::string symbol_id = create_symbol_id() + "@" + name;
    symbolt *symbol = context.find_symbol(symbol_id);
    assert(symbol);
    lhs = symbol_expr(*symbol);
  }

  if (is_constructor_call(ast_node["value"]))
    ref_instance = &lhs;

  // Get RHS
  exprt rhs = get_expr(ast_node["value"]);
  if (lhs_symbol)
    lhs_symbol->value = rhs;

  /* If the right-hand side (rhs) of the assignment is a function call, such as: x : int = func()
   * we need to adjust the left-hand side (lhs) of the function call to refer to the lhs of the current assignment.
   */
  if (rhs.is_function_call())
  {
    // op0() refers to the left-hand side (lhs) of the function call
    rhs.op0() = lhs;
    target_block.copy_to_operands(rhs);
    return;
  }

  adjust_statement_types(lhs, rhs);

  code_assignt code_assign(lhs, rhs);
  code_assign.location() = location_begin;
  target_block.copy_to_operands(code_assign);

  ref_instance = nullptr;
}

void python_converter::get_compound_assign(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  // Get type from declaration node
  std::string var_name = ast_node["target"]["id"].get<std::string>();
  nlohmann::json ref = find_var_decl(var_name, ast_json);
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
  if (expr.is_code())
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
  if (ast_node["body"].is_array())
    then = get_block(ast_node["body"]);
  else
    then = get_expr(ast_node["body"]);

  locationt location = get_location_from_decl(ast_node);
  then.location() = location;

  // Extract 'else' block from AST
  exprt else_expr;
  if (ast_node.contains("orelse") && !ast_node["orelse"].empty())
  {
    // Append 'else' block to the statement
    if (ast_node["orelse"].is_array())
      else_expr = get_block(ast_node["orelse"]);
    else
      else_expr = get_expr(ast_node["orelse"]);
  }

  auto type = ast_node["_type"];

  // ternary operator
  if (type == "IfExp")
  {
    exprt if_expr("if", current_element_type);
    if_expr.copy_to_operands(cond, then, else_expr);
    return if_expr;
  }

  // Create if or while code
  codet code;
  if (type == "If")
    code.set_statement("ifthenelse");
  else if (type == "While")
    code.set_statement("while");

  // Append "then" block
  code.copy_to_operands(cond, then);
  if (!else_expr.id_string().empty())
    code.copy_to_operands(else_expr);

  return code;
}

void python_converter::get_function_definition(
  const nlohmann::json &function_node)
{
  // Function return type
  code_typet type;
  nlohmann::json return_node = function_node["returns"];
  if (return_node.contains("id"))
  {
    type.return_type() = get_typet(return_node["id"].get<std::string>());
  }
  else if (
    return_node.is_null() ||
    (return_node.contains("value") && return_node["value"].is_null()))
  {
    type.return_type() = empty_typet();
  }
  else
  {
    log_error("Return type undefined\n");
    abort();
  }

  // Copy caller function name
  const std::string caller_func_name = current_func_name;

  // Function location
  locationt location = get_location_from_decl(function_node);

  current_element_type = type.return_type();
  current_func_name = function_node["name"].get<std::string>();

  // __init__() is renamed to Classname()
  if (current_func_name == "__init__")
  {
    current_func_name = current_class_name;
    typet ctor_type("constructor");
    type.return_type() = ctor_type;
  }

  std::string id = create_symbol_id();
  std::string module_name =
    python_filename.substr(0, python_filename.find_last_of("."));

  // Iterate over function arguments
  for (const nlohmann::json &element : function_node["args"]["args"])
  {
    // Argument name
    std::string arg_name = element["arg"].get<std::string>();
    // Argument type
    typet arg_type;
    if (arg_name == "self")
      arg_type = gen_pointer_type(get_typet(current_class_name));
    else
      arg_type = get_typet(element["annotation"]["id"].get<std::string>());

    assert(arg_type != typet());

    code_typet::argumentt arg;
    arg.type() = arg_type;

    arg.cmt_base_name(arg_name);

    // Argument id
    std::string arg_id = id + "@" + arg_name;
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

void python_converter::get_attributes_from_self(
  const nlohmann::json &method_body,
  struct_typet &clazz)
{
  for (const auto &stmt : method_body)
  {
    if (
      stmt["_type"] == "AnnAssign" && stmt["target"]["_type"] == "Attribute" &&
      stmt["target"]["value"]["id"] == "self")
    {
      std::string attr_name = stmt["target"]["attr"];
      struct_typet::componentt comp(
        attr_name,
        attr_name,
        get_typet(stmt["annotation"]["id"].get<std::string>()));
      comp.type().set("#member_name", std::string("tag-") + current_class_name);
      comp.set_access("private");
      clazz.components().push_back(comp);
    }
  }
}

void python_converter::get_class_definition(const nlohmann::json &class_node)
{
  struct_typet clazz;
  current_class_name = class_node["name"].get<std::string>();
  clazz.tag(current_class_name);
  std::string id = "tag-" + current_class_name;

  if (context.find_symbol(id) != nullptr)
    return;

  locationt location_begin = get_location_from_decl(class_node);
  std::string module_name = location_begin.get_file().as_string();

  // Add class to symbol table
  symbolt symbol =
    create_symbol(module_name, current_class_name, id, location_begin, clazz);
  symbol.is_type = true;

  symbolt *added_symbol = context.move_symbol_to_context(symbol);

  // Iterate over class members
  for (auto &class_member : class_node["body"])
  {
    if (class_member["_type"] == "FunctionDef")
    {
      get_attributes_from_self(class_member["body"], clazz);
      added_symbol->type = clazz;

      std::string method_name = class_member["name"].get<std::string>();
      if (method_name == "__init__")
        method_name = current_class_name;

      current_func_name = method_name;
      get_function_definition(class_member);

      exprt added_method =
        symbol_expr(*context.find_symbol(create_symbol_id()));
      struct_typet::componentt method(added_method.name(), added_method.type());
      clazz.methods().push_back(method);
      current_func_name.clear();
    }
  }
  added_symbol->type = clazz;
  current_class_name.clear();
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

  // Iterate over block statements
  for (auto &element : ast_block)
  {
    StatementType type = get_statement_type(element);

    switch (type)
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
      if (expr != empty)
      {
        block.move_to_operands(expr);
      }
      break;
    }
    case StatementType::CLASS_DEFINITION:
    {
      get_class_definition(element);
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

python_converter::python_converter(
  contextt &_context,
  const nlohmann::json &ast)
  : context(_context),
    ast_json(ast),
    current_func_name(""),
    current_class_name(""),
    ref_instance(nullptr)
{
}

bool python_converter::convert()
{
  python_filename = ast_json["filename"].get<std::string>();

  exprt block_expr;

  // Handle --function option
  const std::string function = config.options.get_option("function");
  if (!function.empty())
  {
    /* If the user passes --function, we add only a call to the
     * respective function in __ESBMC_main instead of entire Python program
     */

    nlohmann::json function_node;
    // Find function node in AST
    for (const auto &element : ast_json["body"])
    {
      if (element["_type"] == "FunctionDef" && element["name"] == function)
      {
        function_node = element;
        break;
      }
    }

    if (function_node.empty())
    {
      log_error("Function \"{}\" not found\n", function);
      return true;
    }

    // Convert a single function
    get_function_definition(function_node);

    // Get function symbol
    std::string symbol_id = create_symbol_id() + "@F@" + function;
    symbolt *symbol = context.find_symbol(symbol_id);
    if (!symbol)
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

    // Function args are nondet values
    for (const code_typet::argumentt &arg : arguments)
    {
      exprt arg_value = exprt("sideeffect", arg.type());
      arg_value.statement("nondet");
      call.arguments().push_back(arg_value);
    }

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

  if (context.move(main_symbol))
  {
    log_error("main already defined by another language module");
    return true;
  }

  return false;
}
