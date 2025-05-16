#include <python-frontend/python_converter.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/type_utils.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/function_call_builder.h>
#include <ansi-c/convert_float_literal.h>
#include <util/std_code.h>
#include <util/c_types.h>
#include <util/c_typecast.h>
#include <util/arith_tools.h>
#include <util/expr_util.h>
#include <util/message.h>
#include <util/encoding.h>

#include <algorithm>
#include <fstream>
#include <regex>
#include <unordered_map>

#include <boost/filesystem.hpp>

using namespace json_utils;
namespace fs = boost::filesystem;

static const std::unordered_map<std::string, std::string> operator_map = {
  {"add", "+"},         {"sub", "-"},          {"subtract", "-"},
  {"mult", "*"},        {"multiply", "*"},     {"div", "/"},
  {"divide", "/"},      {"mod", "mod"},        {"bitor", "bitor"},
  {"floordiv", "/"},    {"bitand", "bitand"},  {"bitxor", "bitxor"},
  {"invert", "bitnot"}, {"lshift", "shl"},     {"rshift", "ashr"},
  {"usub", "unary-"},   {"eq", "="},           {"lt", "<"},
  {"lte", "<="},        {"noteq", "notequal"}, {"gt", ">"},
  {"gte", ">="},        {"and", "and"},        {"or", "or"},
  {"not", "not"},
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
  {"Pass", StatementType::PASS},
  {"Break", StatementType::BREAK},
  {"Continue", StatementType::CONTINUE},
  {"ImportFrom", StatementType::IMPORT},
  {"Import", StatementType::IMPORT},
  {"Raise", StatementType::RAISE}};

static bool is_char_type(const typet &t)
{
  return (t.is_signedbv() || t.is_unsignedbv()) && t.get("#cpp_type") == "char";
}

static bool is_float_vs_char(const exprt &a, const exprt &b)
{
  const auto &type_a = a.type();
  const auto &type_b = b.type();
  return (type_a.is_floatbv() && is_char_type(type_b)) ||
         (type_b.is_floatbv() && is_char_type(type_a));
}

static bool is_ordered_comparison(const std::string &op)
{
  return op == "Lt" || op == "Gt" || op == "LtE" || op == "GtE";
}

static bool is_relational_op(const std::string &op)
{
  return (
    op == "Eq" || op == "Lt" || op == "LtE" || op == "NotEq" || op == "Gt" ||
    op == "GtE" || op == "And" || op == "Or");
}

static StatementType get_statement_type(const nlohmann::json &element)
{
  if (!element.contains("_type"))
    return StatementType::UNKNOWN;

  auto it = statement_map.find(element["_type"]);
  return (it != statement_map.end()) ? it->second : StatementType::UNKNOWN;
}

static std::string get_op(const std::string &op, const typet &type)
{
  // Convert the operator to lowercase to allow case-insensitive comparison.
  std::string lower_op = op;
  std::transform(
    lower_op.begin(), lower_op.end(), lower_op.begin(), [](unsigned char c) {
      return std::tolower(c);
    });

  // Special case: if the type is floating-point, use IEEE-specific operators.
  if (type.is_floatbv())
  {
    static const std::unordered_map<std::string, std::string> float_ops = {
      {"add", "ieee_add"},
      {"sub", "ieee_sub"},
      {"subtract", "ieee_sub"},
      {"mult", "ieee_mul"},
      {"multiply", "ieee_mul"},
      {"div", "ieee_div"},
      {"divide", "ieee_div"}};

    auto float_it = float_ops.find(lower_op);
    if (float_it != float_ops.end())
      return float_it->second;
  }

  // Look up the operator in the general operator map (for non-floating-point types).
  auto it = operator_map.find(lower_op);
  if (it != operator_map.end())
  {
    return it->second;
  }

  // Operator not found — issue a warning and return an empty string.
  log_warning("Unknown operator: {}", op);
  return {};
}

static struct_typet::componentt build_component(
  const std::string &class_name,
  const std::string &comp_name,
  const typet &type)
{
  struct_typet::componentt component(comp_name, comp_name, type);

  // Add metadata used internally by ESBMC for member-to-class tagging.
  // The key "#member_name" is used by the type system; the value "tag-<class_name>" helps
  // associate this member with its parent class.
  component.type().set("#member_name", "tag-" + class_name);

  // Set the member visibility to public by default.
  component.set_access("public");

  return component;
}

symbolt python_converter::create_symbol(
  const std::string &module,
  const std::string &name,
  const std::string &id,
  const locationt &location,
  const typet &type) const
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
  // Return UNKNOWN if the expected "_type" field is missing
  if (!element.contains("_type"))
    return ExpressionType::UNKNOWN;

  // Map of Python AST "_type" strings to internal expression categories
  static const std::unordered_map<std::string, ExpressionType> type_map = {
    {"UnaryOp", ExpressionType::UNARY_OPERATION},
    {"BinOp", ExpressionType::BINARY_OPERATION},
    {"Compare",
     ExpressionType::BINARY_OPERATION}, // Comparison treated as binary op
    {"BoolOp", ExpressionType::LOGICAL_OPERATION},
    {"Constant", ExpressionType::LITERAL},
    {"Name", ExpressionType::VARIABLE_REF},
    {"Attribute",
     ExpressionType::VARIABLE_REF}, // Both treated as variable references
    {"Call", ExpressionType::FUNC_CALL},
    {"IfExp", ExpressionType::IF_EXPR},
    {"Subscript", ExpressionType::SUBSCRIPT},
    {"List", ExpressionType::LIST}};

  const auto &type = element["_type"];
  auto it = type_map.find(type);
  if (it != type_map.end())
    return it->second;

  // If the type is not recognized, return UNKNOWN
  return ExpressionType::UNKNOWN;
}

exprt python_converter::get_logical_operator_expr(const nlohmann::json &element)
{
  std::string op(element["op"]["_type"].get<std::string>());
  exprt logical_expr(get_op(op, bool_type()), bool_type());

  // Iterate over operands of logical operations (and/or)
  for (const auto &operand : element["values"])
  {
    exprt operand_expr = get_expr(operand);
    logical_expr.copy_to_operands(operand_expr);
  }
  return logical_expr;
}

void python_converter::update_symbol(const exprt &expr) const
{
  // Generate a symbol ID from the expression's name.
  symbol_id sid = create_symbol_id();
  sid.set_object(expr.name().c_str());

  // Try to locate the symbol in the symbol table.
  symbolt *sym = symbol_table_.find_symbol(sid.to_string());

  if (sym == nullptr)
  {
    // Symbol not found, nothing to update.
    return;
  }

  // Update the type of the symbol and its value.
  const typet &expr_type = expr.type();
  sym->type = expr_type;
  sym->value.type() = expr_type;

  // Check if the symbol has a constant or bitvector value.
  if (
    sym->value.is_constant() || sym->value.is_signedbv() ||
    sym->value.is_unsignedbv())
  {
    const std::string &binary_value_str = sym->value.value().c_str();

    try
    {
      // Convert binary string to integer.
      int64_t int_val = std::stoll(binary_value_str, nullptr, 2);

      // Create a new constant expression with the converted value and type.
      exprt new_value = from_integer(int_val, expr_type);

      // Assign the new value to the symbol.
      sym->value = new_value;
    }
    catch (const std::exception &e)
    {
      log_error(
        "update_symbol: Failed to convert binary value '{}' to integer for "
        "symbol '{}'. Error: {}",
        binary_value_str,
        sid.to_string(),
        e.what());
    }
  }
}

/// Promotes an integer expression to a float type (floatbv) when needed,
/// typically for Python-style division where / must always yield a float result,
// even with integer operands.
void python_converter::promote_int_to_float(exprt &op, const typet &target_type)
  const
{
  typet &op_type = op.type();

  // Only promote if operand is an integer type
  if (!(op_type.is_signedbv() || op_type.is_unsignedbv()))
    return;

  // Handle constant integers
  if (op.is_constant())
  {
    try
    {
      const BigInt int_val =
        binary2integer(op.value().as_string(), op_type.is_signedbv());

      // Generate a string like "3.0" for float parsing
      const std::string float_literal =
        std::to_string(int_val.to_int64()) + ".0";

      // Convert string literal to float expression
      convert_float_literal(float_literal, op);
    }
    catch (const std::exception &e)
    {
      log_error(
        "promote_int_to_float: Failed to promote constant to float: {}",
        e.what());
      return;
    }
  }

  // Update the operand type
  op.type() = target_type;

  // Update symbol type info if necessary
  if (op.is_symbol())
    update_symbol(op);
}

void python_converter::adjust_statement_types(exprt &lhs, exprt &rhs) const
{
  typet &lhs_type = lhs.type();
  typet &rhs_type = rhs.type();

  // Case 1: Promote RHS integer constant to float if LHS expects a float
  if (
    lhs_type.is_floatbv() && rhs.is_constant() &&
    (rhs_type.is_signedbv() || rhs_type.is_unsignedbv()))
  {
    try
    {
      // Convert binary string value to integer
      BigInt value(
        binary2integer(rhs.value().as_string(), rhs_type.is_signedbv()));

      // Create a float literal string (e.g., "42.0")
      std::string rhs_float = std::to_string(value.to_int64()) + ".0";

      // Replace RHS with a float expression
      convert_float_literal(rhs_float, rhs);

      // Update the symbol table entry for RHS if needed
      update_symbol(rhs);
    }
    catch (const std::exception &e)
    {
      log_error(
        "adjust_statement_types: Failed to promote integer to float: {}",
        e.what());
    }
  }
  /// Case 2: Handles Python’s / operator by promoting operands to floats
  /// to ensure floating-point division, preventing division by zero, and
  /// setting the result type to floatbv.
  else if (rhs.id() == "/" && rhs.operands().size() == 2)
  {
    auto &ops = rhs.operands();
    exprt &lhs_op = ops[0];
    exprt &rhs_op = ops[1];

    // Only optimize when both operands are compile-time constants
    if (!lhs_op.is_constant() || !rhs_op.is_constant())
      return;

    // Check if the right-hand side is a constant zero to prevent division-by-zero
    if (
      (rhs_op.type().is_signedbv() || rhs_op.type().is_unsignedbv()) &&
      binary2integer(rhs_op.value().as_string(), rhs_op.type().is_signedbv()) ==
        0)
      return;

    // Promote both operands to IEEE float (double precision) to match Python semantics
    const typet float_type =
      double_type(); // Python default float is double-precision

    for (exprt &op : ops)
      promote_int_to_float(op, float_type);

    // Update LHS type if it's a symbol, so it holds a float result
    lhs.type() = float_type;
    if (lhs.is_symbol())
      update_symbol(lhs);

    // Update the division expression type and operator ID
    rhs.type() = float_type;
    rhs.id(get_op("div", float_type));
  }
  // Case 3: Align bit-widths between LHS and RHS if they differ
  else if (lhs_type.width() != rhs_type.width())
  {
    try
    {
      const int lhs_width = std::stoi(lhs_type.width().c_str());
      const int rhs_width = std::stoi(rhs_type.width().c_str());

      if (lhs_width > rhs_width)
      {
        // Promote RHS to LHS type
        rhs_type = lhs_type;
        if (rhs.is_symbol())
          update_symbol(rhs);
      }
      else
      {
        // Promote LHS to RHS type
        lhs_type = rhs_type;
        if (lhs.is_symbol())
          update_symbol(lhs);
      }
    }
    catch (const std::exception &e)
    {
      log_error(
        "adjust_statement_types: Failed to parse type widths: {}", e.what());
    }
  }
}

symbol_id python_converter::create_symbol_id(const std::string &filename) const
{
  return symbol_id(filename, current_class_name_, current_func_name_);
}

symbol_id python_converter::create_symbol_id() const
{
  return symbol_id(
    current_python_file, current_class_name_, current_func_name_);
}

exprt python_converter::compute_math_expr(const exprt &expr) const
{
  auto resolve_symbol = [this](const exprt &operand) -> exprt {
    if (operand.is_symbol())
    {
      symbolt *s = symbol_table_.find_symbol(operand.identifier());
      assert(s && "Symbol not found in symbol table");
      return s->value;
    }
    return operand;
  };

  // Resolve operands
  const exprt lhs = resolve_symbol(expr.operands().at(0));
  const exprt rhs = resolve_symbol(expr.operands().at(1));

  // Convert to BigInt
  const BigInt op1 =
    binary2integer(lhs.value().as_string(), lhs.type().is_signedbv());
  const BigInt op2 =
    binary2integer(rhs.value().as_string(), rhs.type().is_signedbv());

  // Perform the math operation
  BigInt result;
  if (expr.id() == "+")
    result = op1 + op2;
  else if (expr.id() == "-")
    result = op1 - op2;
  else if (expr.id() == "*")
    result = op1 * op2;
  else if (expr.id() == "/")
    result = op1 / op2;
  else
    throw std::runtime_error("Unsupported math operation");

  // Return the result as a constant expression
  return constant_exprt(result, lhs.type());
}

inline bool is_math_expr(const exprt &expr)
{
  const std::string &id = expr.id().as_string();
  return id == "+" || id == "-" || id == "*" || id == "/";
}

// Attach source location from symbol table if expr is a symbol
static void attach_symbol_location(exprt &expr, contextt &symbol_table)
{
  if (!expr.is_symbol())
    return;

  const irep_idt &id = expr.identifier();
  symbolt *sym = symbol_table.find_symbol(id);
  if (sym != nullptr)
    expr.location() = sym->location;
}

exprt handle_floor_division(
  const exprt &lhs,
  const exprt &rhs,
  const exprt &bin_expr)
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

exprt python_converter::handle_power_operator(exprt lhs, exprt rhs)
{
  if (lhs.is_symbol())
  {
    symbolt *s = symbol_table_.find_symbol(lhs.identifier());
    assert(s);
    if (!s->value.value().empty())
      lhs = s->value;
  }
  if (rhs.is_symbol())
  {
    symbolt *s = symbol_table_.find_symbol(rhs.identifier());
    assert(s);
    if (!s->value.value().empty())
      rhs = s->value;
  }
  else if (is_math_expr(rhs))
  {
    rhs = compute_math_expr(rhs);
  }
  BigInt base(
    binary2integer(lhs.value().as_string(), lhs.type().is_signedbv()));
  BigInt exp(binary2integer(rhs.value().as_string(), rhs.type().is_signedbv()));
  constant_exprt pow_expr(power(base, exp), lhs.type());
  return pow_expr;
}

exprt handle_float_vs_string(exprt &bin_expr, const std::string &op)
{
  if (op == "Eq")
  {
    // float == str → False (no exception)
    bin_expr.make_false();
  }
  else if (op == "NotEq")
  {
    // float != str → True (no exception)
    bin_expr.make_true();
  }
  else if (is_ordered_comparison(op))
  {
    // Python-style error: float < str → TypeError
    std::string lower_op = op;
    std::transform(
      lower_op.begin(), lower_op.end(), lower_op.begin(), [](unsigned char c) {
        return std::tolower(c);
      });

    const auto &loc = bin_expr.location();
    const auto it = operator_map.find(lower_op);
    assert(it != operator_map.end());

    std::ostringstream error;
    error << "'" << it->second
          << "' not supported between instances of 'float' and 'str'";

    if (loc.is_not_nil())
      error << " at " << loc.get_file() << ":" << loc.get_line();
    else
      error << " at <unknown location>";

    throw std::runtime_error(error.str());
  }

  return bin_expr;
}

void python_converter::handle_float_division(
  exprt &lhs,
  exprt &rhs,
  exprt &bin_expr) const
{
  const typet float_type = double_type();

  auto promote_to_float = [&](exprt &e) {
    const typet &t = e.type();
    const bool is_integer = t.is_signedbv() || t.is_unsignedbv();

    if (!is_integer)
      return;

    // Handle constant integers: convert them to float literals
    if (e.is_constant())
    {
      try
      {
        const bool is_signed = t.is_signedbv();
        const BigInt val = binary2integer(e.value().as_string(), is_signed);
        const double float_val = static_cast<double>(val.to_int64());
        convert_float_literal(std::to_string(float_val), e);
      }
      catch (const std::exception &ex)
      {
        log_error(
          "handle_float_division: failed to promote constant to float: {}",
          ex.what());
      }
    }

    // Update expression type to float (double)
    e.type() = float_type;

    // Update symbol table if this is a symbol
    if (e.is_symbol())
      update_symbol(e);
  };

  promote_to_float(lhs);
  promote_to_float(rhs);

  // Set the result type and operator ID to reflect float division
  bin_expr.type() = float_type;
  bin_expr.id(get_op("div", float_type));
}

BigInt python_converter::get_string_size(const exprt &expr)
{
  const auto &arr_type = to_array_type(expr.type());
  return binary2integer(arr_type.size().value().as_string(), false);
}

exprt python_converter::handle_string_concatenation(
  const exprt &lhs,
  const exprt &rhs,
  const nlohmann::json &left,
  const nlohmann::json &right)
{
  BigInt lhs_size = get_string_size(lhs);
  BigInt rhs_size = get_string_size(rhs);
  BigInt total_size = lhs_size + rhs_size;

  typet t = type_handler_.get_typet("str", total_size.to_uint64());
  exprt result = gen_zero(t);
  unsigned int i = 0;

  auto append_from_symbol = [&](const std::string &id) {
    symbolt *symbol = symbol_table_.find_symbol(id);
    assert(symbol);
    for (const exprt &ch : symbol->value.operands())
      result.operands().at(i++) = ch;
  };

  auto append_from_json = [&](const nlohmann::json &json) {
    std::string value = json["value"].get<std::string>();
    typet &char_type = t.subtype();

    for (char ch : value)
    {
      BigInt v(ch);
      exprt char_expr = constant_exprt(
        integer2binary(v, bv_width(char_type)), integer2string(v), char_type);
      result.operands().at(i++) = char_expr;
    }
  };

  if (left["_type"] == "Name")
    append_from_symbol(lhs.identifier().as_string());
  else if (left["_type"] == "Constant")
    append_from_json(left);

  if (right["_type"] == "Name")
    append_from_symbol(rhs.identifier().as_string());
  else if (right["_type"] == "Constant")
    append_from_json(right);

  return result;
}

bool python_converter::is_zero_length_array(const exprt &expr)
{
  if (!expr.type().is_array())
    return false;

  const auto &arr_type = to_array_type(expr.type());
  if (!arr_type.size().is_constant())
    return false;

  BigInt size = binary2integer(arr_type.size().value().as_string(), false);
  return size == 0;
}

exprt python_converter::handle_string_comparison(
  const std::string &op,
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &element)
{
  if (lhs.type() != rhs.type())
    return gen_boolean(op == "NotEq");

  if (lhs.is_constant() && rhs.is_constant() && lhs == rhs)
    return gen_boolean(op == "Eq");

  if (is_zero_length_array(lhs) && is_zero_length_array(rhs))
    return gen_boolean(op == "Eq");

  const auto &array_type = to_array_type(lhs.type());
  BigInt string_size =
    binary2integer(array_type.size().value().as_string(), false);

  symbolt *strncmp_symbol = symbol_table_.find_symbol("c:@F@strncmp");
  assert(strncmp_symbol);

  side_effect_expr_function_callt strncmp_call;
  strncmp_call.function() = symbol_expr(*strncmp_symbol);
  strncmp_call.arguments() = {
    lhs, rhs, from_integer(string_size, long_uint_type())};
  strncmp_call.location() = get_location_from_decl(element);
  strncmp_call.type() = int_type();

  lhs = strncmp_call;
  rhs = gen_zero(int_type());

  return nil_exprt(); // continue with lhs OP rhs
}

void python_converter::ensure_string_array(exprt &expr)
{
  if (!expr.type().is_array())
  {
    typet t = type_handler_.build_array(expr.type(), 1);
    exprt arr = gen_zero(t);
    arr.operands().at(0) = expr;
    expr = arr;
  }
}

exprt python_converter::handle_string_operations(
  const std::string &op,
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &left,
  const nlohmann::json &right,
  const nlohmann::json &element)
{
  ensure_string_array(lhs);
  ensure_string_array(rhs);

  assert(lhs.type().is_array());
  assert(rhs.type().is_array());

  if (op == "Eq" || op == "NotEq")
    return handle_string_comparison(op, lhs, rhs, element);

  if (op == "Add")
    return handle_string_concatenation(lhs, rhs, left, right);

  return nil_exprt();
}

exprt python_converter::get_binary_operator_expr(const nlohmann::json &element)
{
  auto left = (element.contains("left")) ? element["left"] : element["target"];

  decltype(left) right;
  if (element.contains("right"))
    right = element["right"];
  else if (element.contains("comparators"))
    right = element["comparators"][0];
  else if (element.contains("value"))
    right = element["value"];

  exprt lhs = get_expr(left);
  exprt rhs = get_expr(right);

  attach_symbol_location(lhs, symbol_table());
  attach_symbol_location(rhs, symbol_table());

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

  // Get LHS and RHS types
  std::string lhs_type = type_handler_.type_to_string(lhs.type());
  std::string rhs_type = type_handler_.type_to_string(rhs.type());

  // Infer the missing operand type if one side is explicitly a string and the operation is Eq or NotEq
  if (
    (op == "Eq" || op == "NotEq") && ((lhs_type.empty() && rhs_type == "str") ||
                                      (rhs_type.empty() && lhs_type == "str")))
  {
    // Infer lhs_type if it is empty
    if (lhs_type.empty() && element.contains("left"))
    {
      const auto &lhs_expr = element["left"];
      if (lhs_expr.contains("value") && lhs_expr["value"].is_string())
        lhs_type = "str";
      else if (lhs_expr.contains("id") && lhs_expr["id"].is_string())
        lhs_type = "str";
    }
    // Infer rhs_type if it is empty
    else if (
      rhs_type.empty() && element.contains("comparators") &&
      element["comparators"].is_array() && !element["comparators"].empty())
    {
      const auto &rhs_expr = element["comparators"][0];
      if (rhs_expr.contains("value") && rhs_expr["value"].is_string())
        rhs_type = "str";
      else if (rhs_expr.contains("id") && rhs_expr["id"].is_string())
        rhs_type = "str";
    }
  }

  if (lhs_type == "str" && rhs_type == "str")
  {
    const exprt &result =
      handle_string_operations(op, lhs, rhs, left, right, element);
    if (!result.is_nil())
      return result;
  }

  // Replace ** operation with the resultant constant.
  if (op == "Pow" || op == "power")
    return handle_power_operator(lhs, rhs);

  // Determine the result type of the binary operation:
  // If it's a relational operation (e.g., <, >, ==), the result is a boolean type.
  // Otherwise, it inherits the type of the left-hand side (lhs).
  typet type = (is_relational_op(op)) ? bool_type() : lhs.type();

  // Create a binary expression node with the determined type and location.
  exprt bin_expr(get_op(op, type), type);
  if (lhs.is_symbol())
    bin_expr.location() = lhs.location();
  else if (rhs.is_symbol())
    bin_expr.location() = rhs.location();

  // Handle type promotion for mixed signed/unsigned comparisons:
  // If lhs is unsigned and rhs is signed, convert rhs to match lhs's type.
  // This prevents signed-unsigned comparison issues.
  if (lhs.type().is_unsignedbv() && rhs.type().is_signedbv())
    rhs.make_typecast(lhs.type());

  // Promote both operands to float for Python-style true division ("/")
  if (lhs.type().is_floatbv() && (op == "Div" || op == "div"))
    handle_float_division(lhs, rhs, bin_expr);

  // Add lhs and rhs as operands to the binary expression.
  bin_expr.copy_to_operands(lhs, rhs);

  // According to the Python 3 Language Reference (Section 6.10, "Comparisons"):
  // - Equality comparisons (==, !=) between built-in incompatible types like float and str do not raise an exception.
  //   Instead, they typically evaluate to False for `==` and True for `!=`.
  // - However, ordered comparisons (<, <=, >, >=) between incompatible types such as float and str
  //   raise a TypeError at runtime.
  //
  // References:
  // - https://docs.python.org/3/reference/expressions.html#comparisons
  // - https://docs.python.org/3/library/stdtypes.html#typeerror
  //
  // This block emulates that behavior in ESBMC's symbolic execution: converting equality comparisons to
  // constants (false or true), and rejecting invalid ordered comparisons with a runtime error.
  if (is_float_vs_char(lhs, rhs))
    return handle_float_vs_string(bin_expr, op);

  // floor division (//) operation corresponds to an int division with floor rounding
  // So we need to emulate this behavior here:
  // int result = (num/div) - (num%div != 0 && ((num < 0) ^ (den<0)) ? 1 : 0)
  // e.g.: -5//2 equals to -3, and 5//2 equals to 2
  if (op == "FloorDiv")
    return handle_floor_division(lhs, rhs, bin_expr);

  // Handle chained comparisons like: assert 0 <= x <= 1
  if (element.contains("comparators") && element["comparators"].size() > 1)
  {
    exprt cond("and", bool_type());
    cond.move_to_operands(
      bin_expr); // bin_expr compares left and comparators[0]
    for (size_t i = 0; i + 1 < element["comparators"].size(); i += 2)
    {
      std::string op(element["ops"][i + 1]["_type"].get<std::string>());
      exprt logical_expr(get_op(op, bool_type()), bool_type());
      exprt operand = get_expr(element["comparators"][i]);
      logical_expr.copy_to_operands(operand);
      operand = get_expr(element["comparators"][i + 1]);
      logical_expr.copy_to_operands(operand);

      cond.move_to_operands(logical_expr);
    }
    return cond;
  }

  return bin_expr;
}

exprt python_converter::get_unary_operator_expr(const nlohmann::json &element)
{
  typet type = current_element_type;
  if (
    element["operand"].contains("value") &&
    element["operand"]["_type"] == "Constant")
  {
    type = type_handler_.get_typet(element["operand"]["value"]);
  }
  else if (element["operand"]["_type"] == "Name")
  {
    const std::string var_type =
      type_handler_.get_var_type(element["operand"]["id"].get<std::string>());
    type = type_handler_.get_typet(var_type);
  }

  exprt unary_expr(
    get_op(element["op"]["_type"].get<std::string>(), type), type);

  // get subexpr
  exprt unary_sub = get_expr(element["operand"]);
  unary_expr.operands().push_back(unary_sub);

  return unary_expr;
}

locationt
python_converter::get_location_from_decl(const nlohmann::json &ast_node)
{
  locationt location;
  if (ast_node.contains("lineno"))
    location.set_line(ast_node["lineno"].get<int>());

  if (ast_node.contains("col_offset"))
    location.set_column(ast_node["col_offset"].get<int>());

  location.set_file(current_python_file.c_str());
  location.set_function(current_func_name_);
  return location;
}

symbolt *python_converter::find_function_in_base_classes(
  const std::string &class_name,
  const std::string &symbol_id,
  std::string method_name,
  bool is_ctor) const
{
  symbolt *func = nullptr;

  // Find class node in the AST
  auto class_node = json_utils::find_class(ast_json["body"], class_name);

  if (class_node != nlohmann::json())
  {
    std::string current_class = class_name;
    std::string current_func_name = (is_ctor) ? class_name : method_name;
    std::string sym_id = symbol_id;
    // Search for method in all bases classes
    for (const auto &base_class_node : class_node["bases"])
    {
      const std::string &base_class = base_class_node["id"].get<std::string>();
      if (is_ctor)
        method_name = base_class;

      std::size_t pos = sym_id.rfind("@C@" + current_class);

      sym_id.replace(
        pos,
        std::string("@C@" + current_class + "@F@" + current_func_name).length(),
        std::string("@C@" + base_class + "@F@" + method_name));

      if ((func = symbol_table_.find_symbol(sym_id.c_str())))
        return func;

      current_class = base_class;
    }
  }

  return func;
}

symbolt *
python_converter::find_imported_symbol(const std::string &symbol_id) const
{
  for (const auto &obj : ast_json["body"])
  {
    if (obj["_type"] == "ImportFrom" || obj["_type"] == "Import")
    {
      std::regex pattern("py:(.*?)@");
      std::string imported_symbol = std::regex_replace(
        symbol_id, pattern, "py:" + obj["full_path"].get<std::string>() + "@");

      if (
        symbolt *func_symbol =
          symbol_table_.find_symbol(imported_symbol.c_str()))
        return func_symbol;
    }
  }
  return nullptr;
}

symbolt *python_converter::find_symbol(const std::string &symbol_id) const
{
  if (symbolt *symbol = symbol_table_.find_symbol(symbol_id))
    return symbol;
  if (symbolt *symbol = find_symbol_in_global_scope(symbol_id))
    return symbol;
  return find_imported_symbol(symbol_id);
}

symbolt *python_converter::find_symbol_in_global_scope(
  const std::string &symbol_id) const
{
  std::size_t class_start_pos = symbol_id.find("@C@");
  std::size_t func_start_pos = symbol_id.find("@F@");
  std::string sid = symbol_id;

  // Remove class name from symbol
  if (class_start_pos != std::string::npos)
    sid.erase(class_start_pos, func_start_pos - class_start_pos);

  func_start_pos = sid.find("@F@");
  std::size_t func_end_pos = sid.rfind("@");

  // Remove function name from symbol
  if (func_start_pos != std::string::npos)
    sid.erase(func_start_pos, func_end_pos - func_start_pos);

  return symbol_table_.find_symbol(sid);
}

bool python_converter::is_imported_module(const std::string &module_name) const
{
  if (imported_modules.find(module_name) != imported_modules.end())
    return true;

  return json_utils::is_module(module_name, ast_json);
}

exprt python_converter::get_function_call(const nlohmann::json &element)
{
  if (!element.contains("func") || element["_type"] != "Call")
  {
    throw std::runtime_error("Invalid function call");
  }

  const std::string function = config.options.get_option("function");
  // To verify a specific function, it is necessary to load the definitions of functions it calls.
  if (!function.empty() && !is_loading_models)
  {
    std::string func_name("");
    if (element["func"]["_type"] == "Name")
      func_name = element["func"]["id"];
    else if (element["func"]["_type"] == "Attribute")
      func_name = element["func"]["attr"];

    if (
      !type_utils::is_builtin_type(func_name) &&
      !type_utils::is_consensus_type(func_name) &&
      !type_utils::is_consensus_func(func_name) &&
      !type_utils::is_python_model_func(func_name) &&
      !is_class(func_name, ast_json))
    {
      const auto &func_node = find_function(ast_json["body"], func_name);
      assert(!func_node.empty());
      get_function_definition(func_node);
    }
  }

  function_call_builder call_builder(*this, element);
  return call_builder.build();
}

exprt python_converter::make_char_array_expr(
  const std::vector<unsigned char> &string_literal,
  const typet &t)
{
  exprt expr = gen_zero(t);
  const typet &char_type = t.subtype();

  for (size_t i = 0; i < string_literal.size(); ++i)
  {
    uint8_t ch = string_literal[i];
    exprt char_value = constant_exprt(
      integer2binary(BigInt(ch), bv_width(char_type)),
      integer2string(BigInt(ch)),
      char_type);
    expr.operands().at(i) = char_value;
  }

  return expr;
}

/// Convert a Python AST literal element to an expression.
/// This function handles various Python 3 literal types, including:
///   - Integers (e.g., `42`)
///   - Booleans (`True`, `False`)
///   - Floats (e.g., `3.14`)
///   - Characters (e.g., `'a'`)
///   - Strings (e.g., `"hello"`)
///   - Byte literals (e.g., `b"data"`)
///   - Ignores docstrings or other unsupported formats
///
/// Example JSON input:
/// { "_type": "Constant", "value": 42 }               → returns integer constant expr
/// { "_type": "Constant", "value": "a" }              → returns char literal expr
/// { "_type": "Constant", "value": "hello" }          → returns string literal expr
/// { "_type": "Constant", "value": true }             → returns boolean expr
/// { "_type": "UnaryOp", "op": "USub", "operand":
/// { "_type": "Constant", "value": 42 } }           → returns -42 as integer expr
exprt python_converter::get_literal(const nlohmann::json &element)
{
  // Determine the source of the literal's value.
  const auto &value = (element["_type"] == "UnaryOp")
                        ? element["operand"]["value"]
                        : element["value"];

  // Handle integer literals (int)
  if (value.is_number_integer())
    return from_integer(value.get<long long>(), long_long_int_type());

  // Handle boolean literals (True/False)
  if (value.is_boolean())
    return gen_boolean(value.get<bool>());

  // Handle floating-point literals (float)
  if (value.is_number_float())
  {
    exprt expr;
    convert_float_literal(
      value.dump(), expr); // `value.dump()` converts it to string
    return expr;
  }

  // Handle single-character string as char literal
  if (value.is_string() && value.get<std::string>().size() == 1)
  {
    const std::string &str = value.get<std::string>();
    typet t = type_handler_.get_typet("str", str.size());
    return from_integer(static_cast<unsigned char>(str[0]), t);
  }

  // Handle empty strings or docstrings (often beginning with a newline)
  if (
    value.is_string() && !value.get<std::string>().empty() &&
    value.get<std::string>()[0] == '\n')
  {
    return exprt(); // Return empty expression
  }

  // Handle string or byte literals
  if (value.is_string())
  {
    typet t = current_element_type;
    std::vector<uint8_t> string_literal;

    // Detect and decode "bytes" literal from encoded base64 content
    if (element.contains("encoded_bytes"))
    {
      string_literal =
        base64_decode(element["encoded_bytes"].get<std::string>());
    }
    else // Handle Python str literals
    {
      const std::string &str_val = value.get<std::string>();
      t = type_handler_.get_typet("str", str_val.size());
      string_literal.assign(str_val.begin(), str_val.end());
    }

    exprt expr = make_char_array_expr(string_literal, t);

    return expr;
  }

  throw std::runtime_error("Unsupported literal " + value.get<std::string>());
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
    expr = get_literal(element);
    break;
  }
  case ExpressionType::LIST:
  {
    exprt zero = gen_zero(size_type());
    exprt &list_size = static_cast<array_typet &>(current_element_type).size();
    typet list_type = type_handler_.get_list_type(element);

    if (list_size == zero)
    {
      current_element_type = list_type;
    }

    expr = gen_zero(list_type);

    unsigned int i = 0;
    for (auto &e : element["elts"])
    {
      if (e["_type"] == "List")
        expr.operands().at(i++) = get_expr(e);
      else
        expr.operands().at(i++) = get_expr(e);
    }

    break;
  }
  case ExpressionType::VARIABLE_REF:
  {
    std::string var_name;
    bool is_class_attr = false;
    if (element["_type"] == "Name")
    {
      var_name = element["id"].get<std::string>();
    }
    else if (element["_type"] == "Attribute")
    {
      var_name = element["value"]["id"].get<std::string>();
      if (is_class(var_name, ast_json))
      {
        // Found a class attribute
        var_name = "C@" + var_name;
        is_class_attr = true;
      }
    }

    assert(!var_name.empty());

    symbol_id sid = create_symbol_id();
    sid.set_object(var_name);

    if (element.contains("attr") && is_class_attr)
      sid.set_attribute(element["attr"].get<std::string>());

    std::string sid_str = sid.to_string();

    symbolt *symbol = nullptr;
    if (!(symbol = find_symbol(sid_str)))
    {
      // Fallback for global variables accessed inside functions
      if (!is_class_attr && element["_type"] == "Name")
      {
        sid.set_function(""); // remove function scope
        sid_str = sid.to_string();
        symbol = find_symbol(sid_str);
      }
      if (!symbol)
      {
        log_error("Symbol not found {}", sid_str);
        abort();
      }
    }

    expr = symbol_expr(*symbol);

    // Get instance attribute
    if (!is_class_attr && element["_type"] == "Attribute")
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
      symbolt *class_symbol = symbol_table_.find_symbol(obj_type_name);
      if (!class_symbol)
      {
        throw std::runtime_error("Class \"" + obj_type_name + "\" not found");
      }

      struct_typet &class_type =
        static_cast<struct_typet &>(class_symbol->type);

      if (is_converting_lhs)
      {
        // Add member in the class
        if (!class_type.has_component(attr_name))
        {
          struct_typet::componentt comp = build_component(
            class_type.tag().as_string(), attr_name, current_element_type);
          class_type.components().push_back(comp);
        }
        // Add instance attribute in the objects map
        instance_attr_map[symbol->id.as_string()].insert(attr_name);
      }

      auto is_instance_attr = [&]() -> bool {
        auto it = instance_attr_map.find(symbol->id.as_string());
        if (it != instance_attr_map.end())
        {
          for (const auto &attr : it->second)
          {
            if (attr == attr_name)
              return true;
          }
        }
        return false;
      };

      // Get instance attribute from class component
      if (
        class_type.has_component(attr_name) &&
        (is_instance_attr() || is_converting_rhs))
      {
        const typet &attr_type = class_type.get_component(attr_name).type();
        expr = member_exprt(
          symbol_exprt(symbol->id, symbol->type), attr_name, attr_type);
      }
      // Fallback to class attribute when instance attribute is not found
      else
      {
        // All class attributes are static symbols with ids in the format: filename@C@classname@varname
        sid.set_function("");
        sid.set_class(obj_type_name.substr(4));
        sid.set_object(attr_name);
        symbolt *class_attr_symbol = symbol_table_.find_symbol(sid.to_string());

        if (!class_attr_symbol)
        {
          throw std::runtime_error("Attribute \"" + attr_name + "\" not found");
        }
        expr = symbol_expr(*class_attr_symbol);
      }
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
  case ExpressionType::SUBSCRIPT:
  {
    exprt array = get_expr(element["value"]);
    typet t = array.type().subtype();

    const nlohmann::json &slice = element["slice"];

    if (slice["_type"] == "Slice")
    {
      const size_t &upper = slice["upper"]["value"].get<size_t>();
      const size_t &lower = slice["lower"]["value"].get<size_t>();

      typet list_type = type_handler_.build_array(t, upper - lower);

      expr = constant_exprt(list_type);

      const auto &list = json_utils::find_var_decl(
        element["value"]["id"], current_func_name_, ast_json);

      assert(!list.empty());

      const auto &list_elts = list["value"]["elts"];

      for (size_t j = lower; j < upper; ++j)
        expr.operands().push_back(get_expr(list_elts[j]));
    }
    else
    {
      exprt pos = get_expr(slice);

      // Adjust negative indexes
      if (slice.contains("op") && slice["op"]["_type"] == "USub")
      {
        BigInt v = binary2integer(pos.op0().value().c_str(), true);
        v *= -1;

        array_typet t = static_cast<array_typet &>(array.type());
        BigInt s = binary2integer(t.size().value().c_str(), true);

        v += s;
        pos = from_integer(v, pos.type());
      }

      expr = index_exprt(array, pos, t);
    }
    break;
  }
  default:
  {
    const auto &lineno = element["lineno"].template get<int>();
    std::ostringstream oss;
    oss << "Unsupported expression ";
    if (element.contains("_type"))
      oss << element["_type"].get<std::string>();

    oss << " at line " << lineno;
    throw std::runtime_error(oss.str());
  }
  }

  return expr;
}

void python_converter::update_instance_from_self(
  const std::string &class_name,
  const std::string &func_name,
  const std::string &obj_symbol_id)
{
  symbol_id sid(current_python_file, class_name, func_name);
  sid.set_object("self");

  auto self_instance = instance_attr_map.find(sid.to_string());

  if (self_instance != instance_attr_map.end())
  {
    std::set<std::string> &attr_list = instance_attr_map[obj_symbol_id];
    attr_list.insert(
      self_instance->second.begin(), self_instance->second.end());
  }
}

size_t get_type_size(const nlohmann::json &ast_node)
{
  size_t type_size = 0;
  if (ast_node["value"].contains("value"))
  {
    if (ast_node["annotation"]["id"] == "bytes")
    {
      const std::string &str =
        ast_node["value"]["encoded_bytes"].get<std::string>();
      std::vector<uint8_t> decoded = base64_decode(str);
      type_size = decoded.size();
    }
    else if (ast_node["value"]["value"].is_string())
      type_size = ast_node["value"]["value"].get<std::string>().size();
  }
  else if (
    ast_node["value"].contains("args") &&
    ast_node["value"]["args"].size() > 0 &&
    ast_node["value"]["args"][0].contains("value") &&
    ast_node["value"]["args"][0]["value"].is_string())
    type_size = ast_node["value"]["args"][0]["value"].get<std::string>().size();
  else if (
    ast_node["value"].contains("_type") && ast_node["value"]["_type"] == "List")
  {
    type_size = ast_node["value"]["elts"].size();
  }

  return type_size;
}

const nlohmann::json &get_return_statement(const nlohmann::json &function)
{
  for (const auto &stmt : function["body"])
  {
    if (get_statement_type(stmt) == StatementType::RETURN)
      return stmt;
  }

  throw std::runtime_error(
    "Function " + function["name"].get<std::string>() +
    " has no return statement");
}

void python_converter::get_var_assign(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  std::string lhs_type("");
  if (ast_node.contains("annotation"))
  {
    // Get type from annotation node
    size_t type_size = get_type_size(ast_node);
    if (ast_node["annotation"]["_type"] == "Subscript")
      lhs_type = ast_node["annotation"]["value"]["id"];
    else
      lhs_type = ast_node["annotation"]["id"];

    if (lhs_type == "list")
      current_element_type = type_handler_.get_list_type(ast_node["value"]);
    else
      current_element_type = type_handler_.get_typet(lhs_type, type_size);
  }

  exprt lhs;
  symbolt *lhs_symbol = nullptr;
  locationt location_begin;
  symbol_id sid = create_symbol_id();

  const auto &target = (ast_node.contains("targets")) ? ast_node["targets"][0]
                                                      : ast_node["target"];
  const auto &target_type = target["_type"];

  if (ast_node["_type"] == "AnnAssign")
  {
    // Id and name
    std::string name;

    if (!target.is_null())
    {
      if (target_type == "Name")
        name = target["id"];
      else if (target_type == "Attribute")
        name = target["attr"];
      else if (target_type == "Subscript")
        name = target["value"]["id"];

      assert(!name.empty());

      sid.set_object(name);
    }

    // Location
    location_begin = get_location_from_decl(target);

    lhs_symbol = symbol_table_.find_symbol(sid.to_string().c_str());

    if (!lhs_symbol)
    {
      // Debug module name
      std::string module_name = location_begin.get_file().as_string();

      // Create/init symbol
      symbolt symbol = create_symbol(
        module_name,
        name,
        sid.to_string(),
        location_begin,
        current_element_type);
      symbol.lvalue = true;
      symbol.static_lifetime = false;
      symbol.file_local = true;
      symbol.is_extern = false;

      lhs_symbol = symbol_table_.move_symbol_to_context(symbol);
    }

    if (target_type == "Attribute" || target_type == "Subscript")
    {
      is_converting_lhs = true;
      lhs = get_expr(target); // lhs is a obj.member expression
    }
    else
      lhs = symbol_expr(*lhs_symbol); // lhs is a simple variable

    lhs.location() = location_begin;
  }
  else if (ast_node["_type"] == "Assign")
  {
    const auto &target = ast_node["targets"][0];

    const auto &name = (target["_type"] == "Subscript")
                         ? target["value"]["id"].get<std::string>()
                         : target["id"].get<std::string>();

    sid.set_object(name);
    lhs_symbol = symbol_table_.find_symbol(sid.to_string());

    if (!lhs_symbol)
      throw std::runtime_error("Type undefined for \"" + name + "\"");

    lhs = symbol_expr(*lhs_symbol);
  }

  bool is_ctor_call = type_handler_.is_constructor_call(ast_node["value"]);

  current_lhs = &lhs;

  is_converting_lhs = false;

  // Get RHS
  exprt rhs;
  bool has_value = false;
  if (!ast_node["value"].is_null())
  {
    is_converting_rhs = true;
    rhs = get_expr(ast_node["value"]);
    has_value = true;
    is_converting_rhs = false;
  }

  if (has_value && rhs != exprt("_init_undefined"))
  {
    if (lhs_symbol)
    {
      if (
        lhs_type == "str" || lhs_type == "chr" || lhs_type == "ord" ||
        lhs_type == "list" || rhs.type().is_array())
      {
        /* When a string is assigned the result of a concatenation, we initially
         * create the LHS type as a zero-size array: "current_element_type = get_typet(lhs_type, type_size);"
         * After parsing the RHS, we need to adjust the LHS type size to match
         * the size of the resulting RHS string.*/
        lhs_symbol->type = rhs.type();
        lhs.type() = rhs.type();
      }
      lhs_symbol->value = rhs;
    }

    /* If the right-hand side (rhs) of the assignment is a function call, such as: x : int = func()
     * we need to adjust the left-hand side (lhs) of the function call to refer to the lhs of the current assignment.
     */
    if (rhs.is_function_call())
    {
      // If rhs is a constructor call so it is necessary to update lhs instance attributes with members added in self
      if (is_ctor_call)
      {
        std::string func_name = ast_node["value"]["func"]["id"];

        if (base_ctor_called)
        {
          auto class_node = json_utils::find_class(ast_json["body"], func_name);
          func_name = class_node["bases"][0]["id"].get<std::string>();
          base_ctor_called = false;
        }

        update_instance_from_self(
          func_name, func_name, lhs_symbol->id.as_string());
      }
      // op0() refers to the left-hand side (lhs) of the function call
      rhs.op0() = lhs;
      target_block.copy_to_operands(rhs);
      return;
    }

    adjust_statement_types(lhs, rhs);

    code_assignt code_assign(lhs, rhs);
    code_assign.location() = location_begin;
    target_block.copy_to_operands(code_assign);
  }
  else
  {
    lhs_symbol->value = gen_zero(current_element_type, true);
    lhs_symbol->value.zero_initializer(true);

    code_declt decl(symbol_expr(*lhs_symbol));
    decl.location() = location_begin;
    target_block.copy_to_operands(decl);
  }

  current_lhs = nullptr;
}

void python_converter::get_compound_assign(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  // Get type from declaration node
  std::string var_name = ast_node["target"]["id"].get<std::string>();
  nlohmann::json ref = get_var_node(var_name, ast_json);
  assert(!ref.empty());
  current_element_type =
    type_handler_.get_typet(ref["annotation"]["id"].get<std::string>());

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
  const nlohmann::json &return_node = function_node["returns"];

  if (
    return_node.is_null() ||
    (return_node["_type"] == "Constant" && return_node["value"].is_null()))
  {
    type.return_type() = empty_typet();
  }
  else if (return_node.contains("id") || return_node["_type"] == "Subscript")
  {
    const nlohmann::json &return_type = (return_node["_type"] == "Subscript")
                                          ? return_node["value"]["id"]
                                          : return_node["id"];
    if (return_type == "list")
    {
      const auto &return_stmt = get_return_statement(function_node);
      if (
        return_stmt["value"]["_type"] == "Name" ||
        return_stmt["value"]["_type"] == "Subscript")
      {
        const std::string &var_name =
          (return_stmt["value"]["_type"].get<std::string>() == "Subscript")
            ? return_stmt["value"]["value"]["id"].get<std::string>()
            : return_stmt["value"]["id"].get<std::string>();

        const auto &return_var = get_var_node(var_name, function_node);

        assert(!return_var.empty());

        if (return_var["_type"] == "arg")
        {
          if (return_stmt["value"]["_type"] == "Subscript")
            type.return_type() = type_handler_.get_typet(
              return_var["annotation"]["slice"]["id"].get<std::string>());
          else
            type.return_type() = type_handler_.get_list_type(return_var);
        }
        else
          type.return_type() = type_handler_.get_list_type(return_var["value"]);
      }
    }
    else
    {
      type.return_type() =
        type_handler_.get_typet(return_node["id"].get<std::string>());
    }
  }
  else
  {
    throw std::runtime_error("Return type undefined");
  }

  // Copy caller function name
  const std::string caller_func_name = current_func_name_;

  // Function location
  locationt location = get_location_from_decl(function_node);

  current_element_type = type.return_type();
  current_func_name_ = function_node["name"].get<std::string>();

  // __init__() is renamed to Classname()
  if (current_func_name_ == "__init__")
  {
    current_func_name_ = current_class_name_;
    typet ctor_type("constructor");
    type.return_type() = ctor_type;
  }

  symbol_id id = create_symbol_id();

  std::string module_name =
    current_python_file.substr(0, current_python_file.find_last_of("."));

  // Iterate over function arguments
  for (const nlohmann::json &element : function_node["args"]["args"])
  {
    // Argument name
    std::string arg_name = element["arg"].get<std::string>();
    // Argument type
    typet arg_type;
    if (arg_name == "self")
      arg_type = gen_pointer_type(type_handler_.get_typet(current_class_name_));
    else if (arg_name == "cls")
      arg_type = pointer_typet(empty_typet());
    else
    {
      if (!element.contains("annotation") || element["annotation"].is_null())
      {
        throw std::runtime_error(
          "All parameters in function \"" + current_func_name_ +
          "\" must be type annotated");
      }

      if (element["annotation"]["_type"] == "Subscript")
        arg_type = type_handler_.get_list_type(element);
      else
        arg_type = type_handler_.get_typet(
          element["annotation"]["id"].get<std::string>());
    }

    if (arg_type.is_array())
      arg_type = gen_pointer_type(arg_type.subtype());

    assert(arg_type != typet());

    code_typet::argumentt arg;
    arg.type() = arg_type;

    arg.cmt_base_name(arg_name);

    // Argument id
    std::string arg_id = id.to_string() + "@" + arg_name;
    arg.cmt_identifier(arg_id);

    // Location
    locationt location = get_location_from_decl(element);
    arg.location() = location;

    // Push arg
    type.arguments().push_back(arg);

    // Create and add symbol to symbol_table_
    symbolt param_symbol = create_symbol(
      location.get_file().as_string(), arg_name, arg_id, location, arg_type);
    param_symbol.lvalue = true;
    param_symbol.is_parameter = true;
    param_symbol.file_local = true;
    param_symbol.static_lifetime = false;
    param_symbol.is_extern = false;
    symbol_table_.add(param_symbol);
  }

  // Create symbol
  symbolt symbol = create_symbol(
    module_name, current_func_name_, id.to_string(), location, type);
  symbol.lvalue = true;
  symbol.is_extern = false;
  symbol.file_local = false;

  symbolt *added_symbol = symbol_table_.move_symbol_to_context(symbol);

  // Function body
  exprt function_body = get_block(function_node["body"]);
  added_symbol->value = function_body;

  // Restore caller function name
  current_func_name_ = caller_func_name;
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
      typet type =
        type_handler_.get_typet(stmt["annotation"]["id"].get<std::string>());
      struct_typet::componentt comp =
        build_component(current_class_name_, attr_name, type);

      auto &class_components = clazz.components();
      if (
        std::find(class_components.begin(), class_components.end(), comp) ==
        class_components.end())
        class_components.push_back(comp);
    }
  }
}

void python_converter::get_class_definition(
  const nlohmann::json &class_node,
  codet &target_block)
{
  struct_typet clazz;
  current_class_name_ = class_node["name"].get<std::string>();
  clazz.tag(current_class_name_);
  std::string id = "tag-" + current_class_name_;

  if (symbol_table_.find_symbol(id) != nullptr)
    return;

  locationt location_begin = get_location_from_decl(class_node);
  std::string module_name = location_begin.get_file().as_string();

  // Add class to symbol table
  symbolt symbol =
    create_symbol(module_name, current_class_name_, id, location_begin, clazz);
  symbol.is_type = true;

  symbolt *added_symbol = symbol_table_.move_symbol_to_context(symbol);

  // Iterate over base classes
  for (auto &base_class : class_node["bases"])
  {
    const std::string &base_class_name = base_class["id"].get<std::string>();
    /* TODO: Define OMs for built-in type classes.
     * This will allow us to add their definitions to the symbol_table_
     * inherit from them, and extend their functionality. */
    if (
      type_utils::is_builtin_type(base_class_name) ||
      type_utils::is_consensus_type(base_class_name))
      continue;

    // Get class definition from symbols table
    symbolt *class_symbol = symbol_table_.find_symbol("tag-" + base_class_name);
    if (!class_symbol)
    {
      throw std::runtime_error("Base class not found: " + base_class_name);
    }

    struct_typet &class_type = static_cast<struct_typet &>(class_symbol->type);
    for (const auto &component : class_type.components())
      clazz.components().emplace_back(component);
  }

  // Iterate over class members
  for (auto &class_member : class_node["body"])
  {
    // Process methods
    if (class_member["_type"] == "FunctionDef")
    {
      get_attributes_from_self(class_member["body"], clazz);
      added_symbol->type = clazz;

      std::string method_name = class_member["name"].get<std::string>();
      if (method_name == "__init__")
        method_name = current_class_name_;

      current_func_name_ = method_name;
      get_function_definition(class_member);

      exprt added_method =
        symbol_expr(*symbol_table_.find_symbol(create_symbol_id().to_string()));

      struct_typet::componentt method(added_method.name(), added_method.type());
      clazz.methods().push_back(method);
      current_func_name_.clear();
    }
    // Process class attributes
    else if (class_member["_type"] == "AnnAssign")
    {
      /* Ensure the attribute's type is defined by checking for its symbol.
       * If the symbol for the type is not found, attempt to locate
       * the class definition in the AST and convert it if available. */
      const std::string &class_name = class_member["annotation"]["id"];
      if (!symbol_table_.find_symbol("tag-" + class_name))
      {
        const auto &class_node = find_class(ast_json["body"], class_name);
        if (!class_node.empty())
        {
          std::string current_class = current_class_name_;
          get_class_definition(class_node, target_block);
          current_class_name_ = current_class;
        }
      }

      get_var_assign(class_member, target_block);

      symbol_id sid = create_symbol_id();
      sid.set_object(class_member["target"]["id"].get<std::string>());
      symbolt *class_attr_symbol = symbol_table_.find_symbol(sid.to_string());

      if (!class_attr_symbol)
        throw std::runtime_error("Class attribute not found");

      class_attr_symbol->static_lifetime = true;
    }
  }
  added_symbol->type = clazz;
  current_class_name_.clear();
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
      if (!test.type().is_bool())
        test.make_typecast(current_element_type);
      code_assertt assert_code;
      assert_code.assertion() = test;
      assert_code.location() = get_location_from_decl(element);
      block.move_to_operands(assert_code);
      break;
    }
    case StatementType::EXPR:
    {
      // Function calls are handled here
      exprt empty;
      exprt expr = get_expr(element["value"]);
      if (expr != empty)
        block.move_to_operands(expr);

      break;
    }
    case StatementType::CLASS_DEFINITION:
    {
      get_class_definition(element, block);
      break;
    }
    case StatementType::BREAK:
    {
      code_breakt break_expr;
      block.move_to_operands(break_expr);
      break;
    }
    case StatementType::CONTINUE:
    {
      code_continuet continue_expr;
      block.move_to_operands(continue_expr);
      break;
    }
    /* "https://docs.python.org/3/tutorial/controlflow.html:
     * "The pass statement does nothing. It can be used when a statement
     *  is required syntactically but the program requires no action." */
    case StatementType::PASS:
    // Imports are handled by parser.py so we can just ignore here.
    case StatementType::IMPORT:
    // TODO: Raises are ignored for now. Handling case to avoid calling abort() on default.
    case StatementType::RAISE:
      break;
    case StatementType::UNKNOWN:
    default:
      throw std::runtime_error(
        element["_type"].get<std::string>() + " statements are not supported");
    }
  }

  return block;
}

python_converter::python_converter(
  contextt &_context,
  const nlohmann::json &ast,
  const global_scope &gs)
  : symbol_table_(_context),
    ast_json(ast),
    global_scope_(gs),
    type_handler_(*this),
    ns(_context),
    current_func_name_(""),
    current_class_name_(""),
    current_lhs(nullptr)
{
}

void python_converter::append_models_from_directory(
  std::list<std::string> &file_list,
  const std::string &dir_path)
{
  fs::path directory(dir_path);

  // Checks if the directory exists
  if (!fs::exists(directory) || !fs::is_directory(directory))
    return;

  // Iterates over the files in the directory
  for (fs::directory_iterator it(directory), end_it; it != end_it; ++it)
  {
    if (fs::is_regular_file(*it) && it->path().extension() == ".json")
    {
      std::string file_name =
        directory.filename().string() + "/" +
        it->path().stem().string(); // File name without the extension
      file_list.push_back(file_name);

      imported_modules[it->path().stem().string()] = it->path().string();
    }
  }
}

void python_converter::load_c_intrisics()
{
  // Add symbols required by the C models

  typet t = long_long_int_type();
  locationt l;
  l.set_file("esbmc_intrinsics.h");
  symbolt symbol;
  symbol.mode = "C";
  symbol.module = "esbmc_intrinsics";
  symbol.location = l;
  symbol.type = t;
  symbol.name = "__ESBMC_rounding_mode";
  symbol.id = "c:@__ESBMC_rounding_mode";
  symbol.lvalue = true;
  symbol.static_lifetime = true;

  symbolt *new_symbol = symbol_table_.move_symbol_to_context(symbol);
  exprt value = from_integer(BigInt(0), t);
  new_symbol->value = value;
}

void python_converter::convert()
{
  code_typet main_type;
  main_type.return_type() = empty_typet();

  symbolt main_symbol;
  main_symbol.id = "__ESBMC_main";
  main_symbol.name = "__ESBMC_main";
  main_symbol.type.swap(main_type);
  main_symbol.lvalue = true;
  main_symbol.is_extern = false;
  main_symbol.file_local = false;

  main_python_file = ast_json["filename"].get<std::string>();
  current_python_file = main_python_file;

  if (!config.options.get_bool_option("no-library"))
  {
    // Load operational models -----
    const std::string &ast_output_dir =
      ast_json["ast_output_dir"].get<std::string>();
    std::list<std::string> model_files = {
      "range", "int", "consensus", "random"};
    std::list<std::string> model_folders = {"os"};

    for (const auto &folder : model_folders)
    {
      append_models_from_directory(model_files, ast_output_dir + "/" + folder);
    }

    is_loading_models = true;

    for (const auto &file : model_files)
    {
      std::stringstream model_path;
      model_path << ast_output_dir << "/" << file << ".json";

      std::ifstream model_file(model_path.str());
      nlohmann::json model_json;
      model_file >> model_json;
      model_file.close();

      size_t pos = file.rfind("/");
      if (pos != std::string::npos)
      {
        std::string filename = file.substr(pos + 1);
        if (imported_modules.find(filename) != imported_modules.end())
          current_python_file = imported_modules[filename];
      }

      exprt model_code = get_block(model_json["body"]);
      convert_expression_to_code(model_code);

      // Add imported code to main symbol
      main_symbol.value.swap(model_code);
      current_python_file = main_python_file;
    }
    is_loading_models = false;
  }

  // Load C intrisics
  load_c_intrisics();

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
      throw std::runtime_error("Function " + function + " not found");
    }

    code_blockt block;
    // Convert classes referenced by the function
    for (const auto &clazz : global_scope_.classes())
    {
      const auto &class_node = find_class(ast_json["body"], clazz);
      get_class_definition(class_node, block);
      current_class_name_.clear();
    }

    // Convert only the global variables referenced by the function
    for (const auto &global_var : global_scope_.variables())
    {
      const auto &var_node = find_var_decl(global_var, "", ast_json);
      get_var_assign(var_node, block);
    }

    // Convert function arguments types
    for (const auto &arg : function_node["args"]["args"])
    {
      auto node = find_class(ast_json["body"], arg["annotation"]["id"]);
      if (!node.empty())
        get_class_definition(node, block);
    }

    // Convert a single function
    get_function_definition(function_node);

    // Get function symbol
    symbol_id sid = create_symbol_id();
    sid.set_function(function);
    symbolt *symbol = symbol_table_.find_symbol(sid.to_string());

    if (!symbol)
    {
      throw std::runtime_error("Symbol " + sid.to_string() + " not found");
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

    convert_expression_to_code(call);
    convert_expression_to_code(block);

    main_symbol.value.swap(
      block); // Add class definitions and global variable assignments
    main_symbol.value.copy_to_operands(call); // Add function call
  }
  else
  {
    // Convert imported modules
    for (const auto &elem : ast_json["body"])
    {
      if (elem["_type"] == "ImportFrom" || elem["_type"] == "Import")
      {
        is_importing_module = true;
        const std::string &module_name = (elem["_type"] == "ImportFrom")
                                           ? elem["module"]
                                           : elem["names"][0]["name"];
        std::stringstream module_path;
        module_path << ast_json["ast_output_dir"].get<std::string>() << "/"
                    << module_name << ".json";
        std::ifstream imported_file(module_path.str());
        imported_file >> imported_module_json;

        current_python_file =
          imported_module_json["filename"].get<std::string>();
        imported_modules.emplace(module_name, current_python_file);

        exprt imported_code = get_block(imported_module_json["body"]);
        convert_expression_to_code(imported_code);

        // Add imported code to main symbol
        main_symbol.value.swap(imported_code);
        imported_module_json.clear();
      }
    }

    is_importing_module = false;
    current_python_file = main_python_file;

    // Convert main statements
    exprt main_block = get_block(ast_json["body"]);
    codet main_code = convert_expression_to_code(main_block);

    if (main_symbol.value.is_code())
      main_symbol.value.copy_to_operands(main_code);
    else
      main_symbol.value.swap(main_code);
  }

  if (symbol_table_.move(main_symbol))
  {
    throw std::runtime_error(
      "The main function is already defined in another module");
  }
}
