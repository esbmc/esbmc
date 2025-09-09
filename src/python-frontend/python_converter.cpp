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
#include <cmath>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <regex>
#include <sstream>
#include <unordered_map>

#include <boost/filesystem.hpp>

using namespace json_utils;
namespace fs = boost::filesystem;

static const std::unordered_map<std::string, std::string> operator_map = {
  {"add", "+"},         {"sub", "-"},         {"subtract", "-"},
  {"mult", "*"},        {"multiply", "*"},    {"dot", "*"},
  {"div", "/"},         {"divide", "/"},      {"mod", "mod"},
  {"bitor", "bitor"},   {"floordiv", "/"},    {"bitand", "bitand"},
  {"bitxor", "bitxor"}, {"invert", "bitnot"}, {"lshift", "shl"},
  {"rshift", "ashr"},   {"usub", "unary-"},   {"eq", "="},
  {"lt", "<"},          {"lte", "<="},        {"noteq", "notequal"},
  {"gt", ">"},          {"gte", ">="},        {"and", "and"},
  {"or", "or"},         {"not", "not"},       {"uadd", "unary+"},
  {"is", "="},          {"isnot", "not"}};

static const std::unordered_map<std::string, StatementType> statement_map = {
  {"AnnAssign", StatementType::VARIABLE_ASSIGN},
  {"Assign", StatementType::VARIABLE_ASSIGN},
  {"FunctionDef", StatementType::FUNC_DEFINITION},
  {"If", StatementType::IF_STATEMENT},
  {"AugAssign", StatementType::COMPOUND_ASSIGN},
  {"While", StatementType::WHILE_STATEMENT},
  {"For", StatementType::FOR_STATEMENT},
  {"Expr", StatementType::EXPR},
  {"Return", StatementType::RETURN},
  {"Assert", StatementType::ASSERT},
  {"ClassDef", StatementType::CLASS_DEFINITION},
  {"Pass", StatementType::PASS},
  {"Break", StatementType::BREAK},
  {"Continue", StatementType::CONTINUE},
  {"ImportFrom", StatementType::IMPORT},
  {"Import", StatementType::IMPORT},
  {"Raise", StatementType::RAISE},
  {"Global", StatementType::GLOBAL},
  {"Try", StatementType::TRY},
  {"ExceptHandler", StatementType::EXCEPTHANDLER}};

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
      {"dot", "ieee_mul"},
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

static codet convert_expression_to_code(exprt &expr)
{
  if (expr.is_code())
    return static_cast<codet &>(expr);

  codet code("expression");
  code.location() = expr.location();
  code.move_to_operands(expr);
  return code;
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
    {"List", ExpressionType::LIST},
    {"Lambda", ExpressionType::FUNC_CALL},
    {"JoinedStr", ExpressionType::FSTRING}};

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
  // Case 2: For Python assignments, if RHS is float but LHS is integer,
  // promote LHS to float to maintain Python's dynamic typing semantics
  else if (
    rhs_type.is_floatbv() &&
    (lhs_type.is_signedbv() || lhs_type.is_unsignedbv()))
  {
    // Update LHS variable type to match RHS float type
    lhs.type() = rhs_type;

    // Update symbol table if LHS is a symbol
    if (lhs.is_symbol())
      update_symbol(lhs);
  }
  // Case 3: Handles Python's / operator by promoting operands to floats
  // to ensure floating-point division, preventing division by zero, and
  // setting the result type to floatbv.
  else if (
    (rhs.id() == "/" || rhs.id() == "ieee_div") && rhs.operands().size() == 2)
  {
    auto &ops = rhs.operands();
    exprt &lhs_op = ops[0];
    exprt &rhs_op = ops[1];

    // Promote both operands to IEEE float (double precision) to match Python semantics
    const typet float_type =
      double_type(); // Python default float is double-precision

    // Handle constant operands
    if (
      lhs_op.is_constant() &&
      (lhs_op.type().is_signedbv() || lhs_op.type().is_unsignedbv()))
      promote_int_to_float(lhs_op, float_type);
    // For non-constant operands, create explicit typecast
    else if (!lhs_op.type().is_floatbv())
      lhs_op = typecast_exprt(lhs_op, float_type);

    if (
      rhs_op.is_constant() &&
      (rhs_op.type().is_signedbv() || rhs_op.type().is_unsignedbv()))
      promote_int_to_float(rhs_op, float_type);
    else if (!rhs_op.type().is_floatbv())
      rhs_op = typecast_exprt(rhs_op, float_type);

    // For in-place division (like x /= y), ensure LHS variable is promoted to float
    lhs.type() = float_type;
    if (lhs.is_symbol())
      update_symbol(lhs);

    // Update the division expression type and operator ID
    rhs.type() = float_type;
    rhs.id(get_op("div", float_type));
  }
  // Case 4: Special case for IEEE division results - ensure LHS is float
  else if (rhs.id() == "ieee_div" && !lhs_type.is_floatbv())
  {
    // For any IEEE division result assigned to an integer variable,
    // promote the variable to float to avoid truncation
    const typet float_type = double_type();
    lhs.type() = float_type;

    if (lhs.is_symbol())
      update_symbol(lhs);

    // Ensure RHS type is also float
    if (!rhs_type.is_floatbv())
      rhs.type() = float_type;
  }
  // Case 5: Align bit-widths between LHS and RHS if they differ
  else if (lhs_type.width() != rhs_type.width())
  {
    try
    {
      const int lhs_width = type_handler_.get_type_width(lhs_type);
      const int rhs_width = type_handler_.get_type_width(rhs_type);

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

inline bool is_ieee_op(const exprt &expr)
{
  const std::string &id = expr.id().as_string();
  return id == "ieee_add" || id == "ieee_mul" || id == "ieee_sub" ||
         id == "ieee_div";
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

exprt python_converter::handle_power_operator_sym(exprt base, exprt exp)
{
  // Find the pow function symbol
  symbolt *pow_symbol = symbol_table_.find_symbol("c:@F@pow");
  if (!pow_symbol)
    throw std::runtime_error("pow function not found in symbol table");

  // Convert arguments to double type if needed
  exprt double_base = base;
  exprt double_exp = exp;

  if (!base.type().is_floatbv())
  {
    double_base = exprt("typecast", double_type());
    double_base.copy_to_operands(base);
  }

  if (!exp.type().is_floatbv())
  {
    double_exp = exprt("typecast", double_type());
    double_exp.copy_to_operands(exp);
  }

  // Create the function call
  side_effect_expr_function_callt pow_call;
  pow_call.function() = symbol_expr(*pow_symbol);
  pow_call.arguments() = {double_base, double_exp};
  pow_call.type() = double_type();

  // Always return double result: Python power with float operands returns float
  return pow_call;
}

exprt python_converter::handle_power_operator(exprt lhs, exprt rhs)
{
  // Handle pow symbolically if one of the operands is floatbv
  if (lhs.type().is_floatbv() || rhs.type().is_floatbv())
    return handle_power_operator_sym(lhs, rhs);

  // Try to resolve constant values of both lhs and rhs
  exprt resolved_lhs = lhs;
  if (lhs.is_symbol())
  {
    const symbolt *s = symbol_table_.find_symbol(lhs.identifier());
    if (s && !s->value.value().empty())
      resolved_lhs = s->value;
  }
  else if (is_math_expr(lhs))
    resolved_lhs = compute_math_expr(lhs);

  exprt resolved_rhs = rhs;
  if (rhs.is_symbol())
  {
    const symbolt *s = symbol_table_.find_symbol(rhs.identifier());
    if (s && !s->value.value().empty())
      resolved_rhs = s->value;
  }
  else if (is_math_expr(rhs))
    resolved_rhs = compute_math_expr(rhs);

  // If rhs is still not constant, we need to handle this case
  if (!resolved_rhs.is_constant())
  {
    log_warning(
      "ESBMC-Python does not support power expressions with non-constant "
      "exponents");
    return from_integer(1, lhs.type());
  }

  // Check if the exponent is a floating-point number
  if (resolved_rhs.type().is_floatbv())
  {
    log_warning("ESBMC-Python does not support floating-point exponents yet");
    return from_integer(1, lhs.type());
  }

  // Convert rhs to integer exponent
  BigInt exponent;
  try
  {
    exponent = binary2integer(
      resolved_rhs.value().as_string(), resolved_rhs.type().is_signedbv());
  }
  catch (...)
  {
    log_warning("Failed to convert exponent to integer");
    return from_integer(1, lhs.type());
  }

  // Handle negative exponents more gracefully
  if (exponent < 0)
  {
    log_warning(
      "ESBMC-Python does not support power expressions with negative "
      "exponents, treating as symbolic");
    return from_integer(1, lhs.type());
  }

  // Handle special cases first
  if (exponent == 0)
    return from_integer(1, lhs.type());
  if (exponent == 1)
    return lhs;

  // Check resolved base for special cases
  if (resolved_lhs.is_constant())
  {
    BigInt base = binary2integer(
      resolved_lhs.value().as_string(), resolved_lhs.type().is_signedbv());

    // Special cases for constant base
    if (base == 0 && exponent > 0)
      return from_integer(0, lhs.type());
    if (base == 1)
      return from_integer(1, lhs.type());
    if (base == -1)
      return from_integer((exponent % 2 == 0) ? 1 : -1, lhs.type());
  }

  // Build symbolic multiplication tree using exponentiation by squaring for efficiency
  return build_power_expression(lhs, exponent);
}

// Helper function for efficient exponentiation
exprt python_converter::build_power_expression(
  const exprt &base,
  const BigInt &exp)
{
  if (exp == 0)
    return from_integer(1, base.type());
  if (exp == 1)
    return base;

  // For small exponents, use simple multiplication chain
  if (exp <= 10)
  {
    exprt result = base;
    for (BigInt i = 1; i < exp; ++i)
    {
      exprt mul_expr("*", base.type());
      mul_expr.copy_to_operands(result, base);
      result = mul_expr;
    }
    return result;
  }

  // For larger exponents, use exponentiation by squaring
  // This reduces the number of operations from O(n) to O(log n)
  if (exp % 2 == 0)
  {
    // Even exponent: (base^2)^(exp/2)
    exprt square("*", base.type());
    square.copy_to_operands(base, base);
    return build_power_expression(square, exp / 2);
  }
  else
  {
    // Odd exponent: base * base^(exp-1)
    exprt mul_expr("*", base.type());
    exprt sub_power = build_power_expression(base, exp - 1);
    mul_expr.copy_to_operands(base, sub_power);
    return mul_expr;
  }
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
  else if (type_utils::is_ordered_comparison(op))
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
    else
    {
      // For non-constant operands (like function parameters), create explicit typecast expression
      e = typecast_exprt(e, float_type);
    }
  };

  promote_to_float(lhs);
  promote_to_float(rhs);

  // Set the result type and operator ID to reflect float division
  bin_expr.type() = float_type;
  bin_expr.id(get_op("div", float_type));
}

BigInt python_converter::get_string_size(const exprt &expr)
{
  if (!expr.type().is_array())
  {
    // For non-array types in f-strings, convert them first to get actual size
    if (
      expr.is_constant() &&
      (expr.type().is_signedbv() || expr.type().is_unsignedbv()))
    {
      // Convert the actual integer to string to get real size
      BigInt value =
        binary2integer(expr.value().as_string(), expr.type().is_signedbv());
      std::string str_repr = std::to_string(value.to_int64());
      return BigInt(str_repr.size() + 1); // +1 for null terminator
    }

    if (expr.is_symbol())
    {
      const symbolt *symbol = symbol_table_.find_symbol(expr.identifier());
      if (symbol && symbol->type.is_array())
      {
        const auto &arr_type = to_array_type(symbol->type);
        return binary2integer(arr_type.size().value().as_string(), false);
      }
      // For non-array symbols, we need a reasonable default since we can't compute actual size
      return BigInt(20); // Conservative default
    }

    // For other types, use conservative defaults
    if (expr.type().is_bool())
      return BigInt(6); // "False" + null terminator

    // Default fallback
    return BigInt(20);
  }

  const auto &arr_type = to_array_type(expr.type());
  return binary2integer(arr_type.size().value().as_string(), false);
}

std::string
python_converter::process_format_spec(const nlohmann::json &format_spec)
{
  if (format_spec.is_null() || !format_spec.contains("_type"))
    return "";

  // Handle direct Constant format spec
  if (format_spec["_type"] == "Constant" && format_spec.contains("value"))
    return format_spec["value"].get<std::string>();

  // Handle JoinedStr format spec (which contains Constant values)
  if (format_spec["_type"] == "JoinedStr" && format_spec.contains("values"))
  {
    std::string result;
    for (const auto &value : format_spec["values"])
      if (value["_type"] == "Constant" && value.contains("value"))
        result += value["value"].get<std::string>();
    return result;
  }

  // Log warning for unsupported format specifications
  std::string spec_type = format_spec.contains("_type")
                            ? format_spec["_type"].get<std::string>()
                            : "unknown";
  log_warning("Unsupported f-string format specification type: {}", spec_type);

  return "";
}

exprt python_converter::apply_format_specification(
  const exprt &expr,
  const std::string &format)
{
  // Basic format specification handling
  if (format.empty())
    return convert_to_string(expr);

  // Handle integer formatting
  if (format == "d" || format == "i")
    return convert_to_string(expr);

  // Handle float formatting with precision
  else if (format.find(".") != std::string::npos && format.back() == 'f')
  {
    // Extract precision from format string (e.g., ".2f" -> 2)
    size_t dot_pos = format.find(".");
    size_t f_pos = format.find("f");
    if (
      dot_pos != std::string::npos && f_pos != std::string::npos &&
      f_pos > dot_pos)
    {
      std::string precision_str =
        format.substr(dot_pos + 1, f_pos - dot_pos - 1);
      int precision = 6; // default
      try
      {
        precision = std::stoi(precision_str);
      }
      catch (...)
      {
        precision = 6;
      }

      // Handle floatbv expressions (both constant and symbols)
      if (expr.type().is_floatbv())
      {
        const typet &t = expr.type();
        const std::size_t float_width = bv_width(t);

        // Support common floating point widths
        if (t.is_floatbv() && (float_width == 32 || float_width == 64))
        {
          const std::string *float_bits = nullptr;

          // Handle constant expressions
          if (expr.is_constant())
            float_bits = &expr.value().as_string();
          // Handle symbol expressions
          else if (expr.is_symbol())
          {
            const symbol_exprt &sym_expr = to_symbol_expr(expr);
            const symbolt *symbol =
              symbol_table_.find_symbol(sym_expr.get_identifier());

            if (symbol && symbol->value.is_constant())
              float_bits = &symbol->value.value().as_string();
          }

          if (float_bits && float_bits->length() == float_width)
          {
            double val = 0.0;

            // Handle different floating point widths
            if (float_width == 32)
            {
              // IEEE 754 single precision
              uint32_t bits = 0;
              for (std::size_t i = 0; i < float_width; ++i)
                if ((*float_bits)[i] == '1')
                  bits |= (1U << (float_width - 1 - i));

              float float_val;
              std::memcpy(&float_val, &bits, sizeof(float));
              val = static_cast<double>(float_val);
            }
            else if (float_width == 64)
            {
              // IEEE 754 double precision
              uint64_t bits = 0;
              for (std::size_t i = 0; i < float_width; ++i)
                if ((*float_bits)[i] == '1')
                  bits |= (1ULL << (float_width - 1 - i));

              std::memcpy(&val, &bits, sizeof(double));
            }

            // Use proper rounding to avoid IEEE 754 precision issues
            double multiplier = std::pow(10.0, precision);
            double rounded = std::round(val * multiplier) / multiplier;

            std::ostringstream oss;
            oss << std::fixed << std::setprecision(precision) << rounded;
            std::string formatted_str = oss.str();

            typet string_type =
              type_handler_.build_array(char_type(), formatted_str.size() + 1);
            std::vector<unsigned char> chars(
              formatted_str.begin(), formatted_str.end());
            chars.push_back('\0');

            return make_char_array_expr(chars, string_type);
          }
        }
      }
    }
  }

  // Default: just convert to string
  return convert_to_string(expr);
}

exprt python_converter::convert_to_string(const exprt &expr)
{
  const typet &t = expr.type();

  // Already a string/char array - return as is
  if (t.is_array() && t.subtype() == char_type())
    return expr;

  // Handle symbol references
  if (expr.is_symbol())
  {
    const symbolt *symbol = symbol_table_.find_symbol(expr.identifier());
    if (symbol)
    {
      // If symbol has string type, return it
      if (symbol->type.is_array() && symbol->type.subtype() == char_type())
        return expr;

      // If symbol has a constant value, convert that
      if (symbol->value.is_constant())
        return convert_to_string(symbol->value);
    }
  }

  // Handle constants
  if (expr.is_constant())
  {
    if (t.is_signedbv() || t.is_unsignedbv())
    {
      BigInt value = binary2integer(expr.value().as_string(), t.is_signedbv());
      std::string str_value = std::to_string(value.to_int64());

      typet string_type =
        type_handler_.build_array(char_type(), str_value.size() + 1);
      std::vector<unsigned char> chars(str_value.begin(), str_value.end());
      chars.push_back('\0'); // null terminator

      return make_char_array_expr(chars, string_type);
    }
    else if (t.is_floatbv())
    {
      std::string str_value = "0.0";
      if (expr.is_constant() && !expr.value().empty())
      {
        const std::string &float_bits = expr.value().as_string();
        if (t.is_floatbv() && bv_width(t) == 64 && float_bits.length() == 64)
        {
          uint64_t bits = 0;
          for (size_t i = 0; i < 64; ++i)
          {
            if (float_bits[i] == '1')
              bits |= (1ULL << (63 - i));
          }
          double val;
          std::memcpy(&val, &bits, sizeof(double));

          // Use proper rounding for default precision (6 decimal places)
          double multiplier = std::pow(10.0, 6);
          double rounded = std::round(val * multiplier) / multiplier;

          std::ostringstream oss;
          oss << std::fixed << std::setprecision(6) << rounded;
          str_value = oss.str();
        }
      }

      typet string_type =
        type_handler_.build_array(char_type(), str_value.size() + 1);
      std::vector<unsigned char> chars(str_value.begin(), str_value.end());
      chars.push_back('\0');

      return make_char_array_expr(chars, string_type);
    }
    else if (t.is_bool())
    {
      // Convert boolean to string
      bool value = expr.is_true();
      std::string str_value = value ? "True" : "False";

      typet string_type =
        type_handler_.build_array(char_type(), str_value.size() + 1);
      std::vector<unsigned char> chars(str_value.begin(), str_value.end());
      chars.push_back('\0');

      return make_char_array_expr(chars, string_type);
    }
  }

  // For non-constant expressions, we'd need runtime conversion
  // For now, create a placeholder string
  std::string placeholder = "<expr>";
  typet string_type =
    type_handler_.build_array(char_type(), placeholder.size() + 1);
  std::vector<unsigned char> chars(placeholder.begin(), placeholder.end());
  chars.push_back('\0');

  return make_char_array_expr(chars, string_type);
}

exprt python_converter::get_fstring_expr(const nlohmann::json &element)
{
  if (!element.contains("values") || element["values"].empty())
  {
    // Empty f-string
    typet empty_string_type = type_handler_.build_array(char_type(), 1);
    exprt empty_str = gen_zero(empty_string_type);
    empty_str.operands().at(0) = from_integer(0, char_type());
    return empty_str;
  }

  const auto &values = element["values"];
  std::vector<exprt> parts;
  BigInt total_estimated_size = BigInt(1); // Start with 1 for null terminator

  for (const auto &value : values)
  {
    exprt part_expr;

    try
    {
      if (value["_type"] == "Constant")
      {
        // String literal part
        part_expr = get_literal(value);
      }
      else if (value["_type"] == "FormattedValue")
      {
        // Expression to be formatted
        exprt expr = get_expr(value["value"]);

        // Handle format specification if present
        if (value.contains("format_spec") && !value["format_spec"].is_null())
        {
          std::string format = process_format_spec(value["format_spec"]);
          part_expr = apply_format_specification(expr, format);
        }
        else
          part_expr = convert_to_string(expr);
      }
      else
      {
        // Other expression types
        exprt expr = get_expr(value);
        part_expr = convert_to_string(expr);
      }

      parts.push_back(part_expr);
      total_estimated_size += get_string_size(part_expr) -
                              1; // -1 to avoid double counting terminators
    }
    catch (const std::exception &e)
    {
      log_warning("Error processing f-string part: {}", e.what());
      // Create error placeholder
      std::string error_str = "<error>";
      typet error_type =
        type_handler_.build_array(char_type(), error_str.size() + 1);
      std::vector<unsigned char> chars(error_str.begin(), error_str.end());
      chars.push_back('\0');
      parts.push_back(make_char_array_expr(chars, error_type));
      total_estimated_size += BigInt(error_str.size());
    }
  }

  // If only one part, return it directly
  if (parts.size() == 1)
    return parts[0];

  // Concatenate all parts
  exprt result = parts[0];
  for (size_t i = 1; i < parts.size(); ++i)
  {
    nlohmann::json empty_left, empty_right;
    result =
      handle_string_concatenation(result, parts[i], empty_left, empty_right);
  }

  return result;
}

exprt python_converter::handle_string_concatenation(
  const exprt &lhs,
  const exprt &rhs,
  const nlohmann::json &left,
  const nlohmann::json &right)
{
  // Validate inputs and calculate sizes safely
  BigInt lhs_size = get_string_size(lhs);
  BigInt rhs_size = get_string_size(rhs);

  if (lhs_size < 1 || rhs_size < 1)
    throw std::runtime_error("Invalid string size in concatenation");

  // Account for null terminators properly
  BigInt content_size = (lhs_size - 1) + (rhs_size - 1);
  BigInt total_size = content_size + 1; // +1 for final null terminator

  typet t = type_handler_.get_typet("str", total_size.to_uint64());
  exprt result = gen_zero(t);
  unsigned int i = 0;

  // Helper function with consistent bounds checking and null handling
  auto safe_append_char = [&](const exprt &ch) -> bool {
    if (i >= result.operands().size())
      return false; // Buffer full
    if (ch != gen_zero(ch.type()))
      result.operands().at(i++) = ch;
    return true;
  };

  auto append_from_symbol = [&](const std::string &id) -> bool {
    symbolt *symbol = symbol_table_.find_symbol(id);
    if (!symbol)
      return false;

    for (const exprt &ch : symbol->value.operands())
    {
      if (!safe_append_char(ch))
        return false;
    }
    return true;
  };

  auto append_from_json = [&](const nlohmann::json &json) -> bool {
    if (!json.contains("value"))
      return false;

    std::string value = json["value"].get<std::string>();
    typet char_type = t.subtype();

    for (char ch : value)
    {
      if (ch == 0)
        break; // Stop at null terminator
      if (i >= result.operands().size())
        return false; // Buffer full

      BigInt v(ch);
      exprt char_expr = constant_exprt(
        integer2binary(v, bv_width(char_type)), integer2string(v), char_type);
      result.operands().at(i++) = char_expr;
    }
    return true;
  };

  auto append_from_expr = [&](const exprt &expr) -> bool {
    if (expr.is_constant() && expr.type().is_array())
    {
      for (const auto &ch : expr.operands())
      {
        if (ch.is_constant() && !safe_append_char(ch))
          return false;
      }
    }
    else if (expr.is_symbol())
      return append_from_symbol(expr.identifier().as_string());
    return true;
  };

  auto append_from_json_or_expr =
    [&](const nlohmann::json &node, const exprt &expr) -> bool {
    if (node.contains("_type"))
    {
      const std::string type = node["_type"].get<std::string>();
      if (type == "Constant" && node.contains("value"))
        return append_from_json(node);
    }

    // Handle all other cases (including symbols) consistently
    return append_from_expr(expr);
  };

  // Use the helper for both operands
  append_from_json_or_expr(left, lhs);
  append_from_json_or_expr(right, rhs);

  // Add null terminator
  if (i < result.operands().size())
    result.operands().at(i) = gen_zero(t.subtype());

  return result;
}

bool python_converter::is_zero_length_array(const exprt &expr)
{
  if (expr.id() == "sideeffect")
    return false;

  if (!expr.type().is_array())
    return false;

  const auto &arr_type = to_array_type(expr.type());
  if (!arr_type.size().is_constant())
    return false;

  BigInt size = binary2integer(arr_type.size().value().as_string(), false);
  return size == 0;
}

std::string python_converter::extract_string_from_array_operands(
  const exprt &array_expr) const
{
  std::string result;
  for (const auto &op : array_expr.operands())
  {
    if (op.is_constant())
    {
      BigInt val =
        binary2integer(op.value().as_string(), op.type().is_signedbv());
      if (val == 0)
        break;
      result += static_cast<char>(val.to_uint64());
    }
  }
  return result;
}

std::pair<exprt, exprt> python_converter::resolve_comparison_operands_internal(
  const exprt &lhs,
  const exprt &rhs)
{
  // Try to resolve symbols to their constant values
  exprt lhs_resolved = get_resolved_value(lhs);
  exprt rhs_resolved = get_resolved_value(rhs);

  exprt final_lhs = lhs_resolved.is_nil() ? lhs : lhs_resolved;
  exprt final_rhs = rhs_resolved.is_nil() ? rhs : rhs_resolved;

  // Resolve array element symbols
  if (final_lhs.is_constant() && final_lhs.type().is_array())
  {
    exprt::operandst &lhs_ops = final_lhs.operands();
    for (size_t i = 0; i < lhs_ops.size(); ++i)
    {
      if (lhs_ops[i].is_symbol())
      {
        exprt resolved_elem = get_resolved_value(lhs_ops[i]);
        if (!resolved_elem.is_nil())
          lhs_ops[i] = resolved_elem;
      }
    }
  }

  if (final_rhs.is_constant() && final_rhs.type().is_array())
  {
    exprt::operandst &rhs_ops = final_rhs.operands();
    for (size_t i = 0; i < rhs_ops.size(); ++i)
    {
      if (rhs_ops[i].is_symbol())
      {
        exprt resolved_elem = get_resolved_value(rhs_ops[i]);
        if (!resolved_elem.is_nil())
          rhs_ops[i] = resolved_elem;
      }
    }
  }

  return {final_lhs, final_rhs};
}

bool python_converter::has_unsupported_side_effects_internal(
  const exprt &lhs,
  const exprt &rhs)
{
  auto has_unsupported_side_effect = [](const exprt &expr) {
    return expr.id() == "sideeffect" &&
           expr.get("statement") != "function_call";
  };

  return has_unsupported_side_effect(lhs) || has_unsupported_side_effect(rhs);
}

exprt python_converter::compare_constants_internal(
  const std::string &op,
  const exprt &lhs,
  const exprt &rhs)
{
  if (!lhs.is_constant() || !rhs.is_constant())
    return nil_exprt();

  // Single character comparisons
  if (
    (lhs.type().is_unsignedbv() || lhs.type().is_signedbv()) &&
    (rhs.type().is_unsignedbv() || rhs.type().is_signedbv()))
  {
    bool equal = (lhs == rhs);
    return gen_boolean((op == "Eq") ? equal : !equal);
  }

  // Mixed character vs array comparisons
  if (
    (lhs.type().is_unsignedbv() || lhs.type().is_signedbv()) &&
    rhs.type().is_array())
  {
    const exprt::operandst &rhs_ops = rhs.operands();
    if (rhs_ops.size() == 1)
    {
      bool equal =
        (lhs == rhs_ops[0]) || (lhs.get("value") == rhs_ops[0].get("value"));
      return gen_boolean((op == "Eq") ? equal : !equal);
    }
    return gen_boolean(op == "NotEq");
  }

  if (
    lhs.type().is_array() &&
    (rhs.type().is_unsignedbv() || rhs.type().is_signedbv()))
  {
    const exprt::operandst &lhs_ops = lhs.operands();
    if (lhs_ops.size() == 1)
    {
      bool equal =
        (lhs_ops[0] == rhs) || (lhs_ops[0].get("value") == rhs.get("value"));
      return gen_boolean((op == "Eq") ? equal : !equal);
    }
    return gen_boolean(op == "NotEq");
  }

  return nil_exprt();
}

exprt python_converter::handle_indexed_comparison_internal(
  const std::string &op,
  const exprt &lhs,
  const exprt &rhs)
{
  if (lhs.id() != "index" || !rhs.is_constant() || !rhs.type().is_array())
    return nil_exprt();

  const exprt &index = lhs.operands()[1];
  BigInt idx =
    binary2integer(index.value().as_string(), index.type().is_signedbv());

  std::string rhs_str = extract_string_from_array_operands(rhs);

  const exprt &array = lhs.operands()[0];
  exprt resolved_array = get_resolved_value(array);

  if (resolved_array.is_nil() && array.is_symbol())
  {
    const symbolt *symbol = symbol_table_.find_symbol(array.identifier());
    if (symbol)
    {
      resolved_array = symbol->value;
      if (symbol->value.is_symbol())
      {
        const symbolt *compound =
          symbol_table_.find_symbol(symbol->value.identifier());
        if (compound && compound->value.is_constant())
          resolved_array = compound->value;
      }
    }
  }

  if (
    !resolved_array.is_nil() && resolved_array.is_constant() &&
    resolved_array.type().is_array() && idx >= 0 &&
    idx < (BigInt)resolved_array.operands().size())
  {
    const exprt &string_element = resolved_array.operands()[idx.to_uint64()];
    std::string lhs_str = extract_string_from_array_operands(string_element);
    bool strings_equal = (lhs_str == rhs_str);
    return gen_boolean((op == "Eq") ? strings_equal : !strings_equal);
  }

  return nil_exprt();
}

exprt python_converter::handle_type_mismatches(
  const std::string &op,
  const exprt &lhs,
  const exprt &rhs)
{
  // Skip if either operand is a member expression or types match exactly
  if (lhs.is_member() || rhs.is_member() || lhs.type() == rhs.type())
    return nil_exprt();

  // Both operands are arrays - need to distinguish between lists and strings
  if (lhs.type().is_array() && rhs.type().is_array())
  {
    // Check if these are different semantic types (list vs string)
    bool lhs_is_string = (lhs.type().subtype() == char_type());
    bool rhs_is_string = (rhs.type().subtype() == char_type());

    // If one is a string array and the other is not, they're different types
    if (lhs_is_string != rhs_is_string)
      return gen_boolean(op == "NotEq");

    // Both are strings: compare based on content
    // check if empty
    bool lhs_empty = is_zero_length_array(lhs) ||
                     (lhs.is_constant() && lhs.operands().size() <= 1);
    bool rhs_empty = is_zero_length_array(rhs) ||
                     (rhs.is_constant() && rhs.operands().size() <= 1);

    if (lhs_empty && rhs_empty)
      return gen_boolean(op == "Eq");
  }

  // Mixed types (array vs non-array) = not equal
  return gen_boolean(op == "NotEq");
}

exprt python_converter::handle_string_comparison(
  const std::string &op,
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &element)
{
  // Resolve symbols to their constant values
  auto [resolved_lhs, resolved_rhs] =
    resolve_comparison_operands_internal(lhs, rhs);

  // Check for unsupported side effects
  if (has_unsupported_side_effects_internal(resolved_lhs, resolved_rhs))
    throw std::runtime_error("Cannot compare non-function side effects");

  // Handle zero-length arrays early
  if (is_zero_length_array(resolved_lhs) && is_zero_length_array(resolved_rhs))
    return gen_boolean(op == "Eq");

  // Try constant comparisons
  exprt constant_result =
    compare_constants_internal(op, resolved_lhs, resolved_rhs);
  if (!constant_result.is_nil())
    return constant_result;

  // Try indexed string comparison
  exprt indexed_result =
    handle_indexed_comparison_internal(op, resolved_lhs, resolved_rhs);
  if (!indexed_result.is_nil())
    return indexed_result;

  // Handle type mismatches
  exprt mismatch_result =
    handle_type_mismatches(op, resolved_lhs, resolved_rhs);
  if (!mismatch_result.is_nil())
    return mismatch_result;

  // At this point, both operands should be strings (arrays of char)
  if (resolved_lhs.type().is_array())
    resolved_lhs = get_array_base_address(resolved_lhs);
  if (resolved_rhs.type().is_array())
    resolved_rhs = get_array_base_address(resolved_rhs);

  symbolt *strncmp_symbol = symbol_table_.find_symbol("c:@F@strcmp");
  if (!strncmp_symbol)
    throw std::runtime_error(
      "strcmp function not found in symbol table for string comparison");

  side_effect_expr_function_callt strncmp_call;
  strncmp_call.function() = symbol_expr(*strncmp_symbol);
  strncmp_call.arguments() = {resolved_lhs, resolved_rhs};
  strncmp_call.location() = get_location_from_decl(element);
  strncmp_call.type() = int_type();

  lhs = strncmp_call;
  rhs = gen_zero(int_type());

  return nil_exprt(); // continue with lhs OP rhs
}

// Helper method to resolve symbol values to constants
exprt python_converter::get_resolved_value(const exprt &expr)
{
  // Handle direct function call expressions
  if (expr.id() == "sideeffect")
  {
    const side_effect_exprt &side_effect = to_side_effect_expr(expr);
    if (
      side_effect.get_statement() == "function_call" &&
      side_effect.operands().size() >= 2)
      // Structure: operand 0 = function symbol, operand 1 = arguments
      return resolve_function_call(
        side_effect.operands()[0], side_effect.operands()[1]);
  }

  // Handle symbols that contain function calls or constants
  if (!expr.is_symbol())
    return nil_exprt();

  const symbol_exprt &sym = to_symbol_expr(expr);
  const symbolt *symbol = symbol_table_.find_symbol(sym.get_identifier());

  if (!symbol || symbol->value.is_nil())
    return nil_exprt();

  // Return constant values directly
  if (symbol->value.is_constant())
    return symbol->value;

  // Handle function calls stored as code
  if (symbol->value.is_code())
  {
    const codet &code = to_code(symbol->value);

    if (code.get_statement() == "function_call" && code.operands().size() >= 3)
    {
      // Structure: operand 1 = function symbol, operand 2 = arguments
      exprt result =
        resolve_function_call(code.operands()[1], code.operands()[2]);
      if (!result.is_nil())
        return result;
    }
  }

  return nil_exprt();
}

// Helper to resolve function calls (both identity functions and constant-returning functions)
exprt python_converter::resolve_function_call(
  const exprt &func_expr,
  const exprt &args_expr)
{
  if (!func_expr.is_symbol())
    return nil_exprt();

  const symbol_exprt &func_sym = to_symbol_expr(func_expr);
  const symbolt *func_symbol =
    symbol_table_.find_symbol(func_sym.get_identifier());

  if (!func_symbol || func_symbol->value.is_nil())
    return nil_exprt();

  // First check if this function returns a constant value
  exprt constant_result = get_function_constant_return(func_symbol->value);
  if (!constant_result.is_nil())
    return constant_result;

  // Then check if this function is an identity function (returns its parameter)
  if (!is_identity_function(
        func_symbol->value, func_sym.get_identifier().as_string()))
    return nil_exprt();

  // Extract the first argument for identity functions
  if (args_expr.id() != "arguments" || args_expr.operands().empty())
    return nil_exprt();

  exprt arg = args_expr.operands()[0];

  // Handle address_of wrapper
  if (arg.is_address_of() && arg.operands().size() > 0)
    arg = arg.operands()[0];

  // If the argument is itself a function call, recursively resolve it
  if (arg.id() == "sideeffect")
  {
    exprt nested_resolved = get_resolved_value(arg);
    if (!nested_resolved.is_nil())
      arg = nested_resolved;
  }

  // If the argument is a symbol, try to resolve it to its constant value
  if (arg.is_symbol())
  {
    const symbol_exprt &sym = to_symbol_expr(arg);
    const symbolt *symbol = symbol_table_.find_symbol(sym.get_identifier());
    if (symbol && symbol->value.is_constant())
      arg = symbol->value;
  }

  // Return string constants, array constants, and single character constants
  if (
    arg.id() == "string-constant" || (arg.is_constant() && arg.is_array()) ||
    (arg.is_constant() && arg.type().is_array()) ||
    (arg.is_constant() &&
     (arg.type().is_unsignedbv() || arg.type().is_signedbv())))
  {
    return arg;
  }

  return nil_exprt();
}

// Helper to check if a function returns a constant value
exprt python_converter::get_function_constant_return(const exprt &func_value)
{
  if (!func_value.is_code())
    return nil_exprt();

  const codet &func_code = to_code(func_value);

  // Check if it's a simple return statement with a constant
  if (func_code.get_statement() == "return")
  {
    const code_returnt &ret = to_code_return(func_code);
    if (ret.has_return_value())
    {
      const exprt &return_val = ret.return_value();
      if (
        return_val.id() == "string-constant" ||
        (return_val.is_constant() && return_val.is_array()) ||
        (return_val.is_constant() && return_val.type().is_array()) ||
        (return_val.is_constant() && (return_val.type().is_unsignedbv() ||
                                      return_val.type().is_signedbv())))
      {
        return return_val;
      }
    }
  }

  // Check nested code structures
  for (const auto &operand : func_value.operands())
  {
    if (operand.is_code())
    {
      const codet &sub_code = to_code(operand);
      if (sub_code.get_statement() == "return")
      {
        const code_returnt &ret = to_code_return(sub_code);
        if (ret.has_return_value())
        {
          const exprt &return_val = ret.return_value();
          if (
            return_val.id() == "string-constant" ||
            (return_val.is_constant() && return_val.is_array()) ||
            (return_val.is_constant() && return_val.type().is_array()) ||
            (return_val.is_constant() && (return_val.type().is_unsignedbv() ||
                                          return_val.type().is_signedbv())))
          {
            return return_val;
          }
        }
      }
    }
  }

  return nil_exprt();
}

// Helper to check if a function is an identity function (returns its parameter)
bool python_converter::is_identity_function(
  const exprt &func_value,
  const std::string &func_identifier)
{
  if (!func_value.is_code())
    return false;

  const codet &func_code = to_code(func_value);

  // Check if it's a simple return statement
  if (func_code.get_statement() == "return")
  {
    const code_returnt &ret = to_code_return(func_code);
    if (ret.has_return_value() && ret.return_value().is_symbol())
    {
      const symbol_exprt &return_sym = to_symbol_expr(ret.return_value());
      std::string return_identifier = return_sym.get_identifier().as_string();
      std::string parameter_prefix = func_identifier + "@";

      // Check if the returned symbol is a parameter of this function
      // Parameter pattern: func_identifier + "@" + parameter_name
      if (
        return_identifier.size() >= parameter_prefix.size() &&
        return_identifier.compare(
          0, parameter_prefix.size(), parameter_prefix) == 0)
        return true;
    }
  }

  // Check nested code structures
  for (const auto &operand : func_value.operands())
  {
    if (operand.is_code())
    {
      const codet &sub_code = to_code(operand);
      if (sub_code.get_statement() == "return")
      {
        const code_returnt &ret = to_code_return(sub_code);
        if (ret.has_return_value() && ret.return_value().is_symbol())
        {
          const symbol_exprt &return_sym = to_symbol_expr(ret.return_value());
          std::string return_identifier =
            return_sym.get_identifier().as_string();
          std::string parameter_prefix = func_identifier + "@";

          // Check if the returned symbol is a parameter of this function
          if (
            return_identifier.size() >= parameter_prefix.size() &&
            return_identifier.compare(
              0, parameter_prefix.size(), parameter_prefix) == 0)
            return true;
        }
      }
    }
  }

  return false;
}

void python_converter::ensure_string_array(exprt &expr)
{
  if (expr.type().is_pointer())
    return;

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

  assert(lhs.type().is_array() || lhs.type().is_pointer());
  assert(rhs.type().is_array() || rhs.type().is_pointer());

  if (op == "Eq" || op == "NotEq")
    return handle_string_comparison(op, lhs, rhs, element);

  if (op == "Add")
    return handle_string_concatenation(lhs, rhs, left, right);

  return nil_exprt();
}

/// Construct the expression for Python 'is' operator
exprt python_converter::get_binary_operator_expr_for_is(
  const exprt &lhs,
  const exprt &rhs)
{
  typet bool_type_result = bool_type();
  exprt is_expr("=", bool_type_result);

  if (lhs.type().is_array() && rhs.type().is_array())
  {
    // Compare base addresses of the arrays
    is_expr.copy_to_operands(
      get_array_base_address(lhs), get_array_base_address(rhs));
  }
  else
  {
    // Default identity comparison
    is_expr.copy_to_operands(lhs, rhs);
  }

  return is_expr;
}

/// Get address of the first element of an array
exprt python_converter::get_array_base_address(const exprt &arr)
{
  exprt index = index_exprt(arr, from_integer(0, index_type()));
  return address_of_exprt(index);
}

/// Construct the negation of an 'is' expression, used for 'is not'
exprt python_converter::get_negated_is_expr(const exprt &lhs, const exprt &rhs)
{
  exprt is_expr = get_binary_operator_expr_for_is(lhs, rhs);
  exprt not_expr("not", bool_type());
  not_expr.copy_to_operands(is_expr);
  return not_expr;
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

  if (lhs.type().is_array() || rhs.type().is_array())
  {
    // Check for zero-length arrays
    if (
      is_zero_length_array(lhs) && is_zero_length_array(rhs) &&
      (op == "Eq" || op == "NotEq"))
      return gen_boolean(op == "Eq");

    // Compute resulting list
    if (op == "Mult")
    {
      if (is_right)
        return nil_exprt();

      BigInt list_size;
      exprt list_elem;

      // Get list size from lhs (e.g.: 3 * [1])
      if (!lhs.type().is_array())
      {
        if (lhs.is_symbol())
        {
          symbolt *symbol =
            find_symbol(to_symbol_expr(lhs).get_identifier().as_string());
          assert(symbol);
          list_size = std::stoi(symbol->value.value().as_string(), nullptr, 2);
        }
        else if (lhs.is_constant())
          list_size = std::stoi(lhs.value().as_string(), nullptr, 2);

        // List element is the rhs
        list_elem = get_expr(right["elts"][0]);
      }

      // Get list size from rhs (e.g.: 3 * [1])
      if (!rhs.type().is_array())
      {
        // List element is the rhs
        list_elem = get_expr(left["elts"][0]);

        if (rhs.is_symbol())
        {
          symbolt *symbol =
            find_symbol(to_symbol_expr(rhs).get_identifier().as_string());

          assert(symbol);

          if (symbol->value.is_code())
          {
            // Build array and initialize VLA

            symbolt &vla_symbol = create_tmp_symbol(
              element,
              "vla",
              current_lhs->type(),
              gen_zero(current_lhs->type()));

            code_declt vla_decl(symbol_expr(vla_symbol));
            vla_decl.location() = get_location_from_decl(element);
            current_block->copy_to_operands(vla_decl);

            // Add counter for while loop
            symbolt &counter = create_tmp_symbol(
              element, "counter", int_type(), gen_zero(int_type()));

            code_assignt counter_code(
              symbol_expr(counter), gen_zero(int_type()));
            current_block->copy_to_operands(counter_code);

            // Build conditional for while loop (counter < len(arr))
            exprt cond("<", bool_type());
            cond.operands().push_back(symbol_expr(counter));
            cond.operands().push_back(symbol_expr(*symbol));

            // Build block with arr[i] assignment and counter increment
            code_blockt then;

            // arr[counter] assignment
            index_exprt arr_index(
              symbol_expr(vla_symbol),
              symbol_expr(counter),
              symbol_expr(vla_symbol).type().subtype());
            code_assignt arr_assign(arr_index, list_elem);
            then.copy_to_operands(arr_assign);

            // increment counter
            exprt incr("+");
            incr.copy_to_operands(symbol_expr(counter));
            incr.copy_to_operands(gen_one(int_type()));
            code_assignt update(symbol_expr(counter), incr);
            then.copy_to_operands(update);

            // add while block to initialize vla
            codet while_cod;
            while_cod.set_statement("while");
            while_cod.copy_to_operands(cond, then);
            current_block->copy_to_operands(while_cod);

            // return initialized vla
            return symbol_expr(vla_symbol);
          }

          list_size = std::stoi(symbol->value.value().as_string(), nullptr, 2);
        }
        else if (rhs.is_constant())
          list_size = std::stoi(rhs.value().as_string(), nullptr, 2);
      }

      typet st =
        (lhs.type().is_array()) ? lhs.type().subtype() : rhs.type().subtype();
      typet list_type = array_typet(st, from_integer(list_size, int_type()));
      exprt list = gen_zero(list_type);

      for (int64_t i = 0; i < list_size.to_int64(); ++i)
        list.operands().at(i) = list_elem;

      return list;
    }
  }

  /// Handle 'is' and 'is not' Python identity comparisons
  if (op == "Is")
    return get_binary_operator_expr_for_is(lhs, rhs);
  else if (op == "IsNot")
    return get_negated_is_expr(lhs, rhs);

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
  // For Python division ("/"), the result is always a float regardless of operand types.
  // Otherwise, it inherits the type of the left-hand side (lhs).
  typet type;
  if (type_utils::is_relational_op(op))
    type = bool_type();
  else if (op == "Div" || op == "div")
    type = double_type(); // Python division always returns float
  else
    type = lhs.type();

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
  // In Python 3, the '/' operator ALWAYS performs floating-point division,
  // regardless of operand types. Only '//' performs floor division.
  if (op == "Div" || op == "div")
    handle_float_division(lhs, rhs, bin_expr);

  // Add lhs and rhs as operands to the binary expression.
  bin_expr.copy_to_operands(lhs, rhs);

  // Python 3 comparison behavior: equality (==, !=) between incompatible types
  // returns False/True, but ordered comparisons (<, >, etc.) raise TypeError.
  // This emulates that behavior in ESBMC's symbolic execution.
  if (type_utils::is_float_vs_char(lhs, rhs))
    return handle_float_vs_string(bin_expr, op);

  // floor division (//) operation corresponds to an int division with floor rounding
  // So we need to emulate this behavior here:
  // int result = (num/div) - (num%div != 0 && ((num < 0) ^ (den<0)) ? 1 : 0)
  // e.g.: -5//2 equals to -3, and 5//2 equals to 2
  if (op == "FloorDiv")
    return handle_floor_division(lhs, rhs, bin_expr);

  // Promote operands to floating point if IEEE operator is used
  if (is_ieee_op(bin_expr))
  {
    const typet &target_type =
      lhs.type().is_floatbv() ? lhs.type() : rhs.type();
    if (!lhs.type().is_floatbv())
      bin_expr.op0() = typecast_exprt(lhs, target_type);
    if (!rhs.type().is_floatbv())
      bin_expr.op1() = typecast_exprt(rhs, target_type);
  }

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
      exprt op1 = get_expr(element["comparators"][i]);
      exprt op2 = get_expr(element["comparators"][i + 1]);

      if (op1.is_function_call())
        to_side_effect_call(op1);
      if (op2.is_function_call())
        to_side_effect_call(op2);

      std::string op1_type = type_handler_.type_to_string(op1.type());
      std::string op2_type = type_handler_.type_to_string(op2.type());

      if (op1_type == "str" && op2_type == "str")
      {
        handle_string_comparison(op, op1, op2, element);
        exprt expr(get_op(op, bool_type()), type);
        expr.copy_to_operands(op1, op2);
        cond.move_to_operands(expr);
      }
      else
      {
        logical_expr.copy_to_operands(op1);
        logical_expr.copy_to_operands(op2);
        cond.move_to_operands(logical_expr);
      }
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
  auto class_node = json_utils::find_class((*ast_json)["body"], class_name);

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
  for (const auto &obj : (*ast_json)["body"])
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

  return json_utils::is_module(module_name, *ast_json);
}

exprt python_converter::get_function_call(const nlohmann::json &element)
{
  if (!element.contains("func") || element["_type"] != "Call")
    throw std::runtime_error("Invalid function call");

  // Handle indirect calls through function pointer variables
  if (element["func"]["_type"] == "Name")
  {
    std::string func_name = element["func"]["id"].get<std::string>();

    // Try to find as a variable first
    symbol_id var_sid = create_symbol_id();
    var_sid.set_object(func_name);
    symbolt *var_symbol = find_symbol(var_sid.to_string());

    if (var_symbol && var_symbol->type.is_pointer())
    {
      // This is an indirect call through function pointer
      side_effect_expr_function_callt call;
      call.location() = get_location_from_decl(element);

      // The function pointer itself, not dereferenced
      exprt func_ptr_expr = symbol_expr(*var_symbol);
      call.function() = func_ptr_expr;

      // Set return type
      if (var_symbol->type.subtype().is_code())
      {
        const code_typet &func_type = to_code_type(var_symbol->type.subtype());
        call.type() = func_type.return_type();
      }

      // Process arguments
      if (element.contains("args"))
      {
        for (const auto &arg_element : element["args"])
        {
          exprt arg_expr = get_expr(arg_element);
          call.arguments().push_back(arg_expr);
        }
      }

      return call;
    }
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
      !is_class(func_name, *ast_json))
    {
      const auto &func_node = find_function((*ast_json)["body"], func_name);
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
/// Convert Python AST literal to expression.
/// Handles integers, booleans, floats, chars, strings, and byte literals.
/// Example: {"_type": "Constant", "value": 42} -> integer constant expr
exprt python_converter::get_literal(const nlohmann::json &element)
{
  // Determine the source of the literal's value.
  const auto &value = (element["_type"] == "UnaryOp")
                        ? element["operand"]["value"]
                        : element["value"];

  // Handle None literals (null values)
  if (value.is_null())
  {
    // Create a null pointer expression to represent NoneType
    constant_exprt null_expr(pointer_type());
    // Sets the value to "NULL"
    null_expr.set_value("0");
    return null_expr;
  }

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

  if (!value.is_string())
    return exprt(); // Not a string, no handling

  const std::string &str_val = value.get<std::string>();

  // Handle single-character string as char literal
  if (str_val.size() == 1 && !is_bytes_literal(element))
  {
    typet t = type_handler_.get_typet("str", str_val.size());
    return from_integer(static_cast<unsigned char>(str_val[0]), t);
  }

  // Handle string or byte literals
  typet t = current_element_type;
  std::vector<uint8_t> string_literal;

  if (is_bytes_literal(element))
  {
    // Handle bytes literal - check for encoded_bytes field first
    if (element.contains("encoded_bytes"))
      string_literal =
        base64_decode(element["encoded_bytes"].get<std::string>());
    else
      string_literal.assign(str_val.begin(), str_val.end());

    // Set appropriate bytes type
    t = type_handler_.get_typet("bytes", string_literal.size());
  }
  else
  {
    // Python str literal - add null terminator
    string_literal.assign(str_val.begin(), str_val.end());
    string_literal.push_back('\0');

    t = type_handler_.build_array(char_type(), string_literal.size());
  }

  return make_char_array_expr(string_literal, t);
}

// Helper function to detect bytes literals
bool python_converter::is_bytes_literal(const nlohmann::json &element)
{
  // Check if element has encoded_bytes field (explicit bytes)
  if (element.contains("encoded_bytes"))
    return true;

  // Check if element has bytes type annotation
  if (
    element.contains("annotation") && element["annotation"].contains("id") &&
    element["annotation"]["id"] == "bytes")
    return true;

  // Check if element has a parent context indicating bytes
  if (element.contains("kind") && element["kind"] == "bytes")
    return true;

  // Check if this is part of a bytes assignment/initialization
  if (current_element_type.id() == "bytes")
    return true;

  // Check if this is an array of uint8 (bytes representation)
  if (current_element_type.id() == "array")
  {
    const typet &subtype = current_element_type.subtype();
    if (subtype.id() == "unsignedbv")
    {
      // Convert dstring width to integer
      const irep_idt &width_str = subtype.width();
      try
      {
        int width = std::stoi(width_str.as_string());
        if (width == 8)
          return true;
      }
      catch (const std::exception &)
      {
        // If conversion fails, continue with other checks
      }
    }
  }

  return false;
}

// Helper function to extract class name from tag (removes "tag-" prefix)
std::string
python_converter::extract_class_name_from_tag(const std::string &tag_name)
{
  if (tag_name.size() > 4 && tag_name.substr(0, 4) == "tag-")
    return tag_name.substr(4);
  return tag_name;
}

// Helper function to create normalized self key for cross-method access
std::string
python_converter::create_normalized_self_key(const std::string &class_tag)
{
  std::string class_name = extract_class_name_from_tag(class_tag);
  return "self@" + class_name;
}

// Helper function to clean attribute type by removing internal annotations
typet python_converter::clean_attribute_type(const typet &attr_type)
{
  typet clean_type = attr_type;
  clean_type.remove("#member_name");
  clean_type.remove("#location");
  clean_type.remove("#identifier");
  return clean_type;
}

// Helper function to create member expression with cleaned type
exprt python_converter::create_member_expression(
  const symbolt &symbol,
  const std::string &attr_name,
  const typet &attr_type)
{
  typet clean_type = clean_attribute_type(attr_type);
  return member_exprt(
    symbol_exprt(symbol.id, symbol.type), attr_name, clean_type);
}

// Helper function to register instance attribute in maps
void python_converter::register_instance_attribute(
  const std::string &symbol_id,
  const std::string &attr_name,
  const std::string &var_name,
  const std::string &class_tag)
{
  // Add to regular instance attribute map
  instance_attr_map[symbol_id].insert(attr_name);

  // For 'self' parameters, also track with normalized key for cross-method access
  if (var_name == "self")
  {
    std::string normalized_key = create_normalized_self_key(class_tag);
    instance_attr_map[normalized_key].insert(attr_name);
  }
}

// Helper function to check if attribute is an instance attribute
bool python_converter::is_instance_attribute(
  const std::string &symbol_id,
  const std::string &attr_name,
  const std::string &var_name,
  const std::string &class_tag)
{
  // Check regular per-symbol lookup
  auto it = instance_attr_map.find(symbol_id);
  if (
    it != instance_attr_map.end() &&
    it->second.find(attr_name) != it->second.end())
    return true;

  // For 'self' parameters, check normalized key for cross-method access
  if (var_name == "self")
  {
    std::string normalized_key = create_normalized_self_key(class_tag);
    auto self_it = instance_attr_map.find(normalized_key);
    if (self_it != instance_attr_map.end())
      return self_it->second.find(attr_name) != self_it->second.end();
  }

  return false;
}

symbolt &python_converter::create_tmp_symbol(
  const nlohmann::json &element,
  const std::string var_name,
  const typet &symbol_type,
  const exprt &symbol_value)
{
  locationt location = get_location_from_decl(element);
  std::string path = location.file().as_string();
  std::string name_prefix =
    path + ":" + location.get_line().as_string() + var_name;
  symbolt &cl =
    sym_generator_.new_symbol(symbol_table_, symbol_type, name_prefix);
  cl.mode = "Python";
  std::string module_name = location.get_file().as_string();
  cl.module = module_name;
  cl.location = location;
  cl.static_lifetime = false;
  cl.is_extern = false;
  cl.file_local = true;
  cl.value = symbol_value;
  return cl;
}

exprt python_converter::get_lambda_expr(const nlohmann::json &element)
{
  // Generate unique lambda name
  static int lambda_counter = 0;
  std::string lambda_name = "lam" + std::to_string(++lambda_counter);

  locationt location = get_location_from_decl(element);

  // Save current context and set lambda context
  std::string old_func_name = current_func_name_;
  current_func_name_ = lambda_name;

  // Create function type with proper return type detection
  code_typet lambda_type;
  typet return_type = double_type();
  // TODO: Try to infer better return type from the body if possible
  if (element.contains("body"))
    current_element_type = return_type;
  lambda_type.return_type() = return_type;

  std::string module_name = location.get_file().as_string();
  std::string lambda_id = "py:" + module_name + "@F@" + lambda_name;

  // Process arguments and create parameter symbols
  if (element.contains("args") && element["args"].contains("args"))
  {
    for (const auto &arg : element["args"]["args"])
    {
      std::string arg_name = arg["arg"].get<std::string>();

      // Determine parameter type
      // TODO: try to infer from usage or default to double
      typet param_type = double_type();

      // Create function argument
      code_typet::argumentt argument;
      argument.type() = param_type;
      argument.cmt_base_name(arg_name);

      std::string param_id = lambda_id + "@" + arg_name;
      argument.cmt_identifier(param_id);
      argument.location() = location;
      lambda_type.arguments().push_back(argument);

      // Create parameter symbol with all necessary fields
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
      symbol_table_.add(param_symbol);
    }
  }

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

  symbolt *added_symbol = symbol_table_.move_symbol_to_context(lambda_symbol);

  // Process lambda body
  if (element.contains("body"))
  {
    code_blockt lambda_block;

    // Process the body expression
    exprt body_expr = get_expr(element["body"]);

    // Create return statement
    code_returnt return_stmt;
    return_stmt.return_value() = body_expr;
    return_stmt.location() = location;

    lambda_block.copy_to_operands(return_stmt);
    added_symbol->value = lambda_block;
  }

  // Restore context
  current_func_name_ = old_func_name;

  return symbol_expr(*added_symbol);
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
      current_element_type = list_type;

    exprt list = gen_zero(list_type);

    unsigned int i = 0;
    for (auto &e : element["elts"])
      list.operands().at(i++) = get_expr(e);

    symbolt &cl =
      create_tmp_symbol(element, "$compound-literal$", list_type, list);

    expr = symbol_expr(cl);
    code_declt decl(expr);
    decl.operands().push_back(list);
    assert(current_block);
    current_block->copy_to_operands(decl);

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
      if (is_class(var_name, *ast_json))
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
    {
      sid.set_attribute(element["attr"].get<std::string>());
      sid.set_function("");
    }

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
        // Add member in the class if not exists
        if (!class_type.has_component(attr_name))
        {
          struct_typet::componentt comp = build_component(
            class_type.tag().as_string(), attr_name, current_element_type);
          class_type.components().push_back(comp);
        }

        // Register instance attribute for both regular and normalized keys
        register_instance_attribute(
          symbol->id.as_string(),
          attr_name,
          var_name,
          class_type.tag().as_string());
      }

      // Check if this is an instance attribute
      if (
        class_type.has_component(attr_name) && is_instance_attribute(
                                                 symbol->id.as_string(),
                                                 attr_name,
                                                 var_name,
                                                 class_type.tag().as_string()))
      {
        const typet &attr_type = class_type.get_component(attr_name).type();
        expr = create_member_expression(*symbol, attr_name, attr_type);
      }
      // Fallback to class attribute when instance attribute is not found
      else
      {
        // All class attributes are static symbols with ids in the format: filename@C@classname@varname
        sid.set_function("");
        sid.set_class(extract_class_name_from_tag(obj_type_name));
        sid.set_object(attr_name);
        symbolt *class_attr_symbol = symbol_table_.find_symbol(sid.to_string());

        if (!class_attr_symbol)
        {
          // If no class attribute exists and we're on LHS, create instance attribute
          if (is_converting_lhs && class_type.has_component(attr_name))
          {
            const typet &attr_type = class_type.get_component(attr_name).type();
            expr = create_member_expression(*symbol, attr_name, attr_type);
          }
          else
          {
            throw std::runtime_error(
              "Attribute \"" + attr_name + "\" not found");
          }
        }
        else
        {
          expr = symbol_expr(*class_attr_symbol);
        }
      }
    }

    // Tracks global reads within a function
    if (
      element["_type"] == "Name" &&
      sid.to_string().find("@C") == std::string::npos &&
      sid.to_string().find("@F") != std::string::npos && is_right &&
      !symbol_table_.find_symbol(sid.to_string().c_str()))
    {
      local_loads.push_back(sid.to_string());
    }
    break;
  }
  case ExpressionType::FUNC_CALL:
  {
    // Check if this is a lambda expression
    if (element["_type"] == "Lambda")
      expr = get_lambda_expr(element);
    else
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
        element["value"]["id"], current_func_name_, *ast_json);

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
  case ExpressionType::FSTRING:
    expr = get_fstring_expr(element);
    break;
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

size_t python_converter::get_type_size(const nlohmann::json &ast_node)
{
  size_t type_size = 0;

  // Handle lambda functions - they don't have a meaningful size
  if (
    ast_node.contains("value") && ast_node["value"].contains("_type") &&
    ast_node["value"]["_type"] == "Lambda")
    return 0;

  if (ast_node.contains("value") && ast_node["value"].contains("value"))
  {
    // Handle bytes literals
    if (
      ast_node.contains("annotation") &&
      ast_node["annotation"].contains("id") &&
      ast_node["annotation"]["id"] == "bytes")
    {
      if (ast_node["value"].contains("encoded_bytes"))
      {
        const std::string &str =
          ast_node["value"]["encoded_bytes"].get<std::string>();
        std::vector<uint8_t> decoded = base64_decode(str);
        type_size = decoded.size();
      }
      else if (ast_node["value"]["value"].is_string())
      {
        // Direct bytes literal like b'A'
        type_size = ast_node["value"]["value"].get<std::string>().size();
      }
    }
    else if (ast_node["value"]["value"].is_string())
      type_size = ast_node["value"]["value"].get<std::string>().size();
  }
  else if (
    ast_node["value"].contains("args") &&
    ast_node["value"]["args"].is_array() &&
    ast_node["value"]["args"].size() > 0 &&
    ast_node["value"]["args"][0].contains("value") &&
    ast_node["value"]["args"][0]["value"].is_string())
  {
    type_size = ast_node["value"]["args"][0]["value"].get<std::string>().size();
  }
  else if (
    ast_node["value"].contains("_type") && ast_node["value"]["_type"] == "List")
  {
    type_size = ast_node["value"]["elts"].size();
  }
  // Handle cases where size cannot be determined from AST structure
  else if (
    ast_node["value"].contains("value") &&
    ast_node["value"]["value"].is_string())
  {
    // Fallback for direct string values
    type_size = ast_node["value"]["value"].get<std::string>().size();
  }

  return type_size;
}

symbolt python_converter::create_return_temp_variable(
  const typet &return_type,
  const locationt &location,
  const std::string &func_name)
{
  static int temp_counter = 0;
  temp_counter++;

  symbol_id temp_sid = create_symbol_id();
  std::string temp_name =
    "return_value$_" + func_name + "$" + std::to_string(temp_counter);
  temp_sid.set_object(temp_name);

  symbolt temp_symbol;
  temp_symbol.id = temp_sid.to_string();
  temp_symbol.name = temp_sid.to_string();
  temp_symbol.type = return_type;
  temp_symbol.lvalue = true;
  temp_symbol.static_lifetime = false;
  temp_symbol.location = location;
  temp_symbol.mode = "Python";
  temp_symbol.module = location.get_file().as_string();
  temp_symbol.file_local = true;
  temp_symbol.is_extern = false;

  return temp_symbol;
}

const nlohmann::json &get_return_statement(const nlohmann::json &function)
{
  for (const auto &stmt : function["body"])
    if (get_statement_type(stmt) == StatementType::RETURN)
      return stmt;

  throw std::runtime_error(
    "Function " + function["name"].get<std::string>() +
    " has no return statement");
}

// Helper method to extract type information from annotations
std::pair<std::string, typet>
python_converter::extract_type_info(const nlohmann::json &ast_node)
{
  std::string lhs_type("");
  typet element_type;

  if (ast_node.contains("annotation"))
  {
    size_t type_size = get_type_size(ast_node);
    if (ast_node["annotation"]["_type"] == "Subscript")
      lhs_type = ast_node["annotation"]["value"]["id"];
    else
      lhs_type = ast_node["annotation"]["id"];

    if (lhs_type == "list")
      element_type = type_handler_.get_list_type(ast_node["value"]);
    else
      element_type = type_handler_.get_typet(lhs_type, type_size);
  }

  return {lhs_type, element_type};
}

// Helper method to create LHS expression based on target type
exprt python_converter::create_lhs_expression(
  const nlohmann::json &target,
  symbolt *lhs_symbol,
  const locationt &location)
{
  exprt lhs;
  const auto &target_type = target["_type"];

  if (target_type == "Attribute" || target_type == "Subscript")
  {
    is_converting_lhs = true;
    lhs = get_expr(target);
    is_converting_lhs = false;
  }
  else
    lhs = symbol_expr(*lhs_symbol);

  lhs.location() = location;
  return lhs;
}

// Helper method to handle post-assignment type adjustments
void python_converter::handle_assignment_type_adjustments(
  symbolt *lhs_symbol,
  exprt &lhs,
  exprt &rhs,
  const std::string &lhs_type,
  const nlohmann::json &ast_node,
  bool is_ctor_call)
{
  // Handle lambda assignments
  if (
    ast_node.contains("value") && ast_node["value"].contains("_type") &&
    ast_node["value"]["_type"] == "Lambda" && rhs.is_symbol())
  {
    const symbolt *lambda_func_symbol =
      symbol_table_.find_symbol(rhs.identifier());
    if (lambda_func_symbol && lhs_symbol)
    {
      if (lambda_func_symbol->type.is_code())
      {
        typet func_ptr_type = gen_pointer_type(lambda_func_symbol->type);
        lhs_symbol->type = func_ptr_type;
        lhs.type() = func_ptr_type;
        rhs = address_of_exprt(rhs);
      }
      else
      {
        throw std::runtime_error(
          "Lambda function symbol does not have code type");
      }
    }
  }
  else if (lhs_symbol)
  {
    // Handle string-to-string variable assignments
    if (lhs_type == "str" && rhs.is_symbol())
    {
      symbolt *rhs_symbol = symbol_table_.find_symbol(rhs.identifier());
      if (
        rhs_symbol && rhs_symbol->value.is_constant() &&
        rhs_symbol->value.type().is_array())
      {
        rhs = rhs_symbol->value;
        lhs_symbol->type = rhs.type();
        lhs.type() = rhs.type();
      }
    }
    // Array to pointer decay
    else if (lhs.type().id().empty() && rhs.type().is_array())
    {
      const typet &element_type = to_array_type(rhs.type()).subtype();
      typet pointer_type = gen_pointer_type(element_type);
      lhs_symbol->type = pointer_type;
      lhs.type() = pointer_type;
      rhs = get_array_base_address(rhs);
    }
    // String and list type size adjustments
    else if (
      lhs_type == "str" || lhs_type == "chr" || lhs_type == "ord" ||
      lhs_type == "list" || rhs.type().is_array())
    {
      if (!rhs.type().is_empty())
      {
        lhs_symbol->type = rhs.type();
        lhs.type() = rhs.type();
      }
    }

    if (!rhs.type().is_empty() && !is_ctor_call)
      lhs_symbol->value = rhs;
  }
}

void python_converter::get_var_assign(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  // Extract type information
  auto [lhs_type, element_type] = extract_type_info(ast_node);
  current_element_type = element_type;

  exprt lhs;
  symbolt *lhs_symbol = nullptr;
  locationt location_begin;
  symbol_id sid = create_symbol_id();

  const auto &target = (ast_node.contains("targets")) ? ast_node["targets"][0]
                                                      : ast_node["target"];
  const auto &target_type = target["_type"];

  if (ast_node["_type"] == "AnnAssign")
  {
    // Extract name and set in symbol ID
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

    // Process RHS before LHS if in function scope
    exprt rhs;
    if (
      sid.to_string().find("@F") != std::string::npos &&
      sid.to_string().find("@C") == std::string::npos)
    {
      is_right = true;
      if (!ast_node["value"].is_null())
      {
        if (ast_node["_type"] != "Call")
          rhs = get_expr(ast_node["value"]);
      }
      is_right = false;
    }

    // Location and symbol lookup
    location_begin = get_location_from_decl(target);
    lhs_symbol = symbol_table_.find_symbol(sid.to_string().c_str());

    bool is_global = false;
    for (std::string &s : global_declarations)
    {
      if (s == sid.global_to_string())
      {
        is_global = true;
        lhs_symbol = symbol_table_.find_symbol(sid.global_to_string().c_str());
        break;
      }
    }

    // Symbol creation
    if (!lhs_symbol || !is_global)
    {
      std::string module_name = location_begin.get_file().as_string();
      symbolt symbol = create_symbol(
        module_name,
        name,
        sid.to_string(),
        location_begin,
        current_element_type);
      symbol.lvalue = true;
      symbol.file_local = true;
      symbol.is_extern = false;
      lhs_symbol = symbol_table_.move_symbol_to_context(symbol);
    }

    // Check for uninitialized usage
    for (std::string &s : local_loads)
    {
      if (lhs_symbol->id.as_string() == s)
      {
        throw std::runtime_error(
          "Variable " + sid.get_object() + " in function " +
          current_func_name_ + " is uninitialized.");
      }
      continue;
    }

    // Create LHS expression
    lhs = create_lhs_expression(target, lhs_symbol, location_begin);
  }
  else if (ast_node["_type"] == "Assign")
  {
    // Assign logic
    const auto &target = ast_node["targets"][0];
    const auto &name = (target["_type"] == "Subscript")
                         ? target["value"]["id"].get<std::string>()
                         : target["id"].get<std::string>();

    sid.set_object(name);
    lhs_symbol = symbol_table_.find_symbol(sid.to_string());

    bool is_global = false;
    for (std::string &s : global_declarations)
    {
      if (s == sid.global_to_string())
      {
        is_global = true;
        break;
      }
    }

    // Special handling for lambda assignments
    if (
      !lhs_symbol && !is_global && ast_node.contains("value") &&
      ast_node["value"].contains("_type") &&
      ast_node["value"]["_type"] == "Lambda")
    {
      locationt location = get_location_from_decl(target);
      std::string module_name = location.get_file().as_string();
      typet placeholder_type = pointer_typet(empty_typet());

      symbolt symbol = create_symbol(
        module_name, name, sid.to_string(), location, placeholder_type);
      symbol.lvalue = true;
      symbol.file_local = true;
      symbol.is_extern = false;
      lhs_symbol = symbol_table_.move_symbol_to_context(symbol);
    }

    if (!lhs_symbol && !is_global)
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
    // Handle type adjustments
    handle_assignment_type_adjustments(
      lhs_symbol, lhs, rhs, lhs_type, ast_node, is_ctor_call);

    // Function call handling
    if (rhs.is_function_call())
    {
      if (is_ctor_call)
      {
        std::string func_name = ast_node["value"]["func"]["id"];
        if (base_ctor_called)
        {
          auto class_node =
            json_utils::find_class((*ast_json)["body"], func_name);
          func_name = class_node["bases"][0]["id"].get<std::string>();
          base_ctor_called = false;
        }
        update_instance_from_self(
          func_name, func_name, lhs_symbol->id.as_string());
      }

      if (!rhs.type().is_pointer() && !rhs.type().is_empty())
        rhs.op0() = lhs;

      target_block.copy_to_operands(rhs);
      current_lhs = nullptr;
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

/// Resolve the type of a variable from AST annotations or the symbol table
typet python_converter::resolve_variable_type(
  const std::string &var_name,
  const locationt &loc)
{
  nlohmann::json decl_node = get_var_node(var_name, *ast_json);

  if (!decl_node.empty())
  {
    std::string type_annotation =
      decl_node["annotation"]["id"].get<std::string>();
    return type_handler_.get_typet(type_annotation);
  }

  std::string filename = loc.get_file().as_string();
  std::string function = loc.get_function().as_string();
  std::string symbol_id = "py:" + filename + "@F@" + function + "@" + var_name;

  const symbolt *sym = symbol_table_.find_symbol(symbol_id);
  if (sym != nullptr)
    return sym->type;
  else
  {
    log_error(
      "Variable '{}' not found in symbol table; cannot determine type.",
      symbol_id);
    abort();
  }
}

void python_converter::get_compound_assign(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  locationt loc = get_location_from_decl(ast_node);

  // Set flags for LHS processing
  is_converting_lhs = true;

  // Get the target expression first
  exprt lhs = get_expr(ast_node["target"]);

  // Reset LHS flag and set RHS flag
  is_converting_lhs = false;
  is_converting_rhs = true;

  std::string var_name;

  // Extract variable name based on target type
  if (ast_node["target"].contains("id"))
  {
    // Simple variable assignment: x += 1
    var_name = ast_node["target"]["id"].get<std::string>();
  }
  else if (ast_node["target"]["_type"] == "Attribute")
  {
    // Don't extract just the attribute name for type resolution
    // The type should come from the LHS expression we just created
    if (ast_node["target"].contains("attr"))
      var_name = ast_node["target"]["attr"].get<std::string>();
  }
  else if (ast_node["target"]["_type"] == "Subscript")
  {
    // Subscript assignment: arr[i] += 1
    throw std::runtime_error(
      "Subscript assignment not supported in compound assignment");
  }
  else
  {
    throw std::runtime_error(
      "Unsupported target type in compound assignment: " +
      ast_node["target"]["_type"].get<std::string>());
  }

  // For attribute assignments, use the type from the LHS expression
  // For other assignments, resolve the variable type
  if (!lhs.type().is_nil() && !lhs.type().id().empty())
    current_element_type = lhs.type();
  else
  {
    // Fallback to resolving the variable type from AST or symbol table
    current_element_type = resolve_variable_type(var_name, loc);
  }

  exprt rhs = get_binary_operator_expr(ast_node);

  // Reset RHS flag
  is_converting_rhs = false;

  code_assignt code_assign(lhs, rhs);
  code_assign.location() = loc;
  target_block.copy_to_operands(code_assign);
}

typet resolve_ternary_type(
  const typet &then_type,
  const typet &else_type,
  const typet &default_type)
{
  if (then_type == else_type)
    return then_type;

  // Enhanced numeric promotion: int < float
  if (
    (then_type.is_signedbv() || then_type.is_unsignedbv()) &&
    else_type.is_floatbv())
    return else_type;
  if (
    (else_type.is_signedbv() || else_type.is_unsignedbv()) &&
    then_type.is_floatbv())
    return then_type;

  // Both arrays (strings)
  if (then_type.is_array() && else_type.is_array())
    return then_type;

  // Mixed signed/unsigned integers - prefer signed for safety
  if (then_type.is_signedbv() && else_type.is_unsignedbv())
    return then_type;
  if (then_type.is_unsignedbv() && else_type.is_signedbv())
    return else_type;

  // Incompatible types
  log_warning(
    "Ternary branches have incompatible types: {} vs {}, using default {}",
    then_type.id_string(),
    else_type.id_string(),
    default_type.id_string());

  return default_type;
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
    // Resolve result type based on branch types
    typet result_type =
      resolve_ternary_type(then.type(), else_expr.type(), current_element_type);

    // Create fully symbolic if expression
    exprt if_expr("if", result_type);
    if_expr.copy_to_operands(cond, then, else_expr);
    return if_expr;
  }

  // Create if or while code
  codet code;
  if (type == "If")
    code.set_statement("ifthenelse");
  else if (type == "While")
    code.set_statement("while");

  // Set location for the conditional statement
  code.location() = get_location_from_decl(ast_node);

  // Append "then" block
  code.copy_to_operands(cond, then);
  if (!else_expr.id_string().empty())
    code.copy_to_operands(else_expr);

  return std::move(code);
}

typet python_converter::get_type_from_annotation(
  const nlohmann::json &annotation_node,
  const nlohmann::json &element)
{
  if (annotation_node["_type"] == "Subscript")
    return type_handler_.get_list_type(element);
  else if (
    annotation_node["_type"] == "Constant" || annotation_node["_type"] == "Str")
  {
    // Handle string annotations like "CoordinateData | None" (forward references)
    std::string type_string = annotation_node["value"].get<std::string>();
    type_string = type_utils::remove_quotes(type_string);
    return type_handler_.get_typet(type_string);
  }
  else if (annotation_node.contains("id"))
    return type_handler_.get_typet(annotation_node["id"].get<std::string>());
  else
  {
    throw std::runtime_error(
      "Unsupported annotation type: " +
      annotation_node["_type"].get<std::string>());
  }
}

bool python_converter::function_has_missing_return_paths(
  const nlohmann::json &function_node)
{
  const auto &body = function_node["body"];
  if (body.empty())
    return true;

  // Check if the last statement is a return
  const auto &last_stmt = body.back();
  if (last_stmt["_type"] == "Return")
    return false;

  // Check for if-else structures at the end
  if (last_stmt["_type"] == "If")
  {
    // Check if both if and else branches have returns
    bool if_has_return = false;
    bool else_has_return = false;

    // Check if branch
    if (!last_stmt["body"].empty())
    {
      const auto &if_last = last_stmt["body"].back();
      if_has_return = (if_last["_type"] == "Return");
    }

    // Check else branch
    if (last_stmt.contains("orelse") && !last_stmt["orelse"].empty())
    {
      const auto &else_last = last_stmt["orelse"].back();
      else_has_return = (else_last["_type"] == "Return");
    }

    return !(if_has_return && else_has_return);
  }

  return true; // No explicit return found
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
    // Handles both cases
    if (return_type == "list" || return_type == "List")
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
      else if (return_stmt["value"]["_type"] == "List")
      {
        // Handle direct list literal returns like "return [1, 2, 3]"
        type.return_type() = type_handler_.get_list_type(return_stmt["value"]);
      }
    }
    else
    {
      type.return_type() =
        type_handler_.get_typet(return_type.get<std::string>());
    }
  }
  else if (return_node["_type"] == "Tuple")
  {
    // Handle tuple return types like (int, str)
    // TODO: we must still handle tuple types!
    type.return_type() = type_handler_.get_typet(std::string("tuple"));
  }
  else if (return_node["_type"] == "Constant" || return_node["_type"] == "Str")
  {
    // Handle string annotations using the helper method
    std::string type_string = return_node["value"].get<std::string>();
    type_string = type_utils::remove_quotes(type_string);
    type.return_type() = type_handler_.get_typet(type_string);
  }
  else
    throw std::runtime_error("Return type undefined");

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

      // Use the helper method to get type from annotation
      arg_type = get_type_from_annotation(element["annotation"], element);
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

  // Add missing return statement validation
  // Check if function has non-void return type and missing return paths
  // Skip constructors (they don't need explicit returns)
  if (
    !type.return_type().is_empty() &&
    type.return_type().id() != typet::t_empty &&
    type.return_type().id() != "constructor" && // Skip constructors
    function_has_missing_return_paths(function_node))
  {
    // Add assertion that fails to indicate missing return statement
    locationt missing_return_loc = get_location_from_decl(function_node);

    // Create a false assertion
    exprt false_expr = gen_boolean(false);
    code_assertt missing_return_assert;
    missing_return_assert.assertion() = false_expr;
    missing_return_assert.location() = missing_return_loc;

    // Add descriptive comment to help users understand the error
    missing_return_assert.location().comment(
      "Missing return statement detected in function '" + current_func_name_ +
      "'");

    function_body.copy_to_operands(missing_return_assert);
  }

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
      const std::string &annotated_type =
        stmt["annotation"]["id"].get<std::string>();
      typet type;
      if (annotated_type == "str")
        type = gen_pointer_type(char_type());
      else
        type =
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
  irept::subt &base_ids = clazz.add("bases").get_sub();
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

    base_ids.emplace_back(class_symbol->id);

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
        const auto &class_node = find_class((*ast_json)["body"], class_name);
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
  exprt return_value = get_expr(ast_node["value"]);
  locationt location = get_location_from_decl(ast_node);

  // Check if return value is a function call
  if (return_value.is_function_call() && ast_node["value"]["_type"] == "Call")
  {
    // Extract function name for temporary variable naming
    std::string func_name;
    if (ast_node["value"]["func"]["_type"] == "Name")
      func_name = ast_node["value"]["func"]["id"].get<std::string>();
    else if (ast_node["value"]["func"]["_type"] == "Attribute")
      func_name = ast_node["value"]["func"]["attr"].get<std::string>();
    else
      func_name = "func"; // fallback

    // Create temporary variable to store function call result
    symbolt temp_symbol =
      create_return_temp_variable(return_value.type(), location, func_name);
    symbol_table_.add(temp_symbol);
    exprt temp_var_expr = symbol_expr(temp_symbol);

    // Create declaration for temporary variable
    code_declt temp_decl(temp_var_expr);
    temp_decl.location() = location;
    target_block.copy_to_operands(temp_decl);

    // Set the LHS of the function call to our temporary variable
    if (!return_value.type().is_empty())
      return_value.op0() = temp_var_expr;

    // Add the function call statement to the block
    target_block.copy_to_operands(return_value);

    // Return the temporary variable
    code_returnt return_code;
    return_code.return_value() = temp_var_expr;
    return_code.location() = location;
    target_block.copy_to_operands(return_code);
  }
  else
  {
    // Original behavior for non-function-call returns
    code_returnt return_code;
    return_code.return_value() = return_value;
    return_code.location() = location;
    target_block.copy_to_operands(return_code);
  }
}

// Helper function to create temporary variable for function call results
symbolt python_converter::create_assert_temp_variable(const locationt &location)
{
  symbol_id temp_sid = create_symbol_id();
  temp_sid.set_object("__assert_temp");
  std::string temp_sid_str = temp_sid.to_string();

  symbolt temp_symbol;
  temp_symbol.id = temp_sid_str;
  temp_symbol.name = temp_sid_str;
  temp_symbol.type = bool_type();
  temp_symbol.lvalue = true;
  temp_symbol.static_lifetime = false;
  temp_symbol.location = location;

  return temp_symbol;
}

// Helper function to create function call statement from function call expression
code_function_callt create_function_call_statement(
  const exprt &func_call_expr,
  const exprt &lhs_var,
  const locationt &location)
{
  code_function_callt function_call;
  function_call.lhs() = lhs_var;
  function_call.function() = func_call_expr.operands()[1];

  const exprt &args_operand = func_call_expr.operands()[2];
  code_function_callt::argumentst arguments;
  for (const auto &arg : args_operand.operands())
  {
    arguments.push_back(arg);
  }
  function_call.arguments() = arguments;
  function_call.location() = location;

  return function_call;
}

exprt python_converter::get_block(const nlohmann::json &ast_block)
{
  code_blockt block, *old_block = current_block;
  current_block = &block;

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
    case StatementType::FOR_STATEMENT:
    {
      // For loops are transformed to while loops by the preprocessor
      // This case should not be reached in normal operation
      throw std::runtime_error(
        "For loops should be preprocessed before reaching converter");
    }
    case StatementType::COMPOUND_ASSIGN:
    {
      get_compound_assign(element, block);
      break;
    }
    case StatementType::FUNC_DEFINITION:
    {
      get_function_definition(element);
      global_declarations.clear();
      local_loads.clear();
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

      // Check for function calls in assertions (direct or negated)
      const exprt *func_call_expr = nullptr;
      bool is_negated = false;

      // Case 1: Direct function call - assert func()
      if (test.id() == "code" && test.get("statement") == "function_call")
      {
        func_call_expr = &test;
        is_negated = false;
      }
      // Case 2: Negated function call - assert not func()
      else if (
        test.id() == "not" && test.operands().size() == 1 &&
        test.operands()[0].id() == "code" &&
        test.operands()[0].get("statement") == "function_call")
      {
        func_call_expr = &test.operands()[0];
        is_negated = true;
      }

      // Transform function call assertions to prevent solver crashes
      if (func_call_expr != nullptr)
      {
        locationt location = get_location_from_decl(element);

        // Create temporary variable
        symbolt temp_symbol = create_assert_temp_variable(location);
        symbol_table_.add(temp_symbol);
        exprt temp_var_expr = symbol_expr(temp_symbol);

        // Create function call statement: temp_var = func(args)
        code_function_callt function_call = create_function_call_statement(
          *func_call_expr, temp_var_expr, location);
        block.move_to_operands(function_call);

        // Create appropriate assertion based on negation
        exprt assertion_expr;
        if (is_negated)
        {
          // assert not func() -> assert !temp_var
          assertion_expr = not_exprt(temp_var_expr);
        }
        else
        {
          // assert func() -> assert temp_var == 1
          exprt cast_expr = typecast_exprt(temp_var_expr, signedbv_typet(32));
          exprt one_expr = constant_exprt("1", signedbv_typet(32));
          assertion_expr = equality_exprt(cast_expr, one_expr);
        }

        code_assertt assert_code;
        assert_code.assertion() = assertion_expr;
        assert_code.location() = location;
        block.move_to_operands(assert_code);
      }
      else
      {
        // For expressions without function calls, use direct assertion
        code_assertt assert_code;
        assert_code.assertion() = test;
        assert_code.location() = get_location_from_decl(element);
        block.move_to_operands(assert_code);
      }
      break;
    }
    case StatementType::EXPR:
    {
      // Function calls are handled here
      exprt empty;
      exprt expr = get_expr(element["value"]);
      if (expr != empty)
      {
        codet code_stmt = convert_expression_to_code(expr);
        block.move_to_operands(code_stmt);
      }

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
    case StatementType::GLOBAL:
    {
      symbol_id sid = create_symbol_id();
      for (const auto &item : element["names"])
      {
        sid.set_object(item);
        global_declarations.push_back(sid.global_to_string());
      }
      break;
    }
    case StatementType::TRY:
    {
      exprt new_expr = codet("cpp-catch");
      exprt try_block = get_block(element["body"]);
      exprt handler = get_block(element["handlers"]);
      new_expr.move_to_operands(try_block);

      for (const auto &op : handler.operands())
        new_expr.copy_to_operands(op);

      block.move_to_operands(new_expr);
      break;
    }
    case StatementType::EXCEPTHANDLER:
    {
      exprt catch_block = get_block(element["body"]);

      // Check if the exception handler specifies an exception type
      if (!element["type"].is_null())
      {
        typet type =
          type_handler_.get_typet(element["type"]["id"].get<std::string>());
        catch_block.type() = type;

        std::string name;
        symbol_id sid = create_symbol_id();
        locationt location = get_location_from_decl(element);
        std::string module_name = location.get_file().as_string();
        // Check if the exception handler binds the exception to a variable
        if (!element["name"].is_null())
          name = element["name"].get<std::string>();
        else
          name = "__anon_exc_var_" + location.get_line().as_string();

        sid.set_object(name);
        symbolt symbol = create_symbol(
          module_name, current_func_name_, sid.to_string(), location, type);
        symbol.name = name;
        symbol.lvalue = true;
        symbol.is_extern = false;
        symbol.file_local = false;
        symbolt *s = symbol_table_.move_symbol_to_context(symbol);

        exprt sym = symbol_expr(*s);
        code_declt decl(sym);
        exprt decl_code = convert_expression_to_code(decl);
        decl_code.location() = location;
        codet::operandst &ops = catch_block.operands();
        ops.insert(ops.begin(), decl_code);
      }

      block.move_to_operands(catch_block);
      break;
    }
    case StatementType::RAISE:
    {
      exprt raise = get_expr(element["exc"]);
      typet type = type_handler_.get_typet(
        element["exc"]["func"]["id"].get<std::string>());
      locationt location = get_location_from_decl(element);
      if (raise.is_function_call() && raise.type().id() == "constructor")
      {
        // This logic should be applied to all constructor calls
        // Using sideeffect will convert class(); into
        // DECL tmp; class(tmp);
        code_function_callt call =
          to_code_function_call(convert_expression_to_code(raise));
        side_effect_expr_function_callt tmp;
        tmp.function() = call.function();
        tmp.arguments() = call.arguments();
        tmp.type() = type;
        tmp.location() = location;
        raise = tmp;
      }

      exprt side = side_effect_exprt("cpp-throw", type);
      side.location() = location;
      side.move_to_operands(raise);

      codet code_expr("expression");
      code_expr.operands().push_back(side);
      block.move_to_operands(code_expr);
      break;
    }

    /* "https://docs.python.org/3/tutorial/controlflow.html:
     * "The pass statement does nothing. It can be used when a statement
     *  is required syntactically but the program requires no action." */
    case StatementType::PASS:
    // Imports are handled by parser.py so we can just ignore here.
    case StatementType::IMPORT:
      // TODO: Raises are ignored for now. Handling case to avoid calling abort() on default.
      break;
    case StatementType::UNKNOWN:
    default:
      throw std::runtime_error(
        element["_type"].get<std::string>() + " statements are not supported");
    }
  }

  current_block = old_block;

  return std::move(block);
}

python_converter::python_converter(
  contextt &_context,
  const nlohmann::json *ast,
  const global_scope &gs)
  : symbol_table_(_context),
    ast_json(ast),
    global_scope_(gs),
    type_handler_(*this),
    sym_generator_("python_converter::"),
    ns(_context),
    current_func_name_(""),
    current_class_name_(""),
    current_block(nullptr),
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

static void add_global_static_variable(
  contextt &ctx,
  const typet t,
  const std::string &name)
{
  std::string id = "c:@" + name;
  symbolt symbol;
  symbol.mode = "C";
  symbol.type = std::move(t);
  symbol.name = name;
  symbol.id = id;

  symbol.lvalue = true;
  symbol.static_lifetime = true;
  symbol.is_extern = false;
  symbol.file_local = false;
  symbol.value = gen_zero(t, true);
  symbol.value.zero_initializer(true);

  symbolt *added_symbol = ctx.move_symbol_to_context(symbol);
  assert(added_symbol);
}

void python_converter::load_c_intrisics()
{
  // Add symbols required by the C models

  add_global_static_variable(
    symbol_table_, long_long_int_type(), "__ESBMC_rounding_mode");

  auto type1 = array_typet(bool_type(), exprt("infinity"));
  add_global_static_variable(symbol_table_, type1, "__ESBMC_alloc");
  add_global_static_variable(symbol_table_, type1, "__ESBMC_deallocated");
  add_global_static_variable(symbol_table_, type1, "__ESBMC_is_dynamic");

  auto type2 = array_typet(size_type(), exprt("infinity"));
  add_global_static_variable(symbol_table_, type2, "__ESBMC_alloc_size");
}

///  Only addresses __name__; other Python built-ins such as
/// __file__, __doc__, __package__ are unsupported
void python_converter::create_builtin_symbols()
{
  // Create __name__ symbol
  symbol_id name_sid(current_python_file, "", "");
  name_sid.set_object("__name__");

  locationt location;
  location.set_file(current_python_file.c_str());
  location.set_line(1);

  std::string module_name =
    current_python_file.substr(0, current_python_file.find_last_of("."));

  // Determine the value of __name__ based on whether this is the main module or imported
  std::string name_value;
  if (current_python_file == main_python_file)
    name_value = "__main__";
  else
  {
    // Extract module name from filename (e.g., "/path/to/other.py" -> "other")
    size_t last_slash = current_python_file.find_last_of("/\\");
    size_t last_dot = current_python_file.find_last_of(".");
    if (
      last_slash != std::string::npos && last_dot != std::string::npos &&
      last_dot > last_slash)
    {
      name_value =
        current_python_file.substr(last_slash + 1, last_dot - last_slash - 1);
    }
    else if (last_dot != std::string::npos)
      name_value = current_python_file.substr(0, last_dot);
    else
      name_value = current_python_file; // fallback
  }

  typet string_type =
    type_handler_.build_array(char_type(), name_value.size() + 1);

  // Create the symbol
  symbolt name_symbol = create_symbol(
    module_name, "__name__", name_sid.to_string(), location, string_type);

  name_symbol.lvalue = true;
  name_symbol.static_lifetime = true;
  name_symbol.is_extern = false;
  name_symbol.file_local = false;

  // Set the value
  exprt name_expr = gen_zero(string_type);
  const typet &char_type_ref = string_type.subtype();

  for (size_t i = 0; i < name_value.size(); ++i)
  {
    uint8_t ch = name_value[i];
    exprt char_value = constant_exprt(
      integer2binary(BigInt(ch), bv_width(char_type_ref)),
      integer2string(BigInt(ch)),
      char_type_ref);
    name_expr.operands().at(i) = char_value;
  }

  // Add null terminator
  exprt null_char = constant_exprt(
    integer2binary(BigInt(0), bv_width(char_type_ref)),
    integer2string(BigInt(0)),
    char_type_ref);
  name_expr.operands().at(name_value.size()) = null_char;

  name_symbol.value = name_expr;

  // Add to symbol table
  symbol_table_.add(name_symbol);
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

  main_python_file = (*ast_json)["filename"].get<std::string>();
  current_python_file = main_python_file;

  // Create built-in symbols for main module (__name__ = "__main__")
  create_builtin_symbols();

  if (!config.options.get_bool_option("no-library"))
  {
    // Load operational models -----
    const std::string &ast_output_dir =
      (*ast_json)["ast_output_dir"].get<std::string>();
    std::list<std::string> model_files = {
      "range", "int", "consensus", "random", "exceptions"};
    std::list<std::string> model_folders = {"os", "numpy"};

    for (const auto &folder : model_folders)
      append_models_from_directory(model_files, ast_output_dir + "/" + folder);

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

      exprt model_code =
        with_ast(&model_json, [&]() { return get_block((*ast_json)["body"]); });

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
    for (const auto &element : (*ast_json)["body"])
    {
      if (element["_type"] == "FunctionDef" && element["name"] == function)
      {
        function_node = element;
        break;
      }
    }

    if (function_node.empty())
      throw std::runtime_error("Function " + function + " not found");

    code_blockt block;
    // Convert classes referenced by the function
    for (const auto &clazz : global_scope_.classes())
    {
      const auto &class_node = find_class((*ast_json)["body"], clazz);
      get_class_definition(class_node, block);
      current_class_name_.clear();
    }

    // Convert only the global variables referenced by the function
    for (const auto &global_var : global_scope_.variables())
    {
      const auto &var_node = find_var_decl(global_var, "", *ast_json);
      get_var_assign(var_node, block);
    }

    // Convert function arguments types
    for (const auto &arg : function_node["args"]["args"])
    {
      // Check if annotation exists and is not null before accessing "id"
      if (
        arg.contains("annotation") && !arg["annotation"].is_null() &&
        arg["annotation"].contains("id"))
      {
        auto node = find_class((*ast_json)["body"], arg["annotation"]["id"]);
        if (!node.empty())
          get_class_definition(node, block);
      }
    }

    // Convert a single function
    get_function_definition(function_node);

    // Get function symbol
    symbol_id sid = create_symbol_id();
    sid.set_function(function);
    symbolt *symbol = symbol_table_.find_symbol(sid.to_string());

    if (!symbol)
      throw std::runtime_error("Symbol " + sid.to_string() + " not found");

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
    for (const auto &elem : (*ast_json)["body"])
    {
      if (elem["_type"] == "ImportFrom" || elem["_type"] == "Import")
      {
        is_importing_module = true;
        const std::string &module_name = (elem["_type"] == "ImportFrom")
                                           ? elem["module"]
                                           : elem["names"][0]["name"];
        std::stringstream module_path;
        module_path << (*ast_json)["ast_output_dir"].get<std::string>() << "/"
                    << module_name << ".json";
        std::ifstream imported_file(module_path.str());
        imported_file >> imported_module_json;

        current_python_file =
          imported_module_json["filename"].get<std::string>();
        imported_modules.emplace(module_name, current_python_file);

        // Create built-in symbols for imported module (__name__ = module_name)
        create_builtin_symbols();

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
    exprt main_block = get_block((*ast_json)["body"]);
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
