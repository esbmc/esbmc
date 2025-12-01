#include <python-frontend/char_utils.h>
#include <python-frontend/convert_float_literal.h>
#include <python-frontend/function_call_builder.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/module_locator.h>
#include <python-frontend/python_annotation.h>
#include <python-frontend/python_class_builder.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_dict_handler.h>
#include <python-frontend/python_list.h>
#include <python-frontend/python_typechecking.h>
#include <python-frontend/string_builder.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/tuple_handler.h>
#include <python-frontend/type_utils.h>
#include <util/arith_tools.h>
#include <util/base_type.h>
#include <util/c_typecast.h>
#include <util/c_types.h>
#include <util/encoding.h>
#include <util/expr_util.h>
#include <util/message.h>
#include <util/python_types.h>
#include <util/std_code.h>
#include <util/string_constant.h>
#include <util/symbolic_types.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <regex>
#include <stdexcept>
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
  {"is", "="},          {"isnot", "not"},     {"in", "="}};

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
  {"ExceptHandler", StatementType::EXCEPTHANDLER},
  {"Delete", StatementType::DELETE}};

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

  // If the type is floating-point, use IEEE-specific operators.
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

codet python_converter::convert_expression_to_code(exprt &expr)
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
    {"Set", ExpressionType::LIST},
    {"Lambda", ExpressionType::FUNC_CALL},
    {"JoinedStr", ExpressionType::FSTRING},
    {"Tuple", ExpressionType::TUPLE},
    {"Dict", ExpressionType::LITERAL}};

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
  bool contains_non_boolean = false;
  // Iterate over operands of logical operations (and/or)
  for (const auto &operand : element["values"])
  {
    exprt operand_expr = get_expr(operand);
    logical_expr.copy_to_operands(operand_expr);
    contains_non_boolean |= !operand_expr.is_boolean();
  }
  // Shockingly enough, a BoolOp may not return a boolean.
  if (contains_non_boolean)
  {
    typet t = extract_type_from_boolean_op(logical_expr).type();
    // Are we dealing with an actual bool expression?
    if (t.is_bool())
      return logical_expr;
    // Result expression starts from last operand as default else branch
    exprt result_expr = logical_expr.operands().back();
    for (int i = logical_expr.operands().size() - 2; i >= 0; i--)
    {
      const exprt &current = logical_expr.operands()[i];
      exprt if_expr("if", t);
      if (logical_expr.is_and())
        if_expr.copy_to_operands(current, result_expr, current);
      else
        if_expr.copy_to_operands(current, current, result_expr);

      result_expr = if_expr;
    }
    return result_expr;
  }
  return logical_expr;
}

void python_converter::update_symbol(const exprt &expr) const
{
  // Don't update if expression has no name
  // prevents corruption of function symbols
  if (expr.name().empty())
  {
    log_debug(
      "python-frontend",
      "[update_symbol]: skipping symbol update since expression has no name");
    return;
  }

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

void python_converter::adjust_statement_types(exprt &lhs, exprt &rhs) const
{
  typet &lhs_type = lhs.type();
  typet &rhs_type = rhs.type();

  // Case 1: Promote RHS integer constant to float if LHS expects a float
  if (
    lhs_type.is_floatbv() && rhs.is_constant() &&
    type_utils::is_integer_type(rhs_type))
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
  else if (rhs_type.is_floatbv() && type_utils::is_integer_type(lhs_type))
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
    if (lhs_op.is_constant() && type_utils::is_integer_type(lhs_op.type()))
      math_handler_.promote_int_to_float(lhs_op, float_type);
    // For non-constant operands, create explicit typecast
    else if (!lhs_op.type().is_floatbv())
      lhs_op = typecast_exprt(lhs_op, float_type);

    if (rhs_op.is_constant() && type_utils::is_integer_type(rhs_op.type()))
      math_handler_.promote_int_to_float(rhs_op, float_type);
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

inline bool is_ieee_op(const exprt &expr)
{
  const std::string &id = expr.id().as_string();
  return id == "ieee_add" || id == "ieee_mul" || id == "ieee_sub" ||
         id == "ieee_div";
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

std::pair<exprt, exprt> python_converter::resolve_comparison_operands_internal(
  const exprt &lhs,
  const exprt &rhs)
{
  exprt resolved_lhs = lhs;
  exprt resolved_rhs = rhs;

  // Only resolve constant arrays, not pointers
  if (lhs.is_symbol() && lhs.type().is_array())
  {
    const symbolt *sym = symbol_table_.find_symbol(lhs.identifier());
    if (sym && sym->value.is_constant())
      resolved_lhs = sym->value;
  }

  if (rhs.is_symbol() && rhs.type().is_array())
  {
    const symbolt *sym = symbol_table_.find_symbol(rhs.identifier());
    if (sym && sym->value.is_constant())
      resolved_rhs = sym->value;
  }

  return {resolved_lhs, resolved_rhs};
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

  std::string rhs_str = string_handler_.extract_string_from_array_operands(rhs);

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
    std::string lhs_str =
      string_handler_.extract_string_from_array_operands(string_element);
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
  // Skip if either operand is a member expression
  if (lhs.is_member() || rhs.is_member())
    return nil_exprt();

  // Check if both are string types (either array or pointer to char)
  bool lhs_is_string =
    (lhs.type().is_array() && lhs.type().subtype() == char_type()) ||
    (lhs.type().is_pointer() && lhs.type().subtype() == char_type());
  bool rhs_is_string =
    (rhs.type().is_array() && rhs.type().subtype() == char_type()) ||
    (rhs.type().is_pointer() && rhs.type().subtype() == char_type());

  // If both are strings (regardless of array vs pointer), let strcmp handle it
  if (lhs_is_string && rhs_is_string)
    return nil_exprt();

  // Types match exactly
  if (lhs.type() == rhs.type())
    return nil_exprt();

  // Both operands are arrays - need to distinguish between lists and strings
  if (lhs.type().is_array() && rhs.type().is_array())
  {
    // Check if these are different semantic types (list vs string)
    bool lhs_is_string_array = (lhs.type().subtype() == char_type());
    bool rhs_is_string_array = (rhs.type().subtype() == char_type());

    // If one is a string array and the other is not, they're different types
    if (lhs_is_string_array != rhs_is_string_array)
      return gen_boolean(op == "NotEq");

    // Both are string arrays: compare based on content
    bool lhs_empty = string_handler_.is_zero_length_array(lhs) ||
                     (lhs.is_constant() && lhs.operands().size() <= 1);
    bool rhs_empty = string_handler_.is_zero_length_array(rhs) ||
                     (rhs.is_constant() && rhs.operands().size() <= 1);

    if (lhs_empty != rhs_empty)
      return gen_boolean(op == "NotEq");

    if (lhs.size() != rhs.size())
      return gen_boolean(op == "NotEq");

    return nil_exprt();
  }

  // Mixed types (array vs non-array, but not both strings)
  // Let strcmp handle the comparison if they're both strings
  return nil_exprt();
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
  if (
    string_handler_.is_zero_length_array(resolved_lhs) &&
    string_handler_.is_zero_length_array(resolved_rhs))
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
    resolved_lhs = string_handler_.get_array_base_address(resolved_lhs);
  if (resolved_rhs.type().is_array())
    resolved_rhs = string_handler_.get_array_base_address(resolved_rhs);

  symbolt *strncmp_symbol = symbol_table_.find_symbol("c:@F@strcmp");
  if (!strncmp_symbol)
    throw std::runtime_error(
      "strcmp function not found in symbol table for string comparison");

  side_effect_expr_function_callt strcmp_call;
  strcmp_call.function() = symbol_expr(*strncmp_symbol);
  strcmp_call.arguments() = {resolved_lhs, resolved_rhs};
  strcmp_call.location() = get_location_from_decl(element);
  strcmp_call.type() = int_type();

  lhs = strcmp_call;
  rhs = gen_zero(int_type());

  return nil_exprt(); // continue with lhs OP rhs
}

exprt python_converter::create_char_comparison_expr(
  const std::string &op,
  const exprt &lhs_char_value,
  const exprt &rhs_char_value,
  const exprt &lhs_source,
  const exprt &rhs_source) const
{
  // Create comparison expression with integer operands
  exprt comp_expr(get_op(op, bool_type()), bool_type());
  comp_expr.copy_to_operands(lhs_char_value, rhs_char_value);

  // Preserve location from original operands
  if (!lhs_source.location().is_nil())
    comp_expr.location() = lhs_source.location();
  else if (!rhs_source.location().is_nil())
    comp_expr.location() = rhs_source.location();

  return comp_expr;
}

exprt python_converter::handle_single_char_comparison(
  const std::string &op,
  exprt &lhs,
  exprt &rhs)
{
  exprt lhs_char_value = python_char_utils::get_char_value_as_int(lhs, false);
  exprt rhs_char_value = python_char_utils::get_char_value_as_int(rhs, false);

  if (lhs_char_value.is_nil() || rhs_char_value.is_nil())
    return nil_exprt();

  return create_char_comparison_expr(
    op, lhs_char_value, rhs_char_value, lhs, rhs);
}

exprt python_converter::unwrap_optional_if_needed(const exprt &expr)
{
  if (!expr.type().is_struct())
    return expr;

  const struct_typet &struct_type = to_struct_type(expr.type());
  std::string tag = struct_type.tag().as_string();

  if (tag.starts_with("tag-Optional_"))
  {
    // Extract the value field
    member_exprt value_field(expr, "value", struct_type.components()[1].type());
    return value_field;
  }

  return expr;
}

exprt python_converter::handle_none_comparison(
  const std::string &op,
  const exprt &lhs,
  const exprt &rhs)
{
  const bool is_eq = (op == "Eq" || op == "Is");
  const bool lhs_is_none = (lhs.type() == none_type());
  const bool rhs_is_none = (rhs.type() == none_type());

  // Only handle actual None comparisons
  // If neither side is None, this is NOT a None comparison
  if (!lhs_is_none && !rhs_is_none)
    return exprt();

  // Handle None == None and None != None
  // Create isnone expression
  exprt isnone_expr("isnone", typet("bool"));
  isnone_expr.copy_to_operands(lhs);
  isnone_expr.copy_to_operands(rhs);

  // If checking inequality, wrap with not
  if (!is_eq)
  {
    exprt not_expr("not", typet("bool"));
    not_expr.move_to_operands(isnone_expr);
    return not_expr;
  }

  return isnone_expr;
}

exprt python_converter::handle_str_join(const nlohmann::json &call_json)
{
  // Validate JSON structure: ensure we have the required keys
  if (!call_json.contains("args") || call_json["args"].empty())
    throw std::runtime_error("join() missing required argument: 'iterable'");

  if (!call_json.contains("func"))
    throw std::runtime_error("invalid join() call");

  const auto &func = call_json["func"];

  // Verify this is an Attribute call (method call syntax: obj.method())
  // and has the value (the separator object)
  if (
    !func.contains("_type") || func["_type"] != "Attribute" ||
    !func.contains("value"))
    throw std::runtime_error("invalid join() call");

  // Extract separator: for " ".join(l), func["value"] is the Constant " "
  exprt separator = get_expr(func["value"]);
  string_handler_.ensure_string_array(separator);

  // Get the list argument (the iterable to join)
  const nlohmann::json &list_arg = call_json["args"][0];

  // Currently only support Name references (e.g., variable names)
  // TODO: Support direct List literals such as " ".join(["a", "b"])
  if (
    list_arg.contains("_type") && list_arg["_type"] == "Name" &&
    list_arg.contains("id"))
  {
    std::string var_name = list_arg["id"].get<std::string>();

    // Look up the variable in the AST to get its initialization value
    nlohmann::json var_decl =
      json_utils::find_var_decl(var_name, current_func_name_, *ast_json);

    if (var_decl.empty())
      throw std::runtime_error(
        "NameError: name '" + var_name + "' is not defined");

    // Ensure the variable is a list with elements array
    if (!var_decl.contains("value"))
      throw std::runtime_error("join() requires a list");

    const nlohmann::json &list_value = var_decl["value"];

    if (
      !list_value.contains("_type") || list_value["_type"] != "List" ||
      !list_value.contains("elts"))
      throw std::runtime_error("join() requires a list");

    // Get the list elements from the AST
    const auto &elements = list_value["elts"];

    // Edge case: empty list returns empty string
    if (elements.empty())
    {
      // Create a proper null-terminated empty string
      typet empty_string_type = type_handler_.build_array(char_type(), 1);
      exprt empty_str = gen_zero(empty_string_type);
      // Explicitly set the first (and only) element to null terminator
      empty_str.operands().at(0) = from_integer(0, char_type());
      return empty_str;
    }

    // Convert JSON elements to ESBMC expressions
    std::vector<exprt> elem_exprs;
    for (const auto &elem : elements)
    {
      exprt elem_expr = get_expr(elem);
      string_handler_.ensure_string_array(elem_expr);
      elem_exprs.push_back(elem_expr);
    }

    // Edge case: single element returns the element itself (no separator)
    if (elem_exprs.size() == 1)
      return elem_exprs[0];

    // Main algorithm: Build the joined string by extracting characters
    // from all elements and separators, then constructing a single string.
    // This avoids multiple concatenation operations which could cause
    // null terminator issues.
    string_builder &sb = get_string_builder();
    std::vector<exprt> all_chars;

    // Start with the first element
    std::vector<exprt> first_chars = sb.extract_string_chars(elem_exprs[0]);
    all_chars.insert(all_chars.end(), first_chars.begin(), first_chars.end());

    // For each remaining element: add separator, then add element
    for (size_t i = 1; i < elem_exprs.size(); ++i)
    {
      // Insert separator characters
      std::vector<exprt> sep_chars = sb.extract_string_chars(separator);
      all_chars.insert(all_chars.end(), sep_chars.begin(), sep_chars.end());

      // Insert element characters
      std::vector<exprt> elem_chars = sb.extract_string_chars(elem_exprs[i]);
      all_chars.insert(all_chars.end(), elem_chars.begin(), elem_chars.end());
    }

    // Build final null-terminated string from all collected characters
    return sb.build_null_terminated_string(all_chars);
  }

  throw std::runtime_error("join() argument must be a list of strings");
}

// Resolve symbol values to constants
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

// Resolve function calls (both identity functions and constant-returning functions)
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

// Check if a function returns a constant value
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

// Check if a function is an identity function (returns its parameter)
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
      string_handler_.get_array_base_address(lhs),
      string_handler_.get_array_base_address(rhs));
  }
  else
  {
    // Default identity comparison
    is_expr.copy_to_operands(lhs, rhs);
  }

  return is_expr;
}

/// Construct the negation of an 'is' expression, used for 'is not'
exprt python_converter::get_negated_is_expr(const exprt &lhs, const exprt &rhs)
{
  exprt is_expr = get_binary_operator_expr_for_is(lhs, rhs);
  exprt not_expr("not", bool_type());
  not_expr.copy_to_operands(is_expr);
  return not_expr;
}

/// Convert function calls to side effects
void python_converter::convert_function_calls_to_side_effects(
  exprt &lhs,
  exprt &rhs)
{
  auto to_side_effect_call = [](exprt &expr) {
    side_effect_expr_function_callt side_effect;
    code_function_callt &code = static_cast<code_function_callt &>(expr);
    side_effect.function() = code.function();
    side_effect.location() = code.location();
    side_effect.type() = code.type();
    side_effect.arguments() = code.arguments();
    expr = side_effect;
  };

  if (lhs.is_function_call())
    to_side_effect_call(lhs);
  if (rhs.is_function_call())
    to_side_effect_call(rhs);
}

/// Handle chained comparisons
exprt python_converter::handle_chained_comparisons_logic(
  const nlohmann::json &element,
  exprt &bin_expr)
{
  exprt cond("and", bool_type());
  cond.move_to_operands(bin_expr); // bin_expr compares left and comparators[0]

  for (size_t i = 0; i + 1 < element["comparators"].size(); ++i)
  {
    std::string op(element["ops"][i + 1]["_type"].get<std::string>());
    exprt logical_expr(get_op(op, bool_type()), bool_type());
    exprt op1 = get_expr(element["comparators"][i]);
    exprt op2 = get_expr(element["comparators"][i + 1]);

    convert_function_calls_to_side_effects(op1, op2);

    std::string op1_type = type_handler_.type_to_string(op1.type());
    std::string op2_type = type_handler_.type_to_string(op2.type());

    if (op1_type == "str" && op2_type == "str")
    {
      handle_string_comparison(op, op1, op2, element);
      exprt expr(get_op(op, bool_type()), bool_type());
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

exprt python_converter::handle_membership_operator(
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &element,
  bool invert)
{
  // Check if rhs is a dictionary (struct type with dict tag)
  typet rhs_resolved_type = rhs.type();
  if (rhs.is_symbol())
  {
    const symbolt *sym = symbol_table_.find_symbol(rhs.identifier());
    if (sym)
      rhs_resolved_type = sym->type;
  }

  if (rhs_resolved_type.id() == "symbol")
    rhs_resolved_type = ns.follow(rhs_resolved_type);

  if (rhs_resolved_type.is_struct())
  {
    const struct_typet &struct_type = to_struct_type(rhs_resolved_type);
    std::string tag = struct_type.tag().as_string();

    if (
      tag.find("dict_") != std::string::npos ||
      tag.find("tag-dict") != std::string::npos)
    {
      return dict_handler_->handle_dict_membership(lhs, rhs, invert);
    }
  }

  typet list_type = type_handler_.get_list_type();

  // Handle set/list membership:
  // "item" in [list/set] or "item" not in [list/set]
  if (rhs.type() == list_type)
  {
    python_list list(*this, element);
    exprt contains_expr = list.contains(lhs, rhs);
    return invert ? not_exprt(contains_expr) : contains_expr;
  }

  // Get string type identifiers
  std::string lhs_type = type_handler_.type_to_string(lhs.type());
  std::string rhs_type = type_handler_.type_to_string(rhs.type());

  // Handle string membership testing: "substr" in "string" or "substr" not in "string"
  if (
    lhs.type().is_pointer() || rhs.type().is_pointer() ||
    lhs.type().is_array() || rhs.type().is_array() || lhs_type == "str" ||
    rhs_type == "str")
  {
    exprt membership_expr =
      string_handler_.handle_string_membership(lhs, rhs, element);
    return invert ? not_exprt(membership_expr) : membership_expr;
  }

  throw std::runtime_error(
    std::string("Unsupported expression for '") + (invert ? "not in" : "in") +
    "' operation");
}

exprt python_converter::handle_string_type_mismatch(
  const exprt &lhs,
  const exprt &rhs,
  const std::string &op)
{
  bool lhs_is_string = type_utils::is_string_type(lhs.type());
  bool rhs_is_string = type_utils::is_string_type(rhs.type());

  // Check if we have a type mismatch
  if (!((lhs_is_string && !rhs_is_string) || (!lhs_is_string && rhs_is_string)))
    return nil_exprt(); // No mismatch, return nil to indicate no action taken

  exprt lhs_char_value = python_char_utils::get_char_value_as_int(lhs, false);
  exprt rhs_char_value = python_char_utils::get_char_value_as_int(rhs, false);

  if (!lhs_char_value.is_nil() && !rhs_char_value.is_nil())
  {
    return create_char_comparison_expr(
      op, lhs_char_value, rhs_char_value, lhs, rhs);
  }

  // Handle equality/inequality comparisons for other type mismatches
  if (op == "Eq" || op == "NotEq")
  {
    // Python allows this comparison but it always returns False for Eq and True for NotEq
    // For verification purposes, we model this as returning the expected constant value
    // This represents Python's behavior: str == int always evaluates to False
    return gen_boolean(op == "NotEq");
  }

  return nil_exprt(); // No action taken for other operators
}

void python_converter::resolve_dict_subscript_types(
  const nlohmann::json &left,
  const nlohmann::json &right,
  exprt &lhs,
  exprt &rhs)
{
  bool lhs_is_dict_subscript = type_utils::is_dict_subscript(left);
  bool rhs_is_dict_subscript = type_utils::is_dict_subscript(right);

  bool lhs_is_ptr = lhs.type().is_pointer();
  bool rhs_is_ptr = rhs.type().is_pointer();

  auto is_primitive_type = [](const typet &t) {
    return t.is_signedbv() || t.is_unsignedbv() || t.is_bool() ||
           t.is_floatbv();
  };

  bool lhs_is_primitive = is_primitive_type(lhs.type());
  bool rhs_is_primitive = is_primitive_type(rhs.type());

  // Case 1: LHS is dict subscript (returning pointer) and RHS is primitive
  if (lhs_is_dict_subscript && lhs_is_ptr && rhs_is_primitive)
  {
    exprt dict_expr = get_expr(left["value"]);
    if (
      dict_expr.type().is_struct() &&
      dict_handler_->is_dict_type(dict_expr.type()))
    {
      lhs = dict_handler_->handle_dict_subscript(
        dict_expr, left["slice"], rhs.type());
    }
  }

  // Case 2: RHS is dict subscript (returning pointer) and LHS is primitive
  if (rhs_is_dict_subscript && rhs_is_ptr && lhs_is_primitive)
  {
    exprt dict_expr = get_expr(right["value"]);
    if (
      dict_expr.type().is_struct() &&
      dict_handler_->is_dict_type(dict_expr.type()))
    {
      rhs = dict_handler_->handle_dict_subscript(
        dict_expr, right["slice"], lhs.type());
    }
  }

  // Case 3: Both sides are dict subscripts (returning pointers)
  // Default to long_int_type for dict-to-dict comparisons
  if (
    lhs_is_dict_subscript && rhs_is_dict_subscript && lhs_is_ptr && rhs_is_ptr)
  {
    typet default_type = long_int_type();

    exprt lhs_dict = get_expr(left["value"]);
    if (
      lhs_dict.type().is_struct() &&
      dict_handler_->is_dict_type(lhs_dict.type()))
    {
      lhs = dict_handler_->handle_dict_subscript(
        lhs_dict, left["slice"], default_type);
    }

    exprt rhs_dict = get_expr(right["value"]);
    if (
      rhs_dict.type().is_struct() &&
      dict_handler_->is_dict_type(rhs_dict.type()))
    {
      rhs = dict_handler_->handle_dict_subscript(
        rhs_dict, right["slice"], default_type);
    }
  }
}

// (annotation type collection & inheritance helpers moved to python_typechecking)

exprt python_converter::get_binary_operator_expr(const nlohmann::json &element)
{
  // Extract left and right operands from AST
  auto left = element.contains("left") ? element["left"] : element["target"];

  decltype(left) right;
  if (element.contains("right"))
    right = element["right"];
  else if (element.contains("comparators"))
    right = element["comparators"][0];
  else if (element.contains("value"))
    right = element["value"];

  // Convert operands to expressions
  exprt lhs = get_expr(left);
  exprt rhs = get_expr(right);

  // Resolve dictionary subscript types for proper comparison
  resolve_dict_subscript_types(left, right, lhs, rhs);

  // Extract operator
  std::string op;
  if (element.contains("op"))
    op = element["op"]["_type"].get<std::string>();
  else if (element.contains("ops"))
    op = element["ops"][0]["_type"].get<std::string>();
  assert(!op.empty());

  // Handle None comparisons (don't unwrap optionals for identity checks)
  bool is_none_check = handle_none_check_setup(op, lhs, rhs);
  if (!is_none_check)
  {
    lhs = unwrap_optional_if_needed(lhs);
    rhs = unwrap_optional_if_needed(rhs);
  }

  if (lhs.type() == none_type() || rhs.type() == none_type())
    return handle_none_comparison(op, lhs, rhs);

  // Handle exceptions
  if (lhs.statement() == "cpp-throw")
    return lhs;
  if (rhs.statement() == "cpp-throw")
    return rhs;

  attach_symbol_location(lhs, symbol_table());
  attach_symbol_location(rhs, symbol_table());

  // Handle set operations (difference, intersection, union)
  typet list_type = type_handler_.get_list_type();
  if (
    (lhs.type() == list_type || rhs.type() == list_type) &&
    (op == "Sub" || op == "BitAnd" || op == "BitOr"))
  {
    exprt set_result =
      handle_set_operations(op, lhs, rhs, left, right, element);
    if (!set_result.is_nil())
      return set_result;
  }

  // Handle membership operators
  if (op == "In")
    return handle_membership_operator(lhs, rhs, element, false);
  if (op == "NotIn")
    return handle_membership_operator(lhs, rhs, element, true);

  // Convert function calls to side effects
  convert_function_calls_to_side_effects(lhs, rhs);

  // Handle array/string operations
  if (lhs.type().is_array() || rhs.type().is_array())
  {
    exprt result = handle_array_operations(op, lhs, rhs, left, right, element);
    if (!result.is_nil())
      return result;
  }

  // Handle list operations
  exprt list_result =
    handle_list_operations(op, lhs, rhs, left, right, element);
  if (!list_result.is_nil())
    return list_result;

  // Handle identity comparisons
  if (op == "Is")
    return get_binary_operator_expr_for_is(lhs, rhs);
  if (op == "IsNot")
    return get_negated_is_expr(lhs, rhs);

  // Handle relational operation type mismatches
  if (type_utils::is_relational_op(op))
  {
    exprt result = handle_relational_type_mismatches(op, lhs, rhs, element);
    if (!result.is_nil())
      return result;
  }

  // Handle string operations
  exprt string_result =
    handle_string_binary_operations(op, lhs, rhs, left, right, element);
  if (!string_result.is_nil())
    return string_result;

  // Handle type mismatches
  exprt type_mismatch_result = handle_string_type_mismatch(lhs, rhs, op);
  if (!type_mismatch_result.is_nil())
    return type_mismatch_result;

  // Handle special mathematical operations
  if (op == "Pow" || op == "power")
    return math_handler_.handle_power(lhs, rhs);

  if (op == "Mod" && (lhs.type().is_floatbv() || rhs.type().is_floatbv()))
    return math_handler_.handle_modulo(lhs, rhs, element);

  // Build the binary expression
  exprt bin_expr = build_binary_expression(op, lhs, rhs);

  // Handle float vs char comparisons
  if (type_utils::is_float_vs_char(lhs, rhs))
    return handle_float_vs_string(bin_expr, op);

  // Handle floor division
  if (op == "FloorDiv")
    return math_handler_.handle_floor_division(lhs, rhs, bin_expr);

  // Promote operands for IEEE operations
  promote_ieee_operands(bin_expr, lhs, rhs);

  // Handle chained comparisons
  if (element.contains("comparators") && element["comparators"].size() > 1)
    return handle_chained_comparisons_logic(element, bin_expr);

  return bin_expr;
}

bool python_converter::handle_none_check_setup(
  const std::string &op,
  const exprt &lhs,
  const exprt &rhs)
{
  bool is_none_check = (op == "Is" || op == "IsNot") &&
                       (lhs.type() == none_type() || rhs.type() == none_type());

  if (!is_none_check && (op == "Eq" || op == "NotEq"))
  {
    if (lhs.type() == none_type() || rhs.type() == none_type())
      is_none_check = true;
  }

  return is_none_check;
}

exprt python_converter::handle_array_operations(
  const std::string &op,
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &left,
  const nlohmann::json &right,
  const nlohmann::json & /*element*/)
{
  if (!lhs.type().is_array() && !rhs.type().is_array())
    return nil_exprt();

  // Check for zero-length array comparisons
  if (
    string_handler_.is_zero_length_array(lhs) &&
    string_handler_.is_zero_length_array(rhs) && (op == "Eq" || op == "NotEq"))
  {
    return gen_boolean(op == "Eq");
  }

  // Handle string concatenation
  if (op == "Add")
    return string_handler_.handle_string_concatenation_with_promotion(
      lhs, rhs, left, right);

  return nil_exprt();
}

exprt python_converter::handle_list_operations(
  const std::string &op,
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &left,
  const nlohmann::json &right,
  const nlohmann::json &element)
{
  typet list_type = type_handler_.get_list_type();

  // Resolve function calls that return lists to temporary variables
  auto resolve_list_call = [&](exprt &expr) -> bool {
    // Check if this is a side effect function call
    if (expr.id().as_string() != "sideeffect")
      return false;

    if (expr.get("statement") != "function_call")
      return false;

    if (expr.type() != list_type)
      return false;

    locationt location = get_location_from_decl(element);

    // Create temporary variable for the list
    symbolt &tmp_var_symbol = create_tmp_symbol(
      element, "tmp_func_ret", list_type, gen_zero(list_type));

    // Declare the temporary
    code_declt tmp_var_decl(symbol_expr(tmp_var_symbol));
    tmp_var_decl.location() = location;
    current_block->copy_to_operands(tmp_var_decl);

    // Build function call statement from side effect expression
    side_effect_expr_function_callt &side_effect =
      to_side_effect_expr_function_call(expr);

    code_function_callt call;
    call.function() = side_effect.function();
    call.arguments() = side_effect.arguments();
    call.lhs() = symbol_expr(tmp_var_symbol);
    call.type() = list_type;
    call.location() = location;

    current_block->copy_to_operands(call);

    // Replace expr with the temp variable
    expr = symbol_expr(tmp_var_symbol);
    return true;
  };

  // Resolve both sides if they are function calls
  resolve_list_call(lhs);
  resolve_list_call(rhs);

  // List comparison
  if (
    lhs.type() == list_type && rhs.type() == list_type &&
    (op == "Eq" || op == "NotEq"))
  {
    python_list list(*this, element);
    return list.compare(lhs, rhs, op);
  }

  // List concatenation
  if (lhs.type() == list_type && rhs.type() == list_type && op == "Add")
  {
    python_list list(*this, element);
    return list.build_concat_list_call(lhs, rhs, element);
  }

  // List repetition
  if ((lhs.type() == list_type || rhs.type() == list_type) && op == "Mult")
  {
    if (is_right)
      return nil_exprt();
    python_list list(*this, element);
    return list.list_repetition(left, right, lhs, rhs);
  }

  return nil_exprt();
}

exprt python_converter::handle_relational_type_mismatches(
  const std::string &op,
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &element)
{
  // Single character comparisons
  if (type_utils::is_ordered_comparison(op))
  {
    exprt char_comp_result = handle_single_char_comparison(op, lhs, rhs);
    if (!char_comp_result.is_nil())
      return char_comp_result;
  }

  // Float vs string comparisons
  bool lhs_is_float = lhs.type().is_floatbv();
  bool rhs_is_float = rhs.type().is_floatbv();
  bool lhs_is_str = type_utils::is_string_type(lhs.type());
  bool rhs_is_str = type_utils::is_string_type(rhs.type());

  if ((lhs_is_float && rhs_is_str) || (lhs_is_str && rhs_is_float))
  {
    exprt binary_expr(get_op(op, bool_type()), bool_type());

    locationt loc = get_location_from_decl(element);
    if (loc.is_nil() || loc.get_line().empty())
    {
      if (!lhs.location().is_nil())
        loc = lhs.location();
      else if (!rhs.location().is_nil())
        loc = rhs.location();
    }
    binary_expr.location() = loc;

    return handle_float_vs_string(binary_expr, op);
  }

  return nil_exprt();
}

exprt python_converter::handle_string_binary_operations(
  const std::string &op,
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &left,
  const nlohmann::json &right,
  const nlohmann::json &element)
{
  std::string lhs_type = type_handler_.type_to_string(lhs.type());
  std::string rhs_type = type_handler_.type_to_string(rhs.type());

  // Infer string types for equality comparisons
  if (
    (op == "Eq" || op == "NotEq") && ((lhs_type.empty() && rhs_type == "str") ||
                                      (rhs_type.empty() && lhs_type == "str")))
  {
    if (lhs_type.empty() && element.contains("left"))
    {
      const auto &lhs_expr = element["left"];
      if (
        (lhs_expr.contains("value") && lhs_expr["value"].is_string()) ||
        (lhs_expr.contains("id") && lhs_expr["id"].is_string()))
        lhs_type = "str";
    }
    else if (
      rhs_type.empty() && element.contains("comparators") &&
      element["comparators"].is_array() && !element["comparators"].empty())
    {
      const auto &rhs_expr = element["comparators"][0];
      if (
        (rhs_expr.contains("value") && rhs_expr["value"].is_string()) ||
        (rhs_expr.contains("id") && rhs_expr["id"].is_string()))
        rhs_type = "str";
    }
  }

  // Check for string literals in Add operations
  if (op == "Add")
  {
    if (
      element.contains("left") && element["left"].contains("value") &&
      element["left"]["value"].is_string())
      lhs_type = "str";

    if (
      element.contains("right") && element["right"].contains("value") &&
      element["right"]["value"].is_string())
      rhs_type = "str";

    if (!lhs_type.empty() || !rhs_type.empty())
    {
      if (lhs_type.empty())
        lhs_type = "str";
      if (rhs_type.empty())
        rhs_type = "str";
    }
  }

  // Check if both operands are strings
  bool lhs_is_string =
    (lhs_type == "str") || type_utils::is_string_type(lhs.type());
  bool rhs_is_string =
    (rhs_type == "str") || type_utils::is_string_type(rhs.type());

  if (
    (lhs_is_string && rhs_is_string) ||
    (op == "Mult" &&
     (lhs_is_string || rhs_is_string || type_utils::is_char_type(lhs.type()) ||
      type_utils::is_char_type(rhs.type()))))
  {
    return string_handler_.handle_string_operations(
      op, lhs, rhs, left, right, element);
  }

  return nil_exprt();
}

exprt python_converter::build_binary_expression(
  const std::string &op,
  exprt &lhs,
  exprt &rhs)
{
  // Adjust types for non-relational operations
  if (!type_utils::is_relational_op(op))
  {
    // Check for critical type incompatibilities
    const typet &lhs_type = lhs.type();
    const typet &rhs_type = rhs.type();

    // Check for bitvector width mismatch
    if (
      (lhs_type.is_signedbv() || lhs_type.is_unsignedbv()) &&
      (rhs_type.is_signedbv() || rhs_type.is_unsignedbv()) &&
      lhs_type.width() != rhs_type.width())
    {
      adjust_statement_types(lhs, rhs);
    }
  }

  // Determine result type
  typet type;
  if (type_utils::is_relational_op(op))
    type = bool_type();
  else if (op == "Div" || op == "div")
    type = double_type();
  else if (lhs.type().is_floatbv() || rhs.type().is_floatbv())
    type = lhs.type().is_floatbv() ? lhs.type() : rhs.type();
  else
    type = lhs.type();

  // Create expression
  exprt bin_expr(get_op(op, type), type);

  // Set location
  if (lhs.is_symbol())
    bin_expr.location() = lhs.location();
  else if (rhs.is_symbol())
    bin_expr.location() = rhs.location();

  // Handle signed/unsigned promotion
  if (lhs.type().is_unsignedbv() && rhs.type().is_signedbv())
    rhs.make_typecast(lhs.type());

  // Handle division promotion
  if (op == "Div" || op == "div")
    math_handler_.handle_float_division(lhs, rhs, bin_expr);

  // Add operands
  bin_expr.copy_to_operands(lhs, rhs);

  return bin_expr;
}

void python_converter::promote_ieee_operands(
  exprt &bin_expr,
  const exprt &lhs,
  const exprt &rhs)
{
  if (!is_ieee_op(bin_expr))
    return;

  const typet &target_type = lhs.type().is_floatbv() ? lhs.type() : rhs.type();

  if (!lhs.type().is_floatbv())
    bin_expr.op0() = typecast_exprt(lhs, target_type);
  if (!rhs.type().is_floatbv())
    bin_expr.op1() = typecast_exprt(rhs, target_type);
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
    if (
      (obj["_type"] == "ImportFrom" || obj["_type"] == "Import") &&
      obj.contains("full_path") && !obj["full_path"].is_null())
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

exprt python_converter::wrap_in_optional(
  const exprt &value,
  const typet &optional_type)
{
  assert(optional_type.is_struct());
  const struct_typet &struct_type = to_struct_type(optional_type);

  // Create struct expression
  struct_exprt optional_value(struct_type);

  // Set is_none field based on whether value is None
  exprt is_none_value;
  if (value.type() == none_type())
  {
    is_none_value = gen_boolean(true);
    // Set value field to zero for None case
    optional_value.operands().push_back(is_none_value);
    optional_value.operands().push_back(
      gen_zero(struct_type.components()[1].type()));
  }
  else
  {
    is_none_value = gen_boolean(false);
    optional_value.operands().push_back(is_none_value);
    optional_value.operands().push_back(value);
  }

  return optional_value;
}

exprt python_converter::get_function_call(const nlohmann::json &element)
{
  if (!element.contains("func") || element["_type"] != "Call")
    throw std::runtime_error("Invalid function call");

  // Handle str.join() method calls
  // Python syntax: separator.join(iterable), e.g., " ".join(["a", "b"])
  if (
    element["func"]["_type"] == "Attribute" &&
    element["func"]["attr"] == "join")
  {
    const auto &func = element["func"];
    // Check if the caller is a string (Constant like " " or a Name variable)
    if (
      func.contains("value") && (func["value"]["_type"] == "Constant" ||
                                 func["value"]["_type"] == "Name"))
    {
      return handle_str_join(element);
    }
  }

  // Check for forward-referenced constructor calls
  if (type_handler_.is_constructor_call(element))
  {
    code_blockt temp_block;
    process_forward_reference(element["func"], temp_block);
  }

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

  // Handle empty set() creation
  if (
    element["func"]["_type"] == "Name" && element["func"]["id"] == "set" &&
    (!element.contains("args") || element["args"].empty()))
  {
    // Create an empty set (modeled as list)
    python_set set_handler(*this, element);
    return set_handler.get_empty_set();
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
  exprt call_expr = call_builder.build();

  auto handle_keywords = [&](exprt &call_expr) {
    if (!element.contains("keywords") || element["keywords"].empty())
      return;

    const exprt &func =
      call_expr.operands().size() > 1 ? call_expr.operands()[1] : exprt();

    if (!func.is_symbol())
      return;

    const symbolt *func_symbol = symbol_table_.find_symbol(func.identifier());
    if (!func_symbol || !func_symbol->type.is_code())
      return;

    const code_typet &func_type = to_code_type(func_symbol->type);
    const code_typet::argumentst &params = func_type.arguments();

    code_function_callt &call = static_cast<code_function_callt &>(call_expr);
    auto &args = call.arguments();

    size_t positional_count =
      element.contains("args") && element["args"].is_array()
        ? element["args"].size()
        : 0;

    std::map<std::string, size_t> param_positions;
    for (size_t i = 0; i < params.size(); ++i)
    {
      std::string param_name = params[i].get_base_name().as_string();
      assert(!param_name.empty());
      param_positions[param_name] = i;
    }

    if (args.size() < params.size())
      args.resize(params.size(), exprt());

    for (const auto &kw : element["keywords"])
    {
      std::string arg_name = kw["arg"].get<std::string>();
      exprt arg_expr = get_expr(kw["value"]);

      auto it = param_positions.find(arg_name);
      if (it == param_positions.end())
      {
        throw std::runtime_error(
          "Unknown keyword argument: " + arg_name + " in function " +
          func_symbol->name.as_string());
      }

      // Convert array to pointer to match parameter type
      const typet &param_type = params[it->second].type();
      if (arg_expr.type().is_array() && param_type.is_pointer())
        arg_expr = string_handler_.get_array_base_address(arg_expr);

      args[it->second] = arg_expr;
    }

    // we need to check if the argument is provided despite being optional
    auto is_optional_type = [&](const typet &param_type) {
      if (!param_type.is_struct())
        return false;
      const struct_typet &struct_type = to_struct_type(param_type);
      const std::string &tag = struct_type.tag().as_string();
      return tag.starts_with("tag-Optional_");
    };

    std::vector<size_t> missing_required;
    std::vector<bool> provided(params.size(), false);

    size_t bound_params = 0;
    if (!params.empty())
    {
      const std::string &first_param_name =
        params[0].get_base_name().as_string();
      if (first_param_name == "self" || first_param_name == "cls")
        bound_params = 1;
    }

    for (size_t i = 0; i < bound_params && i < provided.size(); ++i)
      provided[i] = true;

    for (size_t i = 0; i < positional_count; ++i)
    {
      size_t param_idx = bound_params + i;
      if (param_idx < provided.size())
        provided[param_idx] = true;
    }

    for (const auto &entry : param_positions)
    {
      size_t index = entry.second;
      if (
        index < provided.size() &&
        !(args[index].is_nil() || args[index].id().empty()))
        provided[index] = true;
    }

    // check if any argument is missing
    for (size_t i = 0; i < params.size(); ++i)
    {
      if (provided[i])
        continue;

      bool has_default = params[i].has_default_value();
      bool optional_param = is_optional_type(params[i].type());

      if (!has_default && !optional_param)
      {
        missing_required.push_back(i); // add the index of the missing argument
      }
    }

    if (!missing_required.empty())
    {
      std::vector<std::string> missing_names;
      missing_names.reserve(missing_required.size());
      for (size_t idx : missing_required)
        missing_names.push_back(params[idx].get_base_name().as_string());

      std::ostringstream msg;
      if (missing_names.size() == 1)
      {
        msg << "TypeError: " << func_symbol->name.as_string()
            << "() missing 1 required positional argument: '"
            << missing_names.front() << "'";
      }
      else
      {
        msg << "TypeError: " << func_symbol->name.as_string() << "() missing "
            << missing_names.size() << " required positional arguments: ";
        for (size_t i = 0; i < missing_names.size(); ++i)
        {
          msg << "'" << missing_names[i] << "'";
          if (i + 2 < missing_names.size())
            msg << ", ";
          else if (i + 2 == missing_names.size())
            msg << " and ";
        }
      }

      throw std::runtime_error(msg.str());
    }

    // Fill empty arguments with proper Optional values or None for optional parameters
    for (size_t i = 0; i < args.size(); ++i)
    {
      if (args[i].is_nil() || args[i].id().empty())
      {
        const typet &param_type = params[i].type();

        // Check if this is an Optional type (struct with "is_none" field)
        if (is_optional_type(param_type))
        {
          // Create Optional value with is_none=true
          constant_exprt none_expr(none_type());
          none_expr.set_value("NULL");
          args[i] = wrap_in_optional(none_expr, param_type);
        }
        else
        {
          // Non-struct type - use NULL for None
          constant_exprt none_expr(none_type());
          none_expr.set_value("NULL");
          args[i] = none_expr;
        }
      }
    }
  };

  handle_keywords(call_expr);

  // Convert struct arguments to pointers for union-typed parameters
  // This handles both positional and keyword arguments
  if (call_expr.id() == "code" && call_expr.get("statement") == "function_call")
  {
    code_function_callt &call = static_cast<code_function_callt &>(call_expr);
    // Get function symbol to access parameter types
    const exprt &func = call.function();
    if (func.is_symbol())
    {
      const symbolt *func_symbol = symbol_table_.find_symbol(func.identifier());
      if (func_symbol && func_symbol->type.is_code())
      {
        const code_typet &func_type = to_code_type(func_symbol->type);
        const code_typet::argumentst &params = func_type.arguments();
        auto &args = call.arguments();
        for (size_t i = 0; i < args.size() && i < params.size(); ++i)
        {
          const typet &param_type = params[i].type();
          exprt &arg = args[i];

          // Get the actual type of the argument (resolve symbols)
          typet arg_actual_type = arg.type();
          if (arg.is_symbol())
          {
            const symbolt *arg_symbol =
              symbol_table_.find_symbol(arg.identifier());
            if (arg_symbol)
            {
              arg_actual_type = arg_symbol->type;
              // Follow symbol type references using namespace
              if (arg_actual_type.id() == "symbol")
                arg_actual_type = ns.follow(arg_actual_type);
            }
          }
          // Handle union types: if param is pointer and arg is struct (or symbol to struct), take address
          if (
            param_type.is_pointer() && arg_actual_type.is_struct() &&
            !arg.is_address_of() && !arg_actual_type.is_pointer())
            arg = gen_address_of(arg);
        }
      }
    }
  }

  return call_expr;
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
    constant_exprt null_expr(none_type());
    null_expr.set_value("NULL");
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

  // Handle string or byte literals
  typet t = current_element_type;
  std::vector<uint8_t> string_literal;

  if (is_bytes_literal(element))
  {
    std::vector<uint8_t> bytes;
    if (element.contains("encoded_bytes"))
      bytes = base64_decode(element["encoded_bytes"].get<std::string>());
    else
      bytes.assign(str_val.begin(), str_val.end());

    return string_builder_->build_raw_byte_array(bytes);
  }
  else
  {
    // Strings are null-terminated
    return string_builder_->build_string_literal(str_val);
  }

  return make_char_array_expr(string_literal, t);
}

// Detect bytes literals
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

std::string
python_converter::extract_class_name_from_tag(const std::string &tag_name)
{
  if (tag_name.size() > 4 && tag_name.substr(0, 4) == "tag-")
    return tag_name.substr(4);
  return tag_name;
}

std::string
python_converter::create_normalized_self_key(const std::string &class_tag)
{
  std::string class_name = extract_class_name_from_tag(class_tag);
  return "self@" + class_name;
}

typet python_converter::clean_attribute_type(const typet &attr_type)
{
  typet clean_type = attr_type;
  clean_type.remove("#member_name");
  clean_type.remove("#location");
  clean_type.remove("#identifier");
  return clean_type;
}

exprt python_converter::create_member_expression(
  const symbolt &symbol,
  const std::string &attr_name,
  const typet &attr_type)
{
  typet clean_type = clean_attribute_type(attr_type);
  exprt source = symbol_exprt(symbol.id, symbol.type);
  member_exprt member_expr(source, attr_name, clean_type);

  // Apply adjust_member logic (from Clang frontend): insert dereference if source is pointer
  exprt &base = member_expr.struct_op();
  if (base.type().is_pointer())
  {
    exprt deref("dereference");
    deref.type() = base.type().subtype();
    deref.move_to_operands(base);
    base.swap(deref);
  }

  return member_expr;
}

// Register instance attribute in maps
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
  if (symbol_value != exprt())
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
    if (dict_handler_->is_dict_literal(element))
    {
      expr = dict_handler_->get_dict_literal(element);
      break;
    }

    expr = get_literal(element);
    break;
  }
  case ExpressionType::LIST:
  {
    // For now, treat set literals such as lists
    // Store elements in order they appear (order doesn't matter for sets)
    if (element["_type"] == "Set")
    {
      python_set set_handler(*this, element);
      expr = set_handler.get();
      break;
    }

    // Check if we should use static arrays (for numpy and similar operations)
    if (build_static_lists)
    {
      typet size = type_handler_.get_typet(element["elts"]);
      expr = get_static_array(element, size);
      break;
    }

    // List handling (dynamic lists)
    python_list list(*this, element);
    expr = list.get();
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
      // Handle nested attribute chain (e.g., self.b.a)
      if (element["value"]["_type"] == "Attribute")
      {
        exprt base_expr = get_expr(element["value"]);
        const std::string &attr_name = element["attr"].get<std::string>();

        typet base_type = base_expr.type();
        if (base_type.is_pointer())
          base_type = base_type.subtype();
        if (base_type.id() == "symbol")
          base_type = ns.follow(base_type);

        if (base_type.is_struct())
        {
          const struct_typet &struct_type = to_struct_type(base_type);
          if (struct_type.has_component(attr_name))
          {
            const typet &attr_type =
              struct_type.get_component(attr_name).type();
            typet clean_type = clean_attribute_type(attr_type);

            member_exprt member_expr(base_expr, attr_name, clean_type);

            // Insert dereference if needed
            exprt &base = member_expr.struct_op();
            if (base.type().is_pointer())
            {
              exprt deref("dereference");
              deref.type() = base.type().subtype();
              deref.move_to_operands(base);
              base.swap(deref);
            }

            expr = member_expr;
            break;
          }
        }

        log_error("Cannot resolve nested attribute: {}", attr_name);
        abort();
      }
      else if (element["value"]["_type"] == "Name")
      {
        var_name = element["value"]["id"].get<std::string>();
      }
      else
      {
        log_error(
          "Unsupported Attribute value type: {}",
          element["value"]["_type"].get<std::string>());
        abort();
      }

      // Handle module attribute access (e.g., math.inf)
      if (is_imported_module(var_name))
      {
        std::string attr_name = element["attr"].get<std::string>();
        std::string module_path = imported_modules[var_name];

        // Construct symbol ID for module member: py:module_path@member_name
        symbol_id module_sid(module_path, "", "");
        module_sid.set_object(attr_name);

        symbolt *symbol = find_symbol(module_sid.to_string());
        if (!symbol)
        {
          log_error(
            "Module member '{}' not found in module '{}'", attr_name, var_name);
          abort();
        }

        expr = symbol_expr(*symbol);
        break;
      }

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

      if (symbol_type.id() == "struct")
      {
        // Struct types store class name in "tag" field
        const struct_typet &struct_type = to_struct_type(symbol_type);
        obj_type_name = "tag-" + struct_type.tag().as_string();
      }
      else
      {
        // Search named_sub for identifier
        for (const auto &it : symbol_type.get_named_sub())
        {
          if (it.first == "identifier")
            obj_type_name = it.second.id_string();
        }
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

      // Check if this specific instance has explicitly set this attribute
      bool instance_has_attr = is_instance_attribute(
        symbol->id.as_string(),
        attr_name,
        var_name,
        class_type.tag().as_string());

      // For LHS (writing): always use instance member and register it
      if (is_converting_lhs && class_type.has_component(attr_name))
      {
        const typet &attr_type = class_type.get_component(attr_name).type();
        expr = create_member_expression(*symbol, attr_name, attr_type);

        // Register as instance attribute
        register_instance_attribute(
          symbol->id.as_string(),
          attr_name,
          var_name,
          class_type.tag().as_string());
      }
      // For RHS (reading): use instance member if explicitly set OR if symbol is a parameter
      // This allows parameter objects like 'f: Foo' to access instance attributes
      else if (
        !is_converting_lhs && class_type.has_component(attr_name) &&
        (instance_has_attr || symbol->is_parameter))
      {
        const typet &attr_type = class_type.get_component(attr_name).type();
        expr = create_member_expression(*symbol, attr_name, attr_type);
      }
      // Otherwise use class attribute
      else
      {
        sid.set_function("");
        sid.set_class(extract_class_name_from_tag(obj_type_name));
        sid.set_object(attr_name);
        symbolt *class_attr_symbol = symbol_table_.find_symbol(sid.to_string());

        if (!class_attr_symbol)
        {
          throw std::runtime_error("Attribute \"" + attr_name + "\" not found");
        }

        expr = symbol_expr(*class_attr_symbol);
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
    const nlohmann::json &slice = element["slice"];

    // Handle tuple subscripting - tuples are structs, not arrays
    if (tuple_handler_->is_tuple_type(array.type()))
    {
      expr = tuple_handler_->handle_tuple_subscript(array, slice, element);
      break;
    }

    // Handle dictionary subscript
    if (array.type().is_struct())
    {
      // This is a dictionary access
      expr = dict_handler_->handle_dict_subscript(array, slice);
      break;
    }

    // Handle regular array/list subscripting
    python_list list(*this, element);
    expr = list.index(array, slice);
    break;
  }
  case ExpressionType::FSTRING:
    expr = string_handler_.get_fstring_expr(element);
    break;
  case ExpressionType::TUPLE:
    return get_tuple_expr(element);
  default:
  {
    std::ostringstream oss;
    oss << "Unsupported expression ";
    if (element.contains("_type"))
      oss << element["_type"].get<std::string>();

    if (element.contains("lineno"))
      oss << " at line " << element["lineno"].template get<int>();

    throw std::runtime_error(oss.str());
  }
  }

  return expr;
}

exprt python_converter::get_tuple_expr(const nlohmann::json &element)
{
  return tuple_handler_->get_tuple_expr(element);
}

void python_converter::copy_instance_attributes(
  const std::string &src_obj_id,
  const std::string &target_obj_id)
{
  auto src_attrs = instance_attr_map.find(src_obj_id);

  if (src_attrs != instance_attr_map.end())
  {
    std::set<std::string> &target_attrs = instance_attr_map[target_obj_id];
    target_attrs.insert(src_attrs->second.begin(), src_attrs->second.end());
  }
}

void python_converter::update_instance_from_self(
  const std::string &class_name,
  const std::string &func_name,
  const std::string &obj_symbol_id)
{
  symbol_id sid(current_python_file, class_name, func_name);
  sid.set_object("self");
  copy_instance_attributes(sid.to_string(), obj_symbol_id);
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
        // Direct bytes literal such as b'A'
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
  {
    if (get_statement_type(stmt) == StatementType::RETURN)
      return stmt;
  }

  throw std::runtime_error(
    "Function " + function["name"].get<std::string>() +
    " has no return statement");
}

std::pair<std::string, typet>
python_converter::extract_type_info(const nlohmann::json &var_node)
{
  typet var_typet;
  std::string var_type_str("");

  if (var_node.contains("annotation") && !var_node["annotation"].is_null())
  {
    // Get type from annotation node
    size_t type_size = get_type_size(var_node);
    const auto &ann = var_node["annotation"];

    if (ann.contains("_type") && ann["_type"] == "Subscript")
    {
      if (ann.contains("value") && ann["value"].contains("id"))
        var_type_str = ann["value"]["id"];
    }
    else if (
      ann.contains("_type") && ann["_type"] == "Attribute" &&
      ann.contains("attr"))
      var_type_str = ann["attr"];
    else if (ann.contains("id"))
      var_type_str = var_node["annotation"]["id"];
    else if (ann.contains("_type") && ann["_type"] == "BinOp")
    {
      // Handle union types (e.g., re.Match[str] | None)
      // Use get_type_from_annotation which has proper union handling
      var_typet = get_type_from_annotation(ann, var_node);
      return {var_type_str, var_typet};
    }

    if (var_type_str.empty())
      return {var_type_str, var_typet};

    if (var_type_str == "dict" || var_type_str == "Dict")
      var_typet = dict_handler_->get_dict_struct_type();
    else if (var_type_str == "list" || var_type_str == "List")
      var_typet = type_handler_.get_list_type();
    else
      var_typet = type_handler_.get_typet(var_type_str, type_size);
  }

  return {var_type_str, var_typet};
}

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

void python_converter::handle_assignment_type_adjustments(
  symbolt *lhs_symbol,
  exprt &lhs,
  exprt &rhs,
  const std::string &lhs_type,
  const nlohmann::json &ast_node,
  bool is_ctor_call)
{
  const bool has_annotation =
    ast_node.contains("annotation") && !ast_node["annotation"].is_null();

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
  // Handle tuple assignments with generic tuple annotation
  else if (
    lhs_symbol && lhs_symbol->type.id() == "empty" &&
    rhs.type().id() == "struct")
  {
    const struct_typet &rhs_struct = to_struct_type(rhs.type());

    // Check if RHS is a tuple (has tuple tag pattern)
    if (rhs_struct.tag().as_string().find("tag-tuple") == 0)
    {
      // Update symbol type from empty to concrete tuple type
      lhs_symbol->type = rhs.type();
      lhs.type() = rhs.type();
      lhs_symbol->value = rhs;
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
      // TODO: This case is used to infer an unknown type.
      // Should we model it uniformly using char* ?
      const typet &element_type = to_array_type(rhs.type()).subtype();
      typet pointer_type = gen_pointer_type(element_type);
      lhs_symbol->type = pointer_type;
      lhs.type() = pointer_type;
      rhs = string_handler_.get_array_base_address(rhs);
    }
    else if (
      lhs.type().is_pointer() && rhs.type().is_array() &&
      lhs.type() != type_handler_.get_list_type())
    {
      // Array to pointer typecast
      // skip the list type until the list is moved to symex
      // TODO: remove list condition
      rhs = string_handler_.get_array_base_address(rhs);
    }
    // String and list type size adjustments
    else if (
      lhs_type == "str" || lhs_type == "chr" || lhs_type == "ord" ||
      lhs_type == "list" || rhs.type().is_array() ||
      rhs.type() == type_handler_.get_list_type())
    {
      if (!rhs.type().is_empty())
      {
        lhs_symbol->type = rhs.type();
        lhs.type() = rhs.type();
      }
    }
    else if (rhs.type() == none_type())
    {
      // Adjust pointer_type() to pointer_typet(empty_typet())
      lhs_symbol->type = rhs.type();
      lhs.type() = rhs.type();
    }
    else if (
      !has_annotation && !rhs.type().is_empty() && lhs.type() != rhs.type() &&
      !rhs.type().is_code() &&
      !(rhs.type().is_pointer() && rhs.type().subtype().id() == "empty"))
    {
      // Default case: allow Python's dynamic typing by updating the variable
      // type to match the assigned value. Type annotations are enforced via
      // runtime assertions rather than static typing.
      lhs_symbol->type = rhs.type();
      lhs.type() = rhs.type();
    }

    if (!rhs.type().is_empty() && !is_ctor_call)
      lhs_symbol->value = rhs;
  }
}

exprt python_converter::get_return_from_func(const char *func_symbol_id)
{
  symbolt *func_symbol = symbol_table_.find_symbol(func_symbol_id);
  assert(func_symbol);

  const auto &operands = func_symbol->value.operands();

  for (std::vector<exprt>::const_reverse_iterator it = operands.rbegin();
       it != operands.rend();
       ++it)
  {
    const codet &c = to_code(*it);
    if (c.statement() == "return")
    {
      return c;
    }
  }
  return nil_exprt();
}

void python_converter::handle_array_unpacking(
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  exprt &rhs,
  codet &target_block)
{
  const auto &targets = target["elts"];

  for (size_t i = 0; i < targets.size(); i++)
  {
    if (targets[i]["_type"] != "Name")
    {
      throw std::runtime_error(
        "Array unpacking only supports simple names, not " +
        targets[i]["_type"].get<std::string>());
    }

    std::string var_name = targets[i]["id"].get<std::string>();
    symbol_id var_sid = create_symbol_id();
    var_sid.set_object(var_name);

    symbolt *var_symbol = find_symbol(var_sid.to_string());

    if (!var_symbol)
    {
      locationt loc = get_location_from_decl(targets[i]);
      typet elem_type = rhs.type().subtype();

      symbolt new_symbol = create_symbol(
        loc.get_file().as_string(),
        var_name,
        var_sid.to_string(),
        loc,
        elem_type);
      new_symbol.lvalue = true;
      new_symbol.file_local = true;
      new_symbol.is_extern = false;
      var_symbol = symbol_table_.move_symbol_to_context(new_symbol);
    }

    // Create subscript: rhs[i]
    exprt index_expr = from_integer(i, size_type());
    index_exprt subscript(rhs, index_expr, rhs.type().subtype());

    code_assignt assign(symbol_expr(*var_symbol), subscript);
    assign.location() = get_location_from_decl(ast_node);
    target_block.copy_to_operands(assign);
  }
}

void python_converter::handle_list_literal_unpacking(
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  codet &target_block)
{
  const auto &value_node = ast_node["value"];
  const auto &elements = value_node["elts"];
  const auto &targets = target["elts"];

  if (elements.size() != targets.size())
  {
    throw std::runtime_error(
      "Cannot unpack list: expected " + std::to_string(targets.size()) +
      " values, got " + std::to_string(elements.size()));
  }

  // Create assignments directly from list elements
  for (size_t i = 0; i < targets.size(); i++)
  {
    if (targets[i]["_type"] != "Name")
    {
      throw std::runtime_error(
        "List unpacking only supports simple names, not " +
        targets[i]["_type"].get<std::string>());
    }

    std::string var_name = targets[i]["id"].get<std::string>();
    symbol_id var_sid = create_symbol_id();
    var_sid.set_object(var_name);

    symbolt *var_symbol = find_symbol(var_sid.to_string());

    // Convert the element expression
    is_converting_rhs = true;
    exprt elem_expr = get_expr(elements[i]);
    is_converting_rhs = false;

    if (!var_symbol)
    {
      locationt loc = get_location_from_decl(targets[i]);

      symbolt new_symbol = create_symbol(
        loc.get_file().as_string(),
        var_name,
        var_sid.to_string(),
        loc,
        elem_expr.type());
      new_symbol.lvalue = true;
      new_symbol.file_local = true;
      new_symbol.is_extern = false;
      var_symbol = symbol_table_.move_symbol_to_context(new_symbol);
    }

    code_assignt assign(symbol_expr(*var_symbol), elem_expr);
    assign.location() = get_location_from_decl(ast_node);
    target_block.copy_to_operands(assign);
  }
}

bool python_converter::handle_dict_subscript_assignment(
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  codet &target_block)
{
  if (target["_type"] != "Subscript")
    return false;

  exprt container_expr = get_expr(target["value"]);
  typet container_type = container_expr.type();

  if (container_expr.is_symbol())
  {
    const symbolt *sym = symbol_table_.find_symbol(container_expr.identifier());
    if (sym)
      container_type = sym->type;
  }

  if (container_type.id() == "symbol")
    container_type = ns.follow(container_type);

  if (!dict_handler_->is_dict_type(container_type))
    return false;

  // Handle dict[key] = value assignment
  is_converting_rhs = true;
  exprt rhs = get_expr(ast_node["value"]);
  is_converting_rhs = false;

  dict_handler_->handle_dict_subscript_assign(
    container_expr, target["slice"], rhs, target_block);
  return true;
}

bool python_converter::handle_dict_literal_assignment(
  const nlohmann::json &ast_node,
  const exprt &lhs)
{
  if (!ast_node.contains("value") || ast_node["value"].is_null())
    return false;

  if (!dict_handler_->is_dict_literal(ast_node["value"]))
    return false;

  dict_handler_->create_dict_from_literal(ast_node["value"], lhs);
  current_lhs = nullptr;
  return true;
}

bool python_converter::handle_unannotated_dict_literal(
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  const symbol_id &sid)
{
  if (!ast_node.contains("value") || !ast_node["value"].contains("_type"))
    return false;

  if (!dict_handler_->is_dict_literal(ast_node["value"]))
    return false;

  locationt location = get_location_from_decl(target);
  std::string module_name = location.get_file().as_string();
  std::string name;

  if (target["_type"] == "Name")
    name = target["id"].get<std::string>();
  else if (target["_type"] == "Attribute")
    name = target["attr"].get<std::string>();

  symbolt symbol = create_symbol(
    module_name,
    name,
    sid.to_string(),
    location,
    dict_handler_->get_dict_struct_type());
  symbol.lvalue = true;
  symbol.file_local = true;
  symbol.is_extern = false;
  symbolt *lhs_symbol = symbol_table_.move_symbol_to_context(symbol);

  exprt lhs = create_lhs_expression(target, lhs_symbol, location);
  dict_handler_->create_dict_from_literal(ast_node["value"], lhs);
  current_lhs = nullptr;
  return true;
}

exprt python_converter::get_rhs_with_dict_resolution(
  const nlohmann::json &ast_node,
  const typet &target_type)
{
  if (!type_utils::is_dict_subscript(ast_node["value"]))
    return get_expr(ast_node["value"]);

  // Check if we need special dict subscript handling for typed variables
  if (
    !target_type.is_signedbv() && !target_type.is_unsignedbv() &&
    !target_type.is_bool())
    return get_expr(ast_node["value"]);

  exprt dict_expr = get_expr(ast_node["value"]["value"]);
  if (
    !dict_expr.type().is_struct() ||
    !dict_handler_->is_dict_type(dict_expr.type()))
    return get_expr(ast_node["value"]);

  return dict_handler_->handle_dict_subscript(
    dict_expr, ast_node["value"]["slice"], target_type);
}

std::string python_converter::infer_type_from_any_annotation(
  const nlohmann::json &ast_node,
  const std::string &lhs_type)
{
  if (lhs_type != "Any")
    return lhs_type;

  if (ast_node["value"].is_null() || ast_node["value"]["_type"] != "Call")
    return lhs_type;

  const auto &func_node = ast_node["value"]["func"];
  std::string func_name;

  if (func_node["_type"] == "Name")
    func_name = func_node["id"].get<std::string>();
  else if (func_node["_type"] == "Attribute")
    func_name = func_node["attr"].get<std::string>();

  if (func_name.empty())
    return lhs_type;

  symbol_id func_sid(current_python_file, "", func_name);
  symbolt *func_symbol = symbol_table_.find_symbol(func_sid.to_string());

  if (func_symbol && func_symbol->type.is_code())
  {
    const code_typet &func_type = to_code_type(func_symbol->type);
    current_element_type = func_type.return_type();
    return ""; // Clear to avoid further "Any" processing
  }

  return lhs_type;
}

// (type assertion helpers moved to python_typechecking)

bool python_converter::handle_unpacking_assignment(
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  codet &target_block)
{
  const auto &target_type = target["_type"];

  if (target_type != "Tuple" && target_type != "List")
    return false;

  // Get RHS
  is_converting_rhs = true;
  exprt rhs = get_expr(ast_node["value"]);
  is_converting_rhs = false;

  // Prepare RHS if it's a function call
  rhs = tuple_handler_->prepare_rhs_for_unpacking(ast_node, rhs, target_block);

  // Handle different unpacking types
  if (rhs.type().id() == "struct")
  {
    tuple_handler_->handle_tuple_unpacking(ast_node, target, rhs, target_block);
    return true;
  }
  else if (rhs.type().is_array())
  {
    handle_array_unpacking(ast_node, target, rhs, target_block);
    return true;
  }
  else if (rhs.type().is_pointer())
  {
    const auto &value_node = ast_node["value"];
    if (value_node["_type"] == "List")
    {
      handle_list_literal_unpacking(ast_node, target, target_block);
      return true;
    }
  }

  throw std::runtime_error(
    "Cannot unpack " + rhs.type().id_string() +
    " - only tuples and arrays can be unpacked");
}

symbolt *python_converter::create_symbol_for_unannotated_assign(
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  const symbol_id &sid,
  bool is_global)
{
  if (is_global)
    return nullptr;

  if (!ast_node.contains("value") || !ast_node["value"].contains("_type"))
    return nullptr;

  const std::string &value_type = ast_node["value"]["_type"];
  locationt location = get_location_from_decl(target);
  std::string module_name = location.get_file().as_string();
  std::string name;

  if (target["_type"] == "Name")
    name = target["id"].get<std::string>();
  else if (target["_type"] == "Attribute")
    name = target["attr"].get<std::string>();

  typet inferred_type;

  if (value_type == "Lambda")
  {
    inferred_type = any_type();
  }
  else if (value_type == "Call" || value_type == "BoolOp")
  {
    // Convert RHS first to get its type
    is_converting_rhs = true;
    exprt rhs_expr = get_expr(ast_node["value"]);
    is_converting_rhs = false;

    inferred_type = rhs_expr.type();
    if (inferred_type.is_empty())
      inferred_type = any_type();
  }
  else
  {
    return nullptr;
  }

  symbolt symbol =
    create_symbol(module_name, name, sid.to_string(), location, inferred_type);
  symbol.lvalue = true;
  symbol.file_local = true;
  symbol.is_extern = false;
  return symbol_table_.move_symbol_to_context(symbol);
}

void python_converter::handle_function_call_rhs(
  const nlohmann::json &ast_node,
  symbolt *lhs_symbol,
  exprt &lhs,
  exprt &rhs,
  const locationt &location,
  bool is_ctor_call,
  codet &target_block)
{
  if (is_ctor_call)
  {
    std::string func_name =
      ast_node["value"]["func"].contains("id")
        ? ast_node["value"]["func"]["id"].get<std::string>()
        : ast_node["value"]["func"]["attr"].get<std::string>();

    if (base_ctor_called)
    {
      auto class_node = json_utils::find_class((*ast_json)["body"], func_name);
      func_name = class_node["bases"][0]["id"].get<std::string>();
      base_ctor_called = false;
    }

    update_instance_from_self(func_name, func_name, lhs_symbol->id.as_string());
  }
  else
  {
    symbolt *func_symbol =
      symbol_table_.find_symbol(rhs.op1().identifier().c_str());
    assert(func_symbol);
    if (!static_cast<code_typet &>(func_symbol->type).return_type().is_empty())
    {
      if (auto ret = get_return_from_func(func_symbol->id.c_str());
          !ret.is_nil())
      {
        copy_instance_attributes(
          ret.op0().identifier().as_string(), lhs_symbol->id.as_string());
      }
    }
  }

  // Copy attributes from function arguments
  if (!is_ctor_call)
  {
    const code_function_callt &call =
      static_cast<const code_function_callt &>(rhs);
    for (const auto &arg : call.arguments())
    {
      const exprt *arg_ptr = &arg;
      if (arg.is_address_of())
        arg_ptr = &arg.op0();

      if (arg_ptr->is_symbol())
      {
        copy_instance_attributes(
          arg_ptr->identifier().as_string(), lhs_symbol->id.as_string());
      }
    }
  }

  // Set return destination
  if (rhs.type().is_pointer() && !is_ctor_call)
  {
    rhs.op0() = lhs;
  }
  else if (!rhs.type().is_pointer() && !rhs.type().is_empty() && !is_ctor_call)
    rhs.op0() = lhs;

  // Special handling for list return type
  if (rhs.type() == type_handler_.get_list_type())
  {
    if (auto ret = get_return_from_func(rhs.op1().identifier().c_str());
        !ret.is_nil())
    {
      python_list::copy_type_info(
        ret.op0().identifier().as_string(), lhs.identifier().as_string());
    }

    typet l_type = type_handler_.get_list_type();
    symbolt &tmp_var_symbol =
      create_tmp_symbol(ast_node, "tmp_var", l_type, gen_zero(l_type));

    code_declt tmp_var_decl(symbol_expr(tmp_var_symbol));
    tmp_var_decl.location() = get_location_from_decl(ast_node);
    target_block.copy_to_operands(tmp_var_decl);

    rhs.op0() = symbol_expr(tmp_var_symbol);
    target_block.copy_to_operands(rhs);

    code_assignt code_assign(lhs, symbol_expr(tmp_var_symbol));
    code_assign.location() = location;
    rhs = code_assign;
  }

  target_block.copy_to_operands(rhs);
}

exprt python_converter::handle_string_literal_rhs(
  const nlohmann::json &ast_node,
  const std::string &lhs_type,
  const exprt &rhs)
{
  if (lhs_type != "str" || !type_utils::is_integer_type(rhs.type()))
    return rhs;

  if (
    ast_node["value"]["_type"] != "Constant" ||
    !ast_node["value"]["value"].is_string())
    return rhs;

  std::string str_value = ast_node["value"]["value"].get<std::string>();

  typet string_type =
    type_handler_.build_array(char_type(), str_value.length() + 1);
  exprt str_array = gen_zero(string_type);

  for (size_t i = 0; i < str_value.length(); ++i)
  {
    BigInt char_val(static_cast<unsigned char>(str_value[i]));
    exprt char_expr = constant_exprt(
      integer2binary(char_val, 8), integer2string(char_val), char_type());
    str_array.operands().at(i) = char_expr;
  }

  return str_array;
}

bool python_converter::is_global_variable(const symbol_id &sid) const
{
  for (const std::string &s : global_declarations)
  {
    if (s == sid.global_to_string())
      return true;
  }
  return false;
}

std::string
python_converter::extract_target_name(const nlohmann::json &target) const
{
  const auto &target_type = target["_type"];

  if (target_type == "Name")
    return target["id"].get<std::string>();
  else if (target_type == "Attribute")
    return target["attr"].get<std::string>();
  else if (target_type == "Subscript")
    return target["value"]["id"].get<std::string>();

  throw std::runtime_error(
    "Unsupported assignment target type: " + target_type.get<std::string>());
}

void python_converter::get_var_assign(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  // Extract type information
  auto [lhs_type, element_type] = extract_type_info(ast_node);

  // Check if the RHS is a dictionary literal - set the element type
  if (
    ast_node.contains("value") && !ast_node["value"].is_null() &&
    dict_handler_->is_dict_literal(ast_node["value"]))
  {
    element_type = dict_handler_->get_dict_struct_type();
  }

  current_element_type = element_type;
  typet annotated_type = element_type;
  std::vector<typet> annotation_types;
  bool can_emit_annotation_check = false;
  locationt annotation_location;
  std::string annotated_name;
  std::vector<typet> annotation_candidates;

  exprt lhs;
  symbolt *lhs_symbol = nullptr;
  locationt location_begin;
  symbol_id sid = create_symbol_id();

  const auto &target = (ast_node.contains("targets")) ? ast_node["targets"][0]
                                                      : ast_node["target"];

  // Handle forward references
  if (
    ast_node.contains("value") && !ast_node["value"].is_null() &&
    ast_node["value"]["_type"] == "Call" &&
    type_handler_.is_constructor_call(ast_node["value"]))
  {
    process_forward_reference(ast_node["value"]["func"], target_block);
  }

  // Handle dict subscript assignment: dict[key] = value
  if (handle_dict_subscript_assignment(ast_node, target, target_block))
    return;

  if (ast_node["_type"] == "AnnAssign")
  {
    // Extract name and set in symbol ID
    std::string name = extract_target_name(target);
    sid.set_object(name);
    annotated_name = name;

    // Infer type from function return if annotation is "Any"
    lhs_type = infer_type_from_any_annotation(ast_node, lhs_type);

    // Process RHS before LHS if in function scope
    exprt rhs;
    if (
      sid.to_string().find("@F") != std::string::npos &&
      sid.to_string().find("@C") == std::string::npos)
    {
      is_right = true;
      if (!ast_node["value"].is_null())
      {
        // Skip getting expr for dict literals - handle specially later
        if (!dict_handler_->is_dict_literal(ast_node["value"]))
        {
          if (ast_node["_type"] != "Call")
          {
            rhs = get_rhs_with_dict_resolution(ast_node, current_element_type);
          }
        }
      }
      is_right = false;
    }

    // Location and symbol lookup
    location_begin = get_location_from_decl(target);
    annotation_location = location_begin;
    can_emit_annotation_check = true;
    lhs_symbol = symbol_table_.find_symbol(sid.to_string().c_str());

    bool is_global = is_global_variable(sid);
    if (is_global)
      lhs_symbol = symbol_table_.find_symbol(sid.global_to_string().c_str());

    // Symbol creation
    bool symbol_created = false;
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

      symbol_created = (lhs_symbol == nullptr);
      lhs_symbol = symbol_table_.move_symbol_to_context(symbol);

      // Add declaration statement ONLY for newly created local variables
      if (symbol_created && !current_func_name_.empty() && !is_global)
      {
        code_declt decl(symbol_expr(*lhs_symbol));
        decl.location() = location_begin;
        target_block.copy_to_operands(decl);
      }
    }

    if (lhs_symbol && ast_node.contains("annotation"))
      get_typechecker().cache_annotation_types(
        *lhs_symbol, ast_node["annotation"]);

    if (
      type_assertions_enabled() && lhs_symbol &&
      ast_node.contains("annotation"))
    {
      auto &tc = get_typechecker();
      annotation_types = tc.get_annotation_types(lhs_symbol->id.as_string());
      if (
        !annotation_types.empty() &&
        !tc.should_skip_type_assertion(annotated_type))
      {
        annotated_type = annotation_types.front();
        can_emit_annotation_check = true;
        annotation_location = location_begin;
        annotated_name = name;
        annotation_candidates = annotation_types;
      }
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
    }

    // Create LHS expression
    lhs = create_lhs_expression(target, lhs_symbol, location_begin);

    // Handle dict literal assignment specially - after LHS is created
    if (handle_dict_literal_assignment(ast_node, lhs))
    {
      if (type_assertions_enabled() && can_emit_annotation_check)
        get_typechecker().emit_type_annotation_assertion(
          lhs,
          annotated_type,
          annotation_types,
          annotated_name,
          annotation_location,
          target_block);
      return;
    }
  }
  else if (ast_node["_type"] == "Assign")
  {
    const auto &target = ast_node["targets"][0];
    location_begin = get_location_from_decl(target);

    // Handle tuple/list unpacking
    if (handle_unpacking_assignment(ast_node, target, target_block))
      return;

    // Normal assignment handling
    std::string name = extract_target_name(target);
    sid.set_object(name);
    lhs_symbol = symbol_table_.find_symbol(sid.to_string());

    bool is_global = is_global_variable(sid);

    // Handle unannotated dict literal assignment
    if (!lhs_symbol && handle_unannotated_dict_literal(ast_node, target, sid))
      return;

    // Create symbol for unannotated assignments with inferrable types
    if (!lhs_symbol && !is_global)
    {
      lhs_symbol =
        create_symbol_for_unannotated_assign(ast_node, target, sid, is_global);
    }

    if (!lhs_symbol && !is_global)
      throw std::runtime_error("Type undefined for \"" + name + "\"");

    lhs = create_lhs_expression(target, lhs_symbol, location_begin);

    if (lhs_symbol && ast_node.contains("annotation"))
      get_typechecker().cache_annotation_types(
        *lhs_symbol, ast_node["annotation"]);

    if (type_assertions_enabled() && lhs_symbol)
    {
      auto &tc = get_typechecker();
      annotation_types = tc.get_annotation_types(lhs_symbol->id.as_string());
      if (
        !annotation_types.empty() &&
        !tc.should_skip_type_assertion(lhs_symbol->type))
      {
        annotated_type = annotation_types.front();
        can_emit_annotation_check = true;
        annotation_location = location_begin;
        annotated_name = name;
        annotation_candidates = annotation_types;
      }
    }
  }

  if (
    type_assertions_enabled() && can_emit_annotation_check &&
    annotation_candidates.empty() &&
    !get_typechecker().should_skip_type_assertion(annotated_type))
    annotation_candidates.push_back(annotated_type);

  bool is_ctor_call = type_handler_.is_constructor_call(ast_node["value"]);
  current_lhs = &lhs;
  is_converting_lhs = false;

  // Get RHS
  exprt rhs;
  bool has_value = false;
  if (!ast_node["value"].is_null())
  {
    is_converting_rhs = true;

    if (lhs_symbol)
      rhs = get_rhs_with_dict_resolution(ast_node, lhs_symbol->type);
    else
      rhs = get_expr(ast_node["value"]);

    has_value = true;
    is_converting_rhs = false;

    // Handle string literal conversion
    rhs = handle_string_literal_rhs(ast_node, lhs_type, rhs);
  }

  if (has_value && rhs != exprt("_init_undefined"))
  {
    // Handle throw expression
    if (rhs.statement() == "cpp-throw")
    {
      rhs.location() = location_begin;
      codet code_expr("expression");
      code_expr.operands().push_back(rhs);
      code_declt decl(symbol_expr(*lhs_symbol));
      decl.location() = location_begin;

      target_block.copy_to_operands(code_expr);
      target_block.copy_to_operands(decl);
      current_lhs = nullptr;
      return;
    }

    // Handle type adjustments
    handle_assignment_type_adjustments(
      lhs_symbol, lhs, rhs, lhs_type, ast_node, is_ctor_call);

    // Function call handling
    if (rhs.is_function_call())
    {
      // Static constructor compatibility check for annotated variables:
      // if var is annotated with a class type (e.g., Animal) and the RHS
      // constructor is a different, non-derived class (e.g., Car), inject
      // an assertion failure.
      if (
        type_assertions_enabled() && can_emit_annotation_check &&
        is_ctor_call && ast_node.contains("annotation") &&
        ast_node["annotation"].contains("id"))
      {
        std::string expected_base =
          ast_node["annotation"]["id"].get<std::string>();
        std::string ctor_name =
          get_typechecker().get_constructor_name(ast_node["value"]["func"]);

        if (
          !expected_base.empty() && !ctor_name.empty() &&
          !get_typechecker().class_derives_from(ctor_name, expected_base))
        {
          code_assertt ctor_assert(gen_boolean(false));
          ctor_assert.location() = location_begin;
          ctor_assert.location().comment(
            "Constructor '" + ctor_name +
            "' is incompatible with annotated type '" + expected_base + "'");
          target_block.copy_to_operands(ctor_assert);
        }
      }

      handle_function_call_rhs(
        ast_node,
        lhs_symbol,
        lhs,
        rhs,
        location_begin,
        is_ctor_call,
        target_block);
      if (type_assertions_enabled() && can_emit_annotation_check)
        get_typechecker().emit_type_annotation_assertion(
          lhs,
          annotated_type,
          annotation_types,
          annotated_name,
          annotation_location,
          target_block);
      current_lhs = nullptr;
      return;
    }

    adjust_statement_types(lhs, rhs);

    // Handle list type info propagation
    if (lhs.type() == rhs.type() && lhs.type() == type_handler_.get_list_type())
    {
      const std::string &lhs_identifier = lhs.identifier().as_string();
      const std::string &rhs_identifier = rhs.identifier().as_string();
      python_list::copy_type_info(rhs_identifier, lhs_identifier);
    }
    else if (
      rhs.type() != lhs.type() && lhs.type().is_array() &&
      !rhs.type().is_code())
    {
#ifndef NDEBUG
      const array_typet &thetype = lhs.type();
      thetype.size().is_constant();
      assert(thetype.size().is_nil());
#endif
      lhs_symbol->type = rhs.type();

      code_declt decl(symbol_expr(*lhs_symbol), rhs);
      decl.location() = location_begin;
      target_block.copy_to_operands(decl);
      if (type_assertions_enabled() && can_emit_annotation_check)
        get_typechecker().emit_type_annotation_assertion(
          lhs,
          annotated_type,
          annotation_types,
          annotated_name,
          annotation_location,
          target_block);
      current_lhs = nullptr;
      return;
    }

    code_assignt code_assign(lhs, rhs);
    code_assign.location() = location_begin;
    target_block.copy_to_operands(code_assign);
    if (type_assertions_enabled() && can_emit_annotation_check)
      get_typechecker().emit_type_annotation_assertion(
        lhs,
        annotated_type,
        annotation_types,
        annotated_name,
        annotation_location,
        target_block);
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

typet python_converter::resolve_variable_type(
  const std::string &var_name,
  const locationt &loc)
{
  nlohmann::json decl_node = get_var_node(var_name, *ast_json);

  if (!decl_node.empty())
  {
    if (decl_node.contains("annotation") && !decl_node["annotation"].is_null())
    {
      const auto &annotation = decl_node["annotation"];

      try
      {
        // Handle rich annotations such as Union, Optional, module attributes,
        // etc. via the unified helper.
        return get_type_from_annotation(annotation, decl_node);
      }
      catch (const std::exception &e)
      {
        log_warning(
          "Failed to resolve complex annotation for '{}': {}. Falling back to "
          "simple identifier lookup.",
          var_name,
          e.what());
      }

      if (annotation.contains("id"))
      {
        std::string type_annotation = annotation["id"].get<std::string>();
        return type_handler_.get_typet(type_annotation);
      }
    }
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

  std::string op = ast_node["op"]["_type"].get<std::string>();

  // Check if this is a string concatenation based on variable annotation
  bool is_string_concat = false;
  if (op == "Add")
  {
    // Standard array-based string concatenation
    if (
      (lhs.type().is_array() && lhs.type().subtype() == char_type()) ||
      (current_element_type.is_array() &&
       current_element_type.subtype() == char_type()))
    {
      is_string_concat = true;
    }
    // Pointer-based string
    else if (
      (lhs.type().is_pointer() && lhs.type().subtype() == char_type()) ||
      (current_element_type.is_pointer() &&
       current_element_type.subtype() == char_type()))
    {
      is_string_concat = true;
    }
    // Check if variable is annotated as str but implemented as single char
    else if (
      type_utils::is_integer_type(lhs.type()) &&
      type_utils::is_integer_type(current_element_type))
    {
      // Check if the variable was declared with str annotation
      nlohmann::json decl_node = get_var_node(var_name, *ast_json);
      if (
        !decl_node.empty() && decl_node.contains("annotation") &&
        decl_node["annotation"].contains("id") &&
        decl_node["annotation"]["id"] == "str")
      {
        is_string_concat = true;
      }
    }
  }

  if (is_string_concat)
  {
    exprt rhs_expr = get_expr(ast_node["value"]);
    nlohmann::json left = ast_node["target"];
    nlohmann::json right = ast_node["value"];
    exprt concatenated =
      string_handler_.handle_string_concatenation(lhs, rhs_expr, left, right);

    // Update the variable's type to match the concatenated result
    // Handle both array and pointer results
    if (
      !var_name.empty() && (concatenated.type().is_array() ||
                            (concatenated.type().is_pointer() &&
                             concatenated.type().subtype() == char_type())))
    {
      symbol_id sid = create_symbol_id();
      sid.set_object(var_name);
      symbolt *symbol = symbol_table_.find_symbol(sid.to_string());
      if (symbol)
      {
        // Update the symbol's type to pointer if concatenated returns pointer
        symbol->type = concatenated.type();

        // Update LHS to be a symbol with the new type
        lhs = symbol_exprt(symbol->id, symbol->type);

        // For pointer results, don't update the value
        // (it will be assigned via the assignment statement)
        if (concatenated.type().is_array())
        {
          symbol->value = concatenated;
        }
      }
    }

    code_assignt code_assign(lhs, concatenated);
    code_assign.location() = loc;
    target_block.copy_to_operands(code_assign);

    // Reset RHS flag
    is_converting_rhs = false;
    return;
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
  if (type_utils::is_integer_type(then_type) && else_type.is_floatbv())
    return else_type;
  if (type_utils::is_integer_type(else_type) && then_type.is_floatbv())
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
  log_debug(
    "python-frontend",
    "[resolve_ternary_type] Ternary branches have incompatible types: {} vs "
    "{}, using default {}",
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

  // Check if we need to materialize function calls in the condition
  // This handles cases like: if not math.isnan(x): or if isinstance(x, type):
  auto test_type = ast_node["test"]["_type"].get<std::string>();

  bool has_nested_call = false;
  nlohmann::json call_node;
  bool is_wrapped_in_unary = false;

  // Check for function call wrapped in UnaryOp (e.g., "not func()")
  if (test_type == "UnaryOp" && ast_node["test"].contains("operand"))
  {
    auto operand_type = ast_node["test"]["operand"]["_type"].get<std::string>();
    if (operand_type == "Call")
    {
      has_nested_call = true;
      is_wrapped_in_unary = true;
      call_node = ast_node["test"]["operand"];
    }
  }
  // Check for direct function call
  else if (test_type == "Call")
  {
    has_nested_call = true;
    call_node = ast_node["test"];
  }

  // Extract condition from AST
  exprt cond;

  // Materialize function call if needed
  if (has_nested_call)
  {
    locationt location = get_location_from_decl(call_node);

    // Get the function call expression with special handling
    // Temporarily disable the conditional processing to avoid recursion
    exprt func_call = get_expr(call_node);

    if (func_call.is_function_call())
    {
      // Create temporary variable for function call result
      symbolt temp_symbol =
        create_return_temp_variable(func_call.type(), location, "cond");
      symbol_table_.add(temp_symbol);
      exprt temp_var_expr = symbol_expr(temp_symbol);

      // Create declaration for temporary
      code_declt temp_decl(temp_var_expr);
      temp_decl.location() = location;

      // Set the LHS of the function call
      if (!func_call.type().is_empty())
        func_call.op0() = temp_var_expr;

      // Add both declaration and function call to current_block
      if (current_block)
      {
        current_block->copy_to_operands(temp_decl);
        current_block->copy_to_operands(func_call);
      }

      // Build the final condition expression
      if (is_wrapped_in_unary)
      {
        // Rebuild the UnaryOp with our temp var
        auto op = ast_node["test"]["op"]["_type"].get<std::string>();
        if (op == "Not")
        {
          cond = exprt("not", bool_type());
          cond.copy_to_operands(temp_var_expr);
        }
        else
        {
          // For other unary operators, try to build them manually
          // This avoids calling get_expr which might cause recursion
          cond = temp_var_expr;
        }
      }
      else
      {
        // Direct call: use the temp var
        cond = temp_var_expr;
      }
    }
    else
    {
      // If it's not actually a function call, fall back to normal processing
      cond = get_expr(ast_node["test"]);
    }
  }
  else
  {
    // Normal path: no function call to materialize
    cond = get_expr(ast_node["test"]);
  }

  cond.location() = get_location_from_decl(ast_node["test"]);

  // Recover type
  current_element_type = t;

  // Extract 'then' block from AST
  exprt then;

  // Skip the 'then' block when the condition evaluates to false.
  if (cond.is_constant() && cond.value() == "false")
  {
    then = code_blockt();
  }
  else
  {
    if (ast_node["body"].is_array())
      then = get_block(ast_node["body"]);
    else
      then = get_expr(ast_node["body"]);
  }

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

    // Handle array-to-pointer conversion for ternary expressions
    // When assigning to a pointer (e.g., str field), convert array branches to pointers
    if (
      then.type().is_array() && else_expr.type().is_array() && current_lhs &&
      current_lhs->type().is_pointer())
    {
      then = string_handler_.get_array_base_address(then);
      else_expr = string_handler_.get_array_base_address(else_expr);
      result_type = then.type(); // Use pointer type as result
    }

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

  return code;
}

// Extract non-None type from union
std::string
python_converter::extract_non_none_type(const nlohmann::json &annotation_node)
{
  std::function<std::string(const nlohmann::json &)> extract_type =
    [&](const nlohmann::json &node) -> std::string {
    if (
      node.contains("_type") && node["_type"] == "Constant" &&
      node.contains("value") && node["value"].is_null())
      return ""; // This is None

    if (node.contains("id"))
      return node["id"].get<std::string>();

    // Handle Subscript nodes (such as Literal["bar"] or Sequence[str])
    if (node.contains("_type") && node["_type"] == "Subscript")
    {
      if (node.contains("value") && node["value"].is_object())
      {
        const auto &value_node = node["value"];
        // Handle Name nodes (e.g., List[int], Literal["bar"])
        if (value_node.contains("id"))
        {
          std::string subscript_type = value_node["id"].get<std::string>();
          if (subscript_type == "Literal")
            return "__LITERAL__"; // Special marker for Literal types
          // For Sequence[str], List[int], etc., return "list" as the concrete type
          if (subscript_type == "Sequence" || subscript_type == "List")
            return "list";
          // For other generic types, return the base type
          return subscript_type;
        }
        // Handle Attribute nodes (e.g., re.Match[str], typing.Optional[int])
        if (
          value_node.contains("_type") && value_node["_type"] == "Attribute" &&
          value_node.contains("attr"))
        {
          // Return special marker for external module types that should be
          // treated as opaque/any type (e.g., re.Match, typing.Pattern, etc.)
          return "__EXTERNAL_TYPE__";
        }
      }
      return ""; // Other subscript types
    }

    // Handle standalone Attribute nodes (e.g., module.Type without subscript)
    if (
      node.contains("_type") && node["_type"] == "Attribute" &&
      node.contains("attr"))
    {
      return "__EXTERNAL_TYPE__";
    }

    // Recursively handle nested BinOp (e.g., bool | str in bool | str | None)
    if (node.contains("_type") && node["_type"] == "BinOp")
    {
      if (node.contains("left"))
      {
        std::string left_type = extract_type(node["left"]);
        if (!left_type.empty())
          return left_type;
      }
      if (node.contains("right"))
        return extract_type(node["right"]);
    }

    return "";
  };

  // Guard: ensure annotation_node has left and right before accessing
  if (!annotation_node.contains("left") || !annotation_node.contains("right"))
    return "";

  const auto &left = annotation_node["left"];
  const auto &right = annotation_node["right"];

  // Extract the first non-None type
  std::string inner_type = extract_type(left);
  if (inner_type.empty())
    inner_type = extract_type(right);

  return inner_type;
}

typet python_converter::get_type_from_annotation(
  const nlohmann::json &annotation_node,
  const nlohmann::json &element)
{
  // Be defensive: not all annotation nodes are guaranteed to have the same
  // structure. In particular, forward references or tool-generated annotations
  // may appear as plain strings or objects without a "_type" field. On some
  // platforms (e.g., macOS debug builds) accessing a missing key via
  // operator[] triggers an assertion inside nlohmann::json, so we must guard
  // all such uses.
  if (!annotation_node.is_object())
  {
    // String-like forward reference, e.g. "CoordinateData | None"
    if (annotation_node.is_string())
    {
      std::string type_string = annotation_node.get<std::string>();
      type_string = type_utils::remove_quotes(type_string);
      return type_handler_.get_typet(type_string);
    }

    // Unknown/unsupported shape – fall back to empty type (no assertion
    // should be emitted for this annotation).
    return empty_typet();
  }

  if (!annotation_node.contains("_type"))
  {
    // Minimal object with direct "id" field, e.g. {"id": "int"}
    if (annotation_node.contains("id"))
    {
      std::string type_id = annotation_node["id"].get<std::string>();

      if (type_id == "dict" || type_id == "Dict")
        return dict_handler_->get_dict_struct_type();
      if (type_id == "list" || type_id == "List")
        return type_handler_.get_list_type();

      return type_handler_.get_typet(type_id);
    }

    // Nothing recognizable – treat as empty/unknown
    return empty_typet();
  }

  if (annotation_node["_type"] == "Subscript")
  {
    // Helper to safely get id from value node
    auto get_value_id = [&]() -> std::string {
      if (
        annotation_node.contains("value") &&
        annotation_node["value"].is_object() &&
        annotation_node["value"].contains("id"))
      {
        return annotation_node["value"]["id"].get<std::string>();
      }
      return "";
    };

    std::string value_id = get_value_id();

    if (value_id == "list" || value_id == "List")
      return type_handler_.get_list_type();

    if (value_id == "dict" || value_id == "Dict")
      return dict_handler_->get_dict_struct_type();

    // Handle Literal[T]: extract the type from the literal value
    if (value_id == "Literal")
    {
      // Infer type from a literal constant value
      auto infer_literal_type = [](const nlohmann::json &value) -> typet {
        if (value.is_string())
          return gen_pointer_type(char_type());
        else if (value.is_number_integer())
          return long_long_int_type();
        else if (value.is_boolean())
          return bool_type();
        else if (value.is_number_float())
          return double_type();
        else if (value.is_null())
          return none_type();

        return empty_typet(); // Unsupported type
      };

      // Resolve a slice element to a constant value
      auto resolve_to_constant =
        [this](const nlohmann::json &elem) -> nlohmann::json {
        // Guard: ensure elem is an object with _type
        if (!elem.is_object() || !elem.contains("_type"))
          return nlohmann::json();

        // Direct constant
        if (elem["_type"] == "Constant" && elem.contains("value"))
          return elem["value"];
        // Variable reference: resolve it
        if (elem["_type"] == "Name" && elem.contains("id"))
        {
          std::string var_name = elem["id"].get<std::string>();
          nlohmann::json var_decl =
            json_utils::find_var_decl(var_name, "", *ast_json);
          if (
            !var_decl.empty() && var_decl.contains("value") &&
            var_decl["value"].is_object() &&
            var_decl["value"].contains("_type") &&
            var_decl["value"]["_type"] == "Constant" &&
            var_decl["value"].contains("value"))
          {
            return var_decl["value"]["value"];
          }
        }
        return nlohmann::json(); // Could not resolve
      };

      // Track type flags from a resolved type
      auto update_type_flags = [](
                                 const typet &type,
                                 TypeFlags &flags,
                                 bool &has_string,
                                 bool &has_none) {
        if (type == gen_pointer_type(char_type()))
          has_string = true;
        else if (type == double_type())
          flags.has_float = true;
        else if (type == long_long_int_type())
          flags.has_int = true;
        else if (type == bool_type())
          flags.has_bool = true;
        else if (type == none_type())
          has_none = true;
        else if (type == pointer_type())
        {
          // Mixed type: mark as having both string and numeric
          has_string = true;
          flags.has_int = true;
        }
      };

      if (annotation_node.contains("slice"))
      {
        const auto &slice = annotation_node["slice"];

        // Guard: ensure slice is an object with _type
        if (!slice.is_object() || !slice.contains("_type"))
          return empty_typet();

        // Helper to safely check if node is a Literal subscript
        auto is_literal_subscript_node =
          [](const nlohmann::json &node) -> bool {
          return node.is_object() && node.contains("_type") &&
                 node["_type"] == "Subscript" && node.contains("value") &&
                 node["value"].is_object() && node["value"].contains("id") &&
                 node["value"]["id"] == "Literal";
        };

        // Handle nested Literal (e.g., Literal[Literal["foo"]])
        if (is_literal_subscript_node(slice))
        {
          return get_type_from_annotation(slice, element);
        }
        // Handle Literal with single value (e.g., Literal["foo"] or Literal[NAME])
        if (slice["_type"] == "Constant" && slice.contains("value"))
        {
          typet result = infer_literal_type(slice["value"]);
          if (!result.is_empty())
            return result;
        }
        else if (slice["_type"] == "Name")
        {
          nlohmann::json resolved_value = resolve_to_constant(slice);
          if (!resolved_value.is_null())
          {
            typet result = infer_literal_type(resolved_value);
            if (!result.is_empty())
              return result;
          }
          if (slice.contains("id"))
          {
            throw std::runtime_error(
              "Literal annotation references variable '" +
              slice["id"].get<std::string>() +
              "' which could not be resolved to a constant value.");
          }
          throw std::runtime_error(
            "Literal annotation references variable which could not be "
            "resolved to a constant value.");
        }
        // Handle Literal with multiple values
        else if (slice["_type"] == "Tuple" && slice.contains("elts"))
        {
          const auto &elts = slice["elts"];
          if (elts.empty())
            throw std::runtime_error("Empty Literal tuple is not supported.");

          TypeFlags type_flags;
          bool has_string = false;
          bool has_none = false;

          for (size_t i = 0; i < elts.size(); ++i)
          {
            const auto &elem = elts[i];
            // Handle nested Literal in tuple
            if (is_literal_subscript_node(elem))
            {
              typet nested_type = get_type_from_annotation(elem, element);
              update_type_flags(nested_type, type_flags, has_string, has_none);
              continue;
            }
            // Try to resolve element to constant
            nlohmann::json resolved_value = resolve_to_constant(elem);
            if (resolved_value.is_null())
            {
              std::string error_msg =
                "Literal tuple element at index " + std::to_string(i);
              if (
                elem.is_object() && elem.contains("_type") &&
                elem["_type"] == "Name" && elem.contains("id"))
                error_msg +=
                  " references variable '" + elem["id"].get<std::string>() +
                  "' which could not be resolved to a constant value.";
              else
                error_msg += " is not a constant value.";
              throw std::runtime_error(error_msg);
            }
            typet elem_type = infer_literal_type(resolved_value);
            if (elem_type.is_empty())
            {
              throw std::runtime_error(
                "Unsupported literal type at index " + std::to_string(i) +
                " in Literal tuple.");
            }
            update_type_flags(elem_type, type_flags, has_string, has_none);
          }
          // Determine the widest type: string > float > int > bool > None
          if (has_string)
          {
            if (
              type_flags.has_float || type_flags.has_int || type_flags.has_bool)
              return pointer_type(); // Mixed string and numeric
            return gen_pointer_type(char_type());
          }
          if (type_flags.has_float)
            return double_type();
          if (type_flags.has_int)
            return long_long_int_type();
          if (type_flags.has_bool)
            return bool_type();
          if (has_none)
            return none_type();
          throw std::runtime_error(
            "Could not determine type for Literal tuple.");
        }
      }
      throw std::runtime_error(
        "Unsupported (or malformed) Literal type annotation. "
        "We currently support constant values (string, int, bool, float, or "
        "None).");
    }

    // Handle Optional[T] - extract the inner type T
    if (
      annotation_node.contains("value") &&
      annotation_node["value"].is_object() &&
      annotation_node["value"].contains("id") &&
      annotation_node["value"]["id"] == "Optional")
    {
      if (
        annotation_node.contains("slice") &&
        annotation_node["slice"].is_object() &&
        annotation_node["slice"].contains("id"))
      {
        std::string inner_type =
          annotation_node["slice"]["id"].get<std::string>();
        typet base_type = type_handler_.get_typet(inner_type);
        // Always use pointer type for Optional to properly represent None
        return gen_pointer_type(base_type);
      }
    }

    // Handle external module types in Subscript (e.g., re.Match[str])
    // Treat as opaque/any type
    if (
      annotation_node.contains("value") &&
      annotation_node["value"].is_object() &&
      annotation_node["value"].contains("_type") &&
      annotation_node["value"]["_type"] == "Attribute")
    {
      return any_type();
    }

    return type_handler_.get_list_type(element);
  }
  else if (annotation_node["_type"] == "BinOp")
  {
    // Handle union types such as str | None (PEP 604 syntax)
    std::string inner_type = extract_non_none_type(annotation_node);

    // Special handling for Literal types in unions
    if (inner_type == "__LITERAL__")
    {
      // Find the Literal node and recursively process it
      const auto &left = annotation_node["left"];
      const auto &right = annotation_node["right"];

      // Helper to check if a node is a Literal subscript
      auto is_literal_subscript = [](const nlohmann::json &node) -> bool {
        return node.contains("_type") && node["_type"] == "Subscript" &&
               node.contains("value") && node["value"].is_object() &&
               node["value"].contains("id") && node["value"]["id"] == "Literal";
      };

      const auto &literal_node = is_literal_subscript(left) ? left : right;

      return get_type_from_annotation(literal_node, element);
    }

    // Special handling for external module types (e.g., re.Match[str] | None)
    // Treat them as opaque pointers (any_type)
    if (inner_type == "__EXTERNAL_TYPE__")
    {
      return any_type();
    }

    if (inner_type.empty())
    {
      // All types were None or couldn't be extracted - use any_type (void*)
      return any_type();
    }

    // Count the number of distinct type names in the union
    std::set<std::string> type_names;
    std::function<void(const nlohmann::json &)> collect_types;
    bool contains_none = false;
    collect_types = [&](const nlohmann::json &node) {
      // Guard: only process objects
      if (!node.is_object())
        return;

      if (
        node.contains("_type") && node["_type"] == "Constant" &&
        node.contains("value") && node["value"].is_null())
      {
        // This is None, skip it
        contains_none = true;
        return;
      }
      if (node.contains("id"))
        type_names.insert(node["id"].get<std::string>());
      // Handle Attribute nodes (e.g., re.Match in re.Match[str])
      if (
        node.contains("_type") && node["_type"] == "Attribute" &&
        node.contains("attr"))
        type_names.insert(node["attr"].get<std::string>());
      // Handle Subscript nodes (e.g., re.Match[str], List[int])
      if (node.contains("_type") && node["_type"] == "Subscript")
      {
        if (node.contains("value") && node["value"].is_object())
        {
          const auto &value_node = node["value"];
          if (value_node.contains("id"))
            type_names.insert(value_node["id"].get<std::string>());
          else if (
            value_node.contains("_type") &&
            value_node["_type"] == "Attribute" && value_node.contains("attr"))
            type_names.insert(value_node["attr"].get<std::string>());
        }
      }
      if (node.contains("_type") && node["_type"] == "BinOp")
      {
        if (node.contains("left"))
          collect_types(node["left"]);
        if (node.contains("right"))
          collect_types(node["right"]);
      }
    };
    collect_types(annotation_node);

    // If we have multiple types, treat as untyped pointer
    // This preserves the original behavior for type checking
    if (type_names.size() > 1 && contains_none)
      return gen_pointer_type(char_type());

    // Treat T | ... | None as Optional[T]
    typet base_type = type_handler_.get_typet(inner_type);

    // Single type + None: use Optional wrapper for primitives only
    if (
      base_type == long_long_int_type() || base_type == long_long_uint_type() ||
      base_type == double_type() || base_type == bool_type())
    {
      return type_handler_.build_optional_type(base_type);
    }

    // List types are already pointers
    if (base_type == type_handler_.get_list_type())
      return base_type;

    // For other types (e.g., classes, strings), use pointer type
    return gen_pointer_type(base_type);
  }
  else if (
    annotation_node["_type"] == "Constant" || annotation_node["_type"] == "Str")
  {
    // Handle None annotation: Constant with null value
    if (annotation_node["value"].is_null())
      return none_type();

    // Handle string annotations like "CoordinateData | None" (forward references)
    std::string type_string = annotation_node["value"].get<std::string>();
    type_string = type_utils::remove_quotes(type_string);
    return type_handler_.get_typet(type_string);
  }
  else if (
    annotation_node["_type"] == "Attribute" && annotation_node.contains("attr"))
    return type_handler_.get_typet(annotation_node["attr"].get<std::string>());
  else if (annotation_node.contains("id"))
  {
    std::string type_id = annotation_node["id"].get<std::string>();

    // Special handling for dict type
    if (type_id == "dict" || type_id == "Dict")
      return dict_handler_->get_dict_struct_type();

    // Special handling for list type
    if (type_id == "list" || type_id == "List")
      return type_handler_.get_list_type();

    return type_handler_.get_typet(type_id);
  }
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

TypeFlags
python_converter::infer_types_from_returns(const nlohmann::json &function_body)
{
  TypeFlags flags;

  std::function<void(const nlohmann::json &)> scan =
    [&](const nlohmann::json &body) {
      for (const auto &stmt : body)
      {
        if (stmt["_type"] == "Return" && !stmt["value"].is_null())
        {
          const auto &val = stmt["value"];

          if (val["_type"] == "Constant")
          {
            const auto &constant_val = val["value"];
            if (constant_val.is_number_float())
              flags.has_float = true;
            else if (constant_val.is_number_integer())
              flags.has_int = true;
            else if (constant_val.is_boolean())
              flags.has_bool = true;
            else
            {
              std::string type_name = constant_val.is_string()   ? "string"
                                      : constant_val.is_null()   ? "null"
                                      : constant_val.is_object() ? "object"
                                      : constant_val.is_array()  ? "array"
                                                                 : "unknown";
              throw std::runtime_error(
                "Unsupported return type '" + type_name + "' detected");
            }
          }
          else if (val["_type"] == "BinOp" || val["_type"] == "UnaryOp")
          {
            flags.has_float = true; // Default for expressions
          }
        }

        if (stmt.contains("body") && stmt["body"].is_array())
          scan(stmt["body"]);
        if (stmt.contains("orelse") && stmt["orelse"].is_array())
          scan(stmt["orelse"]);
      }
    };

  scan(function_body);
  return flags;
}

size_t python_converter::register_function_argument(
  const nlohmann::json &element,
  code_typet &type,
  const symbol_id &id,
  const locationt &location,
  bool is_keyword_only)
{
  (void)is_keyword_only;

  // Extract the argument name and resolve its type from the annotation.
  // Special cases: `self` and `cls` are modelled as pointers to the current class
  std::string arg_name = element["arg"].get<std::string>();
  typet arg_type;

  if (arg_name == "self")
    arg_type = gen_pointer_type(type_handler_.get_typet(current_class_name_));
  else if (arg_name == "cls")
    arg_type = any_type();
  else
  {
    if (!element.contains("annotation") || element["annotation"].is_null())
    {
      throw std::runtime_error(
        "All parameters in function \"" + current_func_name_ +
        "\" must be type annotated");
    }
    arg_type = get_type_from_annotation(element["annotation"], element);
  }

  // Arrays are converted to pointers so that the backend receives the same
  // representation regardless of how the parameter is declared.
  if (arg_type.is_array())
    arg_type = gen_pointer_type(arg_type.subtype());

  assert(arg_type != typet());

  code_typet::argumentt arg;
  arg.type() = arg_type;
  arg.cmt_base_name(arg_name);

  // Build a unique identifier for the parameter. The identifier mirrors the
  // scheme used elsewhere in the converter (function-id@parameter-name)
  std::string arg_id = id.to_string() + "@" + arg_name;
  arg.cmt_identifier(arg_id);
  arg.identifier(arg_id);
  arg.location() = get_location_from_decl(element);

  type.arguments().push_back(arg);
  size_t inserted_index = type.arguments().size() - 1;

  // Materialise a symbol for the parameter so that subsequent passes (e.g.
  // attribute access on instances) can resolve it.
  symbolt param_symbol = create_symbol(
    location.get_file().as_string(),
    arg_name,
    arg_id,
    arg.location(),
    arg_type);
  param_symbol.lvalue = true;
  param_symbol.is_parameter = true;
  param_symbol.file_local = true;
  param_symbol.static_lifetime = false;
  param_symbol.is_extern = false;
  symbol_table_.add(param_symbol);
  if (element.contains("annotation") && !element["annotation"].is_null())
    get_typechecker().cache_annotation_types(
      param_symbol, element["annotation"]);

  // If the parameter is class-typed (e.g. Foo), copy instance attributes from
  // the class’ synthetic `self` symbol so method bodies can access members via
  // this parameter.
  if (arg_name != "self" && arg_name != "cls")
  {
    typet base_type = arg_type.is_pointer() ? arg_type.subtype() : arg_type;
    if (base_type.id() == "symbol")
      base_type = ns.follow(base_type);

    if (base_type.is_struct())
    {
      const struct_typet &struct_type = to_struct_type(base_type);
      std::string class_tag = struct_type.tag().as_string();

      std::string class_name = extract_class_name_from_tag(class_tag);

      symbol_id self_sid(
        location.get_file().as_string(), class_name, class_name);
      self_sid.set_object("self");

      copy_instance_attributes(self_sid.to_string(), arg_id);

      std::string normalized_key = create_normalized_self_key(class_tag);
      copy_instance_attributes(normalized_key, arg_id);
    }
  }

  return inserted_index;
}

void python_converter::process_function_arguments(
  const nlohmann::json &function_node,
  code_typet &type,
  const symbol_id &id,
  const locationt &location)
{
  std::vector<size_t> positional_indices;
  std::vector<size_t> kwonly_indices;

  // Extract args node to avoid repeated access
  const nlohmann::json &args_node = function_node["args"];

  // Process regular arguments
  for (const nlohmann::json &element : args_node["args"])
  {
    size_t index =
      register_function_argument(element, type, id, location, false);
    positional_indices.push_back(index);
  }

  // Process keyword-only arguments (parameters after * separator)
  if (args_node.contains("kwonlyargs") && !args_node["kwonlyargs"].is_null())
  {
    for (const nlohmann::json &element : args_node["kwonlyargs"])
    {
      size_t index =
        register_function_argument(element, type, id, location, true);
      kwonly_indices.push_back(index);
    }
  }

  if (
    args_node.contains("defaults") && args_node["defaults"].is_array() &&
    !args_node["defaults"].empty() && !positional_indices.empty())
  {
    const auto &defaults = args_node["defaults"];
    size_t defaults_count = defaults.size();

    if (defaults_count <= positional_indices.size())
    {
      for (size_t i = 0; i < defaults_count; ++i)
      {
        size_t positional_index =
          positional_indices[positional_indices.size() - defaults_count + i];
        if (!defaults[i].is_null())
        {
          exprt default_expr = get_expr(defaults[i]);
          type.arguments()[positional_index].default_value() = default_expr;
        }
      }
    }
  }

  if (
    args_node.contains("kw_defaults") && args_node["kw_defaults"].is_array() &&
    args_node["kw_defaults"].size() == kwonly_indices.size())
  {
    const auto &kw_defaults = args_node["kw_defaults"];
    for (size_t i = 0; i < kw_defaults.size(); ++i)
    {
      if (!kw_defaults[i].is_null())
      {
        exprt default_expr = get_expr(kw_defaults[i]);
        type.arguments()[kwonly_indices[i]].default_value() = default_expr;
      }
    }
  }
}

void python_converter::validate_return_paths(
  const nlohmann::json &function_node,
  const code_typet &type,
  exprt &function_body)
{
  // Skip validation for void returns and constructors
  if (
    type.return_type().is_empty() ||
    type.return_type().id() == typet::t_empty ||
    type.return_type().id() == "constructor" ||
    !function_has_missing_return_paths(function_node))
  {
    return;
  }

  locationt loc = get_location_from_decl(function_node);

  code_assertt missing_return_assert;
  missing_return_assert.assertion() = gen_boolean(false);
  missing_return_assert.location() = loc;
  missing_return_assert.location().comment(
    "Missing return statement detected in function '" + current_func_name_ +
    "'");

  function_body.copy_to_operands(missing_return_assert);
}

void python_converter::get_function_definition(
  const nlohmann::json &function_node)
{
  // Function return type
  code_typet type;
  const nlohmann::json &return_node = function_node["returns"];

  // Determine return type
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

    if (return_type == "Any")
    {
      // Infer type from return statements
      TypeFlags flags = infer_types_from_returns(function_node["body"]);
      type.return_type() = type_utils::select_widest_type(flags, double_type());

      if (!flags.has_float && !flags.has_int && !flags.has_bool)
        log_warning("Default to double since no type could be inferred");
    }
    else if (return_type == "Union")
    {
      // Extract Union member types
      TypeFlags flags = type_utils::extract_union_types(return_node["slice"]);
      type.return_type() = type_utils::select_widest_type(flags, any_type());

      if (!flags.has_float && !flags.has_int && !flags.has_bool)
        log_warning("Union with no recognized types, defaulting to pointer");
    }
    else if (return_type == "list" || return_type == "List")
    {
      type.return_type() = type_handler_.get_list_type();
    }
    else if (return_type == "dict" || return_type == "Dict")
    {
      type.return_type() = dict_handler_->get_dict_struct_type();
    }
    else if (return_type == "str")
    {
      // String return types should be pointers, not arrays
      type.return_type() = gen_pointer_type(char_type());
    }
    else if (
      (return_type == "Tuple" || return_type == "tuple") &&
      return_node["_type"] == "Subscript")
    {
      type.return_type() =
        tuple_handler_->get_tuple_type_from_annotation(return_node);
    }
    else
    {
      type.return_type() =
        type_handler_.get_typet(return_type.get<std::string>());
    }
  }
  else if (return_node["_type"] == "BinOp")
  {
    // Handle PEP 604 union syntax: int | bool
    TypeFlags flags = type_utils::extract_binop_union_types(return_node);
    type.return_type() = type_utils::select_widest_type(flags, any_type());

    if (!flags.has_float && !flags.has_int && !flags.has_bool)
      log_warning("Union with no recognized types, defaulting to pointer");
  }
  else if (return_node["_type"] == "Tuple")
  {
    // Handle tuple return types such as (int, str)
    // TODO: we must still handle tuple types!
    type.return_type() = type_handler_.get_typet(std::string("tuple"));
  }
  else if (return_node["_type"] == "Constant" || return_node["_type"] == "Str")
  {
    std::string type_string =
      type_utils::remove_quotes(return_node["value"].get<std::string>());
    if (type_string == "str")
      type.return_type() = gen_pointer_type(char_type());
    else
      type.return_type() = type_handler_.get_typet(type_string);
  }
  else
    throw std::runtime_error("Return type undefined");

  // Setup function context
  const std::string caller_func_name = current_func_name_;

  // Function location
  locationt location = get_location_from_decl(function_node);

  current_element_type = type.return_type();
  current_func_name_ = function_node["name"].get<std::string>();

  // __init__() is renamed to Classname()
  if (current_func_name_ == "__init__")
  {
    current_func_name_ = current_class_name_;
    type.return_type() = typet("constructor");
  }

  symbol_id id = create_symbol_id();

  std::string module_name =
    current_python_file.substr(0, current_python_file.find_last_of("."));

  // Process function arguments
  process_function_arguments(function_node, type, id, location);

  // Create and register function symbol
  symbolt symbol = create_symbol(
    module_name, current_func_name_, id.to_string(), location, type);
  symbol.lvalue = true;
  symbol.is_extern = false;
  symbol.file_local = false;

  symbolt *added_symbol = symbol_table_.move_symbol_to_context(symbol);

  // Process function body
  exprt function_body = get_block(function_node["body"]);

  // Inject runtime checks for annotated parameters
  if (type_assertions_enabled())
    get_typechecker().inject_parameter_type_assertions(
      function_node, id, type, function_body);

  // Add ESBMC_Hide label for models/imports
  if (is_loading_models || is_importing_module)
  {
    code_labelt esbmc_hide;
    esbmc_hide.set_label("__ESBMC_HIDE");
    esbmc_hide.code() = code_skipt();
    function_body.operands().insert(
      function_body.operands().begin(), esbmc_hide);
  }

  // Validate return paths
  validate_return_paths(function_node, type, function_body);

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
      const std::string &attr_name = stmt["target"]["attr"];

      // Handle both simple names (id) and module-qualified names (Attribute)
      std::string annotated_type;

      if (stmt["annotation"].contains("id"))
      {
        // Simple type annotation such as self._md: Bar
        annotated_type = stmt["annotation"]["id"].get<std::string>();
      }
      else if (
        stmt["annotation"].contains("_type") &&
        stmt["annotation"]["_type"] == "Attribute")
      {
        // Module-qualified type annotation like: self._md: md.Bar
        // Extract just the class name (the attribute part)
        annotated_type = stmt["annotation"]["attr"].get<std::string>();
      }
      else
      {
        log_warning(
          "Skipping attribute '{}' with unsupported annotation type",
          attr_name);
        continue;
      }

      typet type;
      if (annotated_type == "str")
        type = gen_pointer_type(char_type());
      else if (annotated_type == "Optional")
      {
        typet base_type = get_type_from_annotation(stmt["annotation"], stmt);
        type = gen_pointer_type(base_type);
      }
      else
        type = type_handler_.get_typet(annotated_type);

      struct_typet::componentt comp =
        build_component(current_class_name_, attr_name, type);

      auto &class_components = clazz.components();
      if (
        std::find(class_components.begin(), class_components.end(), comp) ==
        class_components.end())
        class_components.push_back(comp);
    }
    else if (
      stmt["_type"] == "Assign" && stmt["targets"][0]["_type"] == "Attribute" &&
      stmt["targets"][0]["value"]["id"] == "self")
    {
      // A member is initialized with something that might be not annotated
      typet type = any_type();
      const std::string &attr_name = stmt["targets"][0]["attr"];
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

// Process forward reference
void python_converter::process_forward_reference(
  const nlohmann::json &annotation,
  codet &target_block)
{
  if (annotation.is_null())
    return;

  std::string referenced_class;

  // Process string form of forward reference: 'Bar'
  if (
    (annotation["_type"] == "Constant" || annotation["_type"] == "Str") &&
    annotation.contains("value") && !annotation["value"].is_null())
  {
    referenced_class =
      type_utils::remove_quotes(annotation["value"].get<std::string>());
  }
  // Process direct name reference: Bar
  else if (annotation["_type"] == "Name" && annotation.contains("id"))
  {
    referenced_class = annotation["id"].get<std::string>();

    if (
      type_utils::is_builtin_type(referenced_class) ||
      type_utils::is_consensus_type(referenced_class))
      return;
  }
  else
  {
    return;
  }

  // If class is already in symbol table, skip
  std::string class_id = "tag-" + referenced_class;
  if (symbol_table_.find_symbol(class_id))
    return;

  // Find and process referenced class definition
  const auto ref_class_node =
    json_utils::find_class((*ast_json)["body"], referenced_class);

  if (!ref_class_node.empty())
  {
    std::string saved_class = current_class_name_;
    std::string saved_func = current_func_name_;
    get_class_definition(ref_class_node, target_block);
    current_class_name_ = saved_class;
    current_func_name_ = saved_func;
  }
}

void python_converter::get_class_definition(
  const nlohmann::json &class_node,
  codet &target_block)
{
  python_class_builder(*this, class_node).build(target_block);
}

void python_converter::get_return_statements(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  if (ast_node["value"].is_null())
  {
    // Handle bare return statement (return with no value)
    locationt location = get_location_from_decl(ast_node);
    code_returnt return_code;
    return_code.location() = location;
    target_block.copy_to_operands(return_code);
    return;
  }

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

    // Determine return type: check if it's empty (forward reference)
    typet return_type = return_value.type();

    if (return_type.is_empty() || return_type.id() == typet::t_empty)
    {
      // Forward reference: function not yet processed
      // Look up return type from AST
      const auto &func_node =
        json_utils::find_function((*ast_json)["body"], func_name);

      if (
        !func_node.empty() && func_node.contains("returns") &&
        !func_node["returns"].is_null())
        return_type = get_type_from_annotation(func_node["returns"], func_node);
      else
      {
        // Default to void* if we can't determine the type
        return_type = any_type();
      }
    }

    // Create temporary variable to store function call result
    symbolt temp_symbol =
      create_return_temp_variable(return_type, location, func_name);
    symbol_table_.add(temp_symbol);
    exprt temp_var_expr = symbol_expr(temp_symbol);

    // Create declaration for temporary variable
    code_declt temp_decl(temp_var_expr);
    temp_decl.location() = location;
    target_block.copy_to_operands(temp_decl);

    // Set the LHS of the function call to our temporary variable
    if (!return_type.is_empty())
      return_value.op0() = temp_var_expr;

    // If a constructor is being invoked, the temporary variable is passed as 'self'
    if (type_handler_.is_constructor_call(ast_node["value"]))
    {
      code_function_callt &call =
        static_cast<code_function_callt &>(return_value);
      call.arguments().emplace(
        call.arguments().begin(), gen_address_of(temp_var_expr));
      update_instance_from_self(
        func_name, func_name, temp_var_expr.identifier().as_string());
    }

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
    // If we're returning an array but the function expects a pointer,
    // convert the array to a pointer (for string literals)
    const typet &expected_return_type = current_element_type;

    if (expected_return_type.is_pointer() && return_value.type().is_array())
    {
      // For constant array literals (string literals), convert to string_constantt
      if (return_value.is_constant())
      {
        // Extract the string content from the constant array
        std::string str_content;
        for (const auto &operand : return_value.operands())
        {
          if (operand.is_constant())
          {
            BigInt char_val = binary2integer(
              operand.value().as_string(), operand.type().is_signedbv());
            if (char_val == 0)
              break; // Stop at null terminator
            str_content += static_cast<char>(char_val.to_int64());
          }
        }

        // Create a string_constantt with proper type
        typet string_type = return_value.type();
        return_value = string_constantt(
          str_content, string_type, string_constantt::k_default);

        // Get its address (converts array to pointer)
        return_value = address_of_exprt(return_value);
      }
      else
      {
        // For non-constant arrays (variables), convert to pointer
        return_value = string_handler_.get_array_base_address(return_value);
      }
    }

    code_returnt return_code;
    return_code.return_value() = return_value;
    return_code.location() = location;
    target_block.copy_to_operands(return_code);
  }
}

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

// function to create function call statement from function call expression
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

void python_converter::handle_list_assertion(
  const nlohmann::json &element,
  const exprt &test,
  code_blockt &block,
  const std::function<void(code_assertt &)> &attach_assert_message)
{
  locationt location = get_location_from_decl(element);

  // Materialize function call if needed
  exprt list_expr = test;
  if (test.is_function_call())
  {
    // Create temp variable to store function result (pointer to list)
    symbolt &list_temp =
      create_tmp_symbol(element, "$list_assert_temp$", test.type(), exprt());
    code_declt list_decl(symbol_expr(list_temp));
    list_decl.location() = location;
    block.move_to_operands(list_decl);

    // Execute function call
    code_function_callt &func_call =
      static_cast<code_function_callt &>(const_cast<exprt &>(test));
    func_call.lhs() = symbol_expr(list_temp);
    block.move_to_operands(func_call);

    list_expr = symbol_expr(list_temp);
  }

  // Get list size using __ESBMC_list_size
  const symbolt *size_sym = symbol_table_.find_symbol("c:@F@__ESBMC_list_size");
  if (!size_sym)
    throw std::runtime_error("__ESBMC_list_size function not found");

  // Create temp var to store size
  symbolt &size_result = create_tmp_symbol(
    element, "$list_size_result$", size_type(), gen_zero(size_type()));
  code_declt size_decl(symbol_expr(size_result));
  size_decl.location() = location;
  block.move_to_operands(size_decl);

  // Call list_size(list)
  code_function_callt size_func_call;
  size_func_call.function() = symbol_expr(*size_sym);
  if (list_expr.type().is_pointer())
    size_func_call.arguments().push_back(list_expr);
  else
    size_func_call.arguments().push_back(address_of_exprt(list_expr));
  size_func_call.lhs() = symbol_expr(size_result);
  size_func_call.type() = size_type();
  size_func_call.location() = location;
  block.move_to_operands(size_func_call);

  // Assert size > 0
  exprt assertion(">", bool_type());
  assertion.copy_to_operands(symbol_expr(size_result), gen_zero(size_type()));

  code_assertt assert_code;
  assert_code.assertion() = assertion;
  assert_code.location() = location;
  attach_assert_message(assert_code);
  block.move_to_operands(assert_code);
}

void python_converter::handle_function_call_assertion(
  const nlohmann::json &element,
  const exprt &func_call_expr,
  bool is_negated,
  code_blockt &block,
  const std::function<void(code_assertt &)> &attach_assert_message)
{
  locationt location = get_location_from_decl(element);

  // Check if function returns None
  const typet &return_type = func_call_expr.type();

  if (return_type == none_type() || return_type.id() == "empty")
  {
    // Function returns None: execute call and assert False
    exprt func_call_copy = func_call_expr;
    codet code_stmt = convert_expression_to_code(func_call_copy);
    block.move_to_operands(code_stmt);

    code_assertt assert_code;
    assert_code.assertion() = false_exprt();
    assert_code.location() = location;
    assert_code.location().comment("Assertion on None-returning function");
    attach_assert_message(assert_code);
    block.move_to_operands(assert_code);
    return;
  }

  // Create temporary variable
  symbolt temp_symbol = create_assert_temp_variable(location);
  symbol_table_.add(temp_symbol);
  exprt temp_var_expr = symbol_expr(temp_symbol);

  // Create function call statement
  code_function_callt function_call =
    create_function_call_statement(func_call_expr, temp_var_expr, location);
  block.move_to_operands(function_call);

  // Create assertion based on negation
  exprt assertion_expr;
  if (is_negated)
  {
    assertion_expr = not_exprt(temp_var_expr);
  }
  else
  {
    exprt cast_expr = typecast_exprt(temp_var_expr, signedbv_typet(32));
    exprt one_expr = constant_exprt("1", signedbv_typet(32));
    assertion_expr = equality_exprt(cast_expr, one_expr);
  }

  code_assertt assert_code;
  assert_code.assertion() = assertion_expr;
  assert_code.location() = location;
  attach_assert_message(assert_code);
  block.move_to_operands(assert_code);
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
      if (test.statement() == "cpp-throw")
      {
        test.location() = get_location_from_decl(element);
        codet code_expr("expression");
        code_expr.operands().push_back(test);
        block.move_to_operands(code_expr);
        break;
      }

      // Attach assertion message if present
      auto attach_assert_message = [&element](code_assertt &assert_code) {
        if (element.contains("msg") && !element["msg"].is_null())
        {
          std::string msg;
          if (
            element["msg"]["_type"] == "Constant" &&
            element["msg"]["value"].is_string())
          {
            msg = element["msg"]["value"].get<std::string>();
          }
          else if (element["msg"]["_type"] == "JoinedStr")
          {
            // For f-strings, this is just a placeholder
            // TODO: Full f-string evaluation would require more complex handling
            msg = "<formatted string message>";
          }

          if (!msg.empty())
            assert_code.location().comment(msg);
        }
      };

      // Handle list assertions
      if (
        test.type() == type_handler_.get_list_type() ||
        (test.type().is_pointer() &&
         test.type().subtype() == type_handler_.get_list_type()))
      {
        handle_list_assertion(element, test, block, attach_assert_message);
        break;
      }

      // Check for function call assertions
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

      if (func_call_expr != nullptr)
      {
        handle_function_call_assertion(
          element, *func_call_expr, is_negated, block, attach_assert_message);
      }
      else
      {
        // Direct assertion
        if (!test.type().is_bool())
          test.make_typecast(current_element_type);

        code_assertt assert_code;
        assert_code.assertion() = test;
        assert_code.location() = get_location_from_decl(element);
        attach_assert_message(assert_code);
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
      symbolt *exception_symbol = nullptr;
      typet exception_type;

      // Create exception variable symbol before processing body
      if (!element["type"].is_null())
      {
        exception_type =
          type_handler_.get_typet(element["type"]["id"].get<std::string>());

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
        // Create and add symbol to symbol table
        symbolt symbol = create_symbol(
          module_name,
          current_func_name_,
          sid.to_string(),
          location,
          exception_type);
        symbol.name = name;
        symbol.lvalue = true;
        symbol.is_extern = false;
        symbol.file_local = false;
        exception_symbol = symbol_table_.move_symbol_to_context(symbol);
      }

      // Process exception handler body (symbol now exists)
      exprt catch_block = get_block(element["body"]);

      // Add type and declaration if exception variable was created
      if (exception_symbol != nullptr)
      {
        catch_block.type() = exception_type;
        exprt sym = symbol_expr(*exception_symbol);
        code_declt decl(sym);
        exprt decl_code = convert_expression_to_code(decl);
        decl_code.location() = exception_symbol->location;

        codet::operandst &ops = catch_block.operands();
        ops.insert(ops.begin(), decl_code);
      }

      block.move_to_operands(catch_block);
      break;
    }
    case StatementType::RAISE:
    {
      typet type = type_handler_.get_typet(
        element["exc"]["func"]["id"].get<std::string>());
      locationt location = get_location_from_decl(element);

      exprt raise;
      if (type_utils::is_python_exceptions(
            element["exc"]["func"]["id"].get<std::string>()))
      {
        // Construct a constant struct to throw:
        // raise { .message=&"Error message" }
        exprt arg = get_expr(element["exc"]["args"][0]);
        arg = string_constantt(
          string_handler_.process_format_spec(element["exc"]["args"][0]),
          arg.type(),
          string_constantt::k_default);

        raise.id("struct");
        raise.type() = type;
        raise.copy_to_operands(address_of_exprt(arg));
      }
      else
      {
        // For custom exceptions:
        // DECL MyException return_value;
        // FUNCTION_CALL:  MyException(&return_value, &"message");
        // Throw MyException return_value;
        raise = get_expr(element["exc"]);
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
    case StatementType::DELETE:
    {
      get_delete_statement(element, block);
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

  return block;
}

exprt python_converter::get_static_array(
  const nlohmann::json &arr,
  const typet &shape)
{
  exprt zero = gen_zero(size_type());
  exprt list = gen_zero(shape);

  unsigned int i = 0;
  for (auto &e : arr["elts"])
  {
    exprt element_expr = get_expr(e);
    list.operands().at(i++) = element_expr;
  }

  symbolt &cl = create_tmp_symbol(arr, "$compound-literal$", shape, list);

  exprt expr = symbol_expr(cl);
  code_declt decl(expr);
  decl.operands().push_back(list);
  assert(current_block);
  current_block->copy_to_operands(decl);

  return expr;
}

python_converter::python_converter(
  contextt &_context,
  const nlohmann::json *ast,
  const global_scope &gs)
  : symbol_table_(_context),
    ast_json(ast),
    global_scope_(gs),
    type_handler_(*this),
    string_builder_(new string_builder(*this, &string_handler_)),
    sym_generator_("python_converter::"),
    ns(_context),
    current_func_name_(""),
    current_class_name_(""),
    current_block(nullptr),
    current_lhs(nullptr),
    string_handler_(*this, symbol_table_, type_handler_, string_builder_),
    math_handler_(*this, symbol_table_, type_handler_),
    tuple_handler_(new tuple_handler(*this, type_handler_)),
    dict_handler_(new python_dict_handler(*this, symbol_table_, type_handler_)),
    typechecker_(new python_typechecking(*this))
{
}

python_converter::~python_converter()
{
  delete string_builder_;
  delete tuple_handler_;
  delete dict_handler_;
  delete typechecker_;
}

python_typechecking &python_converter::get_typechecker()
{
  return *typechecker_;
}

const python_typechecking &python_converter::get_typechecker() const
{
  return *typechecker_;
}

bool python_converter::type_assertions_enabled() const
{
  return config.options.get_bool_option("is-instance-check");
}

string_builder &python_converter::get_string_builder()
{
  if (!string_builder_)
  {
    string_builder_ = new string_builder(*this, &string_handler_);
    string_handler_.set_string_builder(string_builder_);
  }
  return *string_builder_;
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

void python_converter::load_c_intrisics(code_blockt &block)
{
  // Add symbols required by the C models

  add_global_static_variable(
    symbol_table_, int_type(), "__ESBMC_rounding_mode");

  auto type1 = array_typet(bool_type(), exprt("infinity"));
  add_global_static_variable(symbol_table_, type1, "__ESBMC_alloc");
  add_global_static_variable(symbol_table_, type1, "__ESBMC_deallocated");
  add_global_static_variable(symbol_table_, type1, "__ESBMC_is_dynamic");

  auto type2 = array_typet(size_type(), exprt("infinity"));
  add_global_static_variable(symbol_table_, type2, "__ESBMC_alloc_size");

  // Initialize intrinsic variables to match C frontend behavior
  locationt location;
  location.set_file("esbmc_intrinsics.h");

  // ASSIGN __ESBMC_rounding_mode = 0;
  symbol_exprt rounding_symbol("c:@__ESBMC_rounding_mode", int_type());
  code_assignt rounding_assign(rounding_symbol, gen_zero(int_type()));
  rounding_assign.location() = location;
  block.copy_to_operands(rounding_assign);

  // TODO: Consider initializing other intrinsic variables if needed:
  // - __ESBMC_alloc
  // - __ESBMC_deallocated
  // - __ESBMC_is_dynamic
  // - __ESBMC_alloc_size
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
      name_value = current_python_file;
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

void python_converter::process_module_imports(
  const nlohmann::json &module_ast,
  module_locator &locator,
  code_blockt &accumulated_code)
{
  // Process imports in this module first (depth-first)
  for (const auto &elem : module_ast["body"])
  {
    if (elem["_type"] == "ImportFrom" || elem["_type"] == "Import")
    {
      const std::string &module_name = (elem["_type"] == "ImportFrom")
                                         ? elem["module"]
                                         : elem["names"][0]["name"];

      // Skip if already imported
      if (imported_modules.find(module_name) != imported_modules.end())
        continue;

      std::ifstream imported_file = locator.open_module_file(module_name);
      if (!imported_file.is_open())
        continue; // Skip missing modules

      nlohmann::json nested_module_json;
      imported_file >> nested_module_json;

      std::string nested_python_file =
        nested_module_json["filename"].get<std::string>();
      imported_modules.emplace(module_name, nested_python_file);

      // Recursively process nested imports first
      process_module_imports(nested_module_json, locator, accumulated_code);

      // Then process this module's definitions
      std::string saved_file = current_python_file;
      current_python_file = nested_python_file;

      create_builtin_symbols();
      exprt imported_code = with_ast(&nested_module_json, [&]() {
        return get_block(nested_module_json["body"]);
      });
      convert_expression_to_code(imported_code);

      // Accumulate this module's code
      accumulated_code.copy_to_operands(imported_code);

      current_python_file = saved_file;
    }
  }
}

void python_converter::convert()
{
  main_python_file = (*ast_json)["filename"].get<std::string>();
  current_python_file = main_python_file;

  // Create built-in symbols for main module (__name__ = "__main__")
  create_builtin_symbols();

  // Block to accumulate model library code
  code_blockt models_block;

  if (!config.options.get_bool_option("no-library"))
  {
    // Load operational models
    const std::string &ast_output_dir =
      (*ast_json)["ast_output_dir"].get<std::string>();
    std::list<std::string> model_files = {
      "range",
      "int",
      "consensus",
      "random",
      "exceptions",
      "datetime",
      "nondet"};
    std::list<std::string> model_folders = {"os", "numpy"};

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

      exprt model_code =
        with_ast(&model_json, [&]() { return get_block((*ast_json)["body"]); });

      convert_expression_to_code(model_code);

      // Accumulate model code
      models_block.copy_to_operands(model_code);
      current_python_file = main_python_file;
    }
    is_loading_models = false;
  }

  // Create a block to hold intrinsic assignments and load C intrinsics
  code_blockt intrinsic_block;
  load_c_intrisics(intrinsic_block);

  // Variables to hold user code and initialization code
  codet user_code;
  code_blockt init_code;

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

    // Add intrinsic assignments first
    block.copy_to_operands(intrinsic_block);

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

    // Prepare user code: class definitions + function call
    code_blockt user_code_body;
    user_code_body.copy_to_operands(block);
    user_code_body.copy_to_operands(call);
    user_code.swap(user_code_body);

    // Add models to init code
    if (!models_block.operands().empty())
      init_code.copy_to_operands(models_block);
  }
  else
  {
    // Convert imported modules
    module_locator locator((*ast_json)["ast_output_dir"].get<std::string>());

    // Accumulate all imports
    code_blockt all_imports_block;

    for (const auto &elem : (*ast_json)["body"])
    {
      if (elem["_type"] == "ImportFrom" || elem["_type"] == "Import")
      {
        is_importing_module = true;
        const std::string &module_name = (elem["_type"] == "ImportFrom")
                                           ? elem["module"]
                                           : elem["names"][0]["name"];

        // Skip if already processed by recursive import
        if (imported_modules.find(module_name) != imported_modules.end())
          continue;

        std::ifstream imported_file = locator.open_module_file(module_name);
        if (!imported_file.is_open())
        {
          throw std::runtime_error(
            "Cannot open file: " + locator.module_path(module_name));
        }

        imported_file >> imported_module_json;

        current_python_file =
          imported_module_json["filename"].get<std::string>();
        imported_modules.emplace(module_name, current_python_file);

        // Process nested imports recursively first
        process_module_imports(
          imported_module_json, locator, all_imports_block);

        // Create built-in symbols for imported module
        create_builtin_symbols();

        // Annotate types in imported module before conversion
        python_annotation<nlohmann::json> imported_annotator(
          imported_module_json, const_cast<global_scope &>(global_scope_));
        imported_annotator.add_type_annotation();

        exprt imported_code = with_ast(&imported_module_json, [&]() {
          return get_block(imported_module_json["body"]);
        });

        convert_expression_to_code(imported_code);

        // Accumulate imported code instead of overwriting
        all_imports_block.copy_to_operands(imported_code);

        imported_module_json.clear();
      }
    }

    is_importing_module = false;
    current_python_file = main_python_file;

    // Convert main statements
    exprt main_block = get_block((*ast_json)["body"]);
    user_code = convert_expression_to_code(main_block);

    // Prepare initialization code: models + intrinsics + imports
    if (!models_block.operands().empty())
      init_code.copy_to_operands(models_block);
    init_code.copy_to_operands(intrinsic_block);
    if (!all_imports_block.operands().empty())
      init_code.copy_to_operands(all_imports_block);
  }

  /*
   * Create three-function architecture for coverage support (similar to Solidity Frontend):
   *
   * 1. python_init
   *    - Contains models, intrinsics, and imports initialization
   *    - Marked with __ESBMC_HIDE label to exclude from coverage statistics
   *    - Only created if there is initialization code
   *
   * 2. python_user_main
   *    - Contains only user code from the main module
   *    - This is what gets analyzed for branch/decision/assertion coverage
   *
   * 3. __ESBMC_main
   *    - Entry point for ESBMC verification
   *    - Initializes static lifetime variables
   *    - Calls python_init() if it exists
   *    - Calls python_user_main()
   *
   * This architecture ensures that coverage analysis only counts user code,
   * not initialization/library code, making Python behave consistently with C.
   */
  if (!init_code.operands().empty())
  {
    code_typet init_type;
    init_type.return_type() = empty_typet();

    symbolt init_symbol;
    init_symbol.id = "python_init";
    init_symbol.name = "python_init";
    init_symbol.type = init_type;
    init_symbol.lvalue = true;
    init_symbol.is_extern = false;
    init_symbol.file_local = false;
    init_symbol.location = get_location_from_decl(*ast_json);

    // Add __ESBMC_HIDE label to hide from coverage
    code_labelt esbmc_hide;
    esbmc_hide.set_label("__ESBMC_HIDE");
    esbmc_hide.code() = code_skipt();

    code_blockt init_body;
    init_body.copy_to_operands(esbmc_hide);
    init_body.copy_to_operands(init_code);
    init_symbol.value.swap(init_body);

    if (symbol_table_.move(init_symbol))
    {
      throw std::runtime_error("The python_init function is already defined");
    }
  }

  // Create python_user_main function containing only user code
  code_typet user_main_type;
  user_main_type.return_type() = empty_typet();

  symbolt user_main_symbol;
  user_main_symbol.id = "python_user_main";
  user_main_symbol.name = "python_user_main";
  user_main_symbol.type = user_main_type;
  user_main_symbol.lvalue = true;
  user_main_symbol.is_extern = false;
  user_main_symbol.file_local = false;
  user_main_symbol.location = get_location_from_decl(*ast_json);
  user_main_symbol.value = user_code;

  if (symbol_table_.move(user_main_symbol))
  {
    throw std::runtime_error(
      "The python_user_main function is already defined");
  }

  // Create __ESBMC_main that initializes and calls user code
  code_typet main_type;
  main_type.return_type() = empty_typet();

  symbolt main_symbol;
  main_symbol.id = "__ESBMC_main";
  main_symbol.name = "__ESBMC_main";
  main_symbol.type = main_type;
  main_symbol.lvalue = true;
  main_symbol.is_extern = false;
  main_symbol.file_local = false;
  main_symbol.location = get_location_from_decl(*ast_json);

  code_blockt main_body;

  // 1. Initialize static lifetime variables
  symbol_table_.foreach_operand_in_order([&main_body](const symbolt &s) {
    if (s.static_lifetime && !s.value.is_nil() && !s.type.is_code())
    {
      code_assignt assign(symbol_expr(s), s.value);
      assign.location() = s.location;
      main_body.copy_to_operands(assign);
    }
  });

  // 2. Call python_init for initialization
  if (!init_code.operands().empty())
  {
    const symbolt *init_sym = symbol_table_.find_symbol("python_init");
    if (init_sym)
    {
      code_function_callt init_call;
      init_call.function() = symbol_expr(*init_sym);
      main_body.copy_to_operands(init_call);
    }
  }

  // 3. Call python_user_main
  const symbolt *user_main_sym = symbol_table_.find_symbol("python_user_main");
  if (!user_main_sym)
  {
    throw std::runtime_error("python_user_main symbol not found after move");
  }

  code_function_callt user_main_call;
  user_main_call.function() = symbol_expr(*user_main_sym);
  main_body.copy_to_operands(user_main_call);

  main_symbol.value.swap(main_body);

  if (symbol_table_.move(main_symbol))
  {
    throw std::runtime_error(
      "The main function is already defined in another module");
  }
}

exprt python_converter::extract_type_from_boolean_op(const exprt &bool_op)
{
  // Only OR and AND are special
  if (!bool_op.is_and() && !bool_op.is_or())
    return gen_zero(bool_type());

  // Let's try to be smart and guess the type;
  // In the future this could be trivial with an Python Obj struct
  // 1. If there are no non-null constants, then guess any.
  // 2. If there is only one type of constant, then guess it.
  // 3. If there is more than one type of constant, then abort.

  typet found_type = empty_typet();
  assert(found_type.is_empty());

  for (exprt e : bool_op.operands())
  {
    // First, try to solve the underlying type...
    if (!e.is_constant() && !e.is_symbol())
      e = extract_type_from_boolean_op(e);

    assert(e.is_constant() || e.is_symbol());

    if (!e.is_pointer() && e.is_constant())
    {
      if (found_type.is_empty())
        found_type = e.type();
      else if (found_type != e.type() && !e.type().is_array())
      {
        log_error("Boolean expression with more than one constant type");
        bool_op.dump();
        abort();
      }
    }
  }

  // Arrays are special, they have a length property which we don't care about right now
  if (found_type.is_array())
    return gen_zero(any_type());

  return found_type.is_empty() ? gen_zero(any_type()) : gen_zero(found_type);
}

void python_converter::get_delete_statement(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  if (!ast_node.contains("targets") || !ast_node["targets"].is_array())
  {
    throw std::runtime_error("Delete statement missing targets");
  }

  for (const auto &target : ast_node["targets"])
  {
    if (target["_type"] == "Subscript")
    {
      exprt dict_expr = get_expr(target["value"]);
      const nlohmann::json &slice = target["slice"];

      typet dict_type = dict_expr.type();
      if (dict_expr.is_symbol())
      {
        const symbolt *sym = symbol_table_.find_symbol(dict_expr.identifier());
        if (sym)
          dict_type = sym->type;
      }

      if (dict_type.id() == "symbol")
        dict_type = ns.follow(dict_type);

      if (!dict_type.is_struct())
      {
        throw std::runtime_error(
          "del on subscript requires a dictionary (struct) type");
      }

      // Delegate to dict_handler which handles both constant and variable keys
      dict_handler_->handle_dict_delete(dict_expr, slice, target_block);
    }
    else if (target["_type"] == "Name")
    {
      log_warning("del on simple variables is not fully supported");
    }
    else
    {
      throw std::runtime_error(
        "Delete statement target type not supported: " +
        target["_type"].get<std::string>());
    }
  }
}

exprt python_converter::handle_set_operations(
  const std::string &op,
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json & /* left */,
  const nlohmann::json & /* right */,
  const nlohmann::json &element)
{
  typet list_type = type_handler_.get_list_type();

  // Ensure both operands are lists (sets are represented as lists)
  if (lhs.type() != list_type || rhs.type() != list_type)
    return nil_exprt();

  // Resolve function calls to temporary variables
  auto resolve_list_call = [&](exprt &expr) -> bool {
    if (
      expr.id().as_string() != "sideeffect" ||
      expr.get("statement") != "function_call" || expr.type() != list_type)
      return false;

    locationt location = get_location_from_decl(element);

    // Create temporary variable for the list
    symbolt &tmp_var_symbol =
      create_tmp_symbol(element, "tmp_set_op", list_type, gen_zero(list_type));

    code_declt tmp_var_decl(symbol_expr(tmp_var_symbol));
    tmp_var_decl.location() = location;
    current_block->copy_to_operands(tmp_var_decl);

    side_effect_expr_function_callt &side_effect =
      to_side_effect_expr_function_call(expr);

    code_function_callt call;
    call.function() = side_effect.function();
    call.arguments() = side_effect.arguments();
    call.lhs() = symbol_expr(tmp_var_symbol);
    call.type() = list_type;
    call.location() = location;

    current_block->copy_to_operands(call);
    expr = symbol_expr(tmp_var_symbol);
    return true;
  };

  resolve_list_call(lhs);
  resolve_list_call(rhs);

  python_set set_handler(*this, element);

  // Map Python set operations to internal functions
  if (op == "Sub") // Set difference: a - b
    return set_handler.build_set_difference_call(lhs, rhs, element);
  else if (op == "BitAnd") // Set intersection: a & b
    return set_handler.build_set_intersection_call(lhs, rhs, element);
  else if (op == "BitOr") // Set union: a | b
    return set_handler.build_set_union_call(lhs, rhs, element);

  return nil_exprt();
}
