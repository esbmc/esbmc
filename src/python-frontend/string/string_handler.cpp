#include <python-frontend/string/char_utils.h>
#include <python-frontend/exception_utils.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/python_int_overflow.h>
#include <python-frontend/python_list.h>
#include <python-frontend/string/string_method_dispatch.h>
#include <python-frontend/string/string_handler.h>
#include <python-frontend/string/string_handler_utils.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_exception_handler.h>
#include <python-frontend/python_expr_builder.h>
#include <python-frontend/string/string_builder.h>
#include <python-frontend/tuple_handler.h>
#include <python-frontend/type_utils.h>
#include <python-frontend/symbol_id.h>
#include <irep2/irep2_utils.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/migrate.h>
#include <util/python_types.h>
#include <util/std_expr.h>
#include <util/std_code.h>
#include <util/string_constant.h>
#include <util/symbol.h>
#include <util/type.h>

#include <boost/algorithm/string/predicate.hpp>
#include <array>
#include <cmath>
#include <cctype>
#include <climits>
#include <cstring>
#include <iomanip>
#include <limits>
#include <optional>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using namespace python_expr;

string_handler::string_handler(
  python_converter &converter,
  contextt &symbol_table,
  type_handler &type_handler,
  string_builder *str_builder)
  : converter_(converter),
    symbol_table_(symbol_table),
    type_handler_(type_handler),
    string_builder_(str_builder)
{
}

namespace
{
static size_t utf8_codepoint_count(const std::string &text)
{
  size_t count = 0;
  for (unsigned char c : text)
  {
    if ((c & 0xC0) != 0x80)
      ++count;
  }
  return count;
}

static std::optional<std::vector<BigInt>>
extract_constant_char_values(const exprt *array_expr)
{
  if (array_expr == nullptr || !array_expr->type().is_array())
    return std::nullopt;

  const exprt::operandst &ops = array_expr->operands();
  if (ops.empty())
    return std::vector<BigInt>{};

  std::vector<BigInt> values;
  values.reserve(ops.size() - 1);
  for (size_t i = 0; i + 1 < ops.size(); ++i)
  {
    if (!ops[i].is_constant())
      return std::nullopt;

    values.push_back(
      binary2integer(ops[i].value().as_string(), ops[i].type().is_signedbv()));
  }

  return values;
}

static constexpr long long kMembershipMaxHaystackContentLen = 256;
static constexpr long long kMembershipMaxNeedleLen = 64;

static bool contains_subsequence(
  const std::vector<BigInt> &haystack,
  const std::vector<BigInt> &needle)
{
  if (needle.empty())
    return true;
  if (needle.size() > haystack.size())
    return false;

  for (size_t start = 0; start + needle.size() <= haystack.size(); ++start)
  {
    bool match = true;
    for (size_t i = 0; i < needle.size(); ++i)
    {
      if (haystack[start + i] != needle[i])
      {
        match = false;
        break;
      }
    }

    if (match)
      return true;
  }

  return false;
}

static std::optional<BigInt> get_constant_array_extent(const exprt &array_expr)
{
  if (!array_expr.type().is_array())
    return std::nullopt;

  const auto &array_type = to_array_type(array_expr.type());
  if (!array_type.size().is_constant())
    return std::nullopt;

  BigInt sz;
  if (to_integer(array_type.size(), sz))
    return std::nullopt;
  return sz;
}

static exprt
make_binary_bool_expr(const irep_idt &id, const exprt &lhs, const exprt &rhs)
{
  // V.3: build the boolean binop in IREP2 (the only ids used are =/and/or),
  // back-migrated for the legacy callers.
  expr2tc l2, r2;
  migrate_expr(lhs, l2);
  migrate_expr(rhs, r2);
  if (id == "=")
    return migrate_expr_back(equality2tc(l2, r2));
  if (id == "and")
    return migrate_expr_back(and2tc(l2, r2));
  if (id == "or")
    return migrate_expr_back(or2tc(l2, r2));
  exprt out(id, bool_type()); // fallback for any other id
  out.copy_to_operands(lhs, rhs);
  return out;
}

static std::optional<exprt> build_symbolic_membership_from_array(
  const exprt &haystack_array_expr,
  const std::vector<BigInt> &needle_values)
{
  if (!haystack_array_expr.type().is_array())
    return std::nullopt;

  const std::optional<BigInt> extent_opt =
    get_constant_array_extent(haystack_array_expr);
  if (!extent_opt.has_value())
    return std::nullopt;

  const BigInt extent = *extent_opt;
  // V.3: membership constant results built in IREP2.
  if (extent <= 0)
    return migrate_expr_back(
      needle_values.empty() ? gen_true_expr() : gen_false_expr());

  const BigInt haystack_content_len = extent - 1;
  const BigInt needle_len = static_cast<unsigned long>(needle_values.size());
  if (needle_len == 0)
    return migrate_expr_back(gen_true_expr());
  if (needle_len > haystack_content_len)
    return migrate_expr_back(gen_false_expr());

  // Keep this bounded to avoid path explosion in symbolic membership.
  if (
    haystack_content_len > kMembershipMaxHaystackContentLen ||
    needle_len > kMembershipMaxNeedleLen)
    return std::nullopt;

  const long long max_start = (haystack_content_len - needle_len).to_int64();
  exprt disjunction = migrate_expr_back(gen_false_expr()); // V.3

  for (long long start = 0; start <= max_start; ++start)
  {
    // Prefix guard first; only build the full window match if prefix matches.
    exprt prefix_index("index", char_type());
    prefix_index.copy_to_operands(haystack_array_expr);
    prefix_index.copy_to_operands(from_integer(start, int_type()));
    exprt prefix_expected = from_integer(needle_values.front(), char_type());
    exprt conjunction =
      make_binary_bool_expr("=", prefix_index, prefix_expected);

    for (std::size_t i = 1; i < needle_values.size(); ++i)
    {
      exprt index_expr("index", char_type());
      index_expr.copy_to_operands(haystack_array_expr);
      index_expr.copy_to_operands(
        from_integer(static_cast<long long>(start + i), int_type()));

      exprt expected = from_integer(needle_values[i], char_type());
      exprt equal_expr = make_binary_bool_expr("=", index_expr, expected);
      conjunction = make_binary_bool_expr("and", conjunction, equal_expr);
    }
    disjunction = make_binary_bool_expr("or", disjunction, conjunction);
  }

  return disjunction;
}

static int count_name_assignments_in_node(
  const nlohmann::json &node,
  const std::string &var_name,
  bool &has_augassign)
{
  int count = 0;
  if (!node.is_object() && !node.is_array())
    return 0;

  if (node.is_object() && node.contains("_type"))
  {
    const std::string type = node["_type"].get<std::string>();
    if (type == "Assign" && node.contains("targets"))
    {
      for (const auto &tgt : node["targets"])
      {
        if (
          tgt.contains("_type") && tgt["_type"] == "Name" &&
          tgt.contains("id") && tgt["id"] == var_name)
          ++count;
      }
    }
    else if (type == "AnnAssign" && node.contains("target"))
    {
      const auto &tgt = node["target"];
      if (
        tgt.contains("_type") && tgt["_type"] == "Name" && tgt.contains("id") &&
        tgt["id"] == var_name)
      {
        ++count;
      }
    }
    else if (type == "AugAssign" && node.contains("target"))
    {
      const auto &tgt = node["target"];
      if (
        tgt.contains("_type") && tgt["_type"] == "Name" && tgt.contains("id") &&
        tgt["id"] == var_name)
      {
        has_augassign = true;
        ++count;
      }
    }

    // Keep counting scoped to the current body: nested function/class/lambda
    // assignments must not affect fast-path eligibility of outer variables.
    if (
      type == "FunctionDef" || type == "AsyncFunctionDef" ||
      type == "ClassDef" || type == "Lambda")
    {
      return count;
    }
  }

  if (node.is_array())
  {
    for (const auto &elem : node)
      count += count_name_assignments_in_node(elem, var_name, has_augassign);
  }
  else if (node.is_object())
  {
    for (const auto &item : node.items())
      count +=
        count_name_assignments_in_node(item.value(), var_name, has_augassign);
  }

  return count;
}

// Returns true if `var_name` is mutated in place anywhere within `node`: an
// in-place list method call (append/extend/insert/remove/pop/clear/sort/
// reverse) or a subscript store (`var_name[i] = ...`). str.join uses this to
// decide whether the static fold of a list variable's initializer is still
// valid: once the list is mutated, the declaration initialiser no longer
// reflects the runtime contents (e.g. `new_lst = []; new_lst.append(w)` folds
// to "", #5163), so the join must go through the runtime model instead.
// Conservative by design -- a false "mutated" only costs the (correct)
// runtime dispatch, never correctness.
static bool
list_var_is_mutated(const nlohmann::json &node, const std::string &var_name)
{
  if (!node.is_object() && !node.is_array())
    return false;

  if (node.is_object() && node.contains("_type"))
  {
    const std::string type = node["_type"].get<std::string>();

    // <var_name>.<mutator>(...): in-place list mutators
    if (type == "Call" && node.contains("func"))
    {
      const auto &func = node["func"];
      if (
        func.contains("_type") && func["_type"] == "Attribute" &&
        func.contains("attr") && func.contains("value") &&
        func["value"].contains("_type") && func["value"]["_type"] == "Name" &&
        func["value"].contains("id") && func["value"]["id"] == var_name)
      {
        static const std::array<const char *, 8> mutators = {
          {"append",
           "extend",
           "insert",
           "remove",
           "pop",
           "clear",
           "sort",
           "reverse"}};
        const std::string attr = func["attr"].get<std::string>();
        for (const char *m : mutators)
          if (attr == m)
            return true;
      }
    }

    // <var_name>[i] = ...: subscript store
    if (type == "Assign" && node.contains("targets"))
    {
      for (const auto &tgt : node["targets"])
        if (
          tgt.contains("_type") && tgt["_type"] == "Subscript" &&
          tgt.contains("value") && tgt["value"].contains("_type") &&
          tgt["value"]["_type"] == "Name" && tgt["value"].contains("id") &&
          tgt["value"]["id"] == var_name)
          return true;
    }
  }

  if (node.is_array())
  {
    for (const auto &elem : node)
      if (list_var_is_mutated(elem, var_name))
        return true;
  }
  else if (node.is_object())
  {
    for (const auto &item : node.items())
      if (list_var_is_mutated(item.value(), var_name))
        return true;
  }

  return false;
}

} // namespace

// Narrow AST-level constant propagation for a Name receiver:
// when `var_name` has exactly one Assign / AnnAssign to it inside `func_scope`
// AND that assignment's value is a string Constant, return the literal.
//
// Deliberately narrow: we only look at the function's own body, not enclosing
// scopes; we don't trace through aliases; we don't handle augmented
// assignments. The point is to unlock the common pattern
//
//     def f():
//         msg = "hello"
//         ...                       # no reassignment of msg
//         return msg.swapcase()     # receiver looks constant in source
//                                   # but the local symbol's value is nil
//                                   # at conversion time
//
// without taking on the soundness obligations of a real const-prop pass.
// If we find more than one assignment we bail (no flow-sensitive reasoning).
static bool try_const_string_from_single_assignment(
  const std::string &var_name,
  const nlohmann::json &func_scope,
  std::string &out)
{
  if (!func_scope.contains("body") || !func_scope["body"].is_array())
    return false;

  const nlohmann::json *first_const = nullptr;
  int assignment_count = 0;

  std::function<void(const nlohmann::json &)> walk =
    [&](const nlohmann::json &body) {
      for (const auto &stmt : body)
      {
        if (!stmt.contains("_type"))
          continue;

        const std::string &t = stmt["_type"].get<std::string>();

        // Don't descend into nested function/class scopes -- those are
        // separate symbol tables.
        if (t == "FunctionDef" || t == "ClassDef" || t == "Lambda")
          continue;

        // Match `var_name = <expr>` and `var_name: T = <expr>`.
        const nlohmann::json *assigned_value = nullptr;
        if (
          t == "Assign" && stmt.contains("targets") &&
          stmt["targets"].is_array() && !stmt["targets"].empty())
        {
          const auto &tgt = stmt["targets"][0];
          if (
            tgt.contains("_type") && tgt["_type"] == "Name" &&
            tgt.contains("id") && tgt["id"] == var_name &&
            stmt.contains("value"))
            assigned_value = &stmt["value"];
        }
        else if (
          t == "AnnAssign" && stmt.contains("target") &&
          stmt["target"].contains("_type") &&
          stmt["target"]["_type"] == "Name" && stmt["target"].contains("id") &&
          stmt["target"]["id"] == var_name && stmt.contains("value") &&
          !stmt["value"].is_null())
        {
          assigned_value = &stmt["value"];
        }
        // AugAssign (`var_name += ...`) is also a write -- count it so we
        // bail if it appears, but don't capture its value as the constant.
        else if (
          t == "AugAssign" && stmt.contains("target") &&
          stmt["target"].contains("_type") &&
          stmt["target"]["_type"] == "Name" && stmt["target"].contains("id") &&
          stmt["target"]["id"] == var_name)
        {
          ++assignment_count;
          continue;
        }

        if (assigned_value != nullptr)
        {
          ++assignment_count;
          if (first_const == nullptr)
            first_const = assigned_value;
          continue;
        }

        for (const char *key : {"body", "orelse", "finalbody"})
          if (stmt.contains(key) && stmt[key].is_array())
            walk(stmt[key]);

        if (stmt.contains("handlers") && stmt["handlers"].is_array())
          for (const auto &h : stmt["handlers"])
            if (h.contains("body") && h["body"].is_array())
              walk(h["body"]);
      }
    };

  walk(func_scope["body"]);

  if (assignment_count != 1 || first_const == nullptr)
    return false;
  if (!first_const->contains("_type") || (*first_const)["_type"] != "Constant")
    return false;
  if (!first_const->contains("value") || !(*first_const)["value"].is_string())
    return false;

  out = (*first_const)["value"].get<std::string>();
  return true;
}

bool string_handler::try_extract_const_string_expr(
  const exprt &expr,
  std::string &out)
{
  exprt tmp = expr;
  exprt str_expr = ensure_null_terminated_string(tmp);

  if (str_expr.is_symbol())
  {
    const auto &sym_expr = to_symbol_expr(str_expr);
    const symbolt *symbol =
      find_cached_symbol(sym_expr.get_identifier().as_string());
    if (
      symbol && !symbol->get_value().is_nil() &&
      symbol->get_value().type().is_array())
    {
      out = extract_string_from_array_operands(symbol->get_value());
      return true;
    }

    // AST-level fallback: narrow single-assignment constant lookup. Useful
    // for `s = "abc"; ... s.method()` patterns where the local symbol's
    // value isn't yet materialised at conversion time but the assignment
    // is statically determinable from the function's AST. See the helper
    // comment above for the soundness scope.
    const std::string &full_id = sym_expr.get_identifier().as_string();
    auto last_at = full_id.rfind('@');
    if (last_at != std::string::npos && last_at + 1 < full_id.size())
    {
      std::string bare_name = full_id.substr(last_at + 1);
      const std::string &current_scope = converter_.get_current_func_name();
      if (!current_scope.empty())
      {
        const nlohmann::json &ast = converter_.get_ast_json();
        if (ast.contains("body") && ast["body"].is_array())
        {
          auto path = json_utils::split_function_path(current_scope);
          if (!path.empty())
          {
            nlohmann::json func_scope =
              json_utils::find_function(ast["body"], path.back());
            if (
              !func_scope.empty() && try_const_string_from_single_assignment(
                                       bare_name, func_scope, out))
              return true;
          }
        }
      }
    }
    return false;
  }

  if (str_expr.type().is_array())
  {
    out = extract_string_from_array_operands(str_expr);
    return true;
  }

  return false;
}

exprt string_handler::build_nondet_string_fallback(const locationt &location)
{
  // Bare nondet `char *`. Used as a sound over-approximation when a
  // str.*() handler cannot extract a compile-time constant receiver.
  // Subsequent string ops over this value see arbitrary content, which
  // preserves soundness for safety properties (we cannot conclude a
  // specific functional result, but we cannot wrongly conclude SAFE).
  side_effect_expr_nondett nondet(gen_pointer_type(char_type()));
  nondet.location() = location;
  return nondet;
}

BigInt string_handler::get_string_size(const exprt &expr)
{
  if (!expr.type().is_array())
  {
    // For non-array types in f-strings, convert them first to get actual size
    if (expr.is_constant() && type_utils::is_integer_type(expr.type()))
    {
      // Convert the actual integer to string to get real size
      BigInt value =
        binary2integer(expr.value().as_string(), expr.type().is_signedbv());
      std::string str_repr = std::to_string(value.to_int64());
      return BigInt(str_repr.size() + 1); // +1 for null terminator
    }

    if (expr.is_symbol())
    {
      const symbolt *symbol = find_cached_symbol(expr.identifier().as_string());
      if (symbol && symbol->get_type().is_array())
      {
        const auto &arr_type = to_array_type(symbol->get_type());
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
string_handler::process_format_spec(const nlohmann::json &format_spec)
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

std::string string_handler::float_to_string(
  const std::string &float_bits,
  std::size_t width,
  int precision)
{
  double val = 0.0;

  if (width == 32 && float_bits.length() == 32)
  {
    // IEEE 754 single precision
    uint32_t bits = 0;
    for (std::size_t i = 0; i < width; ++i)
      if (float_bits[i] == '1')
        bits |= (1U << (width - 1 - i));

    float float_val;
    std::memcpy(&float_val, &bits, sizeof(float));
    val = static_cast<double>(float_val);
  }
  else if (width == 64 && float_bits.length() == 64)
  {
    // IEEE 754 double precision
    uint64_t bits = 0;
    for (std::size_t i = 0; i < width; ++i)
      if (float_bits[i] == '1')
        bits |= (1ULL << (width - 1 - i));

    std::memcpy(&val, &bits, sizeof(double));
  }
  else
  {
    throw std::runtime_error("Invalid float bit width");
  }

  // Use proper rounding to avoid IEEE 754 precision issues
  double multiplier = std::pow(10.0, precision);
  double rounded = std::round(val * multiplier) / multiplier;

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << rounded;
  return oss.str();
}

// Parse the leading [[fill]align][width] portion of a Python format spec.
// Returns true if any padding parameters were extracted; on success, *rest
// is set to the remainder of the spec (precision/type) for further handling.
static bool parse_format_padding(
  const std::string &format,
  char &fill,
  char &align,
  int &width,
  std::string &rest)
{
  fill = ' ';
  align = '\0';
  width = 0;
  rest = format;

  if (
    format.size() >= 2 && (format[1] == '<' || format[1] == '>' ||
                           format[1] == '^' || format[1] == '='))
  {
    fill = format[0];
    align = format[1];
    rest = format.substr(2);
  }
  else if (
    !format.empty() && (format[0] == '<' || format[0] == '>' ||
                        format[0] == '^' || format[0] == '='))
  {
    align = format[0];
    rest = format.substr(1);
  }

  size_t i = 0;
  while (i < rest.size() && std::isdigit(static_cast<unsigned char>(rest[i])))
    ++i;
  if (i > 0)
  {
    try
    {
      width = std::stoi(rest.substr(0, i));
    }
    catch (...)
    {
      width = 0;
    }
    rest = rest.substr(i);
  }

  return align != '\0' || width > 0;
}

// Apply fill/align/width padding to a literal string per Python's
// format-spec mini-language. Returns padded as a fresh char_array expr.
static std::string
apply_padding(const std::string &input, char fill, char align, int width)
{
  if (width <= 0 || static_cast<int>(input.size()) >= width)
    return input;

  size_t pad = static_cast<size_t>(width) - input.size();
  switch (align)
  {
  case '<':
    return input + std::string(pad, fill);
  case '^':
  {
    size_t left = pad / 2;
    size_t right = pad - left;
    return std::string(left, fill) + input + std::string(right, fill);
  }
  case '>':
  case '=':
  default:
    return std::string(pad, fill) + input;
  }
}

exprt string_handler::apply_format_specification(
  const exprt &expr,
  const std::string &format)
{
  // Basic format specification handling
  if (format.empty())
    return convert_to_string(expr);

  // Pad/align prefix ([[fill]align][width]). Default-align right for
  // numerics and left for strings, matching CPython.
  char fill, align;
  int width;
  std::string rest;
  if (parse_format_padding(format, fill, align, width, rest) && rest.empty())
  {
    exprt body = convert_to_string(expr);
    if (!body.type().is_array() || body.type().subtype() != char_type())
      return body;
    // Extract the literal characters; bail out if the body isn't a
    // string-constant or constant char array we can read.
    std::string content = extract_string_from_array_operands(body);
    if (
      content.empty() && !body.is_constant() && body.id() != "string-constant")
      return body;
    if (align == '\0')
      align = (expr.type().is_signedbv() || expr.type().is_unsignedbv() ||
               expr.type().is_floatbv() || expr.type().is_bool())
                ? '>'
                : '<';
    std::string padded = apply_padding(content, fill, align, width);
    typet string_type =
      type_handler_.build_array(char_type(), padded.size() + 1);
    std::vector<unsigned char> chars(padded.begin(), padded.end());
    chars.push_back('\0');
    return make_char_array_expr(chars, string_type);
  }

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
              find_cached_symbol(sym_expr.get_identifier().as_string());

            if (symbol && symbol->get_value().is_constant())
              float_bits = &symbol->get_value().value().as_string();
          }

          if (float_bits && float_bits->length() == float_width)
          {
            std::string formatted_str =
              float_to_string(*float_bits, float_width, precision);

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

exprt string_handler::make_char_array_expr(
  const std::vector<unsigned char> &chars,
  const typet &type)
{
  exprt arr = gen_zero(type);
  for (size_t i = 0; i < chars.size() && i < arr.operands().size(); ++i)
  {
    arr.operands()[i] = from_integer(chars[i], char_type());
  }
  return arr;
}

exprt string_handler::convert_to_string(const exprt &expr)
{
  const typet &t = expr.type();

  // Already a string/char array - return as is
  if (t.is_array() && t.subtype() == char_type())
    return expr;

  // Handle symbol references
  if (expr.is_symbol())
  {
    const symbolt *symbol = find_cached_symbol(expr.identifier().as_string());
    if (symbol)
    {
      // If symbol has string type, return it
      if (
        symbol->get_type().is_array() &&
        symbol->get_type().subtype() == char_type())
        return expr;

      // If symbol has a constant value, convert that
      if (symbol->get_value().is_constant())
        return convert_to_string(symbol->get_value());
    }
  }

  // Handle constants
  if (expr.is_constant())
  {
    if (type_utils::is_integer_type(t))
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
          str_value = float_to_string(float_bits, 64, 6);
          // Match Python's str(float): drop trailing fractional zeros, keep
          // at least one digit after the dot ("5.0" rather than "5.").
          auto dot = str_value.find('.');
          if (dot != std::string::npos)
          {
            size_t last_nonzero = str_value.find_last_not_of('0');
            if (last_nonzero == dot)
              str_value.resize(dot + 2);
            else
              str_value.resize(last_nonzero + 1);
          }
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

  // Handle pointer types from struct member access (e.g. self.name).
  // Skip code/side-effect expressions (function calls) — their declared
  // return type may be a zero-length array that migrates to empty_type2t,
  // causing a type mismatch when nested inside __python_str_concat.
  if (t.is_pointer() && !expr.is_code())
  {
    typet char_ptr = gen_pointer_type(char_type());
    if (t != char_ptr)
      return build_typecast(expr, char_ptr);
    return expr;
  }

  // Non-constant scalars: dispatch to the matching __python_*_to_str
  // operational model in src/c2goto/library/python/string.c. The model
  // returns a char* whose contents depend on the runtime value of `expr`,
  // so symex builds an accurate string for every reachable case.
  if (t.is_bool())
    return converter_.get_string_builder().build_runtime_str_conversion_call(
      "__python_bool_to_str", bool_type(), expr);

  if (type_utils::is_integer_type(t))
    return converter_.get_string_builder().build_runtime_str_conversion_call(
      "__python_int_to_str", long_long_int_type(), expr);

  if (t.is_floatbv())
    return converter_.get_string_builder().build_runtime_str_conversion_call(
      "__python_float_to_str", double_type(), expr);

  // Anything else (struct, array of non-char, etc.) is currently unsupported.
  std::string placeholder = "<expr>";
  typet string_type =
    type_handler_.build_array(char_type(), placeholder.size() + 1);
  std::vector<unsigned char> chars(placeholder.begin(), placeholder.end());
  chars.push_back('\0');

  return make_char_array_expr(chars, string_type);
}

exprt string_handler::get_fstring_expr(const nlohmann::json &element)
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
        // String literal part - delegate to converter
        part_expr = converter_.get_literal(value);
      }
      else if (value["_type"] == "FormattedValue")
      {
        // Expression to be formatted
        exprt expr = converter_.get_expr(value["value"]);

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
        exprt expr = converter_.get_expr(value);
        part_expr = convert_to_string(expr);
      }

      parts.push_back(part_expr);
      total_estimated_size += get_string_size(part_expr) -
                              1; // -1 to avoid double counting terminators
    }
    catch (const python_int_overflow_excp &)
    {
      // Bignum overflow is a soundness diagnostic, not a recoverable
      // parse hiccup — re-throw so the top-level error path surfaces it
      // instead of letting the f-string silently render as "<e>" and
      // the surrounding assertion succeed for the wrong reason. Issue
      // #4642.
      throw;
    }
    catch (const std::exception &e)
    {
      log_warning("Error processing f-string part: {}", e.what());
      // Create error placeholder
      std::string error_str = "<e>";
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

exprt string_handler::handle_string_concatenation(
  const exprt &lhs,
  const exprt &rhs,
  const nlohmann::json &left,
  const nlohmann::json &right)
{
  return string_builder_->concatenate_strings(lhs, rhs, left, right);
}

exprt string_handler::handle_string_repetition(exprt &lhs, exprt &rhs)
{
  return string_builder_->handle_string_repetition(lhs, rhs);
}

bool string_handler::is_zero_length_array(const exprt &expr)
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

std::string string_handler::extract_string_from_array_operands(
  const exprt &array_expr) const
{
  // constant_exprt char arrays with no operands store their string content
  // in the value attribute (e.g. type identifiers like `int`, `str`).
  // String literals built with build_string_literal() are also constant_exprt
  // but store chars as individual operands with an empty value attribute.
  if (
    array_expr.is_constant() && array_expr.operands().empty() &&
    array_expr.type().is_array() && array_expr.type().subtype() == char_type())
    return to_constant_expr(array_expr).get_value().as_string();

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

void string_handler::ensure_string_array(exprt &expr)
{
  if (expr.type().is_pointer())
    return;

  if (!expr.type().is_array())
  {
    // Explicitly build the array and
    // ensure null-termination (size 2: char + \0)
    typet t = type_handler_.build_array(expr.type(), 2);
    exprt arr = gen_zero(t);
    arr.operands().at(0) = expr;
    expr = arr;
  }
}

exprt string_handler::handle_string_operations(
  const std::string &op,
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &left,
  const nlohmann::json &right,
  const nlohmann::json &element)
{
  if (op == "Mult")
    return handle_string_repetition(lhs, rhs);

  ensure_string_array(lhs);
  ensure_string_array(rhs);

  assert(lhs.type().is_array() || lhs.type().is_pointer());
  assert(rhs.type().is_array() || rhs.type().is_pointer());

  if (op == "Eq" || op == "NotEq" || type_utils::is_ordered_comparison(op))
    return handle_string_comparison(op, lhs, rhs, element);
  else if (op == "Add")
    return handle_string_concatenation(lhs, rhs, left, right);

  return nil_exprt();
}

exprt string_handler::get_array_base_address(const exprt &arr)
{
  if (arr.type().is_pointer())
    return arr;
  // arr is non-pointer (array) here; index2t source ok, address_of of a
  // non-constant/non-address_of expr is a valid address_of2t source.
  exprt index = build_index(arr, from_integer(0, index_type()));
  return build_address_of(index);
}

exprt string_handler::handle_string_concatenation_with_promotion(
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &left,
  const nlohmann::json &right)
{
  if (lhs.type().is_array() && !rhs.type().is_array())
  {
    // LHS is array, RHS is single char - promote RHS to string array
    if (type_utils::is_integer_type(rhs.type()))
    {
      typet string_type = type_handler_.build_array(char_type(), 2);
      exprt str_array = gen_zero(string_type);
      str_array.operands().at(0) = rhs;
      str_array.operands().at(1) = gen_zero(char_type()); // null terminator
      rhs = str_array;
    }
  }
  else if (!lhs.type().is_array() && rhs.type().is_array())
  {
    // RHS is array, LHS is single char - promote LHS to string array
    if (type_utils::is_integer_type(lhs.type()))
    {
      // Extract index/dereference to avoid nested dereferences
      exprt lhs_value = lhs;
      if (lhs.is_index())
      {
        symbolt &temp = converter_.create_tmp_symbol(
          nlohmann::json(), "$char_temp$", lhs.type(), gen_zero(lhs.type()));
        code_assignt assign(build_symbol(temp), lhs);
        converter_.add_instruction(assign);
        lhs_value = build_symbol(temp);
      }

      typet string_type = type_handler_.build_array(char_type(), 2);
      exprt str_array = gen_zero(string_type);
      str_array.operands().at(0) = lhs_value;
      str_array.operands().at(1) = gen_zero(char_type()); // null terminator
      lhs = str_array;
    }
  }

  return handle_string_concatenation(lhs, rhs, left, right);
}

exprt string_handler::ensure_null_terminated_string(exprt &e)
{
  return string_builder_->ensure_null_terminated_string(e);
}

exprt string_handler::handle_string_comparison(
  const std::string &op,
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &element)
{
  return converter_.handle_string_comparison(op, lhs, rhs, element);
}

exprt string_handler::handle_string_membership(
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &element)
{
  bool lhs_is_char_value = false;

  // Get the width of char type from config
  std::size_t char_width = config.ansi_c.char_width;

  // Check if lhs is a symbol holding a character value
  if (lhs.is_symbol())
  {
    const symbolt *sym = find_cached_symbol(lhs.get_string("identifier"));
    if (sym)
    {
      const typet &value_type = sym->get_value().type();
      if (
        (value_type.is_signedbv() || value_type.is_unsignedbv()) &&
        bv_width(value_type) == char_width)
      {
        lhs_is_char_value = true;
      }
    }
  }

  // Use strchr for single character membership testing
  if (lhs_is_char_value)
  {
    symbolt *strchr_symbol = find_cached_c_function_symbol("c:@F@strchr");
    if (!strchr_symbol)
    {
      // Create strchr symbol if it doesn't exist
      symbolt new_symbol;
      new_symbol.name = "strchr";
      new_symbol.id = "c:@F@strchr";
      new_symbol.mode = "C";
      new_symbol.is_extern = true;

      code_typet strchr_type;
      typet char_ptr = gen_pointer_type(char_type());
      strchr_type.return_type() = char_ptr;
      strchr_type.arguments().push_back(code_typet::argumentt(char_ptr));
      strchr_type.arguments().push_back(code_typet::argumentt(int_type()));
      new_symbol.set_type(strchr_type);

      symbol_table_.add(new_symbol);
      strchr_symbol = find_cached_c_function_symbol("c:@F@strchr");
    }

    exprt rhs_str = ensure_null_terminated_string(rhs);
    exprt rhs_addr = get_array_base_address(rhs_str);

    // lhs contains the character value (as void*), cast directly to int
    exprt char_as_int = build_typecast(lhs, int_type());

    // Call strchr(string, character)
    exprt strchr_call = build_call_expr(
      *strchr_symbol, gen_pointer_type(char_type()), {rhs_addr, char_as_int});
    strchr_call.location() = converter_.get_location_from_decl(element);

    // Check if result != NULL (character found). Both operands are synthetic
    // char* values (the strchr() result and a null constant), so build the
    // comparison in IREP2 (V.3).
    constant_exprt null_ptr(gen_pointer_type(char_type()));
    null_ptr.set_value("NULL");

    // V.3: build `strchr(...) != NULL` in IREP2 via the shared build_notequal
    // helper (#5576). The migrate round-trip drops the call operand's location,
    // so re-attach it.
    exprt not_equal = build_notequal(strchr_call, null_ptr);
    not_equal.op0().location() = strchr_call.location();
    return not_equal;
  }

  // Use strstr for substring membership testing
  exprt lhs_str = ensure_null_terminated_string(lhs);
  exprt rhs_str = ensure_null_terminated_string(rhs);

  // Obtain the actual array expression (handle both constants and symbols)
  auto get_array_expr = [this](const exprt &e) -> const exprt * {
    if (e.is_constant() && e.type().is_array())
      return &e;
    if (e.is_symbol())
    {
      const symbolt *sym = find_cached_symbol(e.identifier().as_string());
      if (
        sym && sym->get_value().is_constant() &&
        sym->get_value().type().is_array())
        return &sym->get_value();
    }
    return nullptr;
  };

  const exprt *needle_array = get_array_expr(lhs_str);
  const exprt *haystack_array = get_array_expr(rhs_str);
  auto try_extract_constant_chars_from_ast =
    [this](const nlohmann::json *node) -> std::optional<std::vector<BigInt>> {
    if (node == nullptr)
      return std::nullopt;
    std::string text;
    if (!extract_constant_string(*node, converter_, text))
      return std::nullopt;

    std::vector<BigInt> values;
    values.reserve(text.size());
    for (unsigned char ch : text)
      values.emplace_back(ch);
    return values;
  };

  const nlohmann::json *lhs_node = nullptr;
  const nlohmann::json *rhs_node = nullptr;
  if (
    element.contains("_type") && element["_type"] == "Compare" &&
    element.contains("left") && element.contains("comparators") &&
    element["comparators"].is_array() && !element["comparators"].empty())
  {
    lhs_node = &element["left"];
    rhs_node = &element["comparators"][0];
  }

  const auto contains_embedded_null =
    [](const exprt *array_expr) -> std::optional<bool> {
    if (array_expr == nullptr || !array_expr->type().is_array())
      return std::nullopt;

    const exprt::operandst &ops = array_expr->operands();
    if (ops.empty())
      return false;

    for (size_t i = 0; i + 1 < ops.size(); ++i)
    {
      if (!ops[i].is_constant())
        return std::nullopt;

      BigInt val =
        binary2integer(ops[i].value().as_string(), ops[i].type().is_signedbv());
      if (val == 0)
        return true;
    }
    return false;
  };

  // Fully precise constant path (Python semantics, including embedded '\0').
  std::optional<std::vector<BigInt>> needle_values =
    extract_constant_char_values(needle_array);
  if (!needle_values.has_value())
    needle_values = try_extract_constant_chars_from_ast(lhs_node);

  std::optional<std::vector<BigInt>> haystack_values =
    extract_constant_char_values(haystack_array);
  if (!haystack_values.has_value())
    haystack_values = try_extract_constant_chars_from_ast(rhs_node);

  if (needle_values.has_value() && haystack_values.has_value())
  {
    // V.3: concrete-fold membership result built in IREP2.
    return migrate_expr_back(
      contains_subsequence(*haystack_values, *needle_values)
        ? gen_true_expr()
        : gen_false_expr());
  }

  // C strstr() is not null-aware for embedded '\0'. When one operand is
  // symbolic and the other is known to include embedded nulls, avoid an
  // unsound deterministic result.
  const std::optional<bool> needle_has_embedded_null =
    contains_embedded_null(needle_array);
  const std::optional<bool> haystack_has_embedded_null =
    contains_embedded_null(haystack_array);

  // If the needle is known and haystack is symbolic-but-bounded array,
  // try an explicit bounded membership formula before falling back to nondet.
  if (haystack_array == nullptr && needle_values.has_value())
  {
    if (
      std::optional<exprt> symbolic_membership =
        build_symbolic_membership_from_array(rhs_str, *needle_values))
    {
      symbolic_membership->location() =
        converter_.get_location_from_decl(element);
      return *symbolic_membership;
    }
  }

  if (
    (needle_array == nullptr && haystack_has_embedded_null == true) ||
    (haystack_array == nullptr && needle_has_embedded_null == true))
  {
    side_effect_expr_nondett nondet_contains(bool_type());
    nondet_contains.location() = converter_.get_location_from_decl(element);
    return nondet_contains;
  }

  // Get base addresses for C string functions
  exprt lhs_addr = get_array_base_address(lhs_str);
  exprt rhs_addr = get_array_base_address(rhs_str);

  // Find strstr symbol - returns pointer to first occurrence or NULL
  symbolt *strstr_symbol = find_cached_c_function_symbol("c:@F@strstr");
  if (!strstr_symbol)
    throw std::runtime_error("strstr function not found for 'in' operator");

  // Call strstr(haystack, needle) - in Python "needle in haystack"
  // haystack is rhs, needle is lhs
  exprt strstr_call = build_call_expr(
    *strstr_symbol, gen_pointer_type(char_type()), {rhs_addr, lhs_addr});
  strstr_call.location() = converter_.get_location_from_decl(element);

  // Check if result != NULL (substring found). Both operands are synthetic
  // char* values (the strstr() result and a null constant), so build the
  // comparison in IREP2 (V.3).
  constant_exprt null_ptr(gen_pointer_type(char_type()));
  null_ptr.set_value("NULL");

  // V.3: build `strstr(...) != NULL` in IREP2 via the shared build_notequal
  // helper (#5576, mirrors the strchr membership path above). The migrate
  // round-trip drops the call operand's location, so re-attach it.
  exprt not_equal = build_notequal(strstr_call, null_ptr);
  not_equal.op0().location() = strstr_call.location();
  return not_equal;
}

std::string
string_handler::ensure_string_function_symbol(const std::string &function_name)
{
  symbol_id func_id;
  func_id.set_prefix("c:");
  func_id.set_function(function_name);

  std::string func_symbol_id = func_id.to_string();

  // The operational-model library is linked before conversion runs, so a
  // registered model already has its body in the symbol table here. A missing
  // symbol therefore means the model is absent from the goto allowlist; fail
  // loudly rather than fabricate a body-less declaration that symex would
  // silently treat as an unconstrained nondet value. Models defined under
  // src/c2goto/library/python/ register automatically via
  // scripts/gen_python_c_models.py; external dependencies are listed in
  // python_c_extern_deps in src/c2goto/cprover_library.cpp.
  if (find_cached_symbol(func_symbol_id) == nullptr)
    throw std::runtime_error(
      "Python operational model '" + function_name +
      "' is dispatched but not registered (no body in the symbol table). "
      "Define it under src/c2goto/library/python/ or add it to "
      "python_c_extern_deps in src/c2goto/cprover_library.cpp.");

  return func_symbol_id;
}

symbolt *string_handler::find_cached_symbol(const std::string &symbol_id)
{
  auto cache_it = symbol_cache_.find(symbol_id);
  if (cache_it != symbol_cache_.end())
    return cache_it->second;

  symbolt *symbol = symbol_table_.find_symbol(symbol_id);
  if (symbol != nullptr)
    symbol_cache_.emplace(symbol_id, symbol);

  return symbol;
}

symbolt *
string_handler::find_cached_c_function_symbol(const std::string &symbol_id)
{
  return find_cached_symbol(symbol_id);
}

static bool fold_constant_string(
  const nlohmann::json &node,
  python_converter &converter,
  std::string &out,
  unsigned depth);

// Python str.replace(old, new, count): replace up to `count` non-overlapping
// occurrences of `old` with `new`, scanning left to right; a negative `count`
// replaces every occurrence. Returns false for an empty `old` (Python inserts
// `new` between every character), so the caller defers to the runtime model
// (which still under-approximates split — see #5096).
static bool python_str_replace(
  const std::string &subject,
  const std::string &old_sub,
  const std::string &new_sub,
  long long count,
  std::string &out)
{
  if (old_sub.empty())
    return false;

  out.clear();
  std::size_t pos = 0;
  for (long long done = 0; count < 0 || done < count; ++done)
  {
    std::size_t hit = subject.find(old_sub, pos);
    if (hit == std::string::npos)
      break;
    out.append(subject, pos, hit - pos);
    out += new_sub;
    pos = hit + old_sub.size();
  }
  out.append(subject, pos, std::string::npos);
  return true;
}

// Fold a constant string-valued method call: `sep.join([...])` over a literal
// list/tuple of constant strings, or `subject.replace(old, new[, count])`.
// Only positional, constant-foldable arguments are handled; anything else
// returns false so the caller falls back to the runtime string model (which
// still under-approximates split — see #5096).
static bool fold_string_method_call(
  const nlohmann::json &node,
  python_converter &converter,
  std::string &out,
  unsigned depth)
{
  const auto &func = node["func"];
  if (
    !func.contains("_type") || func["_type"] != "Attribute" ||
    !func.contains("attr") || !func.contains("value"))
    return false;

  // Keyword arguments are not folded.
  if (node.contains("keywords") && !node["keywords"].empty())
    return false;

  const std::string method = func["attr"].get<std::string>();
  const nlohmann::json &args = node["args"];

  if (method == "join")
  {
    std::string sep;
    if (
      args.size() != 1 ||
      !fold_constant_string(func["value"], converter, sep, depth + 1))
      return false;

    // Resolve the iterable, following a single Name binding, to a literal
    // list/tuple of constant strings.
    nlohmann::json seq = args[0];
    if (seq.contains("_type") && seq["_type"] == "Name" && seq.contains("id"))
    {
      nlohmann::json decl = json_utils::get_var_value(
        seq["id"].get<std::string>(),
        converter.get_current_func_name(),
        converter.get_ast_json());
      if (decl.empty() || !decl.contains("value"))
        return false;
      seq = decl["value"];
    }
    if (
      !seq.contains("_type") ||
      (seq["_type"] != "List" && seq["_type"] != "Tuple") ||
      !seq.contains("elts"))
      return false;

    out.clear();
    bool first = true;
    for (const auto &elt : seq["elts"])
    {
      std::string piece;
      if (!fold_constant_string(elt, converter, piece, depth + 1))
        return false;
      if (!first)
        out += sep;
      out += piece;
      first = false;
    }
    return true;
  }

  if (method == "replace")
  {
    std::string subject, old_sub, new_sub;
    if (
      args.size() < 2 || args.size() > 3 ||
      !fold_constant_string(func["value"], converter, subject, depth + 1) ||
      !fold_constant_string(args[0], converter, old_sub, depth + 1) ||
      !fold_constant_string(args[1], converter, new_sub, depth + 1))
      return false;

    long long count = -1;
    if (
      args.size() == 3 && !json_utils::extract_constant_integer(
                            args[2],
                            converter.get_current_func_name(),
                            converter.get_ast_json(),
                            count))
      return false;

    return python_str_replace(subject, old_sub, new_sub, count, out);
  }

  return false;
}

// Recursively fold an AST node into a constant string: a string literal, a Name
// bound to such a value, a Python "+" concatenation, a "*" repetition, or a
// constant-foldable join()/replace() method call. Any non-constant operand
// yields false, so callers fall back to the runtime string model. `depth`
// bounds the recursion against degenerate ASTs.
static bool fold_constant_string(
  const nlohmann::json &node,
  python_converter &converter,
  std::string &out,
  unsigned depth)
{
  if (depth > 64 || !node.contains("_type"))
    return false;

  const auto &type = node["_type"];

  // String literal.
  if (type == "Constant" && node.contains("value") && node["value"].is_string())
  {
    out = node["value"].get<std::string>();
    return true;
  }

  // Name reference: resolve its declaration and fold the bound value.
  // Reassigned names are not foldable:
  // get_var_value returns the first binding, so defer to the runtime string model.
  if (type == "Name" && node.contains("id"))
  {
    const std::string id = node["id"].get<std::string>();
    const std::string &func = converter.get_current_func_name();
    const nlohmann::json &ast = converter.get_ast_json();
    if (json_utils::has_multiple_assignments_in_scope(id, func, ast))
      return false;

    nlohmann::json decl = json_utils::get_var_value(id, func, ast);
    if (!decl.empty() && decl.contains("value"))
      return fold_constant_string(decl["value"], converter, out, depth + 1);
    return false;
  }

  // String concatenation: "a" + "b".
  if (
    type == "BinOp" && node.contains("op") && node["op"].contains("_type") &&
    node["op"]["_type"] == "Add" && node.contains("left") &&
    node.contains("right"))
  {
    std::string lhs, rhs;
    if (
      fold_constant_string(node["left"], converter, lhs, depth + 1) &&
      fold_constant_string(node["right"], converter, rhs, depth + 1))
    {
      out = lhs + rhs;
      return true;
    }
  }

  // String repetition: "ab" * n  or  n * "ab".
  if (
    type == "BinOp" && node.contains("op") && node["op"].contains("_type") &&
    node["op"]["_type"] == "Mult" && node.contains("left") &&
    node.contains("right"))
  {
    auto fold_repeat = [&](
                         const nlohmann::json &str_node,
                         const nlohmann::json &count_node) -> bool {
      std::string s;
      long long n = 0;
      if (
        !fold_constant_string(str_node, converter, s, depth + 1) ||
        !json_utils::extract_constant_integer(
          count_node,
          converter.get_current_func_name(),
          converter.get_ast_json(),
          n))
        return false;
      if (n <= 0)
      {
        out.clear();
        return true;
      }
      // Bound the materialized string so a large factor cannot exhaust memory;
      // beyond the cap, defer to the runtime model.
      constexpr unsigned long long max_len = 1ull << 16;
      if (
        static_cast<unsigned long long>(s.size()) *
          static_cast<unsigned long long>(n) >
        max_len)
        return false;
      out.clear();
      out.reserve(s.size() * static_cast<std::size_t>(n));
      for (long long i = 0; i < n; ++i)
        out += s;
      return true;
    };
    if (
      fold_repeat(node["left"], node["right"]) ||
      fold_repeat(node["right"], node["left"]))
      return true;
  }

  // Constant-foldable string method call: join()/replace().
  if (
    type == "Call" && node.contains("func") && node.contains("args") &&
    node["args"].is_array())
    return fold_string_method_call(node, converter, out, depth);

  return false;
}

bool string_handler::extract_constant_string(
  const nlohmann::json &node,
  python_converter &converter,
  std::string &out)
{
  return fold_constant_string(node, converter, out, 0);
}

exprt string_handler::handle_string_to_int(
  const exprt &string_obj,
  const exprt &base_arg,
  const locationt &location)
{
  // Ensure we have a null-terminated string
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);

  // Get base address of the string
  exprt str_addr = get_array_base_address(str_expr);

  // Determine the base value (default is 10)
  exprt base_expr = base_arg;
  if (base_expr.is_nil())
  {
    // Default base is 10
    base_expr = from_integer(10, int_type());
  }
  else if (!base_expr.type().is_signedbv() && !base_expr.type().is_unsignedbv())
  {
    // Cast base to int if needed
    base_expr = build_typecast(base_expr, int_type());
  }

  // Find the __python_int function symbol
  symbolt *int_symbol = find_cached_c_function_symbol("c:@F@__python_int");
  if (!int_symbol)
  {
    throw std::runtime_error("__python_int function not found in symbol table");
  }

  // Call __python_int(str, base). The result is a 64-bit Python int (matching
  // the model's `long long` return); a 32-bit result type here would make the
  // assigned-to variable 32-bit and truncate a string pointer rebound through
  // it, e.g. `a, b = s.split('-'); a = int(a)` (#5159).
  exprt int_call =
    build_call_expr(*int_symbol, long_long_int_type(), {str_addr, base_expr});
  int_call.location() = location;

  return int_call;
}

exprt string_handler::handle_string_to_int_base10(
  const exprt &string_obj,
  const locationt &location)
{
  // Convenience wrapper for base 10 conversion
  return handle_string_to_int(string_obj, nil_exprt(), location);
}

exprt string_handler::handle_int_conversion(
  const exprt &arg,
  const locationt &location)
{
  // Handle int() with different argument types

  // If argument is already an integer type, return as is
  if (type_utils::is_integer_type(arg.type()))
  {
    return arg;
  }

  // If argument is a float, truncate to integer
  if (arg.type().is_floatbv())
  {
    return build_typecast(arg, int_type());
  }

  // If argument is a boolean, convert to 0 or 1
  if (arg.type().is_bool())
  {
    exprt result("if", int_type());
    result.copy_to_operands(arg);
    result.copy_to_operands(from_integer(1, int_type()));
    result.copy_to_operands(from_integer(0, int_type()));
    return result;
  }

  // If argument is a string or char array, use string conversion
  if (arg.type().is_array() && arg.type().subtype() == char_type())
  {
    return handle_string_to_int_base10(arg, location);
  }

  // If argument is a pointer to char (string pointer)
  if (arg.type().is_pointer() && arg.type().subtype() == char_type())
  {
    // Create a wrapper to ensure null-termination handling
    exprt string_copy = arg;
    return handle_string_to_int(string_copy, nil_exprt(), location);
  }

  // For other types, attempt a typecast
  return build_typecast(arg, int_type());
}

exprt string_handler::handle_int_conversion_with_base(
  const exprt &arg,
  const exprt &base,
  const locationt &location)
{
  // int() with explicit base only works with strings
  if (!arg.type().is_array() && !arg.type().is_pointer())
  {
    throw std::runtime_error("int() with base argument requires string input");
  }

  // Ensure base is an integer
  exprt base_expr = base;
  if (!base_expr.type().is_signedbv() && !base_expr.type().is_unsignedbv())
  {
    base_expr = build_typecast(base_expr, int_type());
  }

  return handle_string_to_int(arg, base_expr, location);
}

exprt string_handler::handle_string_to_float(
  const exprt &string_obj,
  const locationt &location)
{
  // Ensure we have a null-terminated string and take its base address.
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt str_addr = get_array_base_address(str_expr);

  symbolt *float_symbol =
    find_cached_c_function_symbol("c:@F@__python_str_to_float");
  if (!float_symbol)
    throw std::runtime_error(
      "__python_str_to_float function not found in symbol table");

  // Call __python_str_to_float(str)
  exprt float_call = build_call_expr(*float_symbol, double_type(), {str_addr});
  float_call.location() = location;

  return float_call;
}

exprt string_handler::handle_string_is_float(
  const exprt &string_obj,
  const locationt &location)
{
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt str_addr = get_array_base_address(str_expr);

  symbolt *check_symbol =
    find_cached_c_function_symbol("c:@F@__python_str_is_float");
  if (!check_symbol)
    throw std::runtime_error(
      "__python_str_is_float function not found in symbol table");

  // Call __python_str_is_float(str)
  exprt check_call = build_call_expr(*check_symbol, bool_type(), {str_addr});
  check_call.location() = location;

  return check_call;
}

exprt string_handler::handle_chr_conversion(
  const exprt &codepoint_arg,
  const locationt &location)
{
  // Ensure the argument is an integer type
  exprt codepoint_expr = codepoint_arg;

  // If not already an integer, try to convert it
  if (!type_utils::is_integer_type(codepoint_expr.type()))
  {
    // If it's a float, truncate to integer
    if (codepoint_expr.type().is_floatbv())
      codepoint_expr = build_typecast(codepoint_expr, int_type());
    // If it's a boolean, convert to 0 or 1
    else if (codepoint_expr.type().is_bool())
    {
      exprt result("if", int_type());
      result.copy_to_operands(codepoint_expr);
      result.copy_to_operands(from_integer(1, int_type()));
      result.copy_to_operands(from_integer(0, int_type()));
      codepoint_expr = result;
    }
    else
      throw std::runtime_error("chr() argument must be an integer");
  }

  // Cast to int type if it's a different integer width
  if (codepoint_expr.type() != int_type())
    codepoint_expr = build_typecast(codepoint_expr, int_type());

  // Find the __python_chr function symbol
  symbolt *chr_symbol = find_cached_c_function_symbol("c:@F@__python_chr");
  if (!chr_symbol)
    throw std::runtime_error("__python_chr function not found in symbol table");

  // Call __python_chr(codepoint)
  exprt chr_call =
    build_call_expr(*chr_symbol, pointer_typet(char_type()), {codepoint_expr});
  chr_call.location() = location;

  return chr_call;
}

exprt string_handler::handle_ord_conversion(
  const exprt &string_obj,
  const locationt &location)
{
  // Take the base address of the (null-terminated) string.
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt str_addr = get_array_base_address(str_expr);

  // Code point of the single character: (int) *str_addr.
  // V.3: build the dereference in IREP2, back-migrating once. dereference2t
  // carries only (type, value) — no offset/guard — so the round-trip is
  // byte-identical to dereference_exprt(str_addr, char) (mirrors build_index/
  // build_dereference). Restore the exact element type migrate_type may drop.
  expr2tc str_addr2;
  migrate_expr(str_addr, str_addr2);
  exprt first_char =
    migrate_expr_back(dereference2tc(migrate_type(char_type()), str_addr2));
  first_char.type() = char_type();
  first_char.location() = location;
  return build_typecast(first_char, int_type());
}

exprt string_handler::try_handle_len_string_fast_path(
  const nlohmann::json &call_json,
  const exprt &arg_expr)
{
  if (
    !call_json.contains("args") || !call_json["args"].is_array() ||
    call_json["args"].empty())
  {
    return nil_exprt();
  }

  const nlohmann::json &arg_json = call_json["args"][0];
  const std::string current_scope_id = converter_.get_current_func_name();
  if (len_cache_scope_id_ != current_scope_id)
  {
    // Explicit invalidation when scope changes.
    assignment_count_cache_.clear();
    assignment_has_augassign_cache_.clear();
    var_value_cache_.clear();
    len_cache_scope_id_ = current_scope_id;
  }

  if (exprt fast = try_len_fast_path_from_constant_arg(arg_json);
      !fast.is_nil())
    return fast;

  if (exprt fast = try_len_fast_path_from_name_arg(arg_json); !fast.is_nil())
    return fast;

  if (arg_expr.is_symbol())
  {
    symbolt *arg_symbol = converter_.find_symbol(
      to_symbol_expr(arg_expr).get_identifier().as_string());
    if (
      arg_symbol && arg_symbol->get_value().is_not_nil() &&
      arg_symbol->get_value().type().is_array() &&
      arg_symbol->get_value().is_constant())
    {
      const array_typet &arr_type =
        to_array_type(arg_symbol->get_value().type());
      if (
        type_utils::is_char_type(arr_type.subtype()) &&
        arr_type.size().is_constant())
      {
        BigInt sz;
        if (!to_integer(arr_type.size(), sz) && sz > 0)
          return from_integer(sz - 1, size_type());
      }
    }
  }

  if (arg_expr.is_symbol())
  {
    const std::string sym_id =
      to_symbol_expr(arg_expr).get_identifier().as_string();
    const auto &len_map = converter_.input_str_to_len_sym_;
    auto it = len_map.find(sym_id);
    if (it != len_map.end())
    {
      const symbolt *len_sym = converter_.find_symbol(it->second);
      if (len_sym)
        return build_typecast(build_symbol(*len_sym), size_type());
    }
  }

  typet actual_type = arg_expr.type();
  if (actual_type.is_pointer())
    actual_type = actual_type.subtype();
  if (actual_type.id() == "symbol")
    actual_type = converter_.ns.follow(actual_type);

  if (actual_type.id() == "array")
  {
    const array_typet &arr_type = to_array_type(actual_type);
    if (
      type_utils::is_char_type(arr_type.subtype()) &&
      arr_type.size().is_constant())
    {
      BigInt sz;
      if (!to_integer(arr_type.size(), sz) && sz > 0)
        return from_integer(sz - 1, size_type());
    }
  }

  return nil_exprt();
}

exprt string_handler::try_len_fast_path_from_constant_arg(
  const nlohmann::json &arg_json)
{
  const std::string arg_type =
    (arg_json.contains("_type") && arg_json["_type"].is_string())
      ? arg_json["_type"].get<std::string>()
      : "";
  if (
    arg_type == "Constant" && arg_json.contains("value") &&
    arg_json["value"].is_string())
  {
    const std::string text = arg_json["value"].get<std::string>();
    return from_integer(BigInt(utf8_codepoint_count(text)), size_type());
  }
  return nil_exprt();
}

exprt string_handler::try_len_fast_path_from_name_arg(
  const nlohmann::json &arg_json)
{
  const std::string arg_type =
    (arg_json.contains("_type") && arg_json["_type"].is_string())
      ? arg_json["_type"].get<std::string>()
      : "";
  if (!(arg_type == "Name" && arg_json.contains("id")))
    return nil_exprt();

  const std::string var_name = arg_json["id"].get<std::string>();
  const std::string current_scope_id = converter_.get_current_func_name();
  const std::string cache_key = current_scope_id + "::" + var_name;
  int assign_count = 0;
  bool has_augassign = false;

  auto assign_count_it = assignment_count_cache_.find(cache_key);
  auto has_augassign_it = assignment_has_augassign_cache_.find(cache_key);
  if (
    assign_count_it != assignment_count_cache_.end() &&
    has_augassign_it != assignment_has_augassign_cache_.end())
  {
    assign_count = assign_count_it->second;
    has_augassign = has_augassign_it->second;
  }
  else
  {
    const nlohmann::json &ast = converter_.get_ast_json();
    if (current_scope_id.empty())
    {
      assign_count =
        count_name_assignments_in_node(ast["body"], var_name, has_augassign);
    }
    else
    {
      std::vector<std::string> function_path =
        json_utils::split_function_path(current_scope_id);
      nlohmann::json func_node =
        json_utils::find_function_by_path(ast, function_path);
      if (!func_node.empty() && func_node.contains("body"))
      {
        assign_count = count_name_assignments_in_node(
          func_node["body"], var_name, has_augassign);
      }
    }

    assignment_count_cache_[cache_key] = assign_count;
    assignment_has_augassign_cache_[cache_key] = has_augassign;
  }

  if (!(assign_count == 1 && !has_augassign))
  {
    var_value_cache_.erase(cache_key);
    return nil_exprt();
  }

  auto const_string_len_from_symbol =
    [this](const std::string &name) -> std::optional<BigInt> {
    if (name != "__name__")
      return std::nullopt;

    std::string name_value;
    if (converter_.python_file() == converter_.main_python_filename())
      name_value = "__main__";
    else
    {
      const std::string &file = converter_.python_file();
      size_t last_slash = file.find_last_of("/\\");
      size_t last_dot = file.find_last_of(".");
      if (
        last_slash != std::string::npos && last_dot != std::string::npos &&
        last_dot > last_slash)
      {
        name_value = file.substr(last_slash + 1, last_dot - last_slash - 1);
      }
      else if (last_dot != std::string::npos)
        name_value = file.substr(0, last_dot);
      else
        name_value = file;
    }
    return BigInt(utf8_codepoint_count(name_value));
  };

  auto joinedstr_len =
    [&const_string_len_from_symbol](
      const nlohmann::json &joined) -> std::optional<BigInt> {
    if (!joined.contains("values") || !joined["values"].is_array())
      return std::nullopt;

    BigInt total = BigInt(0);
    for (const auto &part : joined["values"])
    {
      if (
        part["_type"] == "Constant" && part.contains("value") &&
        part["value"].is_string())
      {
        const std::string text = part["value"].get<std::string>();
        total += BigInt(utf8_codepoint_count(text));
        continue;
      }
      if (part["_type"] == "FormattedValue" && part.contains("value"))
      {
        const auto &value = part["value"];
        if (value["_type"] == "Name" && value.contains("id"))
        {
          if (
            auto len =
              const_string_len_from_symbol(value["id"].get<std::string>()))
          {
            total += *len;
            continue;
          }
        }
      }
      return std::nullopt;
    }
    return total;
  };

  nlohmann::json var_value;
  auto var_value_it = var_value_cache_.find(cache_key);
  if (var_value_it != var_value_cache_.end())
  {
    var_value = var_value_it->second;
  }
  else
  {
    var_value = json_utils::get_var_value(
      var_name, current_scope_id, converter_.get_ast_json());
    var_value_cache_[cache_key] = var_value;
  }

  if (!var_value.empty() && var_value.contains("value"))
  {
    const auto &value = var_value["value"];
    if (value.contains("_type") && value["_type"] == "JoinedStr")
    {
      if (auto len = joinedstr_len(value))
        return from_integer(*len, size_type());
    }
    if (
      value.contains("_type") && value["_type"] == "Constant" &&
      value.contains("value") && value["value"].is_string())
    {
      const std::string text = value["value"].get<std::string>();
      return from_integer(BigInt(utf8_codepoint_count(text)), size_type());
    }
  }

  return nil_exprt();
}

exprt string_handler::handle_string_attribute_call(
  const nlohmann::json &call_json)
{
  if (
    !call_json.contains("func") || !call_json["func"].contains("_type") ||
    call_json["func"]["_type"] != "Attribute" ||
    !call_json["func"].contains("attr"))
  {
    return nil_exprt();
  }

  const auto &func_json = call_json["func"];
  const auto &receiver_json = func_json["value"];
  const std::string method_name = func_json["attr"].get<std::string>();
  const nlohmann::json empty_json_array = nlohmann::json::array();
  const nlohmann::json &args =
    (call_json.contains("args") && call_json["args"].is_array())
      ? call_json["args"]
      : empty_json_array;
  const nlohmann::json &keywords =
    (call_json.contains("keywords") && call_json["keywords"].is_array())
      ? call_json["keywords"]
      : empty_json_array;

  // Calls on an imported module (e.g. torch.split, numpy.split) are not string
  // methods even though the attribute name overlaps with one (split, count,
  // ...). Defer to the regular dispatch so the module's operational model runs.
  //
  // Use get_imported_module_path() (non-empty only for a name actually present
  // in imported_modules) rather than is_imported_module(), which also returns
  // true when a model .py file merely exists. The latter would misfire for a
  // local variable that shares a module's name (e.g. a parameter named
  // `string`), wrongly diverting `string.lower()` away from the str handler.
  if (
    receiver_json.contains("_type") && receiver_json["_type"] == "Name" &&
    receiver_json.contains("id") && receiver_json["id"].is_string() &&
    !converter_.get_imported_module_path(receiver_json["id"].get<std::string>())
       .empty())
    return nil_exprt();

  std::optional<exprt> cached_receiver_expr;
  auto get_receiver_expr = [&]() -> exprt {
    if (!cached_receiver_expr.has_value())
      cached_receiver_expr = converter_.get_expr(receiver_json);
    return *cached_receiver_expr;
  };

  // Defer count/index to tuple/list handlers when the receiver is one.
  if (method_name == "count" || method_name == "index")
  {
    exprt recv = get_receiver_expr();
    const typet list_type = converter_.get_type_handler().get_list_type();
    if (
      converter_.get_tuple_handler().is_tuple_type(recv.type()) ||
      recv.type() == list_type ||
      (recv.type().is_pointer() && recv.type().subtype() == list_type))
      return nil_exprt();
  }

  // bytes.hex(): fold a constant bytes object to its lowercase hex string
  // (CPython: b"\x01\xab".hex() == "01ab"). str has no .hex() method, so a
  // "hex" attribute is unambiguously a bytes method. Bytes are modelled as an
  // int array; a str is a char array, so the subtype check excludes strings.
  if (method_name == "hex" && args.empty())
  {
    exprt recv = get_receiver_expr();
    if (recv.is_symbol())
    {
      const symbolt *s = converter_.find_symbol(
        to_symbol_expr(recv).get_identifier().as_string());
      if (s && !s->get_value().is_nil())
        recv = s->get_value();
    }
    const typet &rt = recv.type();
    if (rt.is_array() && rt.subtype() != char_type())
    {
      static const char digits[] = "0123456789abcdef";
      std::string hex;
      hex.reserve(recv.operands().size() * 2);
      for (const exprt &op : recv.operands())
      {
        BigInt v;
        if (to_integer(op, v)) // true == not a constant integer
          throw std::runtime_error(
            "bytes.hex() is only supported on a constant bytes object");
        const unsigned byte = static_cast<unsigned>(v.to_int64() & 0xff);
        hex.push_back(digits[(byte >> 4) & 0xf]);
        hex.push_back(digits[byte & 0xf]);
      }
      return string_builder_->build_string_literal(hex);
    }
  }

  // bytes.index(sub) / bytes.rindex(sub): like find/rfind but raise ValueError
  // when sub is absent (CPython). Both are otherwise routed through the str
  // strncmp/strlen machinery, wrong for the int-array bytes representation. Fold
  // over *literal* operands: the receiver must be a bytes([...]) constructor and
  // the argument either a bytes([...]) subsequence or a single integer byte. The
  // match is purely syntactic (AST only), so no symbolic, branch-merged, or
  // partially-evaluated value can reach the fold; a str receiver, a variable
  // receiver, and any other form fall through to the existing dispatch.
  if ((method_name == "index" || method_name == "rindex") && args.size() == 1)
  {
    auto extract_bytes_literal =
      [](const nlohmann::json &n, std::vector<unsigned> &out) -> bool {
      if (!(n.is_object() && n.value("_type", std::string()) == "Call" &&
            n.contains("func") &&
            n["func"].value("_type", std::string()) == "Name" &&
            n["func"].value("id", std::string()) == "bytes" &&
            n.contains("args") && n["args"].is_array() &&
            n["args"].size() == 1 &&
            n["args"][0].value("_type", std::string()) == "List" &&
            n["args"][0].contains("elts") && n["args"][0]["elts"].is_array()))
        return false;
      out.clear();
      for (const auto &e : n["args"][0]["elts"])
      {
        if (!(e.is_object() && e.value("_type", std::string()) == "Constant" &&
              e.contains("value") && e["value"].is_number_unsigned()))
          return false;
        const unsigned long long v = e["value"].get<unsigned long long>();
        if (v > 255)
          return false;
        out.push_back(static_cast<unsigned>(v));
      }
      return true;
    };

    std::vector<unsigned> recv_bytes;
    if (extract_bytes_literal(receiver_json, recv_bytes))
    {
      // The argument is a bytes([...]) subsequence or a single integer byte.
      std::vector<unsigned> sub_bytes;
      bool have_sub = extract_bytes_literal(args[0], sub_bytes);
      if (!have_sub)
      {
        const nlohmann::json &a = args[0];
        if (
          a.is_object() && a.value("_type", std::string()) == "Constant" &&
          a.contains("value") && a["value"].is_number_unsigned() &&
          a["value"].get<unsigned long long>() <= 255)
        {
          sub_bytes.assign(
            1, static_cast<unsigned>(a["value"].get<unsigned>()));
          have_sub = true;
        }
      }

      if (have_sub)
      {
        const std::size_t n = recv_bytes.size();
        const std::size_t m = sub_bytes.size();
        long long result = -1;
        auto matches_at = [&](std::size_t pos) {
          for (std::size_t i = 0; i < m; ++i)
            if (recv_bytes[pos + i] != sub_bytes[i])
              return false;
          return true;
        };
        if (m <= n)
        {
          if (method_name == "index")
          {
            for (std::size_t pos = 0; pos + m <= n; ++pos)
              if (matches_at(pos))
              {
                result = static_cast<long long>(pos);
                break;
              }
          }
          else // rindex, scan from the end
          {
            for (std::size_t pos = n - m + 1; pos-- > 0;)
              if (matches_at(pos))
              {
                result = static_cast<long long>(pos);
                break;
              }
          }
        }
        if (result < 0)
          return converter_.get_exception_handler().gen_exception_raise(
            "ValueError", "subsection not found");
        return from_integer(result, long_long_int_type());
      }
    }
  }

  // str.translate(str.maketrans(x, y[, z])): fold a constant string through a
  // constant translation table. CPython maps each x[i] to y[i] and deletes the
  // characters in z. Only the two/three-string maketrans form over constant
  // operands is modelled; a dict table or a non-constant operand falls through
  // to the regular (unsupported) dispatch.
  if (method_name == "translate" && args.size() == 1)
  {
    const nlohmann::json &mk = args[0];
    if (
      mk.is_object() && mk.value("_type", std::string()) == "Call" &&
      mk.contains("func") &&
      mk["func"].value("_type", std::string()) == "Attribute" &&
      mk["func"].value("attr", std::string()) == "maketrans" &&
      mk["func"].contains("value") &&
      mk["func"]["value"].value("_type", std::string()) == "Name" &&
      mk["func"]["value"].value("id", std::string()) == "str" &&
      mk.contains("args") && mk["args"].is_array())
    {
      const nlohmann::json &mkargs = mk["args"];
      std::string from_chars, to_chars, del_chars, recv;
      auto all_ascii = [](const std::string &s) {
        for (unsigned char ch : s)
          if (ch >= 0x80)
            return false;
        return true;
      };
      if (
        (mkargs.size() == 2 || mkargs.size() == 3) &&
        extract_constant_string(mkargs[0], converter_, from_chars) &&
        extract_constant_string(mkargs[1], converter_, to_chars) &&
        (mkargs.size() == 2 ||
         extract_constant_string(mkargs[2], converter_, del_chars)) &&
        extract_constant_string(receiver_json, converter_, recv) &&
        // The fold is byte-wise; restrict it to ASCII so a multi-byte UTF-8
        // sequence cannot be remapped/deleted one byte at a time (which would
        // corrupt the string). Non-ASCII operands fall through to the regular
        // (unsupported) dispatch — sound, never a wrong verdict.
        all_ascii(from_chars) && all_ascii(to_chars) && all_ascii(del_chars) &&
        all_ascii(recv))
      {
        if (from_chars.size() != to_chars.size())
          throw std::runtime_error(
            "str.maketrans() arguments must have equal length");
        std::string result;
        result.reserve(recv.size());
        for (char c : recv)
        {
          if (del_chars.find(c) != std::string::npos)
            continue;
          const std::size_t pos = from_chars.find(c);
          result.push_back(pos == std::string::npos ? c : to_chars[pos]);
        }
        return string_builder_->build_string_literal(result);
      }
    }
  }

  std::optional<locationt> cached_location;
  auto get_location = [&]() -> locationt {
    if (!cached_location.has_value())
      cached_location = converter_.get_location_from_decl(call_json);
    return *cached_location;
  };

  auto has_keyword_unpacking = [&]() -> bool {
    for (const auto &kw : keywords)
    {
      if (kw.contains("arg") && kw["arg"].is_null())
        return true;
    }
    return false;
  };

  string_call_utils::keyword_valuest keyword_values =
    string_call_utils::collect_keyword_values(method_name, keywords, false);
  if (
    std::optional<exprt> dispatched =
      string_method_dispatch::dispatch_decode_join_method(
        *this,
        method_name,
        call_json,
        receiver_json,
        args,
        keyword_values,
        converter_))
  {
    if (has_keyword_unpacking())
    {
      throw std::runtime_error(
        method_name +
        "() does not support keyword argument unpacking (**kwargs)");
    }
    return *dispatched;
  }

  if (
    std::optional<exprt> dispatched =
      string_method_dispatch::dispatch_no_arg_string_methods(
        *this,
        method_name,
        args,
        keyword_values,
        get_receiver_expr,
        get_location))
  {
    if (has_keyword_unpacking())
    {
      throw std::runtime_error(
        method_name +
        "() does not support keyword argument unpacking (**kwargs)");
    }
    return *dispatched;
  }

  if (
    std::optional<exprt> dispatched =
      string_method_dispatch::dispatch_one_arg_string_methods(
        *this,
        method_name,
        args,
        keyword_values,
        get_receiver_expr,
        get_location,
        converter_))
  {
    if (has_keyword_unpacking())
    {
      throw std::runtime_error(
        method_name +
        "() does not support keyword argument unpacking (**kwargs)");
    }
    return *dispatched;
  }

  if (
    std::optional<exprt> dispatched =
      string_method_dispatch::dispatch_search_string_methods(
        *this,
        method_name,
        call_json,
        args,
        keyword_values,
        get_receiver_expr,
        get_location,
        converter_))
  {
    if (has_keyword_unpacking())
    {
      throw std::runtime_error(
        method_name +
        "() does not support keyword argument unpacking (**kwargs)");
    }
    return *dispatched;
  }

  if (
    std::optional<exprt> dispatched =
      string_method_dispatch::dispatch_spacing_and_padding_methods(
        *this,
        method_name,
        args,
        keyword_values,
        get_receiver_expr,
        get_location,
        converter_))
  {
    if (has_keyword_unpacking())
    {
      throw std::runtime_error(
        method_name +
        "() does not support keyword argument unpacking (**kwargs)");
    }
    return *dispatched;
  }

  if (
    std::optional<exprt> dispatched =
      string_method_dispatch::dispatch_replace_method(
        *this,
        method_name,
        args,
        keyword_values,
        get_receiver_expr,
        get_location,
        converter_))
  {
    if (has_keyword_unpacking())
    {
      throw std::runtime_error(
        method_name +
        "() does not support keyword argument unpacking (**kwargs)");
    }
    return *dispatched;
  }

  if (
    std::optional<exprt> dispatched =
      string_method_dispatch::dispatch_count_method(
        *this,
        method_name,
        args,
        keyword_values,
        get_receiver_expr,
        get_location,
        converter_))
  {
    return *dispatched;
  }

  if (
    std::optional<exprt> dispatched =
      string_method_dispatch::dispatch_splitlines_method(
        *this,
        method_name,
        call_json,
        args,
        keyword_values,
        get_receiver_expr,
        get_location))
  {
    return *dispatched;
  }

  if (
    std::optional<exprt> dispatched =
      string_method_dispatch::dispatch_format_methods(
        *this,
        method_name,
        call_json,
        keyword_values,
        get_receiver_expr,
        get_location))
  {
    return *dispatched;
  }

  if (
    std::optional<exprt> dispatched =
      string_method_dispatch::dispatch_split_method(
        method_name,
        receiver_json,
        call_json,
        args,
        keyword_values,
        get_receiver_expr,
        converter_))
  {
    return *dispatched;
  }

  return nil_exprt();
}

exprt string_handler::handle_str_join(const nlohmann::json &call_json)
{
  // Validate JSON structure: ensure we have the required keys
  if (!call_json.contains("args") || call_json["args"].empty())
    throw std::runtime_error("join() missing required argument: 'iterable'");
  if (call_json["args"].size() != 1)
    throw std::runtime_error("join() takes exactly one argument");

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
  exprt separator = converter_.get_expr(func["value"]);
  ensure_string_array(separator);

  // Get the list argument (the iterable to join)
  const nlohmann::json &list_arg = call_json["args"][0];

  // Resolve the list JSON node from either a Name reference or a direct List literal
  const nlohmann::json *list_node = nullptr;
  nlohmann::json var_decl;

  if (
    list_arg.contains("_type") && list_arg["_type"] == "Name" &&
    list_arg.contains("id"))
  {
    std::string var_name = list_arg["id"].get<std::string>();

    // Look up the variable in the AST to get its initialization value
    var_decl = json_utils::find_var_decl(
      var_name, converter_.get_current_func_name(), converter_.get_ast_json());

    if (var_decl.empty())
      throw std::runtime_error(
        "NameError: name '" + var_name + "' is not defined");

    if (!var_decl.contains("value"))
    {
      // Variable is declared but its initialiser is opaque (e.g. an
      // unannotated parameter, or a function call result we cannot
      // fold). Dispatch to the runtime __python_str_join model with the
      // resolved list variable; it returns "" for an empty list and joins
      // element-by-element otherwise.
      log_debug(
        "python-string",
        "join() variable '{}' has no foldable initialiser: runtime dispatch",
        var_name);
      exprt list_expr = converter_.get_expr(list_arg);
      return string_builder_->build_runtime_str_join_call(separator, list_expr);
    }

    // If the list is mutated in place after initialisation (e.g.
    // new_lst = []; new_lst.append(w)), its declaration initialiser no longer
    // reflects the runtime contents, so the static fold below would join the
    // stale value -- an appended-to empty list folds to "" (#5163). Dispatch
    // to the runtime model, which reads the actual list object. Scope the scan
    // to the variable's own function (module body when at top level), matching
    // how find_var_decl resolved it above.
    {
      const std::string &scope_func = converter_.get_current_func_name();
      const nlohmann::json &ast = converter_.get_ast_json();
      const nlohmann::json func_node =
        scope_func.empty() ? nlohmann::json()
                           : json_utils::find_function(ast["body"], scope_func);
      const nlohmann::json &scan_body =
        func_node.contains("body") ? func_node["body"] : ast["body"];
      if (list_var_is_mutated(scan_body, var_name))
      {
        exprt list_expr = converter_.get_expr(list_arg);
        return string_builder_->build_runtime_str_join_call(
          separator, list_expr);
      }
    }

    list_node = &var_decl["value"];
  }
  else if (list_arg.contains("_type") && list_arg["_type"] == "List")
  {
    list_node = &list_arg;
  }

  // Handle split() calls: resolve the result to a JSON List at compile time
  nlohmann::json resolved_split_list;
  {
    const nlohmann::json *call_to_resolve = nullptr;
    if (
      list_node && list_node->contains("_type") &&
      (*list_node)["_type"] == "Call")
      call_to_resolve = list_node;
    else if (
      !list_node && list_arg.contains("_type") && list_arg["_type"] == "Call")
      call_to_resolve = &list_arg;

    if (
      call_to_resolve && call_to_resolve->contains("func") &&
      (*call_to_resolve)["func"].contains("_type") &&
      (*call_to_resolve)["func"]["_type"] == "Attribute" &&
      (*call_to_resolve)["func"].contains("attr") &&
      (*call_to_resolve)["func"]["attr"] == "split" &&
      (*call_to_resolve)["func"].contains("value"))
    {
      std::string input;
      if (extract_constant_string(
            (*call_to_resolve)["func"]["value"], converter_, input))
      {
        std::string sep;
        if (
          call_to_resolve->contains("args") &&
          !(*call_to_resolve)["args"].empty())
          extract_constant_string(
            (*call_to_resolve)["args"][0], converter_, sep);

        std::vector<std::string> parts;
        if (sep.empty())
        {
          size_t i = 0;
          while (i < input.size())
          {
            while (i < input.size() &&
                   std::isspace(static_cast<unsigned char>(input[i])))
              i++;
            if (i >= input.size())
              break;
            size_t start = i;
            while (i < input.size() &&
                   !std::isspace(static_cast<unsigned char>(input[i])))
              i++;
            parts.push_back(input.substr(start, i - start));
          }
        }
        else
        {
          size_t start = 0;
          while (true)
          {
            size_t pos = input.find(sep, start);
            if (pos == std::string::npos)
            {
              parts.push_back(input.substr(start));
              break;
            }
            parts.push_back(input.substr(start, pos - start));
            start = pos + sep.size();
          }
        }

        resolved_split_list["_type"] = "List";
        resolved_split_list["elts"] = nlohmann::json::array();
        for (const auto &part : parts)
        {
          nlohmann::json elem;
          elem["_type"] = "Constant";
          elem["value"] = part;
          resolved_split_list["elts"].push_back(elem);
        }
        list_node = &resolved_split_list;
      }
    }
  }

  if (
    !list_node || !list_node->contains("_type") ||
    (*list_node)["_type"] != "List" || !list_node->contains("elts"))
  {
    // Iterable arg isn't a literal `List` and couldn't be folded (e.g.
    // a `sorted(...)`, list comprehension, or unannotated parameter).
    // Fall back to a sound nondet `char *` instead of aborting GOTO
    // conversion. Same pattern as PRs #4814 (splitlines), #4815
    // (partition/format), #4818..#4826 (str.* runtime models).
    log_debug(
      "python-string",
      "join() iterable arg not a foldable List literal: nondet fallback");
    return build_nondet_string_fallback(
      converter_.get_location_from_decl(call_json));
  }

  // Get the list elements from the AST
  const auto &elements = (*list_node)["elts"];

  // Edge case: empty list literal
  if (elements.empty())
  {
    // A Name variable whose declared initialiser is an empty list may still be
    // populated at runtime. The preprocessor lowers ''.join(<generator>) and
    // ''.join(<comprehension>) to `tmp = []` followed by appends, so folding
    // the empty initialiser here would wrongly yield "". Dispatch to the
    // runtime __python_str_join model, which reads the list's runtime contents
    // (and still returns "" for a genuinely empty list).
    if (list_arg.contains("_type") && list_arg["_type"] == "Name")
    {
      log_debug(
        "python-string",
        "join() iterable is a runtime-built list: runtime dispatch");
      exprt list_expr = converter_.get_expr(list_arg);
      return string_builder_->build_runtime_str_join_call(separator, list_expr);
    }

    typet empty_string_type = type_handler_.build_array(char_type(), 1);
    exprt empty_str = gen_zero(empty_string_type);
    empty_str.operands().at(0) = from_integer(0, char_type());
    return empty_str;
  }

  // Convert JSON elements to ESBMC expressions
  std::vector<exprt> elem_exprs;
  for (const auto &elem : elements)
  {
    exprt elem_expr = converter_.get_expr(elem);
    ensure_string_array(elem_expr);
    elem_exprs.push_back(elem_expr);
  }

  // Edge case: single element returns the element itself (no separator)
  if (elem_exprs.size() == 1)
    return elem_exprs[0];

  // Build the joined string by extracting characters from all elements
  // and separators, then constructing a single string.
  std::vector<exprt> all_chars;

  std::vector<exprt> first_chars =
    string_builder_->extract_string_chars(elem_exprs[0]);
  all_chars.insert(all_chars.end(), first_chars.begin(), first_chars.end());

  for (size_t i = 1; i < elem_exprs.size(); ++i)
  {
    std::vector<exprt> sep_chars =
      string_builder_->extract_string_chars(separator);
    all_chars.insert(all_chars.end(), sep_chars.begin(), sep_chars.end());

    std::vector<exprt> elem_chars =
      string_builder_->extract_string_chars(elem_exprs[i]);
    all_chars.insert(all_chars.end(), elem_chars.begin(), elem_chars.end());
  }

  return string_builder_->build_null_terminated_string(all_chars);
}

exprt string_handler::create_char_comparison_expr(
  const std::string &op,
  const exprt &lhs_char_value,
  const exprt &rhs_char_value,
  const exprt &lhs_source,
  const exprt &rhs_source) const
{
  // Create comparison expression with integer operands
  exprt comp_expr(converter_.get_op(op, bool_type()), bool_type());
  comp_expr.copy_to_operands(lhs_char_value, rhs_char_value);

  // Preserve location from original operands
  if (!lhs_source.location().is_nil())
    comp_expr.location() = lhs_source.location();
  else if (!rhs_source.location().is_nil())
    comp_expr.location() = rhs_source.location();

  return comp_expr;
}

exprt string_handler::handle_single_char_comparison(
  const std::string &op,
  exprt &lhs,
  exprt &rhs)
{
  // Dereference pointer to character if needed
  auto maybe_dereference = [](const exprt &expr) -> exprt {
    if (
      expr.type().is_pointer() && (expr.type().subtype().is_signedbv() ||
                                   expr.type().subtype().is_unsignedbv()))
    {
      exprt deref("dereference", expr.type().subtype());
      deref.copy_to_operands(expr);
      return deref;
    }
    return expr;
  };

  // Create comparison expression with location info
  auto create_comparison = [&](const exprt &left, const exprt &right) -> exprt {
    exprt comp_expr(converter_.get_op(op, bool_type()), bool_type());
    comp_expr.copy_to_operands(left, right);

    if (!lhs.location().is_nil())
      comp_expr.location() = lhs.location();
    else if (!rhs.location().is_nil())
      comp_expr.location() = rhs.location();

    return comp_expr;
  };

  exprt lhs_to_check = maybe_dereference(lhs);
  exprt rhs_to_check = maybe_dereference(rhs);

  // Try to get character values from the (potentially dereferenced) expressions
  exprt lhs_char_value =
    python_char_utils::get_char_value_as_int(lhs_to_check, false);
  exprt rhs_char_value =
    python_char_utils::get_char_value_as_int(rhs_to_check, false);

  // If both are valid character values, do the comparison
  if (!lhs_char_value.is_nil() && !rhs_char_value.is_nil())
    return create_char_comparison_expr(
      op, lhs_char_value, rhs_char_value, lhs, rhs);

  // Handle mixed cases: dereferenced pointer with valid character value
  if (lhs_to_check.id() == "dereference" && !rhs_char_value.is_nil())
  {
    exprt lhs_as_int = build_typecast(lhs_to_check, rhs_char_value.type());
    return create_comparison(lhs_as_int, rhs_char_value);
  }

  if (!lhs_char_value.is_nil() && rhs_to_check.id() == "dereference")
  {
    exprt rhs_as_int = build_typecast(rhs_to_check, lhs_char_value.type());
    return create_comparison(lhs_char_value, rhs_as_int);
  }

  return nil_exprt();
}