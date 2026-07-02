#include <python-frontend/string/char_utils.h>
#include <python-frontend/exception_utils.h>
#include <python-frontend/python_int_overflow.h>
#include <python-frontend/python_list.h>
#include <python-frontend/string/string_method_dispatch.h>
#include <python-frontend/string/string_handler.h>
#include <python-frontend/string/string_handler_utils.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_expr_builder.h>
#include <python-frontend/tuple_handler.h>
#include <python-frontend/string/string_builder.h>
#include <python-frontend/type_utils.h>
#include <irep2/irep2_utils.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/migrate.h>
#include <util/python_types.h>
#include <util/std_expr.h>
#include <util/std_code.h>
#include <util/string_constant.h>
#include <util/type.h>

#include <boost/algorithm/string/predicate.hpp>
#include <algorithm>
#include <array>
#include <cctype>
#include <cfenv>
#include <cstdio>
#include <cmath>
#include <climits>
#include <iomanip>
#include <limits>
#include <optional>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <util/message.h>

using namespace python_expr;

namespace
{
static bool get_constant_int(const exprt &expr, long long &out)
{
  if (expr.is_nil())
    return false;
  BigInt tmp;
  // to_integer() returns false on success (CBMC convention), so the guard must
  // bail when it returns true. The earlier `!to_integer(...)` was inverted: it
  // rejected every valid integer constant (and accepted non-constants with an
  // unset value), so width/fill string methods — center/ljust/rjust/zfill and
  // expandtabs's tabsize — silently fell back to a nondet/default result.
  if (to_integer(expr, tmp))
    return false;
  out = tmp.to_int64();
  return true;
}

// Extract the constant byte values of a bytes receiver (literal or a variable
// resolved to its literal). Bytes are modelled as a non-char int array, which
// distinguishes them from a str (char array). Returns false for a non-constant
// or non-bytes receiver.
static bool extract_constant_bytes(
  python_converter &converter,
  const nlohmann::json &node,
  std::vector<uint8_t> &out)
{
  exprt recv = converter.get_expr(node);
  if (recv.is_symbol())
  {
    const symbolt *s =
      converter.find_symbol(to_symbol_expr(recv).get_identifier().as_string());
    if (s && !s->get_value().is_nil())
      recv = s->get_value();
  }
  // An unresolved variable (e.g. a bytes parameter, or a value not stored as a
  // constant) stays a bare symbol with the array type but no operands; folding
  // it would silently yield an empty result. Reject so the caller falls through
  // to the runtime model. A genuine b"" literal is a constant array (not a
  // symbol), so empty-bytes folding is preserved.
  if (recv.is_symbol())
    return false;
  const typet &rt = recv.type();
  if (!rt.is_array() || rt.subtype() == char_type())
    return false;
  out.clear();
  for (const exprt &op : recv.operands())
  {
    BigInt v;
    if (to_integer(op, v)) // true == not a constant integer
      return false;
    out.push_back(static_cast<uint8_t>(v.to_int64() & 0xff));
  }
  return true;
}

static char to_lower_char(char c)
{
  return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

static char to_upper_char(char c)
{
  return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
}

static std::string
format_value_from_json(const nlohmann::json &arg, python_converter &converter)
{
  std::string value;
  if (arg.contains("_type") && arg["_type"] == "Constant")
  {
    if (arg.contains("_bigint"))
      throw python_int_overflow_excp(
        "Python int overflow: literal " + arg["_bigint"].get<std::string>() +
        " does not fit in 64-bit int. ESBMC approximates Python int as a "
        "fixed-width bitvector; arbitrary-precision int support is tracked in "
        "issue #4642.");
    if (arg["value"].is_null())
      return "None";
    if (arg["value"].is_string())
      return arg["value"].get<std::string>();
    if (arg["value"].is_boolean())
      return arg["value"].get<bool>() ? "True" : "False";
    if (arg["value"].is_number_integer())
      return std::to_string(arg["value"].get<long long>());
    if (arg["value"].is_number_float())
    {
      std::ostringstream oss;
      oss << arg["value"].get<double>();
      return oss.str();
    }
    throw std::runtime_error("format() unsupported constant type");
  }

  if (string_handler::extract_constant_string(arg, converter, value))
    return value;

  throw std::runtime_error("format() requires constant arguments");
}

// Pins the host floating-point rounding mode to round-to-nearest for its
// lifetime and restores the previous mode on destruction. Float specs are
// folded with host snprintf, whose output depends on the ambient rounding mode;
// a statically linked solver (e.g. CVC5 on the Linux build) can leave that mode
// set to FE_UPWARD, which turns "{:.2f}".format(3.14159) into "3.15" instead of
// CPython's "3.14". Pinning the mode keeps the fold deterministic and matches
// CPython's round-half-to-even.
struct round_to_nearest_guard
{
  round_to_nearest_guard() : saved_(std::fegetround())
  {
    std::fesetround(FE_TONEAREST);
  }
  ~round_to_nearest_guard()
  {
    std::fesetround(saved_);
  }
  int saved_;
};

// Format a constant value per a str.format/format() format spec
// ([[fill]align][sign][#][0][width][.precision][type]). Throws for any value or
// spec feature not modelled here, so the caller can fall back to a sound nondet
// string rather than mis-folding.
std::string apply_format_spec(
  const nlohmann::json &arg,
  const std::string &spec,
  python_converter &converter)
{
  // Only constant numeric/string literals are folded; anything else (a
  // variable, a computed value) throws and degrades to the nondet fallback.
  if (arg.value("_type", std::string()) != "Constant" || !arg.contains("value"))
    throw std::runtime_error("format spec on a non-constant value");
  const auto &val = arg["value"];

  enum
  {
    KIND_INT,
    KIND_FLOAT,
    KIND_STR
  } kind;
  long long ival = 0;
  double dval = 0;
  std::string sval;
  if (val.is_number_integer())
  {
    kind = KIND_INT;
    ival = val.get<long long>();
  }
  else if (val.is_boolean())
  {
    kind = KIND_INT;
    ival = val.get<bool>() ? 1 : 0;
  }
  else if (val.is_number_float())
  {
    kind = KIND_FLOAT;
    dval = val.get<double>();
  }
  else if (val.is_string())
  {
    kind = KIND_STR;
    sval = val.get<std::string>();
  }
  else
    throw std::runtime_error("format spec on an unsupported constant type");

  // Parse the spec.
  size_t p = 0;
  char fill = ' ';
  char align = '\0';
  auto is_align = [](char ch) {
    return ch == '<' || ch == '>' || ch == '^' || ch == '=';
  };
  if (spec.size() >= 2 && is_align(spec[1]))
  {
    fill = spec[0];
    align = spec[1];
    p = 2;
  }
  else if (!spec.empty() && is_align(spec[0]))
    align = spec[p++];
  char sign = '-';
  if (p < spec.size() && (spec[p] == '+' || spec[p] == '-' || spec[p] == ' '))
    sign = spec[p++];
  if (p < spec.size() && spec[p] == '#')
    throw std::runtime_error("unsupported '#' in format spec");
  if (p < spec.size() && spec[p] == '0')
  {
    // The '0' flag forces '0' fill even when an explicit alignment precedes it
    // ("{:<05d}" -> "70000"); align defaults to '=' only when none was given.
    fill = '0';
    if (align == '\0')
      align = '=';
    ++p;
  }
  int width = 0;
  while (p < spec.size() && std::isdigit(static_cast<unsigned char>(spec[p])))
    width = width * 10 + (spec[p++] - '0');
  if (p < spec.size() && (spec[p] == ',' || spec[p] == '_'))
    throw std::runtime_error("unsupported grouping in format spec");
  int prec = -1;
  if (p < spec.size() && spec[p] == '.')
  {
    ++p;
    prec = 0;
    while (p < spec.size() && std::isdigit(static_cast<unsigned char>(spec[p])))
      prec = prec * 10 + (spec[p++] - '0');
  }
  char type = (p < spec.size()) ? spec[p++] : '\0';
  if (p != spec.size())
    throw std::runtime_error("invalid format spec");
  (void)converter;

  // Build the value body (without field padding) and a default alignment.
  std::string body;
  char default_align;
  if (kind == KIND_STR)
  {
    if (type != '\0' && type != 's')
      throw std::runtime_error("unsupported type for str in format spec");
    body = sval;
    if (prec >= 0 && static_cast<int>(body.size()) > prec)
      body.resize(static_cast<size_t>(prec)); // truncate, like %.Ns
    default_align = '<';
  }
  else if (
    kind == KIND_INT && (type == '\0' || type == 'd' || type == 'x' ||
                         type == 'X' || type == 'o' || type == 'b'))
  {
    const bool neg = ival < 0;
    unsigned long long mag = neg ? 0ULL - static_cast<unsigned long long>(ival)
                                 : static_cast<unsigned long long>(ival);
    std::string digits;
    if (type == '\0' || type == 'd')
      digits = std::to_string(mag);
    else
    {
      const unsigned base = (type == 'o') ? 8 : (type == 'b') ? 2 : 16;
      const char *alpha =
        (type == 'X') ? "0123456789ABCDEF" : "0123456789abcdef";
      if (mag == 0)
        digits = "0";
      for (; mag != 0; mag /= base)
        digits.insert(digits.begin(), alpha[mag % base]);
    }
    const std::string sgn =
      neg ? "-" : (sign == '+' ? "+" : (sign == ' ' ? " " : ""));
    body = sgn + digits;
    default_align = '>';
  }
  else if (
    kind == KIND_FLOAT && (type == 'f' || type == 'F' || type == 'e' ||
                           type == 'E' || type == 'g' || type == 'G'))
  {
    // A typeless float spec ("{:8}", "{:.2}") uses CPython's general format,
    // which is not faithfully snprintf-expressible (e.g. 1.0 -> "1.0", not
    // "1"); it is left to the nondet fallback via the final throw below.
    const round_to_nearest_guard rounding_guard;
    const int pr = prec >= 0 ? prec : 6;
    const char t = type;
    int n = 0;
    if (t == 'f' || t == 'F')
      n = std::snprintf(nullptr, 0, "%.*f", pr, dval);
    else if (t == 'e' || t == 'E')
      n = std::snprintf(nullptr, 0, "%.*e", pr, dval);
    else
      n = std::snprintf(nullptr, 0, "%.*g", pr, dval);
    if (n < 0)
      throw std::runtime_error("format spec float error");
    std::string num(static_cast<size_t>(n), '\0');
    if (t == 'f' || t == 'F')
      std::snprintf(&num[0], static_cast<size_t>(n) + 1, "%.*f", pr, dval);
    else if (t == 'e' || t == 'E')
      std::snprintf(&num[0], static_cast<size_t>(n) + 1, "%.*e", pr, dval);
    else
      std::snprintf(&num[0], static_cast<size_t>(n) + 1, "%.*g", pr, dval);
    if (t == 'F' || t == 'E' || t == 'G')
      for (char &ch : num)
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    // Apply +/space sign to a non-negative result.
    if (!num.empty() && num[0] != '-')
    {
      if (sign == '+')
        num = "+" + num;
      else if (sign == ' ')
        num = " " + num;
    }
    body = num;
    default_align = '>';
  }
  else
    throw std::runtime_error("unsupported type/value in format spec");

  if (align == '\0')
    align = default_align;

  // Pad to width.
  if (static_cast<int>(body.size()) >= width)
    return body;
  const int padn = width - static_cast<int>(body.size());
  if (align == '<')
    return body + std::string(padn, fill);
  if (align == '>')
    return std::string(padn, fill) + body;
  if (align == '^')
  {
    const int l = padn / 2;
    return std::string(l, fill) + body + std::string(padn - l, fill);
  }
  // '=' : pad after a leading sign (numeric).
  size_t s =
    (!body.empty() && (body[0] == '-' || body[0] == '+' || body[0] == ' ')) ? 1
                                                                            : 0;
  return body.substr(0, s) + std::string(padn, fill) + body.substr(s);
}
} // namespace

namespace string_method_dispatch
{
using keyword_valuest = string_call_utils::keyword_valuest;
using string_call_utils::collect_keyword_values;
using string_call_utils::ensure_allowed_keywords;
using string_call_utils::find_keyword_value;
using string_call_utils::required_arg_node_or_throw;
using string_call_utils::required_constant_int_arg;
using string_call_utils::resolve_positional_or_keyword_arg;

std::optional<exprt> dispatch_replace_method(
  string_handler &self,
  const std::string &method_name,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  const std::function<exprt()> &get_receiver_expr,
  const std::function<locationt()> &get_location,
  python_converter &converter)
{
  if (method_name != "replace")
    return std::nullopt;

  ensure_allowed_keywords(method_name, keyword_values, {"old", "new", "count"});
  if (args.size() > 3)
  {
    throw std::runtime_error(
      "replace() requires two or three arguments in minimal support");
  }

  const nlohmann::json *old_node = required_arg_node_or_throw(
    method_name,
    args,
    keyword_values,
    "old",
    0,
    "replace() requires two or three arguments in minimal support");
  const nlohmann::json *new_node = required_arg_node_or_throw(
    method_name,
    args,
    keyword_values,
    "new",
    1,
    "replace() requires two or three arguments in minimal support");
  const nlohmann::json *count_node = resolve_positional_or_keyword_arg(
    method_name, args, keyword_values, "count", 2, false);

  exprt count_expr = from_integer(-1, int_type());
  if (count_node != nullptr)
  {
    const long long count_value = required_constant_int_arg(
      *count_node,
      "replace() only supports constant count in minimal support",
      converter);
    count_expr = from_integer(count_value, int_type());
  }

  return self.handle_string_replace(
    get_receiver_expr(),
    converter.get_expr(*old_node),
    converter.get_expr(*new_node),
    count_expr,
    get_location());
}

std::optional<exprt> dispatch_count_method(
  string_handler &self,
  const std::string &method_name,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  const std::function<exprt()> &get_receiver_expr,
  const std::function<locationt()> &get_location,
  python_converter &converter)
{
  if (method_name != "count")
    return std::nullopt;

  ensure_allowed_keywords(method_name, keyword_values, {"sub", "start", "end"});
  if (args.size() > 3)
    throw std::runtime_error("count() requires one to three arguments");

  const nlohmann::json *sub_node = required_arg_node_or_throw(
    method_name,
    args,
    keyword_values,
    "sub",
    0,
    "count() requires one to three arguments");
  const nlohmann::json *start_node = resolve_positional_or_keyword_arg(
    method_name, args, keyword_values, "start", 1, false);
  const nlohmann::json *end_node = resolve_positional_or_keyword_arg(
    method_name, args, keyword_values, "end", 2, false);

  exprt start_arg =
    start_node != nullptr ? converter.get_expr(*start_node) : nil_exprt();
  exprt end_arg =
    end_node != nullptr ? converter.get_expr(*end_node) : nil_exprt();

  return self.handle_string_count(
    get_receiver_expr(),
    converter.get_expr(*sub_node),
    start_arg,
    end_arg,
    get_location());
}

static bool is_falsey_constant(const nlohmann::json &node)
{
  if (
    node.contains("_type") && node["_type"] == "Constant" &&
    node.contains("value"))
  {
    const auto &value = node["value"];
    if (value.is_boolean())
      return !value.get<bool>();
    if (value.is_number_integer())
      return value.get<long long>() == 0;
  }
  return false;
}

std::optional<exprt> dispatch_splitlines_method(
  string_handler &self,
  const std::string &method_name,
  const nlohmann::json &call_json,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  const std::function<exprt()> &get_receiver_expr,
  const std::function<locationt()> &get_location)
{
  if (method_name != "splitlines")
    return std::nullopt;

  ensure_allowed_keywords(method_name, keyword_values, {"keepends"});
  if (args.size() > 1)
    throw std::runtime_error("splitlines() takes zero or one argument");

  const nlohmann::json *keepends_node = resolve_positional_or_keyword_arg(
    method_name, args, keyword_values, "keepends", 0, false);
  if (keepends_node != nullptr && !is_falsey_constant(*keepends_node))
  {
    throw std::runtime_error(
      "splitlines() with keepends=True is not supported");
  }

  return self.handle_string_splitlines(
    call_json, get_receiver_expr(), get_location());
}

struct split_method_argst
{
  std::string separator;
  long long maxsplit = -1;
};

static bool is_none_literal_json(const nlohmann::json &node);

std::optional<exprt> dispatch_split_method(
  const std::string &method_name,
  const nlohmann::json &receiver_json,
  const nlohmann::json &call_json,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  const std::function<exprt()> &get_receiver_expr,
  python_converter &converter)
{
  if (method_name != "split" && method_name != "rsplit")
    return std::nullopt;

  // rsplit() differs from split() only in the direction maxsplit counts from
  // (the rightmost separators rather than the leftmost).
  const bool from_right = (method_name == "rsplit");

  ensure_allowed_keywords(method_name, keyword_values, {"sep", "maxsplit"});
  if (args.size() > 2)
  {
    throw std::runtime_error(
      method_name +
      "() requires zero, one, or two arguments in minimal support");
  }

  split_method_argst parsed;
  const nlohmann::json *sep_node = resolve_positional_or_keyword_arg(
    method_name, args, keyword_values, "sep", 0, false);
  const nlohmann::json *maxsplit_node = resolve_positional_or_keyword_arg(
    method_name, args, keyword_values, "maxsplit", 1, false);

  if (sep_node == nullptr || is_none_literal_json(*sep_node))
  {
    parsed.separator = "";
  }
  else if (!string_handler::extract_constant_string(
             *sep_node, converter, parsed.separator))
  {
    throw std::runtime_error(
      method_name + "() only supports constant sep in minimal support");
  }

  if (maxsplit_node != nullptr)
  {
    parsed.maxsplit = required_constant_int_arg(
      *maxsplit_node,
      method_name + "() only supports constant maxsplit in minimal support",
      converter);
  }

  // The BinOp fast path splits at the FIRST separator (split semantics); it is
  // unsound for rsplit, which would split at the last, so skip it there.
  if (
    !from_right && !parsed.separator.empty() && parsed.maxsplit == 1 &&
    receiver_json.contains("_type") && receiver_json["_type"] == "BinOp" &&
    receiver_json.contains("op") && receiver_json["op"].contains("_type") &&
    receiver_json["op"]["_type"] == "Add")
  {
    const auto &binop = receiver_json;
    std::string right_operand_str;
    if (
      string_handler::extract_constant_string(
        binop["right"], converter, right_operand_str) &&
      right_operand_str.rfind(parsed.separator, 0) == 0)
    {
      bool safe_boundary = true;
      std::string left_const;
      if (string_handler::extract_constant_string(
            binop["left"], converter, left_const))
        safe_boundary = left_const.find(parsed.separator) == std::string::npos;

      if (safe_boundary)
      {
        std::string right_suffix =
          right_operand_str.substr(parsed.separator.size());
        nlohmann::json list_node;
        list_node["_type"] = "List";
        list_node["elts"] = nlohmann::json::array();
        converter.copy_location_fields_from_decl(call_json, list_node);

        nlohmann::json left_node = binop["left"];
        converter.copy_location_fields_from_decl(call_json, left_node);
        nlohmann::json right_node;
        right_node["_type"] = "Constant";
        right_node["value"] = right_suffix;
        converter.copy_location_fields_from_decl(call_json, right_node);

        list_node["elts"].push_back(left_node);
        list_node["elts"].push_back(right_node);

        python_list list(converter, list_node);
        return list.get();
      }
    }
  }

  std::string input;
  if (!string_handler::extract_constant_string(receiver_json, converter, input))
  {
    exprt obj_expr = get_receiver_expr();
    return python_list::build_split_list(
      converter,
      call_json,
      obj_expr,
      parsed.separator,
      parsed.maxsplit,
      from_right);
  }

  return python_list::build_split_list(
    converter, call_json, input, parsed.separator, parsed.maxsplit, from_right);
}

std::optional<exprt> dispatch_no_arg_string_methods(
  string_handler &self,
  const std::string &method_name,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  const std::function<exprt()> &get_receiver_expr,
  const std::function<locationt()> &get_location)
{
  using no_arg_handler_t =
    exprt (string_handler::*)(const exprt &, const locationt &);
  static constexpr std::array<std::pair<const char *, no_arg_handler_t>, 13>
    no_arg_handlers = {{
      {"capitalize", &string_handler::handle_string_capitalize},
      {"title", &string_handler::handle_string_title},
      {"swapcase", &string_handler::handle_string_swapcase},
      {"casefold", &string_handler::handle_string_casefold},
      {"isdigit", &string_handler::handle_string_isdigit},
      {"isalnum", &string_handler::handle_string_isalnum},
      {"isupper", &string_handler::handle_string_isupper},
      {"isnumeric", &string_handler::handle_string_isnumeric},
      {"isidentifier", &string_handler::handle_string_isidentifier},
      {"islower", &string_handler::handle_string_islower},
      {"lower", &string_handler::handle_string_lower},
      {"upper", &string_handler::handle_string_upper},
      {"isalpha", &string_handler::handle_string_isalpha},
    }};

  for (const auto &[name, handler] : no_arg_handlers)
  {
    if (method_name != name)
      continue;

    ensure_allowed_keywords(method_name, keyword_values, {});
    if (!args.empty())
      throw std::runtime_error(method_name + "() takes no arguments");
    return (self.*handler)(get_receiver_expr(), get_location());
  }

  return std::nullopt;
}

std::optional<exprt> dispatch_one_arg_string_methods(
  string_handler &self,
  const std::string &method_name,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  const std::function<exprt()> &get_receiver_expr,
  const std::function<locationt()> &get_location,
  python_converter &converter)
{
  using one_arg_handler_t =
    exprt (string_handler::*)(const exprt &, const exprt &, const locationt &);
  struct one_arg_handler_entryt
  {
    const char *name;
    const char *arg_name;
    one_arg_handler_t handler;
  };
  static constexpr std::array<one_arg_handler_entryt, 6> one_arg_handlers = {{
    {"startswith", "prefix", &string_handler::handle_string_startswith},
    {"endswith", "suffix", &string_handler::handle_string_endswith},
    {"removeprefix", "prefix", &string_handler::handle_string_removeprefix},
    {"removesuffix", "suffix", &string_handler::handle_string_removesuffix},
    {"partition", "sep", &string_handler::handle_string_partition},
    {"rpartition", "sep", &string_handler::handle_string_rpartition},
  }};

  // startswith()/endswith() accept optional start/end position arguments:
  // s.startswith(prefix, start[, end]). Handle the 2/3-arg forms before the
  // single-argument table below.
  if (
    (method_name == "startswith" || method_name == "endswith") &&
    args.size() >= 2)
  {
    if (args.size() > 3)
      throw std::runtime_error(method_name + "() takes at most 3 arguments");
    return self.handle_startswith_endswith_with_pos(
      get_receiver_expr(),
      args,
      converter,
      get_location(),
      method_name == "endswith");
  }

  for (const auto &[name, arg_name, handler] : one_arg_handlers)
  {
    if (method_name != name)
      continue;

    ensure_allowed_keywords(method_name, keyword_values, {arg_name});
    if (args.size() > 1)
      throw std::runtime_error(method_name + "() requires one argument");

    const nlohmann::json *arg_node = required_arg_node_or_throw(
      method_name,
      args,
      keyword_values,
      arg_name,
      0,
      method_name + "() requires one argument");
    return (self.*handler)(
      get_receiver_expr(), converter.get_expr(*arg_node), get_location());
  }

  return std::nullopt;
}

struct search_args_parsedt
{
  exprt needle;
  exprt start;
  exprt end;
  bool has_range;
};

static search_args_parsedt parse_string_search_args(
  const std::string &method_name,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  python_converter &converter)
{
  ensure_allowed_keywords(method_name, keyword_values, {"sub", "start", "end"});
  if (args.size() > 3)
    throw std::runtime_error(
      method_name + "() requires one to three arguments");

  const nlohmann::json *sub_node = required_arg_node_or_throw(
    method_name,
    args,
    keyword_values,
    "sub",
    0,
    method_name + "() requires one to three arguments");
  const nlohmann::json *start_node = resolve_positional_or_keyword_arg(
    method_name, args, keyword_values, "start", 1, false);
  const nlohmann::json *end_node = resolve_positional_or_keyword_arg(
    method_name, args, keyword_values, "end", 2, false);

  search_args_parsedt parsed{
    converter.get_expr(*sub_node),
    from_integer(0, int_type()),
    from_integer(INT_MIN, int_type()),
    false};

  const bool has_start = start_node != nullptr;
  const bool has_end = end_node != nullptr;
  parsed.has_range = has_start || has_end;

  if (has_start)
    parsed.start = converter.get_expr(*start_node);
  if (has_end)
    parsed.end = converter.get_expr(*end_node);

  return parsed;
}

std::optional<exprt> dispatch_search_string_methods(
  string_handler &self,
  const std::string &method_name,
  const nlohmann::json &call_json,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  const std::function<exprt()> &get_receiver_expr,
  const std::function<locationt()> &get_location,
  python_converter &converter)
{
  if (
    method_name != "find" && method_name != "index" &&
    method_name != "rfind" && method_name != "rindex")
    return std::nullopt;

  exprt obj_expr = get_receiver_expr();
  search_args_parsedt parsed =
    parse_string_search_args(method_name, args, keyword_values, converter);

  if (method_name == "find")
  {
    if (!parsed.has_range)
      return self.handle_string_find(obj_expr, parsed.needle, get_location());
    return self.handle_string_find_range(
      obj_expr, parsed.needle, parsed.start, parsed.end, get_location());
  }

  if (method_name == "index")
  {
    if (!parsed.has_range)
      return self.handle_string_index(
        call_json, obj_expr, parsed.needle, get_location());
    return self.handle_string_index_range(
      call_json,
      obj_expr,
      parsed.needle,
      parsed.start,
      parsed.end,
      get_location());
  }

  if (method_name == "rindex")
  {
    if (!parsed.has_range)
      return self.handle_string_rindex(
        call_json, obj_expr, parsed.needle, get_location());
    return self.handle_string_rindex_range(
      call_json,
      obj_expr,
      parsed.needle,
      parsed.start,
      parsed.end,
      get_location());
  }

  if (!parsed.has_range)
    return self.handle_string_rfind(obj_expr, parsed.needle, get_location());
  return self.handle_string_rfind_range(
    obj_expr, parsed.needle, parsed.start, parsed.end, get_location());
}

static bool is_none_literal_json(const nlohmann::json &node)
{
  if (
    node.contains("_type") && node["_type"] == "Constant" &&
    node.contains("value") && node["value"].is_null())
  {
    return true;
  }
  return (
    node.contains("_type") && node["_type"] == "Name" && node.contains("id") &&
    node["id"].is_string() && node["id"] == "None");
}

static bool is_utf8_literal_json(const nlohmann::json &node)
{
  if (!(node.contains("_type") && node["_type"] == "Constant" &&
        node.contains("value") && node["value"].is_string()))
  {
    return false;
  }

  const std::string encoding = node["value"].get<std::string>();
  return boost::iequals(encoding, "utf-8") || boost::iequals(encoding, "utf8");
}

std::optional<exprt> dispatch_decode_join_method(
  string_handler &self,
  const std::string &method_name,
  const nlohmann::json &call_json,
  const nlohmann::json &receiver_json,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  python_converter &converter)
{
  if (method_name == "decode")
  {
    ensure_allowed_keywords(method_name, keyword_values, {"encoding"});

    if (args.size() > 1)
      throw std::runtime_error("decode() takes at most one argument");

    bool decode_utf8 = args.empty();
    if (args.size() == 1)
      decode_utf8 = is_utf8_literal_json(args[0]);

    if (
      const nlohmann::json *encoding_kw =
        find_keyword_value(keyword_values, "encoding"))
      decode_utf8 = is_utf8_literal_json(*encoding_kw);

    if (
      decode_utf8 && receiver_json.contains("_type") &&
      receiver_json["_type"] == "Call" && receiver_json.contains("func") &&
      receiver_json["func"].contains("_type") &&
      receiver_json["func"]["_type"] == "Attribute" &&
      receiver_json["func"].contains("attr") &&
      receiver_json["func"]["attr"] == "encode")
    {
      bool encode_utf8 =
        !receiver_json.contains("args") || receiver_json["args"].empty();
      if (
        receiver_json.contains("args") && receiver_json["args"].is_array() &&
        receiver_json["args"].size() == 1)
      {
        encode_utf8 = is_utf8_literal_json(receiver_json["args"][0]);
      }

      if (
        receiver_json.contains("keywords") &&
        receiver_json["keywords"].is_array())
      {
        for (const auto &kw : receiver_json["keywords"])
        {
          if (
            kw.contains("arg") && kw["arg"].is_string() &&
            kw["arg"] == "encoding" && kw.contains("value"))
          {
            encode_utf8 = is_utf8_literal_json(kw["value"]);
            break;
          }
        }
      }

      if (
        encode_utf8 && receiver_json["func"].contains("value") &&
        !receiver_json["func"]["value"].is_null())
      {
        return converter.get_expr(receiver_json["func"]["value"]);
      }
    }

    // Standalone b"...".decode(): a constant bytes object decodes to the str
    // of its byte values as characters. ASCII only — UTF-8 multi-byte decoding
    // is deferred, so non-ASCII bytes fall through to the clean error.
    std::vector<uint8_t> bytes;
    if (decode_utf8 && extract_constant_bytes(converter, receiver_json, bytes))
    {
      if (std::all_of(
            bytes.begin(), bytes.end(), [](uint8_t b) { return b < 0x80; }))
        return converter.get_string_builder().build_string_literal(
          std::string(bytes.begin(), bytes.end()));
    }
    return nil_exprt();
  }

  if (method_name == "encode")
  {
    ensure_allowed_keywords(method_name, keyword_values, {"encoding"});
    if (args.size() > 1)
      throw std::runtime_error("encode() takes at most one argument");

    bool encode_utf8 = args.empty();
    if (args.size() == 1)
      encode_utf8 = is_utf8_literal_json(args[0]);
    if (
      const nlohmann::json *encoding_kw =
        find_keyword_value(keyword_values, "encoding"))
      encode_utf8 = is_utf8_literal_json(*encoding_kw);

    // Standalone "...".encode(): a constant str encodes to its bytes. The
    // UTF-8 encoding of an ASCII string is the identical byte sequence; a
    // non-ASCII character needs multi-byte UTF-8 (deferred), so it falls
    // through to the clean error.
    std::string s;
    if (
      encode_utf8 &&
      string_handler::extract_constant_string(receiver_json, converter, s))
    {
      if (std::all_of(s.begin(), s.end(), [](char c) {
            return static_cast<unsigned char>(c) < 0x80;
          }))
        return converter.get_string_builder().build_raw_byte_array(
          std::vector<uint8_t>(s.begin(), s.end()));
    }
    return nil_exprt();
  }

  if (method_name == "join")
  {
    ensure_allowed_keywords(method_name, keyword_values, {});
    if (args.size() != 1)
      throw std::runtime_error("join() takes exactly one argument");
    return self.handle_str_join(call_json);
  }

  return std::nullopt;
}

std::optional<exprt> dispatch_spacing_and_padding_methods(
  string_handler &self,
  const std::string &method_name,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  const std::function<exprt()> &get_receiver_expr,
  const std::function<locationt()> &get_location,
  python_converter &converter)
{
  if (method_name == "isspace")
  {
    ensure_allowed_keywords(method_name, keyword_values, {});
    if (!args.empty())
      throw std::runtime_error("isspace() takes no arguments");

    exprt obj_expr = get_receiver_expr();
    locationt loc = get_location();
    if (obj_expr.type().is_unsignedbv() || obj_expr.type().is_signedbv())
      return self.handle_char_isspace(obj_expr, loc);
    return self.handle_string_isspace(obj_expr, loc);
  }

  using optional_char_arg_handler_t =
    exprt (string_handler::*)(const exprt &, const exprt &, const locationt &);
  static constexpr std::
    array<std::pair<const char *, optional_char_arg_handler_t>, 3>
      optional_char_arg_handlers = {{
        {"lstrip", &string_handler::handle_string_lstrip},
        {"rstrip", &string_handler::handle_string_rstrip},
        {"strip", &string_handler::handle_string_strip},
      }};

  for (const auto &[name, handler] : optional_char_arg_handlers)
  {
    if (method_name != name)
      continue;
    ensure_allowed_keywords(method_name, keyword_values, {"chars"});
    if (args.size() > 1)
      throw std::runtime_error(method_name + "() takes at most one argument");

    const nlohmann::json *chars_node = resolve_positional_or_keyword_arg(
      method_name, args, keyword_values, "chars", 0, false);
    exprt chars_arg =
      chars_node ? converter.get_expr(*chars_node) : nil_exprt();
    return (self.*handler)(get_receiver_expr(), chars_arg, get_location());
  }

  using width_fill_handler_t = exprt (string_handler::*)(
    const exprt &, const exprt &, const exprt &, const locationt &);
  static constexpr std::array<std::pair<const char *, width_fill_handler_t>, 3>
    width_fill_handlers = {{
      {"center", &string_handler::handle_string_center},
      {"ljust", &string_handler::handle_string_ljust},
      {"rjust", &string_handler::handle_string_rjust},
    }};

  for (const auto &[name, handler] : width_fill_handlers)
  {
    if (method_name != name)
      continue;
    ensure_allowed_keywords(method_name, keyword_values, {"width", "fillchar"});
    if (args.size() > 2)
      throw std::runtime_error(
        method_name + "() requires one or two arguments");

    const nlohmann::json *width_node = required_arg_node_or_throw(
      method_name,
      args,
      keyword_values,
      "width",
      0,
      method_name + "() requires one or two arguments");
    const nlohmann::json *fill_node = resolve_positional_or_keyword_arg(
      method_name, args, keyword_values, "fillchar", 1, false);

    exprt width_arg = converter.get_expr(*width_node);
    exprt fill_arg = fill_node ? converter.get_expr(*fill_node) : nil_exprt();
    return (self.*handler)(
      get_receiver_expr(), width_arg, fill_arg, get_location());
  }

  if (method_name == "zfill")
  {
    ensure_allowed_keywords(method_name, keyword_values, {"width"});
    if (args.size() > 1)
      throw std::runtime_error("zfill() requires one argument");

    const nlohmann::json *width_node = required_arg_node_or_throw(
      method_name,
      args,
      keyword_values,
      "width",
      0,
      "zfill() requires one argument");
    return self.handle_string_zfill(
      get_receiver_expr(), converter.get_expr(*width_node), get_location());
  }

  if (method_name == "expandtabs")
  {
    ensure_allowed_keywords(method_name, keyword_values, {"tabsize"});
    if (args.size() > 1)
      throw std::runtime_error("expandtabs() takes zero or one argument");

    const nlohmann::json *tabsize_node = resolve_positional_or_keyword_arg(
      method_name, args, keyword_values, "tabsize", 0, false);
    exprt tabsize_arg =
      tabsize_node ? converter.get_expr(*tabsize_node) : nil_exprt();
    return self.handle_string_expandtabs(
      get_receiver_expr(), tabsize_arg, get_location());
  }

  return std::nullopt;
}

std::optional<exprt> dispatch_format_methods(
  string_handler &self,
  const std::string &method_name,
  const nlohmann::json &call_json,
  const keyword_valuest &keyword_values,
  const std::function<exprt()> &get_receiver_expr,
  const std::function<locationt()> &get_location)
{
  if (method_name == "format")
    return self.handle_string_format(
      call_json, get_receiver_expr(), get_location());

  if (method_name == "format_map")
  {
    ensure_allowed_keywords(method_name, keyword_values, {});
    return self.handle_string_format_map(
      call_json, get_receiver_expr(), get_location());
  }

  return std::nullopt;
}
} // namespace string_method_dispatch

exprt string_handler::handle_startswith_endswith_with_pos(
  const exprt &string_obj,
  const nlohmann::json &args,
  python_converter &converter,
  const locationt &location,
  bool is_suffix)
{
  const char *name = is_suffix ? "endswith" : "startswith";

  // s.startswith(prefix, start[, end]) == s[start:end].startswith(prefix)
  // (and likewise for endswith). Compute the s[start:end] substring on a
  // constant receiver and run the base method on it. start/end must be
  // constant ints; a non-constant receiver is rejected cleanly (matching the
  // constant-only support of the other string methods).
  std::string s;
  if (!try_extract_const_string_expr(string_obj, s))
    throw std::runtime_error(
      std::string(name) +
      "() with start/end is only supported on a constant string");

  const long long len = static_cast<long long>(s.size());
  const long long start = string_call_utils::required_constant_int_arg(
    args[1], std::string(name) + "() start must be a constant int", converter);
  const long long end =
    (args.size() == 3) ? string_call_utils::required_constant_int_arg(
                           args[2],
                           std::string(name) + "() end must be a constant int",
                           converter)
                       : len;

  // CPython: once the raw start runs past the end of the string, both methods
  // return False even for an empty affix. Without this guard the start would
  // clamp to len, yielding an empty slice that the base handler's empty-affix
  // short-circuit would wrongly report as a match. (start == len still matches
  // an empty affix, so the comparison is strict.)
  if (start > len)
    return gen_boolean(false);

  // Python slice clamping for s[start:end].
  auto clamp = [len](long long i) {
    if (i < 0)
      i += len;
    if (i < 0)
      i = 0;
    if (i > len)
      i = len;
    return i;
  };
  const long long lo = clamp(start);
  const long long hi = clamp(end);
  const std::string sub =
    (lo < hi) ? s.substr(static_cast<size_t>(lo), static_cast<size_t>(hi - lo))
              : std::string();

  exprt sub_expr = string_builder_->build_string_literal(sub);
  exprt affix_expr = converter.get_expr(args[0]);
  return is_suffix ? handle_string_endswith(sub_expr, affix_expr, location)
                   : handle_string_startswith(sub_expr, affix_expr, location);
}

exprt string_handler::build_affix_tuple_match(
  const exprt &string_obj,
  const exprt &affix_tuple,
  const locationt &location,
  bool is_suffix)
{
  // Python: s.startswith(t) / s.endswith(t) where t is a tuple of strings is
  // True iff s matches ANY element. Build the disjunction over the per-element
  // single-affix matches. Only tuple literals (a struct_exprt whose operands
  // are the elements) are supported; a tuple passed by symbol has no inline
  // operands here, and silently treating it as a string would be unsound, so
  // we reject it with a clean error instead.
  if (affix_tuple.operands().empty())
    throw std::runtime_error(
      std::string(is_suffix ? "endswith" : "startswith") +
      "() with a tuple argument is only supported for tuple literals");

  exprt result = gen_boolean(false);
  for (const exprt &elem : affix_tuple.operands())
  {
    exprt one = is_suffix
                  ? handle_string_endswith(string_obj, elem, location)
                  : handle_string_startswith(string_obj, elem, location);
    // result and one are synthetic bools (constant / startswith-endswith
    // results), so build the disjunction in IREP2 (V.3).
    result = build_or(result, one);
  }
  return result;
}

exprt string_handler::handle_string_startswith(
  const exprt &string_obj,
  const exprt &prefix_arg,
  const locationt &location)
{
  // A tuple of prefixes: True if the string starts with any of them.
  if (converter_.get_tuple_handler().is_tuple_type(prefix_arg.type()))
    return build_affix_tuple_match(string_obj, prefix_arg, location, false);

  // Ensure both are proper null-terminated strings
  exprt string_copy = string_obj;
  exprt prefix_copy = prefix_arg;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt prefix_expr = ensure_null_terminated_string(prefix_copy);

  // Get string addresses
  exprt str_addr = get_array_base_address(str_expr);
  exprt prefix_addr = get_array_base_address(prefix_expr);

  // Calculate the prefix length (excluding the null terminator).
  // A string literal is lowered to a char array carrying a static size, but a
  // symbolic str (e.g. a function parameter) is a char* with no static size,
  // so to_array_type() would abort on it. Use strlen() for the pointer case,
  // mirroring handle_string_endswith().
  exprt actual_len;
  if (prefix_expr.type().is_array())
  {
    const array_typet &prefix_type = to_array_type(prefix_expr.type());
    exprt prefix_len = prefix_type.size();

    // Subtract 1 for null terminator. prefix_len (the array dimension) and the
    // literal share prefix_len.type(), so build it in IREP2 (V.3).
    actual_len = build_sub(
      prefix_len, from_integer(1, prefix_len.type()), prefix_len.type());
  }
  else
  {
    symbolt *strlen_symbol = find_cached_c_function_symbol("c:@F@strlen");
    if (!strlen_symbol)
      throw std::runtime_error("strlen function not found for startswith()");

    exprt prefix_strlen_call =
      build_call_expr(*strlen_symbol, size_type(), {prefix_addr});
    prefix_strlen_call.location() = location;
    actual_len = prefix_strlen_call;
  }

  // Find strncmp symbol
  symbolt *strncmp_symbol = find_cached_c_function_symbol("c:@F@strncmp");
  if (!strncmp_symbol)
    throw std::runtime_error("strncmp function not found for startswith()");

  // Call strncmp(str, prefix, len(prefix))
  exprt strncmp_call = build_call_expr(
    *strncmp_symbol, int_type(), {str_addr, prefix_addr, actual_len});
  strncmp_call.location() = location;

  // V.3: build `(actual_len == 0) || (strncmp(...) == 0)` in IREP2, back
  // -migrating once. Python treats the empty string as a prefix of every
  // string, so s.startswith("") is always True; the empty-prefix guard keeps
  // the result correct for a symbolic empty prefix and is robust to
  // strncmp(_,_,0), which the operational model evaluates to a non-zero value.
  expr2tc strncmp2, zero2;
  migrate_expr(strncmp_call, strncmp2);
  migrate_expr(gen_zero(int_type()), zero2);
  expr2tc equal2 = equality2tc(strncmp2, zero2);

  expr2tc len2, len_zero2;
  migrate_expr(actual_len, len2);
  migrate_expr(gen_zero(actual_len.type()), len_zero2);
  expr2tc is_empty2 = equality2tc(len2, len_zero2);

  return migrate_expr_back(or2tc(is_empty2, equal2));
}

exprt string_handler::handle_string_endswith(
  const exprt &string_obj,
  const exprt &suffix_arg,
  const locationt &location)
{
  // A tuple of suffixes: True if the string ends with any of them.
  if (converter_.get_tuple_handler().is_tuple_type(suffix_arg.type()))
    return build_affix_tuple_match(string_obj, suffix_arg, location, true);

  // Ensure both are proper null-terminated strings
  exprt string_copy = string_obj;
  exprt suffix_copy = suffix_arg;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt suffix_expr = ensure_null_terminated_string(suffix_copy);

  // Get string addresses
  // Handle both pointer and array types
  exprt str_addr;
  exprt suffix_addr;

  if (str_expr.type().is_pointer())
    str_addr = str_expr;
  else
    str_addr = get_array_base_address(str_expr);

  if (suffix_expr.type().is_pointer())
    suffix_addr = suffix_expr;
  else
    suffix_addr = get_array_base_address(suffix_expr);

  // For length calculation, we need to use strlen for pointer types
  // Find strlen symbol
  symbolt *strlen_symbol = find_cached_c_function_symbol("c:@F@strlen");
  if (!strlen_symbol)
    throw std::runtime_error("strlen function not found for endswith()");

  // Get string length using strlen
  exprt str_strlen_call =
    build_call_expr(*strlen_symbol, size_type(), {str_addr});
  str_strlen_call.location() = location;

  // Get suffix length using strlen
  exprt suffix_strlen_call =
    build_call_expr(*strlen_symbol, size_type(), {suffix_addr});
  suffix_strlen_call.location() = location;

  // Check if suffix is longer than string. Both operands are synthetic
  // size_type strlen() results, so build the comparison in IREP2 (V.3).
  exprt len_check = build_greater_than(suffix_strlen_call, str_strlen_call);

  // Calculate offset: strlen(str) - strlen(suffix)
  exprt offset = build_sub(str_strlen_call, suffix_strlen_call, size_type());

  // Get pointer to the position: str + offset
  exprt offset_ptr = build_add(str_addr, offset, gen_pointer_type(char_type()));

  // Find strncmp symbol
  symbolt *strncmp_symbol = find_cached_c_function_symbol("c:@F@strncmp");
  if (!strncmp_symbol)
    throw std::runtime_error("strncmp function not found for endswith()");

  // Call strncmp(str + offset, suffix, strlen(suffix))
  exprt strncmp_call = build_call_expr(
    *strncmp_symbol, int_type(), {offset_ptr, suffix_addr, suffix_strlen_call});
  strncmp_call.location() = location;

  // V.3: build `!(suffix_len > str_len) && (strncmp(...) == 0)` in IREP2,
  // back-migrating once. Order and operands match the legacy nodes exactly.
  expr2tc strncmp2, zero2;
  migrate_expr(strncmp_call, strncmp2);
  migrate_expr(gen_zero(int_type()), zero2);
  expr2tc strings_equal2 = equality2tc(strncmp2, zero2);

  expr2tc len_check2;
  migrate_expr(len_check, len_check2);
  expr2tc len_ok2 = not2tc(len_check2);

  return migrate_expr_back(and2tc(len_ok2, strings_equal2));
}

exprt string_handler::handle_string_isdigit(
  const exprt &string_obj,
  const locationt &location)
{
  // Check if this is a single character
  if (string_obj.type().is_unsignedbv() || string_obj.type().is_signedbv())
  {
    // Call Python's single-character version
    symbolt *isdigit_symbol =
      find_cached_c_function_symbol("c:@F@__python_char_isdigit");
    if (!isdigit_symbol)
      throw std::runtime_error(
        "__python_char_isdigit function not found in symbol table");

    exprt isdigit_call =
      build_call_expr(*isdigit_symbol, bool_type(), {string_obj});
    isdigit_call.location() = location;

    return isdigit_call;
  }

  // For full strings, use the string version
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);

  // Get base address of the string
  exprt str_addr = get_array_base_address(str_expr);

  // Find the helper function symbol
  symbolt *isdigit_str_symbol =
    find_cached_c_function_symbol("c:@F@__python_str_isdigit");
  if (!isdigit_str_symbol)
    throw std::runtime_error("str_isdigit function not found in symbol table");

  // Call str_isdigit(str) - returns bool (0 or 1)
  exprt isdigit_call =
    build_call_expr(*isdigit_str_symbol, bool_type(), {str_addr});
  isdigit_call.location() = location;

  return isdigit_call;
}

exprt string_handler::handle_string_isalpha(
  const exprt &string_obj,
  const locationt &location)
{
  // Check if this is a single character
  if (string_obj.type().is_unsignedbv() || string_obj.type().is_signedbv())
  {
    // Call Python's single-character version (not C's isalpha)
    symbolt *isalpha_symbol =
      find_cached_c_function_symbol("c:@F@__python_char_isalpha");
    if (!isalpha_symbol)
      throw std::runtime_error(
        "__python_char_isalpha function not found in symbol table");

    exprt isalpha_call =
      build_call_expr(*isalpha_symbol, bool_type(), {string_obj});
    isalpha_call.location() = location;

    return isalpha_call;
  }

  // For full strings, use the string version
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt str_addr = get_array_base_address(str_expr);

  symbolt *isalpha_str_symbol =
    find_cached_c_function_symbol("c:@F@__python_str_isalpha");
  if (!isalpha_str_symbol)
    throw std::runtime_error("str_isalpha function not found in symbol table");

  exprt isalpha_call =
    build_call_expr(*isalpha_str_symbol, bool_type(), {str_addr});
  isalpha_call.location() = location;

  return isalpha_call;
}

exprt string_handler::handle_string_isspace(
  const exprt &str_expr,
  const locationt &location)
{
  exprt string_copy = str_expr;
  exprt str_null = ensure_null_terminated_string(string_copy);
  exprt str_addr = get_array_base_address(str_null);

  const char *isspace_str_symbol_id = "c:@F@__python_str_isspace";
  symbolt *isspace_str_symbol =
    find_cached_c_function_symbol(isspace_str_symbol_id);
  if (!isspace_str_symbol)
    throw std::runtime_error(
      std::string(isspace_str_symbol_id) +
      " function not found in symbol table");

  exprt call = build_call_expr(*isspace_str_symbol, bool_type(), {str_addr});
  call.location() = location;

  return call;
}

exprt string_handler::handle_char_isspace(
  const exprt &char_expr,
  const locationt &location)
{
  // For single characters, use the standard C isspace() function
  std::string func_symbol_id = ensure_string_function_symbol("isspace");

  // Convert char to int for isspace
  exprt char_as_int = char_expr;
  if (char_expr.type() != int_type())
  {
    char_as_int = build_typecast(char_expr, int_type());
  }

  // Create function call to C's isspace
  exprt call = build_call_expr(func_symbol_id, int_type(), {char_as_int});
  call.location() = location;

  // V.3: convert the C isspace() result to a boolean in IREP2 (isspace
  // returns non-zero for whitespace), back-migrating once. Operand order
  // (call != 0) and the bool result type match the legacy node.
  expr2tc call2, zero2;
  migrate_expr(call, call2);
  migrate_expr(from_integer(0, int_type()), zero2);
  return migrate_expr_back(notequal2tc(call2, zero2));
}

exprt string_handler::handle_string_lstrip(
  const exprt &str_expr,
  const exprt &chars_arg,
  const locationt &location)
{
  bool can_fold_constant = false;
  if (str_expr.type().is_array())
  {
    if (str_expr.is_constant())
      can_fold_constant = true;
    else if (str_expr.is_symbol())
    {
      const symbolt *symbol =
        find_cached_symbol(str_expr.identifier().as_string());
      if (
        symbol && symbol->get_value().is_constant() &&
        symbol->get_value().type().is_array())
        can_fold_constant = true;
    }
  }

  if (string_builder_ && can_fold_constant)
  {
    std::vector<exprt> chars = string_builder_->extract_string_chars(str_expr);
    bool all_constant = true;

    for (const auto &ch : chars)
    {
      if (!ch.is_constant())
      {
        all_constant = false;
        break;
      }
    }

    if (all_constant)
    {
      if (chars_arg.is_nil())
      {
        auto is_whitespace = [](const exprt &ch) -> bool {
          BigInt char_val =
            binary2integer(ch.value().as_string(), ch.type().is_signedbv());
          char c = static_cast<char>(char_val.to_uint64());
          return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' ||
                 c == '\r';
        };

        while (!chars.empty() && is_whitespace(chars.front()))
          chars.erase(chars.begin());

        return string_builder_->build_null_terminated_string(chars);
      }

      // With chars argument: fold if chars_arg is also constant
      bool chars_foldable =
        (chars_arg.is_constant() && chars_arg.type().is_array());
      if (!chars_foldable && chars_arg.is_symbol())
      {
        const symbolt *sym =
          find_cached_symbol(chars_arg.identifier().as_string());
        chars_foldable = sym && sym->get_value().is_constant() &&
                         sym->get_value().type().is_array();
      }

      if (chars_foldable)
      {
        std::vector<exprt> strip_set =
          string_builder_->extract_string_chars(chars_arg);

        auto is_in_strip_set = [&strip_set](const exprt &ch) -> bool {
          BigInt cv =
            binary2integer(ch.value().as_string(), ch.type().is_signedbv());
          char c = static_cast<char>(cv.to_uint64());
          for (const auto &sc : strip_set)
          {
            BigInt sv =
              binary2integer(sc.value().as_string(), sc.type().is_signedbv());
            if (c == static_cast<char>(sv.to_uint64()))
              return true;
          }
          return false;
        };

        while (!chars.empty() && is_in_strip_set(chars.front()))
          chars.erase(chars.begin());

        return string_builder_->build_null_terminated_string(chars);
      }
    }
  }

  if (chars_arg.is_not_nil())
  {
    // With chars argument: __python_str_lstrip_chars(str, chars)
    std::string func_symbol_id =
      ensure_string_function_symbol("__python_str_lstrip_chars");

    // Convert arguments to pointers if needed
    exprt str_ptr = str_expr;

    if (str_expr.is_constant() && str_expr.type().is_array())
    {
      str_ptr = exprt("index", pointer_typet(char_type()));
      str_ptr.copy_to_operands(str_expr);
      str_ptr.copy_to_operands(from_integer(0, int_type()));
    }
    else if (str_expr.type().is_array())
    {
      str_ptr = exprt("address_of", pointer_typet(char_type()));
      exprt index_expr("index", char_type());
      index_expr.copy_to_operands(str_expr);
      index_expr.copy_to_operands(from_integer(0, int_type()));
      str_ptr.copy_to_operands(index_expr);
    }
    else if (!str_expr.type().is_pointer())
    {
      str_ptr = exprt("address_of", pointer_typet(char_type()));
      str_ptr.copy_to_operands(str_expr);
    }

    // Create function call
    exprt chars_ptr = chars_arg;
    if (chars_arg.type().is_array())
    {
      chars_ptr = exprt("address_of", pointer_typet(char_type()));
      exprt index_expr("index", char_type());
      index_expr.copy_to_operands(chars_arg);
      index_expr.copy_to_operands(from_integer(0, int_type()));
      chars_ptr.copy_to_operands(index_expr);
    }

    exprt call = build_call_expr(
      func_symbol_id, pointer_typet(char_type()), {str_ptr, chars_ptr});
    call.location() = location;

    return call;
  }
  else
  {
    // Without chars argument: __python_str_lstrip(str) - default whitespace
    std::string func_symbol_id =
      ensure_string_function_symbol("__python_str_lstrip");

    // Get the string pointer
    exprt str_ptr = str_expr;

    if (str_expr.is_constant() && str_expr.type().is_array())
    {
      str_ptr = exprt("index", pointer_typet(char_type()));
      str_ptr.copy_to_operands(str_expr);
      str_ptr.copy_to_operands(from_integer(0, int_type()));
    }
    else if (str_expr.type().is_array())
    {
      str_ptr = exprt("address_of", pointer_typet(char_type()));
      exprt index_expr("index", char_type());
      index_expr.copy_to_operands(str_expr);
      index_expr.copy_to_operands(from_integer(0, int_type()));
      str_ptr.copy_to_operands(index_expr);
    }
    else if (!str_expr.type().is_pointer())
    {
      str_ptr = exprt("address_of", pointer_typet(char_type()));
      str_ptr.copy_to_operands(str_expr);
    }

    // Create function call
    exprt call =
      build_call_expr(func_symbol_id, pointer_typet(char_type()), {str_ptr});
    call.location() = location;

    return call;
  }
}

exprt string_handler::handle_string_strip(
  const exprt &str_expr,
  const exprt &chars_arg,
  const locationt &location)
{
  bool can_fold_constant = false;
  if (str_expr.type().is_array())
  {
    if (str_expr.is_constant())
      can_fold_constant = true;
    else if (str_expr.is_symbol())
    {
      const symbolt *symbol =
        find_cached_symbol(str_expr.identifier().as_string());
      if (
        symbol && symbol->get_value().is_constant() &&
        symbol->get_value().type().is_array())
        can_fold_constant = true;
    }
  }

  if (string_builder_ && can_fold_constant)
  {
    std::vector<exprt> chars = string_builder_->extract_string_chars(str_expr);
    bool all_constant = true;

    for (const auto &ch : chars)
    {
      if (!ch.is_constant())
      {
        all_constant = false;
        break;
      }
    }

    if (all_constant)
    {
      if (chars_arg.is_nil())
      {
        auto is_whitespace = [](const exprt &ch) -> bool {
          BigInt char_val =
            binary2integer(ch.value().as_string(), ch.type().is_signedbv());
          char c = static_cast<char>(char_val.to_uint64());
          return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' ||
                 c == '\r';
        };

        while (!chars.empty() && is_whitespace(chars.front()))
          chars.erase(chars.begin());

        while (!chars.empty() && is_whitespace(chars.back()))
          chars.pop_back();

        return string_builder_->build_null_terminated_string(chars);
      }

      // With chars argument: fold if chars_arg is also constant
      bool chars_foldable =
        (chars_arg.is_constant() && chars_arg.type().is_array());
      if (!chars_foldable && chars_arg.is_symbol())
      {
        const symbolt *sym =
          find_cached_symbol(chars_arg.identifier().as_string());
        chars_foldable = sym && sym->get_value().is_constant() &&
                         sym->get_value().type().is_array();
      }

      if (chars_foldable)
      {
        std::vector<exprt> strip_set =
          string_builder_->extract_string_chars(chars_arg);

        auto is_in_strip_set = [&strip_set](const exprt &ch) -> bool {
          BigInt cv =
            binary2integer(ch.value().as_string(), ch.type().is_signedbv());
          char c = static_cast<char>(cv.to_uint64());
          for (const auto &sc : strip_set)
          {
            BigInt sv =
              binary2integer(sc.value().as_string(), sc.type().is_signedbv());
            if (c == static_cast<char>(sv.to_uint64()))
              return true;
          }
          return false;
        };

        while (!chars.empty() && is_in_strip_set(chars.front()))
          chars.erase(chars.begin());

        while (!chars.empty() && is_in_strip_set(chars.back()))
          chars.pop_back();

        return string_builder_->build_null_terminated_string(chars);
      }
    }
  }

  // If chars_arg is provided, use __python_str_strip_chars
  if (chars_arg.is_not_nil())
  {
    std::string func_symbol_id =
      ensure_string_function_symbol("__python_str_strip_chars");

    exprt str_ptr = str_expr;
    if (str_expr.type().is_array())
    {
      str_ptr = exprt("address_of", pointer_typet(char_type()));
      exprt index_expr("index", char_type());
      index_expr.copy_to_operands(str_expr);
      index_expr.copy_to_operands(from_integer(0, int_type()));
      str_ptr.copy_to_operands(index_expr);
    }

    exprt chars_ptr = chars_arg;
    if (chars_arg.type().is_array())
    {
      chars_ptr = exprt("address_of", pointer_typet(char_type()));
      exprt index_expr("index", char_type());
      index_expr.copy_to_operands(chars_arg);
      index_expr.copy_to_operands(from_integer(0, int_type()));
      chars_ptr.copy_to_operands(index_expr);
    }

    exprt call = build_call_expr(
      func_symbol_id, pointer_typet(char_type()), {str_ptr, chars_ptr});
    call.location() = location;
    return call;
  }

  // Default behavior: strip whitespace
  std::string func_symbol_id =
    ensure_string_function_symbol("__python_str_strip");

  exprt str_ptr = str_expr;

  if (str_expr.is_constant() && str_expr.type().is_array())
  {
    str_ptr = exprt("index", pointer_typet(char_type()));
    str_ptr.copy_to_operands(str_expr);
    str_ptr.copy_to_operands(from_integer(0, int_type()));
  }
  else if (str_expr.type().is_array())
  {
    str_ptr = exprt("address_of", pointer_typet(char_type()));
    exprt index_expr("index", char_type());
    index_expr.copy_to_operands(str_expr);
    index_expr.copy_to_operands(from_integer(0, int_type()));
    str_ptr.copy_to_operands(index_expr);
  }
  else if (!str_expr.type().is_pointer())
  {
    str_ptr = exprt("address_of", pointer_typet(char_type()));
    str_ptr.copy_to_operands(str_expr);
  }

  exprt call =
    build_call_expr(func_symbol_id, pointer_typet(char_type()), {str_ptr});
  call.location() = location;

  return call;
}

exprt string_handler::handle_string_rstrip(
  const exprt &str_expr,
  const exprt &chars_arg,
  const locationt &location)
{
  bool can_fold_constant = str_expr.type().is_array();
  if (!can_fold_constant && str_expr.is_symbol())
  {
    const symbolt *symbol =
      find_cached_symbol(str_expr.identifier().as_string());
    if (
      symbol && symbol->get_value().is_constant() &&
      symbol->get_value().type().is_array())
      can_fold_constant = true;
  }

  if (string_builder_ && can_fold_constant)
  {
    std::vector<exprt> chars = string_builder_->extract_string_chars(str_expr);
    bool all_constant = true;

    for (const auto &ch : chars)
    {
      if (!ch.is_constant())
      {
        all_constant = false;
        break;
      }
    }

    if (all_constant)
    {
      if (chars_arg.is_nil())
      {
        auto is_whitespace = [](const exprt &ch) -> bool {
          BigInt char_val =
            binary2integer(ch.value().as_string(), ch.type().is_signedbv());
          char c = static_cast<char>(char_val.to_uint64());
          return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' ||
                 c == '\r';
        };

        while (!chars.empty() && is_whitespace(chars.back()))
          chars.pop_back();

        return string_builder_->build_null_terminated_string(chars);
      }

      // With chars argument: fold if chars_arg is also constant
      bool chars_foldable =
        (chars_arg.is_constant() && chars_arg.type().is_array());
      if (!chars_foldable && chars_arg.is_symbol())
      {
        const symbolt *sym =
          find_cached_symbol(chars_arg.identifier().as_string());
        chars_foldable = sym && sym->get_value().is_constant() &&
                         sym->get_value().type().is_array();
      }

      if (chars_foldable)
      {
        std::vector<exprt> strip_set =
          string_builder_->extract_string_chars(chars_arg);

        auto is_in_strip_set = [&strip_set](const exprt &ch) -> bool {
          BigInt cv =
            binary2integer(ch.value().as_string(), ch.type().is_signedbv());
          char c = static_cast<char>(cv.to_uint64());
          for (const auto &sc : strip_set)
          {
            BigInt sv =
              binary2integer(sc.value().as_string(), sc.type().is_signedbv());
            if (c == static_cast<char>(sv.to_uint64()))
              return true;
          }
          return false;
        };

        while (!chars.empty() && is_in_strip_set(chars.back()))
          chars.pop_back();

        return string_builder_->build_null_terminated_string(chars);
      }
    }
  }

  // If chars_arg is provided, use __python_str_rstrip_chars
  if (chars_arg.is_not_nil())
  {
    std::string func_symbol_id =
      ensure_string_function_symbol("__python_str_rstrip_chars");

    exprt str_ptr = str_expr;
    if (str_expr.type().is_array())
    {
      str_ptr = exprt("address_of", pointer_typet(char_type()));
      exprt index_expr("index", char_type());
      index_expr.copy_to_operands(str_expr);
      index_expr.copy_to_operands(from_integer(0, int_type()));
      str_ptr.copy_to_operands(index_expr);
    }

    exprt chars_ptr = chars_arg;
    if (chars_arg.type().is_array())
    {
      chars_ptr = exprt("address_of", pointer_typet(char_type()));
      exprt index_expr("index", char_type());
      index_expr.copy_to_operands(chars_arg);
      index_expr.copy_to_operands(from_integer(0, int_type()));
      chars_ptr.copy_to_operands(index_expr);
    }

    exprt call = build_call_expr(
      func_symbol_id, pointer_typet(char_type()), {str_ptr, chars_ptr});
    call.location() = location;
    return call;
  }

  std::string func_symbol_id =
    ensure_string_function_symbol("__python_str_rstrip");

  exprt str_ptr = str_expr;

  if (str_expr.is_constant() && str_expr.type().is_array())
  {
    str_ptr = exprt("index", pointer_typet(char_type()));
    str_ptr.copy_to_operands(str_expr);
    str_ptr.copy_to_operands(from_integer(0, int_type()));
  }
  else if (str_expr.type().is_array())
  {
    str_ptr = exprt("address_of", pointer_typet(char_type()));
    exprt index_expr("index", char_type());
    index_expr.copy_to_operands(str_expr);
    index_expr.copy_to_operands(from_integer(0, int_type()));
    str_ptr.copy_to_operands(index_expr);
  }
  else if (!str_expr.type().is_pointer())
  {
    str_ptr = exprt("address_of", pointer_typet(char_type()));
    str_ptr.copy_to_operands(str_expr);
  }

  exprt call =
    build_call_expr(func_symbol_id, pointer_typet(char_type()), {str_ptr});
  call.location() = location;

  return call;
}

exprt string_handler::handle_string_islower(
  const exprt &string_obj,
  const locationt &location)
{
  // Check if this is a single character
  if (string_obj.type().is_unsignedbv() || string_obj.type().is_signedbv())
  {
    // Call Python's single-character version
    symbolt *islower_symbol =
      find_cached_c_function_symbol("c:@F@__python_char_islower");
    if (!islower_symbol)
      throw std::runtime_error(
        "__python_char_islower function not found in symbol table");

    exprt islower_call =
      build_call_expr(*islower_symbol, bool_type(), {string_obj});
    islower_call.location() = location;

    return islower_call;
  }

  // For full strings, use the string version
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt str_addr = get_array_base_address(str_expr);

  symbolt *islower_str_symbol =
    find_cached_c_function_symbol("c:@F@__python_str_islower");
  if (!islower_str_symbol)
    throw std::runtime_error("str_islower function not found in symbol table");

  exprt islower_call =
    build_call_expr(*islower_str_symbol, bool_type(), {str_addr});
  islower_call.location() = location;

  return islower_call;
}

exprt string_handler::handle_string_lower(
  const exprt &string_obj,
  const locationt &location)
{
  // For single characters, handle directly
  if (string_obj.type().is_unsignedbv() || string_obj.type().is_signedbv())
  {
    symbolt *lower_symbol =
      find_cached_c_function_symbol("c:@F@__python_char_lower");
    if (!lower_symbol)
      throw std::runtime_error(
        "__python_char_lower function not found in symbol table");

    exprt lower_call =
      build_call_expr(*lower_symbol, char_type(), {string_obj});
    lower_call.location() = location;

    return lower_call;
  }

  // For full strings, use the string version
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt str_addr = get_array_base_address(str_expr);

  symbolt *lower_str_symbol =
    find_cached_c_function_symbol("c:@F@__python_str_lower");
  if (!lower_str_symbol)
    throw std::runtime_error("str_lower function not found in symbol table");

  exprt lower_call =
    build_call_expr(*lower_str_symbol, pointer_typet(char_type()), {str_addr});
  lower_call.location() = location;

  return lower_call;
}

exprt string_handler::handle_string_upper(
  const exprt &string_obj,
  const locationt &location)
{
  // For single characters, handle directly
  if (string_obj.type().is_unsignedbv() || string_obj.type().is_signedbv())
  {
    symbolt *upper_symbol =
      find_cached_c_function_symbol("c:@F@__python_char_upper");
    if (!upper_symbol)
      throw std::runtime_error(
        "__python_char_upper function not found in symbol table");

    exprt upper_call =
      build_call_expr(*upper_symbol, char_type(), {string_obj});
    upper_call.location() = location;

    return upper_call;
  }

  // For full strings, use the string version
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt str_addr = get_array_base_address(str_expr);

  symbolt *upper_str_symbol =
    find_cached_c_function_symbol("c:@F@__python_str_upper");
  if (!upper_str_symbol)
    throw std::runtime_error("str_upper function not found in symbol table");

  exprt upper_call =
    build_call_expr(*upper_str_symbol, pointer_typet(char_type()), {str_addr});
  upper_call.location() = location;

  return upper_call;
}

exprt string_handler::handle_string_find(
  const exprt &string_obj,
  const exprt &find_arg,
  const locationt &location)
{
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt str_addr = get_array_base_address(str_expr);

  exprt arg_copy = find_arg;
  exprt arg_expr = ensure_null_terminated_string(arg_copy);
  exprt arg_addr = get_array_base_address(arg_expr);

  symbolt *find_str_symbol =
    find_cached_c_function_symbol("c:@F@__python_str_find");
  if (!find_str_symbol)
    throw std::runtime_error("str_find function not found in symbol table");

  exprt find_call =
    build_call_expr(*find_str_symbol, int_type(), {str_addr, arg_addr});
  find_call.location() = location;

  return find_call;
}

exprt string_handler::handle_string_find_range(
  const exprt &string_obj,
  const exprt &find_arg,
  const exprt &start_arg,
  const exprt &end_arg,
  const locationt &location)
{
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt str_addr = get_array_base_address(str_expr);

  exprt arg_copy = find_arg;
  exprt arg_expr = ensure_null_terminated_string(arg_copy);
  exprt arg_addr = get_array_base_address(arg_expr);

  exprt start_expr = start_arg;
  if (start_expr.type() != int_type())
    start_expr = build_typecast(start_expr, int_type());

  exprt end_expr = end_arg;
  if (end_expr.type() != int_type())
    end_expr = build_typecast(end_expr, int_type());

  symbolt *find_range_symbol =
    find_cached_c_function_symbol("c:@F@__python_str_find_range");
  if (!find_range_symbol)
    throw std::runtime_error(
      "str_find_range function not found in symbol table");

  exprt find_call = build_call_expr(
    *find_range_symbol, int_type(), {str_addr, arg_addr, start_expr, end_expr});
  find_call.location() = location;

  return find_call;
}

exprt string_handler::handle_string_index(
  const nlohmann::json &call,
  const exprt &string_obj,
  const exprt &find_arg,
  const locationt &location)
{
  exprt find_expr = handle_string_find(string_obj, find_arg, location);
  return build_string_index_result(call, find_expr, location);
}

exprt string_handler::handle_string_index_range(
  const nlohmann::json &call,
  const exprt &string_obj,
  const exprt &find_arg,
  const exprt &start_arg,
  const exprt &end_arg,
  const locationt &location)
{
  exprt find_expr = handle_string_find_range(
    string_obj, find_arg, start_arg, end_arg, location);
  return build_string_index_result(call, find_expr, location);
}

exprt string_handler::build_string_index_result(
  const nlohmann::json &call,
  const exprt &find_expr,
  const locationt &location)
{
  symbolt &find_result = converter_.create_tmp_symbol(
    call, "$str_index$", int_type(), gen_zero(int_type()));
  code_declt decl(build_symbol(find_result));
  decl.location() = location;
  converter_.add_instruction(decl);

  code_assignt assign(build_symbol(find_result), find_expr);
  assign.location() = location;
  converter_.add_instruction(assign);

  // V.3: build the `find_result == -1` not-found check in IREP2.
  expr2tc fr2;
  migrate_expr(build_symbol(find_result), fr2);
  exprt not_found = migrate_expr_back(
    equality2tc(fr2, from_integer(BigInt(-1), migrate_type(int_type()))));
  exprt raise = python_exception_utils::make_exception_raise(
    type_handler_, "ValueError", "substring not found", &location);

  code_expressiont raise_code(raise);
  raise_code.location() = location;

  code_ifthenelset if_stmt;
  if_stmt.cond() = not_found;
  if_stmt.then_case() = raise_code;
  if_stmt.location() = location;
  converter_.add_instruction(if_stmt);

  return build_symbol(find_result);
}

exprt string_handler::handle_string_rfind(
  const exprt &string_obj,
  const exprt &find_arg,
  const locationt &location)
{
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt str_addr = get_array_base_address(str_expr);

  exprt arg_copy = find_arg;
  exprt arg_expr = ensure_null_terminated_string(arg_copy);
  exprt arg_addr = get_array_base_address(arg_expr);

  symbolt *rfind_str_symbol =
    find_cached_c_function_symbol("c:@F@__python_str_rfind");
  if (!rfind_str_symbol)
    throw std::runtime_error("str_rfind function not found in symbol table");

  exprt rfind_call =
    build_call_expr(*rfind_str_symbol, int_type(), {str_addr, arg_addr});
  rfind_call.location() = location;

  return rfind_call;
}

exprt string_handler::handle_string_rfind_range(
  const exprt &string_obj,
  const exprt &find_arg,
  const exprt &start_arg,
  const exprt &end_arg,
  const locationt &location)
{
  exprt string_copy = string_obj;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt str_addr = get_array_base_address(str_expr);

  exprt arg_copy = find_arg;
  exprt arg_expr = ensure_null_terminated_string(arg_copy);
  exprt arg_addr = get_array_base_address(arg_expr);

  exprt start_expr = start_arg;
  if (start_expr.type() != int_type())
    start_expr = build_typecast(start_expr, int_type());

  exprt end_expr = end_arg;
  if (end_expr.type() != int_type())
    end_expr = build_typecast(end_expr, int_type());

  symbolt *rfind_range_symbol =
    find_cached_c_function_symbol("c:@F@__python_str_rfind_range");
  if (!rfind_range_symbol)
    throw std::runtime_error(
      "str_rfind_range function not found in symbol table");

  exprt rfind_call = build_call_expr(
    *rfind_range_symbol,
    int_type(),
    {str_addr, arg_addr, start_expr, end_expr});
  rfind_call.location() = location;

  return rfind_call;
}

exprt string_handler::handle_string_rindex(
  const nlohmann::json &call,
  const exprt &string_obj,
  const exprt &find_arg,
  const locationt &location)
{
  // rindex is rfind that raises ValueError when the substring is not found,
  // exactly as index relates to find (build_string_index_result raises on -1).
  exprt rfind_expr = handle_string_rfind(string_obj, find_arg, location);
  return build_string_index_result(call, rfind_expr, location);
}

exprt string_handler::handle_string_rindex_range(
  const nlohmann::json &call,
  const exprt &string_obj,
  const exprt &find_arg,
  const exprt &start_arg,
  const exprt &end_arg,
  const locationt &location)
{
  exprt rfind_expr = handle_string_rfind_range(
    string_obj, find_arg, start_arg, end_arg, location);
  return build_string_index_result(call, rfind_expr, location);
}

exprt string_handler::handle_string_replace(
  const exprt &string_obj,
  const exprt &old_arg,
  const exprt &new_arg,
  const exprt &count_arg,
  const locationt &location)
{
  // Try to handle constant strings directly using string_builder API
  // This avoids the loop unwinding issues in ESBMC

  exprt string_copy = string_obj;
  exprt old_copy = old_arg;
  exprt new_copy = new_arg;

  // Ensure all are proper strings
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt old_expr = ensure_null_terminated_string(old_copy);
  exprt new_expr = ensure_null_terminated_string(new_copy);

  // Extract the count value
  int64_t max_replacements = -1; // -1 means replace all
  if (count_arg.is_constant())
  {
    BigInt count_val = binary2integer(
      count_arg.value().as_string(), count_arg.type().is_signedbv());
    max_replacements = count_val.to_int64();

    // count=0 means no replacements
    if (max_replacements == 0)
      return str_expr;
  }

  // Try to extract constant string values for compile-time replacement
  std::string src_str = extract_string_from_array_operands(str_expr);
  std::string old_str = extract_string_from_array_operands(old_expr);
  std::string new_str = extract_string_from_array_operands(new_expr);

  // If we can extract all strings as constants, do replacement at compile time
  if (!src_str.empty() || str_expr.operands().size() > 0)
  {
    // Handle the case where source string might be empty but valid
    if (str_expr.type().is_array() && str_expr.operands().size() > 0)
    {
      src_str = extract_string_from_array_operands(str_expr);
    }

    // Perform the replacement
    std::string result;
    int64_t replacements_done = 0;

    if (old_str.empty())
    {
      // Special case: empty old string - insert new_str between each char
      for (size_t i = 0; i < src_str.size(); ++i)
      {
        if (max_replacements < 0 || replacements_done < max_replacements)
        {
          result += new_str;
          replacements_done++;
        }
        result += src_str[i];
      }
      // Add new_str at the end if we still have replacements left
      if (max_replacements < 0 || replacements_done < max_replacements)
      {
        result += new_str;
      }
    }
    else
    {
      // Normal replacement
      size_t pos = 0;
      size_t old_len = old_str.length();

      while (pos < src_str.length())
      {
        size_t found = src_str.find(old_str, pos);

        if (
          found == std::string::npos ||
          (max_replacements >= 0 && replacements_done >= max_replacements))
        {
          // No more matches or reached max replacements - copy rest
          result += src_str.substr(pos);
          break;
        }

        // Copy characters before the match
        result += src_str.substr(pos, found - pos);
        // Add replacement string
        result += new_str;
        replacements_done++;
        // Move past the matched substring
        pos = found + old_len;
      }
    }

    // Build the result string using string_builder
    return string_builder_->build_string_literal(result);
  }

  // Fallback to C function for non-constant strings
  exprt str_addr =
    str_expr.type().is_pointer() ? str_expr : get_array_base_address(str_expr);
  exprt old_addr =
    old_expr.type().is_pointer() ? old_expr : get_array_base_address(old_expr);
  exprt new_addr =
    new_expr.type().is_pointer() ? new_expr : get_array_base_address(new_expr);

  std::string func_symbol_id =
    ensure_string_function_symbol("__python_str_replace");

  exprt replace_call = build_call_expr(
    func_symbol_id,
    pointer_typet(char_type()),
    {str_addr, old_addr, new_addr, count_arg});
  replace_call.location() = location;

  return replace_call;
}

exprt string_handler::handle_string_capitalize(
  const exprt &string_obj,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string_expr(string_obj, input))
  {
    // Prefer the runtime operational model __python_str_capitalize when
    // available -- symex's CP folds the call on concrete arguments,
    // and otherwise produces a real symbolic char* rather than an
    // unconstrained nondet pointer.
    const symbolt *capitalize_sym =
      find_cached_c_function_symbol("c:@F@__python_str_capitalize");
    if (capitalize_sym)
    {
      exprt s_copy = string_obj;
      exprt s_expr = ensure_null_terminated_string(s_copy);
      exprt s_addr = get_array_base_address(s_expr);

      exprt call = build_call_expr(
        *capitalize_sym, gen_pointer_type(char_type()), {s_addr});
      call.location() = location;
      return call;
    }
    log_warning(
      "str.capitalize(): runtime model __python_str_capitalize not in the "
      "symbol table -- falling back to nondet. Define it under "
      "src/c2goto/library/python/ and rebuild (Python frontend must be on).");
    return build_nondet_string_fallback(location);
  }

  if (!input.empty())
  {
    input[0] = to_upper_char(input[0]);
    for (size_t i = 1; i < input.size(); ++i)
      input[i] = to_lower_char(input[i]);
  }

  if (!string_builder_)
    throw std::runtime_error("string_builder not set for capitalize()");

  exprt result = string_builder_->build_string_literal(input);
  result.location() = location;
  return result;
}

exprt string_handler::handle_string_title(
  const exprt &string_obj,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string_expr(string_obj, input))
  {
    // Prefer the runtime operational model __python_str_title when
    // available -- symex's CP folds the call on concrete arguments,
    // and otherwise produces a real symbolic char* rather than an
    // unconstrained nondet pointer.
    const symbolt *title_sym =
      find_cached_c_function_symbol("c:@F@__python_str_title");
    if (title_sym)
    {
      exprt s_copy = string_obj;
      exprt s_expr = ensure_null_terminated_string(s_copy);
      exprt s_addr = get_array_base_address(s_expr);

      exprt call =
        build_call_expr(*title_sym, gen_pointer_type(char_type()), {s_addr});
      call.location() = location;
      return call;
    }
    log_warning(
      "str.title(): runtime model __python_str_title not in the symbol "
      "table -- falling back to nondet. Define it under "
      "src/c2goto/library/python/ and rebuild (Python frontend must be on).");
    return build_nondet_string_fallback(location);
  }

  // A letter starts a new word iff the previous character is uncased
  // (CPython semantics -- digits are uncased, so they *end* a word:
  // "3d movie".title() == "3D Movie"). Matches __python_str_title.
  bool prev_cased = false;
  for (char &ch : input)
  {
    bool cased = std::isalpha(static_cast<unsigned char>(ch)) != 0;
    ch = prev_cased ? to_lower_char(ch) : to_upper_char(ch);
    prev_cased = cased;
  }

  if (!string_builder_)
    throw std::runtime_error("string_builder not set for title()");

  exprt result = string_builder_->build_string_literal(input);
  result.location() = location;
  return result;
}

exprt string_handler::handle_string_swapcase(
  const exprt &string_obj,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string_expr(string_obj, input))
  {
    // Prefer the runtime operational model __python_str_swapcase when
    // available -- symex's CP folds the call on concrete arguments
    // and otherwise produces a real symbolic char* rather than an
    // unconstrained nondet pointer.
    const symbolt *swapcase_sym =
      find_cached_c_function_symbol("c:@F@__python_str_swapcase");
    if (swapcase_sym)
    {
      exprt s_copy = string_obj;
      exprt s_expr = ensure_null_terminated_string(s_copy);
      exprt s_addr = get_array_base_address(s_expr);

      exprt call =
        build_call_expr(*swapcase_sym, gen_pointer_type(char_type()), {s_addr});
      call.location() = location;
      return call;
    }
    log_warning(
      "str.swapcase(): runtime model __python_str_swapcase not in the symbol "
      "table -- falling back to nondet. Define it under "
      "src/c2goto/library/python/ and rebuild (Python frontend must be on).");
    return build_nondet_string_fallback(location);
  }

  for (char &ch : input)
  {
    unsigned char uch = static_cast<unsigned char>(ch);
    if (std::islower(uch))
      ch = to_upper_char(ch);
    else if (std::isupper(uch))
      ch = to_lower_char(ch);
  }

  if (!string_builder_)
    throw std::runtime_error("string_builder not set for swapcase()");

  exprt result = string_builder_->build_string_literal(input);
  result.location() = location;
  return result;
}

exprt string_handler::handle_string_casefold(
  const exprt &string_obj,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string_expr(string_obj, input))
  {
    log_debug(
      "python-string", "casefold() on non-constant receiver: nondet fallback");
    return build_nondet_string_fallback(location);
  }

  for (char &ch : input)
    ch = to_lower_char(ch);

  if (!string_builder_)
    throw std::runtime_error("string_builder not set for casefold()");

  exprt result = string_builder_->build_string_literal(input);
  result.location() = location;
  return result;
}

exprt string_handler::handle_string_count(
  const exprt &string_obj,
  const exprt &sub_arg,
  const exprt &start_arg,
  const exprt &end_arg,
  const locationt &location)
{
  std::string input;
  std::string sub;
  const bool input_const = try_extract_const_string_expr(string_obj, input);
  const bool sub_const = try_extract_const_string_expr(sub_arg, sub);
  if (!input_const || !sub_const)
  {
    // Prefer the runtime operational model __python_str_count when
    // start/end are at default (the model doesn't yet handle Python's
    // slice arguments). Symex's own constant propagation can fold the
    // call when concrete values flow in, so this is more precise than
    // a blanket nondet -- and on truly symbolic inputs the result is
    // a real symbolic count rather than an unconstrained nondet.
    if (start_arg.is_nil() && end_arg.is_nil())
    {
      const symbolt *count_sym =
        find_cached_c_function_symbol("c:@F@__python_str_count");
      if (count_sym)
      {
        exprt s_copy = string_obj;
        exprt s_expr = ensure_null_terminated_string(s_copy);
        exprt s_addr = get_array_base_address(s_expr);

        exprt sub_copy = sub_arg;
        exprt sub_expr = ensure_null_terminated_string(sub_copy);
        exprt sub_addr = get_array_base_address(sub_expr);

        exprt call =
          build_call_expr(*count_sym, size_type(), {s_addr, sub_addr});
        call.location() = location;
        return call;
      }
      // The default-range path tried the named model and missed: a silent
      // degradation worth surfacing (#4827). __python_str_count is normally
      // auto-registered from library/python/string.c by gen_python_c_models.py,
      // so a miss means the model is absent or the build lacked the Python
      // frontend. The outer log_debug still covers the start/end path, where
      // no model is attempted.
      log_warning(
        "str.count(): runtime model __python_str_count not in the symbol "
        "table -- falling back to nondet. Define it under "
        "src/c2goto/library/python/ and rebuild (Python frontend must be on).");
    }
    log_debug(
      "python-string", "count() on non-constant receiver/needle: nondet int");
    side_effect_expr_nondett nondet(long_long_int_type());
    nondet.location() = location;
    return nondet;
  }

  long long start = 0;
  long long end = static_cast<long long>(input.size());
  long long tmp = 0;
  if (get_constant_int(start_arg, tmp))
    start = tmp;
  if (get_constant_int(end_arg, tmp))
    end = tmp;

  if (start < 0)
    start = static_cast<long long>(input.size()) + start;
  if (end < 0)
    end = static_cast<long long>(input.size()) + end;

  if (start < 0)
    start = 0;
  if (end < 0)
    end = 0;
  if (start > static_cast<long long>(input.size()))
    start = static_cast<long long>(input.size());
  if (end > static_cast<long long>(input.size()))
    end = static_cast<long long>(input.size());
  if (end < start)
    end = start;

  long long count = 0;
  if (sub.empty())
  {
    count = (end - start) + 1;
  }
  else
  {
    size_t pos = static_cast<size_t>(start);
    size_t limit = static_cast<size_t>(end);
    while (pos <= limit && pos + sub.size() <= limit)
    {
      size_t found = input.find(sub, pos);
      if (found == std::string::npos || found + sub.size() > limit)
        break;
      ++count;
      pos = found + sub.size();
    }
  }

  exprt result = from_integer(count, long_long_int_type());
  result.location() = location;
  return result;
}

exprt string_handler::handle_string_removeprefix(
  const exprt &string_obj,
  const exprt &prefix_arg,
  const locationt &location)
{
  std::string input;
  std::string prefix;
  if (
    !try_extract_const_string_expr(string_obj, input) ||
    !try_extract_const_string_expr(prefix_arg, prefix))
  {
    log_debug(
      "python-string",
      "removeprefix() on non-constant receiver/prefix: nondet string");
    return build_nondet_string_fallback(location);
  }

  if (!prefix.empty() && input.rfind(prefix, 0) == 0)
    input = input.substr(prefix.size());

  if (!string_builder_)
    throw std::runtime_error("string_builder not set for removeprefix()");

  exprt result = string_builder_->build_string_literal(input);
  result.location() = location;
  return result;
}

exprt string_handler::handle_string_removesuffix(
  const exprt &string_obj,
  const exprt &suffix_arg,
  const locationt &location)
{
  std::string input;
  std::string suffix;
  if (
    !try_extract_const_string_expr(string_obj, input) ||
    !try_extract_const_string_expr(suffix_arg, suffix))
  {
    log_debug(
      "python-string",
      "removesuffix() on non-constant receiver/suffix: nondet string");
    return build_nondet_string_fallback(location);
  }

  if (
    !suffix.empty() && input.size() >= suffix.size() &&
    input.compare(input.size() - suffix.size(), suffix.size(), suffix) == 0)
  {
    input.resize(input.size() - suffix.size());
  }

  if (!string_builder_)
    throw std::runtime_error("string_builder not set for removesuffix()");

  exprt result = string_builder_->build_string_literal(input);
  result.location() = location;
  return result;
}

exprt string_handler::handle_string_splitlines(
  const nlohmann::json &call,
  const exprt &string_obj,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string_expr(string_obj, input))
  {
    // Sound-ish fallback: return an empty list. We cannot enumerate the
    // lines of an unknown string at conversion time, and returning a
    // nondet `char *` list isn't expressible through python_list::get()
    // without inventing new list-build primitives. An empty list keeps
    // the program well-typed and lets len() return 0 -- callers that
    // depend on specific line content will still report VFAILED, but
    // GOTO conversion no longer aborts (#4807).
    log_debug(
      "python-string",
      "splitlines() on non-constant receiver: empty-list fallback");
    nlohmann::json empty_list;
    empty_list["_type"] = "List";
    empty_list["elts"] = nlohmann::json::array();
    converter_.copy_location_fields_from_decl(call, empty_list);
    python_list list(converter_, empty_list);
    exprt result = list.get();
    result.location() = location;
    return result;
  }

  std::vector<std::string> parts;
  size_t start = 0;
  for (size_t i = 0; i < input.size(); ++i)
  {
    if (input[i] == '\n' || input[i] == '\r')
    {
      parts.push_back(input.substr(start, i - start));
      if (input[i] == '\r' && (i + 1) < input.size() && input[i + 1] == '\n')
        ++i;
      start = i + 1;
    }
  }
  if (start < input.size())
    parts.push_back(input.substr(start));

  nlohmann::json list_node;
  list_node["_type"] = "List";
  list_node["elts"] = nlohmann::json::array();
  converter_.copy_location_fields_from_decl(call, list_node);

  for (const auto &part : parts)
  {
    nlohmann::json elem;
    elem["_type"] = "Constant";
    elem["value"] = part;
    converter_.copy_location_fields_from_decl(call, elem);
    list_node["elts"].push_back(elem);
  }

  python_list list(converter_, list_node);
  exprt result = list.get();
  result.location() = location;
  return result;
}

exprt string_handler::handle_string_format(
  const nlohmann::json &call,
  const exprt &string_obj,
  const locationt &location)
{
  std::string format_str;
  if (!try_extract_const_string_expr(string_obj, format_str))
  {
    log_debug(
      "python-string",
      "format() on non-constant format string: nondet string fallback");
    return build_nondet_string_fallback(location);
  }

  std::vector<std::string> args;
  std::unordered_map<std::string, std::string> keywords;
  // Parallel AST nodes, kept so a `{:spec}` field can re-format the original
  // value rather than the already-stringified one. call is a const ref held by
  // the caller, so these pointers stay valid for this function.
  std::vector<const nlohmann::json *> arg_nodes;
  std::unordered_map<std::string, const nlohmann::json *> keyword_nodes;
  try
  {
    if (call.contains("args") && call["args"].is_array())
    {
      for (const auto &arg : call["args"])
      {
        args.push_back(format_value_from_json(arg, converter_));
        arg_nodes.push_back(&arg);
      }
    }

    if (call.contains("keywords") && call["keywords"].is_array())
    {
      for (const auto &kw : call["keywords"])
      {
        if (!kw.contains("arg") || kw["arg"].is_null())
          throw std::runtime_error("format() kwargs are not supported");
        std::string key = kw["arg"].get<std::string>();
        if (!kw.contains("value"))
          throw std::runtime_error("format() keyword missing value");
        keywords.emplace(key, format_value_from_json(kw["value"], converter_));
        keyword_nodes.emplace(key, &kw["value"]);
      }
    }
  }
  catch (const python_int_overflow_excp &)
  {
    // Bignum diagnostic (#4642) must not be swallowed -- the test
    // suite asserts on the exact error message. Re-throw.
    throw;
  }
  catch (const std::runtime_error &e)
  {
    // Any non-constant format argument (line ~88) or unsupported
    // keyword spec aborts the whole call. Fall back to a nondet string
    // so GOTO conversion can proceed; the specific formatted value is
    // not preserved (sound over-approximation). #4807.
    log_debug(
      "python-string",
      "format() argument folding failed ({}): nondet string fallback",
      e.what());
    return build_nondet_string_fallback(location);
  }

  std::string result;
  result.reserve(format_str.size());
  size_t arg_index = 0;

  for (size_t i = 0; i < format_str.size();)
  {
    char ch = format_str[i];
    if (ch == '{')
    {
      if ((i + 1) < format_str.size() && format_str[i + 1] == '{')
      {
        result.push_back('{');
        i += 2;
        continue;
      }

      size_t end = format_str.find('}', i + 1);
      if (end == std::string::npos)
        throw std::runtime_error("format() unmatched '{'");

      std::string field = format_str.substr(i + 1, end - (i + 1));

      // A field is `name[:spec]`. Split off the format spec at the first ':'.
      std::string fmt_spec;
      const size_t colon = field.find(':');
      std::string name = field;
      if (colon != std::string::npos)
      {
        name = field.substr(0, colon);
        fmt_spec = field.substr(colon + 1);
      }

      // !r/!s conversions and .attr/[idx] field access are not folded.
      if (
        name.find('!') != std::string::npos ||
        name.find('.') != std::string::npos ||
        name.find('[') != std::string::npos)
        return build_nondet_string_fallback(location);

      // Resolve the field name to its stringified value and AST node.
      std::string str_val;
      const nlohmann::json *node = nullptr;
      if (name.empty())
      {
        if (arg_index >= args.size())
          throw std::runtime_error("format() missing arguments");
        str_val = args[arg_index];
        node = arg_nodes[arg_index];
        ++arg_index;
      }
      else
      {
        bool all_digits = true;
        for (char fc : name)
          if (!std::isdigit(static_cast<unsigned char>(fc)))
          {
            all_digits = false;
            break;
          }

        if (all_digits)
        {
          const size_t idx = static_cast<size_t>(std::stoull(name));
          if (idx >= args.size())
            throw std::runtime_error("format() argument index out of range");
          str_val = args[idx];
          node = arg_nodes[idx];
        }
        else
        {
          auto it = keywords.find(name);
          if (it == keywords.end())
            throw std::runtime_error("format() missing keyword argument");
          str_val = it->second;
          auto nit = keyword_nodes.find(name);
          node = (nit != keyword_nodes.end()) ? nit->second : nullptr;
        }
      }

      if (fmt_spec.empty())
        result += str_val;
      else
      {
        // Apply the format spec to the original value; an unsupported spec or
        // non-constant value degrades the whole call to a sound nondet string.
        try
        {
          if (node == nullptr)
            throw std::runtime_error("format spec without a value node");
          result += apply_format_spec(*node, fmt_spec, converter_);
        }
        catch (const std::runtime_error &)
        {
          return build_nondet_string_fallback(location);
        }
      }

      i = end + 1;
      continue;
    }
    if (ch == '}')
    {
      if ((i + 1) < format_str.size() && format_str[i + 1] == '}')
      {
        result.push_back('}');
        i += 2;
        continue;
      }
      throw std::runtime_error("format() unmatched '}'");
    }

    result.push_back(ch);
    ++i;
  }

  if (!string_builder_)
    throw std::runtime_error("string_builder not set for format()");

  exprt out = string_builder_->build_string_literal(result);
  out.location() = location;
  return out;
}

exprt string_handler::handle_string_partition(
  const exprt &string_obj,
  const exprt &sep_arg,
  const locationt &location)
{
  return build_partition_tuple(string_obj, sep_arg, location, false);
}

exprt string_handler::handle_string_rpartition(
  const exprt &string_obj,
  const exprt &sep_arg,
  const locationt &location)
{
  return build_partition_tuple(string_obj, sep_arg, location, true);
}

exprt string_handler::build_partition_tuple(
  const exprt &string_obj,
  const exprt &sep_arg,
  const locationt &location,
  bool from_right)
{
  const char *method_name = from_right ? "rpartition" : "partition";
  // Build a 3-tuple (before, sep, after) as a struct tagged like a regular
  // Python tuple ("tag-tuple_..."). The tag is what lets the assignment target
  // fixup, len(), and subscript recognise the result as a tuple rather than a
  // string; without it the result is mistyped as a scalar char and len()/index
  // give wrong answers (unsound, #5114). The tag/component layout mirrors
  // tuple_handler::create_tuple_struct_type.
  auto make_tuple3 =
    [&](const exprt &a, const exprt &b, const exprt &c) -> exprt {
    struct_typet tuple_type;
    const std::array<const exprt *, 3> elems = {&a, &b, &c};
    std::string tag = "tag-tuple";
    for (size_t i = 0; i < elems.size(); ++i)
    {
      const std::string comp_name = "element_" + std::to_string(i);
      tuple_type.components().push_back(
        struct_typet::componentt(comp_name, comp_name, elems[i]->type()));
      tag += "_" + elems[i]->type().to_string();
    }
    tuple_type.tag(tag);
    set_python_aggregate_kind(tuple_type, "tuple");

    // V.3: build the tuple struct value in IREP2, back-migrating once, then
    // restore the full type -- migrate_type drops the frontend-only
    // aggregate-kind marker read by the `in`/membership/subscript dispatch
    // (see tuple_handler::get_tuple_expr).
    std::vector<expr2tc> members;
    members.reserve(elems.size());
    for (const exprt *e : elems)
    {
      expr2tc m2;
      migrate_expr(*e, m2);
      members.push_back(std::move(m2));
    }
    exprt tuple_expr =
      migrate_expr_back(constant_struct2tc(migrate_type(tuple_type), members));
    tuple_expr.type() = tuple_type;
    tuple_expr.location() = location;
    return tuple_expr;
  };

  std::string input;
  std::string sep;
  if (
    !try_extract_const_string_expr(string_obj, input) ||
    !try_extract_const_string_expr(sep_arg, sep))
  {
    // Sound-ish fallback: return ("", "", "") -- the shape Python uses
    // when the separator isn't found, but with the receiver elided. We
    // cannot search an unknown string at conversion time; the three
    // empty fields keep the tuple well-typed and let len(t) == 3 hold.
    // Callers that depend on specific partition contents will still
    // report VFAILED, but GOTO conversion no longer aborts (#4807).
    log_debug(
      "python-string",
      "{}() on non-constant receiver/separator: empty-tuple fallback",
      method_name);
    if (!string_builder_)
      throw std::runtime_error(
        std::string("string_builder not set for ") + method_name + "()");
    exprt empty_a = string_builder_->build_string_literal("");
    exprt empty_b = string_builder_->build_string_literal("");
    exprt empty_c = string_builder_->build_string_literal("");
    return make_tuple3(empty_a, empty_b, empty_c);
  }
  if (sep.empty())
    throw std::runtime_error(
      std::string(method_name) + "() separator cannot be empty");

  std::string before;
  std::string after;
  std::string mid;
  // partition() splits at the first occurrence of sep; rpartition() at the
  // last. When sep is absent, partition() returns (input, "", "") and
  // rpartition() returns ("", "", input) — the unmatched receiver goes in the
  // first vs. the last element respectively.
  size_t pos = from_right ? input.rfind(sep) : input.find(sep);
  if (pos == std::string::npos)
  {
    if (from_right)
    {
      before = "";
      mid = "";
      after = input;
    }
    else
    {
      before = input;
      mid = "";
      after = "";
    }
  }
  else
  {
    before = input.substr(0, pos);
    mid = sep;
    after = input.substr(pos + sep.size());
  }

  if (!string_builder_)
    throw std::runtime_error(
      std::string("string_builder not set for ") + method_name + "()");

  exprt before_expr = string_builder_->build_string_literal(before);
  exprt mid_expr = string_builder_->build_string_literal(mid);
  exprt after_expr = string_builder_->build_string_literal(after);

  return make_tuple3(before_expr, mid_expr, after_expr);
}

exprt string_handler::handle_string_isalnum(
  const exprt &string_obj,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string_expr(string_obj, input))
  {
    // Prefer the runtime operational model __python_str_isalnum when
    // available -- symex's CP folds the call on concrete arguments
    // and otherwise produces a real symbolic predicate rather than
    // an unconstrained nondet bool.
    const symbolt *isalnum_sym =
      find_cached_c_function_symbol("c:@F@__python_str_isalnum");
    if (isalnum_sym)
    {
      exprt s_copy = string_obj;
      exprt s_expr = ensure_null_terminated_string(s_copy);
      exprt s_addr = get_array_base_address(s_expr);

      exprt call = build_call_expr(*isalnum_sym, bool_type(), {s_addr});
      call.location() = location;
      return call;
    }
    log_warning(
      "str.isalnum(): runtime model __python_str_isalnum not in the symbol "
      "table -- falling back to nondet. Define it under "
      "src/c2goto/library/python/ and rebuild (Python frontend must be on).");
    side_effect_expr_nondett nondet(bool_type());
    nondet.location() = location;
    return nondet;
  }
  if (input.empty())
    return from_integer(0, bool_type());

  for (char ch : input)
  {
    if (!std::isalnum(static_cast<unsigned char>(ch)))
      return from_integer(0, bool_type());
  }
  return from_integer(1, bool_type());
}

exprt string_handler::handle_string_isupper(
  const exprt &string_obj,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string_expr(string_obj, input))
  {
    // Prefer the runtime operational model __python_str_isupper when
    // available -- symex's CP folds the call when concrete values flow
    // in, and otherwise produces a real symbolic predicate rather than
    // an unconstrained nondet bool.
    const symbolt *isupper_sym =
      find_cached_c_function_symbol("c:@F@__python_str_isupper");
    if (isupper_sym)
    {
      exprt s_copy = string_obj;
      exprt s_expr = ensure_null_terminated_string(s_copy);
      exprt s_addr = get_array_base_address(s_expr);

      exprt call = build_call_expr(*isupper_sym, bool_type(), {s_addr});
      call.location() = location;
      return call;
    }
    log_warning(
      "str.isupper(): runtime model __python_str_isupper not in the symbol "
      "table -- falling back to nondet. Define it under "
      "src/c2goto/library/python/ and rebuild (Python frontend must be on).");
    side_effect_expr_nondett nondet(bool_type());
    nondet.location() = location;
    return nondet;
  }
  bool has_cased = false;
  for (char ch : input)
  {
    unsigned char uch = static_cast<unsigned char>(ch);
    if (std::islower(uch))
      return from_integer(0, bool_type());
    if (std::isupper(uch))
      has_cased = true;
  }
  return from_integer(has_cased ? 1 : 0, bool_type());
}

exprt string_handler::handle_string_isnumeric(
  const exprt &string_obj,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string_expr(string_obj, input))
  {
    log_debug(
      "python-string", "isnumeric() on non-constant receiver: nondet bool");
    side_effect_expr_nondett nondet(bool_type());
    nondet.location() = location;
    return nondet;
  }
  if (input.empty())
    return from_integer(0, bool_type());

  for (char ch : input)
  {
    if (!std::isdigit(static_cast<unsigned char>(ch)))
      return from_integer(0, bool_type());
  }
  return from_integer(1, bool_type());
}

exprt string_handler::handle_string_isidentifier(
  const exprt &string_obj,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string_expr(string_obj, input))
  {
    log_debug(
      "python-string", "isidentifier() on non-constant receiver: nondet bool");
    side_effect_expr_nondett nondet(bool_type());
    nondet.location() = location;
    return nondet;
  }
  if (input.empty())
    return from_integer(0, bool_type());

  unsigned char first = static_cast<unsigned char>(input[0]);
  if (!(std::isalpha(first) || input[0] == '_'))
    return from_integer(0, bool_type());
  for (size_t i = 1; i < input.size(); ++i)
  {
    unsigned char ch = static_cast<unsigned char>(input[i]);
    if (!(std::isalnum(ch) || input[i] == '_'))
      return from_integer(0, bool_type());
  }
  return from_integer(1, bool_type());
}

exprt string_handler::handle_string_center(
  const exprt &string_obj,
  const exprt &width_arg,
  const exprt &fill_arg,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string_expr(string_obj, input))
  {
    log_debug(
      "python-string", "center() on non-constant receiver: nondet string");
    return build_nondet_string_fallback(location);
  }
  if (!string_builder_)
    throw std::runtime_error("string_builder not set for center()");

  long long width = 0;
  if (!get_constant_int(width_arg, width))
  {
    log_debug("python-string", "center() on non-constant width: nondet string");
    return build_nondet_string_fallback(location);
  }

  char fill = ' ';
  if (!fill_arg.is_nil())
  {
    std::string fill_str;
    if (
      !try_extract_const_string_expr(fill_arg, fill_str) ||
      fill_str.size() != 1)
    {
      throw std::runtime_error("center() fillchar must be a single character");
    }
    fill = fill_str[0];
  }

  if (width <= static_cast<long long>(input.size()))
    return string_builder_->build_string_literal(input);

  // CPython puts the extra fill char on the LEFT when both the margin and
  // the width are odd (Objects/unicodeobject.c unicode_center_impl:
  // left = marg/2 + (marg & width & 1)), e.g. "ab".center(7) == "---ab--".
  long long pad = width - static_cast<long long>(input.size());
  long long left = pad / 2 + (pad & width & 1);
  long long right = pad - left;
  std::string result(static_cast<size_t>(left), fill);
  result += input;
  result.append(static_cast<size_t>(right), fill);

  exprt out = string_builder_->build_string_literal(result);
  out.location() = location;
  return out;
}

exprt string_handler::handle_string_ljust(
  const exprt &string_obj,
  const exprt &width_arg,
  const exprt &fill_arg,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string_expr(string_obj, input))
  {
    log_debug(
      "python-string", "ljust() on non-constant receiver: nondet string");
    return build_nondet_string_fallback(location);
  }
  if (!string_builder_)
    throw std::runtime_error("string_builder not set for ljust()");

  long long width = 0;
  if (!get_constant_int(width_arg, width))
  {
    log_debug("python-string", "ljust() on non-constant width: nondet string");
    return build_nondet_string_fallback(location);
  }

  char fill = ' ';
  if (!fill_arg.is_nil())
  {
    std::string fill_str;
    if (
      !try_extract_const_string_expr(fill_arg, fill_str) ||
      fill_str.size() != 1)
    {
      throw std::runtime_error("ljust() fillchar must be a single character");
    }
    fill = fill_str[0];
  }

  if (width <= static_cast<long long>(input.size()))
    return string_builder_->build_string_literal(input);

  std::string result = input;
  result.append(static_cast<size_t>(width - input.size()), fill);
  exprt out = string_builder_->build_string_literal(result);
  out.location() = location;
  return out;
}

exprt string_handler::handle_string_rjust(
  const exprt &string_obj,
  const exprt &width_arg,
  const exprt &fill_arg,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string_expr(string_obj, input))
  {
    log_debug(
      "python-string", "rjust() on non-constant receiver: nondet string");
    return build_nondet_string_fallback(location);
  }
  if (!string_builder_)
    throw std::runtime_error("string_builder not set for rjust()");

  long long width = 0;
  if (!get_constant_int(width_arg, width))
  {
    log_debug("python-string", "rjust() on non-constant width: nondet string");
    return build_nondet_string_fallback(location);
  }

  char fill = ' ';
  if (!fill_arg.is_nil())
  {
    std::string fill_str;
    if (
      !try_extract_const_string_expr(fill_arg, fill_str) ||
      fill_str.size() != 1)
    {
      throw std::runtime_error("rjust() fillchar must be a single character");
    }
    fill = fill_str[0];
  }

  if (width <= static_cast<long long>(input.size()))
    return string_builder_->build_string_literal(input);

  std::string result(static_cast<size_t>(width - input.size()), fill);
  result += input;
  exprt out = string_builder_->build_string_literal(result);
  out.location() = location;
  return out;
}

exprt string_handler::handle_string_zfill(
  const exprt &string_obj,
  const exprt &width_arg,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string_expr(string_obj, input))
  {
    log_debug(
      "python-string", "zfill() on non-constant receiver: nondet string");
    return build_nondet_string_fallback(location);
  }
  if (!string_builder_)
    throw std::runtime_error("string_builder not set for zfill()");

  long long width = 0;
  if (!get_constant_int(width_arg, width))
  {
    log_debug("python-string", "zfill() on non-constant width: nondet string");
    return build_nondet_string_fallback(location);
  }

  if (width <= static_cast<long long>(input.size()))
    return string_builder_->build_string_literal(input);

  size_t pad = static_cast<size_t>(width - input.size());
  std::string result;
  if (!input.empty() && (input[0] == '+' || input[0] == '-'))
  {
    result.push_back(input[0]);
    result.append(pad, '0');
    result.append(input.substr(1));
  }
  else
  {
    result.append(pad, '0');
    result.append(input);
  }

  exprt out = string_builder_->build_string_literal(result);
  out.location() = location;
  return out;
}

exprt string_handler::handle_string_expandtabs(
  const exprt &string_obj,
  const exprt &tabsize_arg,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string_expr(string_obj, input))
  {
    log_debug(
      "python-string", "expandtabs() on non-constant receiver: nondet string");
    return build_nondet_string_fallback(location);
  }
  if (!string_builder_)
    throw std::runtime_error("string_builder not set for expandtabs()");

  long long tabsize = 8;
  long long tmp = 0;
  if (get_constant_int(tabsize_arg, tmp))
    tabsize = tmp;
  if (tabsize < 0)
    tabsize = 0;

  std::string result;
  size_t column = 0;
  for (char ch : input)
  {
    if (ch == '\t')
    {
      size_t spaces =
        tabsize == 0 ? 0 : (tabsize - (column % static_cast<size_t>(tabsize)));
      result.append(spaces, ' ');
      column += spaces;
    }
    else
    {
      result.push_back(ch);
      if (ch == '\n' || ch == '\r')
        column = 0;
      else
        ++column;
    }
  }
  exprt out = string_builder_->build_string_literal(result);
  out.location() = location;
  return out;
}

exprt string_handler::handle_string_format_map(
  const nlohmann::json &call,
  const exprt &string_obj,
  const locationt &location)
{
  if (!call.contains("args") || call["args"].size() != 1)
    throw std::runtime_error("format_map() requires one argument");

  std::string format_str;
  if (!try_extract_const_string_expr(string_obj, format_str))
  {
    log_debug(
      "python-string",
      "format_map() on non-constant format string: nondet string fallback");
    return build_nondet_string_fallback(location);
  }

  const auto &mapping = call["args"][0];
  if (!mapping.contains("_type") || mapping["_type"] != "Dict")
  {
    log_debug(
      "python-string",
      "format_map() on non-Dict mapping: nondet string fallback");
    return build_nondet_string_fallback(location);
  }

  std::unordered_map<std::string, std::string> values;
  const auto &keys = mapping["keys"];
  const auto &vals = mapping["values"];
  try
  {
    for (size_t i = 0; i < keys.size(); ++i)
    {
      std::string key;
      if (!extract_constant_string(keys[i], converter_, key))
        throw std::runtime_error("format_map() keys must be constant strings");
      values.emplace(key, format_value_from_json(vals[i], converter_));
    }
  }
  catch (const python_int_overflow_excp &)
  {
    // Bignum diagnostic (#4642) must not be swallowed.
    throw;
  }
  catch (const std::runtime_error &e)
  {
    log_debug(
      "python-string",
      "format_map() key/value folding failed ({}): nondet string fallback",
      e.what());
    return build_nondet_string_fallback(location);
  }

  std::string result;
  result.reserve(format_str.size());

  for (size_t i = 0; i < format_str.size();)
  {
    char ch = format_str[i];
    if (ch == '{')
    {
      if ((i + 1) < format_str.size() && format_str[i + 1] == '{')
      {
        result.push_back('{');
        i += 2;
        continue;
      }

      size_t end = format_str.find('}', i + 1);
      if (end == std::string::npos)
        throw std::runtime_error("format_map() unmatched '{'");

      std::string field = format_str.substr(i + 1, end - (i + 1));
      if (field.empty())
        throw std::runtime_error("format_map() requires named fields");

      auto it = values.find(field);
      if (it == values.end())
        throw std::runtime_error("format_map() missing key");
      result += it->second;
      i = end + 1;
      continue;
    }
    if (ch == '}')
    {
      if ((i + 1) < format_str.size() && format_str[i + 1] == '}')
      {
        result.push_back('}');
        i += 2;
        continue;
      }
      throw std::runtime_error("format_map() unmatched '}'");
    }

    result.push_back(ch);
    ++i;
  }

  if (!string_builder_)
    throw std::runtime_error("string_builder not set for format_map()");
  exprt out = string_builder_->build_string_literal(result);
  out.location() = location;
  return out;
}