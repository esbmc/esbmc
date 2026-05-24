#include <python-frontend/string/char_utils.h>
#include <python-frontend/exception_utils.h>
#include <python-frontend/python_int_overflow.h>
#include <python-frontend/python_list.h>
#include <python-frontend/string/string_method_dispatch.h>
#include <python-frontend/string/string_handler.h>
#include <python-frontend/string/string_handler_utils.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/string/string_builder.h>
#include <python-frontend/type_utils.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/python_types.h>
#include <util/std_expr.h>
#include <util/std_code.h>
#include <util/string_constant.h>
#include <util/type.h>

#include <boost/algorithm/string/predicate.hpp>
#include <array>
#include <cctype>
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

namespace
{
static bool get_constant_int(const exprt &expr, long long &out)
{
  if (expr.is_nil())
    return false;
  BigInt tmp;
  if (!to_integer(expr, tmp))
    return false;
  out = tmp.to_int64();
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
  if (method_name != "split")
    return std::nullopt;

  ensure_allowed_keywords(method_name, keyword_values, {"sep", "maxsplit"});
  if (args.size() > 2)
  {
    throw std::runtime_error(
      "split() requires zero, one, or two arguments in minimal support");
  }

  split_method_argst parsed;
  const nlohmann::json *sep_node = resolve_positional_or_keyword_arg(
    "split", args, keyword_values, "sep", 0, false);
  const nlohmann::json *maxsplit_node = resolve_positional_or_keyword_arg(
    "split", args, keyword_values, "maxsplit", 1, false);

  if (sep_node == nullptr || is_none_literal_json(*sep_node))
  {
    parsed.separator = "";
  }
  else if (!string_handler::extract_constant_string(
             *sep_node, converter, parsed.separator))
  {
    throw std::runtime_error(
      "split() only supports constant sep in minimal support");
  }

  if (maxsplit_node != nullptr)
  {
    parsed.maxsplit = required_constant_int_arg(
      *maxsplit_node,
      "split() only supports constant maxsplit in minimal support",
      converter);
  }

  if (
    !parsed.separator.empty() && parsed.maxsplit == 1 &&
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
      converter, call_json, obj_expr, parsed.separator, parsed.maxsplit);
  }

  return python_list::build_split_list(
    converter, call_json, input, parsed.separator, parsed.maxsplit);
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
  static constexpr std::array<one_arg_handler_entryt, 5> one_arg_handlers = {{
    {"startswith", "prefix", &string_handler::handle_string_startswith},
    {"endswith", "suffix", &string_handler::handle_string_endswith},
    {"removeprefix", "prefix", &string_handler::handle_string_removeprefix},
    {"removesuffix", "suffix", &string_handler::handle_string_removesuffix},
    {"partition", "sep", &string_handler::handle_string_partition},
  }};

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
  if (method_name != "find" && method_name != "index" && method_name != "rfind")
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

exprt string_handler::handle_string_startswith(
  const exprt &string_obj,
  const exprt &prefix_arg,
  const locationt &location)
{
  // Ensure both are proper null-terminated strings
  exprt string_copy = string_obj;
  exprt prefix_copy = prefix_arg;
  exprt str_expr = ensure_null_terminated_string(string_copy);
  exprt prefix_expr = ensure_null_terminated_string(prefix_copy);

  // Get string addresses
  exprt str_addr = get_array_base_address(str_expr);
  exprt prefix_addr = get_array_base_address(prefix_expr);

  // Calculate prefix length: len(prefix_expr) - 1 (exclude null terminator)
  const array_typet &prefix_type = to_array_type(prefix_expr.type());
  exprt prefix_len = prefix_type.size();

  // Subtract 1 for null terminator
  exprt one = from_integer(1, prefix_len.type());
  exprt actual_len("-", prefix_len.type());
  actual_len.copy_to_operands(prefix_len, one);

  // Find strncmp symbol
  symbolt *strncmp_symbol = find_cached_c_function_symbol("c:@F@strncmp");
  if (!strncmp_symbol)
    throw std::runtime_error("strncmp function not found for startswith()");

  // Call strncmp(str, prefix, len(prefix))
  side_effect_expr_function_callt strncmp_call;
  strncmp_call.function() = symbol_expr(*strncmp_symbol);
  strncmp_call.arguments() = {str_addr, prefix_addr, actual_len};
  strncmp_call.location() = location;
  strncmp_call.type() = int_type();

  // Check if result == 0 (strings match)
  exprt zero = gen_zero(int_type());
  exprt equal("=", bool_type());
  equal.copy_to_operands(strncmp_call, zero);

  return equal;
}

exprt string_handler::handle_string_endswith(
  const exprt &string_obj,
  const exprt &suffix_arg,
  const locationt &location)
{
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
  side_effect_expr_function_callt str_strlen_call;
  str_strlen_call.function() = symbol_expr(*strlen_symbol);
  str_strlen_call.arguments() = {str_addr};
  str_strlen_call.location() = location;
  str_strlen_call.type() = size_type();

  // Get suffix length using strlen
  side_effect_expr_function_callt suffix_strlen_call;
  suffix_strlen_call.function() = symbol_expr(*strlen_symbol);
  suffix_strlen_call.arguments() = {suffix_addr};
  suffix_strlen_call.location() = location;
  suffix_strlen_call.type() = size_type();

  // Check if suffix is longer than string
  exprt len_check(">", bool_type());
  len_check.copy_to_operands(suffix_strlen_call, str_strlen_call);

  // Calculate offset: strlen(str) - strlen(suffix)
  exprt offset("-", size_type());
  offset.copy_to_operands(str_strlen_call, suffix_strlen_call);

  // Get pointer to the position: str + offset
  exprt offset_ptr("+", gen_pointer_type(char_type()));
  offset_ptr.copy_to_operands(str_addr, offset);

  // Find strncmp symbol
  symbolt *strncmp_symbol = find_cached_c_function_symbol("c:@F@strncmp");
  if (!strncmp_symbol)
    throw std::runtime_error("strncmp function not found for endswith()");

  // Call strncmp(str + offset, suffix, strlen(suffix))
  side_effect_expr_function_callt strncmp_call;
  strncmp_call.function() = symbol_expr(*strncmp_symbol);
  strncmp_call.arguments() = {offset_ptr, suffix_addr, suffix_strlen_call};
  strncmp_call.location() = location;
  strncmp_call.type() = int_type();

  // Check if result == 0 (strings match)
  exprt zero = gen_zero(int_type());
  exprt strings_equal("=", bool_type());
  strings_equal.copy_to_operands(strncmp_call, zero);

  // Return: (suffix_len <= str_len) && (strncmp(...) == 0)
  exprt len_ok("not", bool_type());
  len_ok.copy_to_operands(len_check);

  exprt result("and", bool_type());
  result.copy_to_operands(len_ok, strings_equal);

  return result;
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

    side_effect_expr_function_callt isdigit_call;
    isdigit_call.function() = symbol_expr(*isdigit_symbol);
    isdigit_call.arguments().push_back(string_obj);
    isdigit_call.location() = location;
    isdigit_call.type() = bool_type();

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
  side_effect_expr_function_callt isdigit_call;
  isdigit_call.function() = symbol_expr(*isdigit_str_symbol);
  isdigit_call.arguments().push_back(str_addr);
  isdigit_call.location() = location;
  isdigit_call.type() = bool_type();

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

    side_effect_expr_function_callt isalpha_call;
    isalpha_call.function() = symbol_expr(*isalpha_symbol);
    isalpha_call.arguments().push_back(string_obj);
    isalpha_call.location() = location;
    isalpha_call.type() = bool_type();

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

  side_effect_expr_function_callt isalpha_call;
  isalpha_call.function() = symbol_expr(*isalpha_str_symbol);
  isalpha_call.arguments().push_back(str_addr);
  isalpha_call.location() = location;
  isalpha_call.type() = bool_type();

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

  side_effect_expr_function_callt call;
  call.function() = symbol_expr(*isspace_str_symbol);
  call.arguments().push_back(str_addr);
  call.location() = location;
  call.type() = bool_type();

  return call;
}

exprt string_handler::handle_char_isspace(
  const exprt &char_expr,
  const locationt &location)
{
  // For single characters, use the standard C isspace() function
  std::string func_symbol_id = ensure_string_function_symbol(
    "isspace", int_type(), {int_type()}, location);

  // Convert char to int for isspace
  exprt char_as_int = char_expr;
  if (char_expr.type() != int_type())
  {
    char_as_int = typecast_exprt(char_expr, int_type());
  }

  // Create function call to C's isspace
  side_effect_expr_function_callt call;
  call.function() = symbol_exprt(func_symbol_id, code_typet());
  call.arguments().push_back(char_as_int);
  call.type() = int_type();
  call.location() = location;

  // Convert result to boolean (isspace returns non-zero for whitespace)
  exprt result("notequal", bool_type());
  result.copy_to_operands(call);
  result.copy_to_operands(from_integer(0, int_type()));

  return result;
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

  // If chars_arg is empty, strip whitespace (default behavior)
  std::vector<typet> arg_types = {pointer_typet(char_type())};

  if (chars_arg.is_not_nil())
  {
    // With chars argument: __python_str_lstrip_chars(str, chars)
    arg_types.push_back(pointer_typet(char_type()));

    std::string func_symbol_id = ensure_string_function_symbol(
      "__python_str_lstrip_chars",
      pointer_typet(char_type()),
      arg_types,
      location);

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
    side_effect_expr_function_callt call;
    call.function() = symbol_exprt(func_symbol_id, code_typet());
    call.arguments().push_back(str_ptr);

    exprt chars_ptr = chars_arg;
    if (chars_arg.type().is_array())
    {
      chars_ptr = exprt("address_of", pointer_typet(char_type()));
      exprt index_expr("index", char_type());
      index_expr.copy_to_operands(chars_arg);
      index_expr.copy_to_operands(from_integer(0, int_type()));
      chars_ptr.copy_to_operands(index_expr);
    }
    call.arguments().push_back(chars_ptr);

    call.type() = pointer_typet(char_type());
    call.location() = location;

    return call;
  }
  else
  {
    // Without chars argument: __python_str_lstrip(str) - default whitespace
    std::string func_symbol_id = ensure_string_function_symbol(
      "__python_str_lstrip",
      pointer_typet(char_type()),
      {pointer_typet(char_type())},
      location);

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
    side_effect_expr_function_callt call;
    call.function() = symbol_exprt(func_symbol_id, code_typet());
    call.arguments().push_back(str_ptr);
    call.type() = pointer_typet(char_type());
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
    std::string func_symbol_id = ensure_string_function_symbol(
      "__python_str_strip_chars",
      pointer_typet(char_type()),
      {pointer_typet(char_type()), pointer_typet(char_type())},
      location);

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

    side_effect_expr_function_callt call;
    call.function() = symbol_exprt(func_symbol_id, code_typet());
    call.arguments().push_back(str_ptr);
    call.arguments().push_back(chars_ptr);
    call.type() = pointer_typet(char_type());
    call.location() = location;
    return call;
  }

  // Default behavior: strip whitespace
  std::string func_symbol_id = ensure_string_function_symbol(
    "__python_str_strip",
    pointer_typet(char_type()),
    {pointer_typet(char_type())},
    location);

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

  side_effect_expr_function_callt call;
  call.function() = symbol_exprt(func_symbol_id, code_typet());
  call.arguments().push_back(str_ptr);
  call.type() = pointer_typet(char_type());
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
    std::string func_symbol_id = ensure_string_function_symbol(
      "__python_str_rstrip_chars",
      pointer_typet(char_type()),
      {pointer_typet(char_type()), pointer_typet(char_type())},
      location);

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

    side_effect_expr_function_callt call;
    call.function() = symbol_exprt(func_symbol_id, code_typet());
    call.arguments().push_back(str_ptr);
    call.arguments().push_back(chars_ptr);
    call.type() = pointer_typet(char_type());
    call.location() = location;
    return call;
  }

  std::string func_symbol_id = ensure_string_function_symbol(
    "__python_str_rstrip",
    pointer_typet(char_type()),
    {pointer_typet(char_type())},
    location);

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

  side_effect_expr_function_callt call;
  call.function() = symbol_exprt(func_symbol_id, code_typet());
  call.arguments().push_back(str_ptr);
  call.type() = pointer_typet(char_type());
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

    side_effect_expr_function_callt islower_call;
    islower_call.function() = symbol_expr(*islower_symbol);
    islower_call.arguments().push_back(string_obj);
    islower_call.location() = location;
    islower_call.type() = bool_type();

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

  side_effect_expr_function_callt islower_call;
  islower_call.function() = symbol_expr(*islower_str_symbol);
  islower_call.arguments().push_back(str_addr);
  islower_call.location() = location;
  islower_call.type() = bool_type();

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

    side_effect_expr_function_callt lower_call;
    lower_call.function() = symbol_expr(*lower_symbol);
    lower_call.arguments().push_back(string_obj);
    lower_call.location() = location;
    lower_call.type() = char_type();

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

  side_effect_expr_function_callt lower_call;
  lower_call.function() = symbol_expr(*lower_str_symbol);
  lower_call.arguments().push_back(str_addr);
  lower_call.location() = location;
  lower_call.type() = pointer_typet(char_type());

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

    side_effect_expr_function_callt upper_call;
    upper_call.function() = symbol_expr(*upper_symbol);
    upper_call.arguments().push_back(string_obj);
    upper_call.location() = location;
    upper_call.type() = char_type();

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

  side_effect_expr_function_callt upper_call;
  upper_call.function() = symbol_expr(*upper_str_symbol);
  upper_call.arguments().push_back(str_addr);
  upper_call.location() = location;
  upper_call.type() = pointer_typet(char_type());

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

  side_effect_expr_function_callt find_call;
  find_call.function() = symbol_expr(*find_str_symbol);
  find_call.arguments().push_back(str_addr);
  find_call.arguments().push_back(arg_addr);
  find_call.location() = location;
  find_call.type() = int_type();

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
    start_expr = typecast_exprt(start_expr, int_type());

  exprt end_expr = end_arg;
  if (end_expr.type() != int_type())
    end_expr = typecast_exprt(end_expr, int_type());

  symbolt *find_range_symbol =
    find_cached_c_function_symbol("c:@F@__python_str_find_range");
  if (!find_range_symbol)
    throw std::runtime_error(
      "str_find_range function not found in symbol table");

  side_effect_expr_function_callt find_call;
  find_call.function() = symbol_expr(*find_range_symbol);
  find_call.arguments().push_back(str_addr);
  find_call.arguments().push_back(arg_addr);
  find_call.arguments().push_back(start_expr);
  find_call.arguments().push_back(end_expr);
  find_call.location() = location;
  find_call.type() = int_type();

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
  code_declt decl(symbol_expr(find_result));
  decl.location() = location;
  converter_.add_instruction(decl);

  code_assignt assign(symbol_expr(find_result), find_expr);
  assign.location() = location;
  converter_.add_instruction(assign);

  exprt not_found =
    equality_exprt(symbol_expr(find_result), from_integer(-1, int_type()));
  exprt raise = python_exception_utils::make_exception_raise(
    type_handler_, "ValueError", "substring not found", &location);

  code_expressiont raise_code(raise);
  raise_code.location() = location;

  code_ifthenelset if_stmt;
  if_stmt.cond() = not_found;
  if_stmt.then_case() = raise_code;
  if_stmt.location() = location;
  converter_.add_instruction(if_stmt);

  return symbol_expr(find_result);
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

  side_effect_expr_function_callt rfind_call;
  rfind_call.function() = symbol_expr(*rfind_str_symbol);
  rfind_call.arguments().push_back(str_addr);
  rfind_call.arguments().push_back(arg_addr);
  rfind_call.location() = location;
  rfind_call.type() = int_type();

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
    start_expr = typecast_exprt(start_expr, int_type());

  exprt end_expr = end_arg;
  if (end_expr.type() != int_type())
    end_expr = typecast_exprt(end_expr, int_type());

  symbolt *rfind_range_symbol =
    find_cached_c_function_symbol("c:@F@__python_str_rfind_range");
  if (!rfind_range_symbol)
    throw std::runtime_error(
      "str_rfind_range function not found in symbol table");

  side_effect_expr_function_callt rfind_call;
  rfind_call.function() = symbol_expr(*rfind_range_symbol);
  rfind_call.arguments().push_back(str_addr);
  rfind_call.arguments().push_back(arg_addr);
  rfind_call.arguments().push_back(start_expr);
  rfind_call.arguments().push_back(end_expr);
  rfind_call.location() = location;
  rfind_call.type() = int_type();

  return rfind_call;
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

  std::string func_symbol_id = ensure_string_function_symbol(
    "__python_str_replace",
    pointer_typet(char_type()),
    {pointer_typet(char_type()),
     pointer_typet(char_type()),
     pointer_typet(char_type()),
     int_type()},
    location);

  side_effect_expr_function_callt replace_call;
  replace_call.function() = symbol_exprt(func_symbol_id, code_typet());
  replace_call.arguments() = {str_addr, old_addr, new_addr, count_arg};
  replace_call.location() = location;
  replace_call.type() = pointer_typet(char_type());

  return replace_call;
}

exprt string_handler::handle_string_capitalize(
  const exprt &string_obj,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string_expr(string_obj, input))
  {
    log_debug(
      "python-string",
      "capitalize() on non-constant receiver: nondet fallback");
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
    log_debug(
      "python-string", "title() on non-constant receiver: nondet fallback");
    return build_nondet_string_fallback(location);
  }

  bool new_word = true;
  for (char &ch : input)
  {
    if (std::isalpha(static_cast<unsigned char>(ch)))
    {
      ch = new_word ? to_upper_char(ch) : to_lower_char(ch);
      new_word = false;
    }
    else
    {
      new_word = !std::isalnum(static_cast<unsigned char>(ch));
    }
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
    log_debug(
      "python-string", "swapcase() on non-constant receiver: nondet fallback");
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
  if (
    !try_extract_const_string_expr(string_obj, input) ||
    !try_extract_const_string_expr(sub_arg, sub))
  {
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
    throw std::runtime_error("removeprefix() requires constant strings");
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
    throw std::runtime_error("removesuffix() requires constant strings");
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
    throw std::runtime_error("splitlines() requires constant string");

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
    throw std::runtime_error("format() requires constant format string");

  std::vector<std::string> args;
  if (call.contains("args") && call["args"].is_array())
  {
    for (const auto &arg : call["args"])
    {
      args.push_back(format_value_from_json(arg, converter_));
    }
  }

  std::unordered_map<std::string, std::string> keywords;
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
    }
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
      if (field.empty())
      {
        if (arg_index >= args.size())
          throw std::runtime_error("format() missing arguments");
        result += args[arg_index++];
      }
      else
      {
        bool all_digits = true;
        for (char fc : field)
        {
          if (!std::isdigit(static_cast<unsigned char>(fc)))
          {
            all_digits = false;
            break;
          }
        }

        if (all_digits)
        {
          size_t idx = static_cast<size_t>(std::stoull(field));
          if (idx >= args.size())
            throw std::runtime_error("format() argument index out of range");
          result += args[idx];
        }
        else
        {
          auto it = keywords.find(field);
          if (it == keywords.end())
            throw std::runtime_error("format() missing keyword argument");
          result += it->second;
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
  std::string input;
  std::string sep;
  if (
    !try_extract_const_string_expr(string_obj, input) ||
    !try_extract_const_string_expr(sep_arg, sep))
  {
    throw std::runtime_error("partition() requires constant strings");
  }
  if (sep.empty())
    throw std::runtime_error("partition() separator cannot be empty");

  std::string before;
  std::string after;
  std::string mid;
  size_t pos = input.find(sep);
  if (pos == std::string::npos)
  {
    before = input;
    mid = "";
    after = "";
  }
  else
  {
    before = input.substr(0, pos);
    mid = sep;
    after = input.substr(pos + sep.size());
  }

  if (!string_builder_)
    throw std::runtime_error("string_builder not set for partition()");

  exprt before_expr = string_builder_->build_string_literal(before);
  exprt mid_expr = string_builder_->build_string_literal(mid);
  exprt after_expr = string_builder_->build_string_literal(after);

  struct_typet tuple_type;
  tuple_type.components().push_back(
    struct_typet::componentt("element_0", before_expr.type()));
  tuple_type.components().push_back(
    struct_typet::componentt("element_1", mid_expr.type()));
  tuple_type.components().push_back(
    struct_typet::componentt("element_2", after_expr.type()));

  struct_exprt tuple_expr(tuple_type);
  tuple_expr.operands() = {before_expr, mid_expr, after_expr};
  tuple_expr.location() = location;
  return tuple_expr;
}

exprt string_handler::handle_string_isalnum(
  const exprt &string_obj,
  const locationt &location)
{
  std::string input;
  if (!try_extract_const_string_expr(string_obj, input))
  {
    log_debug(
      "python-string", "isalnum() on non-constant receiver: nondet bool");
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
    log_debug(
      "python-string", "isupper() on non-constant receiver: nondet bool");
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
  [[maybe_unused]] const locationt &location)
{
  std::string input;
  if (!try_extract_const_string_expr(string_obj, input))
    throw std::runtime_error("isnumeric() requires constant string");
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
  [[maybe_unused]] const locationt &location)
{
  std::string input;
  if (!try_extract_const_string_expr(string_obj, input))
    throw std::runtime_error("isidentifier() requires constant string");
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
    throw std::runtime_error("center() requires constant string");
  if (!string_builder_)
    throw std::runtime_error("string_builder not set for center()");

  long long width = 0;
  if (!get_constant_int(width_arg, width))
    throw std::runtime_error("center() requires constant width");

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

  long long pad = width - static_cast<long long>(input.size());
  long long left = pad / 2;
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
    throw std::runtime_error("ljust() requires constant string");
  if (!string_builder_)
    throw std::runtime_error("string_builder not set for ljust()");

  long long width = 0;
  if (!get_constant_int(width_arg, width))
    throw std::runtime_error("ljust() requires constant width");

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
    throw std::runtime_error("rjust() requires constant string");
  if (!string_builder_)
    throw std::runtime_error("string_builder not set for rjust()");

  long long width = 0;
  if (!get_constant_int(width_arg, width))
    throw std::runtime_error("rjust() requires constant width");

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
    throw std::runtime_error("zfill() requires constant string");
  if (!string_builder_)
    throw std::runtime_error("string_builder not set for zfill()");

  long long width = 0;
  if (!get_constant_int(width_arg, width))
    throw std::runtime_error("zfill() requires constant width");

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
    throw std::runtime_error("expandtabs() requires constant string");
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
    throw std::runtime_error("format_map() requires constant format string");

  const auto &mapping = call["args"][0];
  if (!mapping.contains("_type") || mapping["_type"] != "Dict")
    throw std::runtime_error("format_map() requires constant dict");

  std::unordered_map<std::string, std::string> values;
  const auto &keys = mapping["keys"];
  const auto &vals = mapping["values"];
  for (size_t i = 0; i < keys.size(); ++i)
  {
    std::string key;
    if (!extract_constant_string(keys[i], converter_, key))
      throw std::runtime_error("format_map() keys must be constant strings");
    values.emplace(key, format_value_from_json(vals[i], converter_));
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
