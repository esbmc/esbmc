#include <python-frontend/char_utils.h>
#include <python-frontend/exception_utils.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/python_list.h>
#include <python-frontend/string_handler.h>
#include <python-frontend/string_handler_utils.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/string_builder.h>
#include <python-frontend/type_utils.h>
#include <python-frontend/symbol_id.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr_util.h>
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
  exprt out(id, bool_type());
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
  if (extent <= 0)
    return gen_boolean(needle_values.empty());

  const BigInt haystack_content_len = extent - 1;
  const BigInt needle_len = static_cast<unsigned long>(needle_values.size());
  if (needle_len == 0)
    return gen_boolean(true);
  if (needle_len > haystack_content_len)
    return gen_boolean(false);

  // Keep this bounded to avoid path explosion in symbolic membership.
  if (
    haystack_content_len > kMembershipMaxHaystackContentLen ||
    needle_len > kMembershipMaxNeedleLen)
    return std::nullopt;

  const long long max_start = (haystack_content_len - needle_len).to_int64();
  exprt disjunction = gen_boolean(false);

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

using keyword_valuest = string_call_utils::keyword_valuest;
using string_call_utils::collect_keyword_values;
using string_call_utils::ensure_allowed_keywords;
using string_call_utils::find_keyword_value;
using string_call_utils::required_arg_node_or_throw;
using string_call_utils::required_constant_int_arg;
using string_call_utils::resolve_positional_or_keyword_arg;

static std::optional<exprt> dispatch_replace_method(
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

static std::optional<exprt> dispatch_count_method(
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

static std::optional<exprt> dispatch_splitlines_method(
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

static std::optional<exprt> dispatch_split_method(
  const std::string &method_name,
  const nlohmann::json &receiver_json,
  const nlohmann::json &call_json,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  const std::function<bool(const nlohmann::json &)> &is_none_literal,
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

  if (sep_node == nullptr || is_none_literal(*sep_node))
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

static std::optional<exprt> dispatch_no_arg_string_methods(
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

static std::optional<exprt> dispatch_one_arg_string_methods(
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

static std::optional<exprt> dispatch_search_string_methods(
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

static std::optional<exprt> dispatch_decode_join_method(
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

static std::optional<exprt> dispatch_spacing_and_padding_methods(
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

static std::optional<exprt> dispatch_format_methods(
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
} // namespace

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
    if (!symbol || symbol->value.is_nil() || !symbol->value.type().is_array())
      return false;

    out = extract_string_from_array_operands(symbol->value);
    return true;
  }

  if (str_expr.type().is_array())
  {
    out = extract_string_from_array_operands(str_expr);
    return true;
  }

  return false;
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

exprt string_handler::apply_format_specification(
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
              find_cached_symbol(sym_expr.get_identifier().as_string());

            if (symbol && symbol->value.is_constant())
              float_bits = &symbol->value.value().as_string();
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
  exprt index = index_exprt(arr, from_integer(0, index_type()));
  return address_of_exprt(index);
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
        code_assignt assign(symbol_expr(temp), lhs);
        converter_.add_instruction(assign);
        lhs_value = symbol_expr(temp);
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

std::string string_handler::ensure_string_function_symbol(
  const std::string &function_name,
  const typet &return_type,
  const std::vector<typet> &arg_types,
  const locationt &location)
{
  symbol_id func_id;
  func_id.set_prefix("c:");
  func_id.set_function(function_name);

  std::string func_symbol_id = func_id.to_string();

  if (find_cached_symbol(func_symbol_id) == nullptr)
  {
    code_typet code_type;
    code_type.return_type() = return_type;

    for (const auto &arg_type : arg_types)
    {
      code_typet::argumentt arg;
      arg.type() = arg_type;
      code_type.arguments().push_back(arg);
    }

    symbolt symbol = converter_.create_symbol(
      "", function_name, func_symbol_id, location, code_type);

    converter_.add_symbol(symbol);
    symbol_cache_[func_symbol_id] = find_cached_symbol(func_symbol_id);
  }

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
        symbol && symbol->value.is_constant() &&
        symbol->value.type().is_array())
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
        chars_foldable =
          sym && sym->value.is_constant() && sym->value.type().is_array();
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
        symbol && symbol->value.is_constant() &&
        symbol->value.type().is_array())
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
        chars_foldable =
          sym && sym->value.is_constant() && sym->value.type().is_array();
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
      symbol && symbol->value.is_constant() && symbol->value.type().is_array())
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
        chars_foldable =
          sym && sym->value.is_constant() && sym->value.type().is_array();
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
      const typet &value_type = sym->value.type();
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
      new_symbol.type = strchr_type;

      symbol_table_.add(new_symbol);
      strchr_symbol = find_cached_c_function_symbol("c:@F@strchr");
    }

    exprt rhs_str = ensure_null_terminated_string(rhs);
    exprt rhs_addr = get_array_base_address(rhs_str);

    // lhs contains the character value (as void*), cast directly to int
    typecast_exprt char_as_int(lhs, int_type());

    // Call strchr(string, character)
    side_effect_expr_function_callt strchr_call;
    strchr_call.function() = symbol_expr(*strchr_symbol);
    strchr_call.arguments() = {rhs_addr, char_as_int};
    strchr_call.location() = converter_.get_location_from_decl(element);
    strchr_call.type() = gen_pointer_type(char_type());

    // Check if result != NULL (character found)
    constant_exprt null_ptr(gen_pointer_type(char_type()));
    null_ptr.set_value("NULL");

    exprt not_equal("notequal", bool_type());
    not_equal.copy_to_operands(strchr_call, null_ptr);

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
      if (sym && sym->value.is_constant() && sym->value.type().is_array())
        return &sym->value;
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
    return gen_boolean(contains_subsequence(*haystack_values, *needle_values));
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
  side_effect_expr_function_callt strstr_call;
  strstr_call.function() = symbol_expr(*strstr_symbol);
  strstr_call.arguments() = {
    rhs_addr, lhs_addr}; // haystack is rhs, needle is lhs
  strstr_call.location() = converter_.get_location_from_decl(element);
  strstr_call.type() = gen_pointer_type(char_type());

  // Check if result != NULL (substring found)
  constant_exprt null_ptr(gen_pointer_type(char_type()));
  null_ptr.set_value("NULL");

  exprt not_equal("notequal", bool_type());
  not_equal.copy_to_operands(strstr_call, null_ptr);

  return not_equal;
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
    throw std::runtime_error("capitalize() requires constant string");

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
    throw std::runtime_error("title() requires constant string");

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
    throw std::runtime_error("swapcase() requires constant string");

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
    throw std::runtime_error("casefold() requires constant string");

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
    throw std::runtime_error("count() requires constant strings");
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
  [[maybe_unused]] const locationt &location)
{
  std::string input;
  if (!try_extract_const_string_expr(string_obj, input))
    throw std::runtime_error("isalnum() requires constant string");
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
  [[maybe_unused]] const locationt &location)
{
  std::string input;
  if (!try_extract_const_string_expr(string_obj, input))
    throw std::runtime_error("isupper() requires constant string");
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

bool string_handler::extract_constant_string(
  const nlohmann::json &node,
  python_converter &converter,
  std::string &out)
{
  if (
    node.contains("_type") && node["_type"] == "Constant" &&
    node.contains("value") && node["value"].is_string())
  {
    out = node["value"].get<std::string>();
    return true;
  }

  if (node.contains("_type") && node["_type"] == "Name" && node.contains("id"))
  {
    const std::string var_name = node["id"].get<std::string>();
    nlohmann::json var_value = json_utils::get_var_value(
      var_name, converter.get_current_func_name(), converter.get_ast_json());

    if (
      !var_value.empty() && var_value.contains("value") &&
      var_value["value"].contains("_type") &&
      var_value["value"]["_type"] == "Constant" &&
      var_value["value"].contains("value") &&
      var_value["value"]["value"].is_string())
    {
      out = var_value["value"]["value"].get<std::string>();
      return true;
    }
  }

  return false;
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
    base_expr = typecast_exprt(base_expr, int_type());
  }

  // Find the __python_int function symbol
  symbolt *int_symbol = find_cached_c_function_symbol("c:@F@__python_int");
  if (!int_symbol)
  {
    throw std::runtime_error("__python_int function not found in symbol table");
  }

  // Call __python_int(str, base)
  side_effect_expr_function_callt int_call;
  int_call.function() = symbol_expr(*int_symbol);
  int_call.arguments().push_back(str_addr);
  int_call.arguments().push_back(base_expr);
  int_call.location() = location;
  int_call.type() = int_type();

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
    return typecast_exprt(arg, int_type());
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
  return typecast_exprt(arg, int_type());
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
    base_expr = typecast_exprt(base_expr, int_type());
  }

  return handle_string_to_int(arg, base_expr, location);
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
      codepoint_expr = typecast_exprt(codepoint_expr, int_type());
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
    codepoint_expr = typecast_exprt(codepoint_expr, int_type());

  // Find the __python_chr function symbol
  symbolt *chr_symbol = find_cached_c_function_symbol("c:@F@__python_chr");
  if (!chr_symbol)
    throw std::runtime_error("__python_chr function not found in symbol table");

  // Call __python_chr(codepoint)
  side_effect_expr_function_callt chr_call;
  chr_call.function() = symbol_expr(*chr_symbol);
  chr_call.arguments().push_back(codepoint_expr);
  chr_call.location() = location;
  chr_call.type() = pointer_typet(char_type());

  return chr_call;
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
      arg_symbol && arg_symbol->value.is_not_nil() &&
      arg_symbol->value.type().is_array() && arg_symbol->value.is_constant())
    {
      const array_typet &arr_type = to_array_type(arg_symbol->value.type());
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
        return typecast_exprt(symbol_expr(*len_sym), size_type());
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

  std::optional<exprt> cached_receiver_expr;
  auto get_receiver_expr = [&]() -> exprt {
    if (!cached_receiver_expr.has_value())
      cached_receiver_expr = converter_.get_expr(receiver_json);
    return *cached_receiver_expr;
  };

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

  keyword_valuest keyword_values =
    collect_keyword_values(method_name, keywords, false);
  if (
    std::optional<exprt> dispatched = dispatch_decode_join_method(
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
    std::optional<exprt> dispatched = dispatch_no_arg_string_methods(
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
    std::optional<exprt> dispatched = dispatch_one_arg_string_methods(
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
    std::optional<exprt> dispatched = dispatch_search_string_methods(
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
    std::optional<exprt> dispatched = dispatch_spacing_and_padding_methods(
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
    std::optional<exprt> dispatched = dispatch_replace_method(
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
    std::optional<exprt> dispatched = dispatch_count_method(
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
    std::optional<exprt> dispatched = dispatch_splitlines_method(
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
    std::optional<exprt> dispatched = dispatch_format_methods(
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
    std::optional<exprt> dispatched = dispatch_split_method(
      method_name,
      receiver_json,
      call_json,
      args,
      keyword_values,
      is_none_literal_json,
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
      throw std::runtime_error("join() requires a list");

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
    throw std::runtime_error("join() argument must be a list of strings");

  // Get the list elements from the AST
  const auto &elements = (*list_node)["elts"];

  // Edge case: empty list returns empty string
  if (elements.empty())
  {
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
    exprt lhs_as_int = typecast_exprt(lhs_to_check, rhs_char_value.type());
    return create_comparison(lhs_as_int, rhs_char_value);
  }

  if (!lhs_char_value.is_nil() && rhs_to_check.id() == "dereference")
  {
    exprt rhs_as_int = typecast_exprt(rhs_to_check, lhs_char_value.type());
    return create_comparison(lhs_char_value, rhs_as_int);
  }

  return nil_exprt();
}
