#include <python-frontend/converter/converter_internal.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/python_converter.h>

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>

// A builtin that only reads an iterable sees a 1-D NumPy view as the list of
// its logical elements: the view's own storage is read once, through its
// shape and strides, into a fresh list the builtin then consumes.

namespace
{
const std::set<std::string> &list_consuming_builtins()
{
  static const std::set<std::string> names = {
    "sum",
    "any",
    "all",
    "min",
    "max",
    "tuple",
    "reversed",
    "print",
    "list",
    "sorted"};
  return names;
}

// A keyword of a list-consuming builtin whose meaning does not depend on the
// elements.
bool is_element_independent_keyword(
  const std::string &builtin,
  const std::string &keyword)
{
  return (builtin == "sum" && keyword == "start") ||
         (builtin == "sorted" && keyword == "reverse") ||
         ((builtin == "min" || builtin == "max") && keyword == "default");
}

bool is_block_key(const std::string &key)
{
  return key == "body" || key == "orelse" || key == "handlers" ||
         key == "finalbody";
}

nlohmann::json tolist_call(const nlohmann::json &value)
{
  nlohmann::json func = {
    {"_type", "Attribute"},
    {"value", value},
    {"attr", "tolist"},
    {"ctx", {{"_type", "Load"}}}};
  nlohmann::json call = {
    {"_type", "Call"},
    {"func", func},
    {"args", nlohmann::json::array()},
    {"keywords", nlohmann::json::array()}};
  for (const char *key :
       {"lineno", "col_offset", "end_lineno", "end_col_offset"})
    if (value.contains(key))
    {
      call[key] = value[key];
      call["func"][key] = value[key];
    }
  return call;
}
std::string assigned_name(const nlohmann::json &stmt)
{
  const std::string kind = stmt.value("_type", "");
  if (kind == "Assign" && stmt["targets"].size() == 1)
    return stmt["targets"][0].value("id", "");
  if (kind == "AnnAssign")
    return stmt["target"].value("id", "");
  return "";
}

// The iterable a `key=` scan of sorted/min/max (preprocessor
// generator_mixin.py) copies into its working sequence. The copy is `it[:]`
// or `it`, which for a NumPy iterable is a view of it, not a list.
const nlohmann::json *key_scan_iterable(const nlohmann::json &stmt)
{
  const std::string name = assigned_name(stmt);
  if (
    !stmt.contains("value") || stmt["value"].is_null() ||
    (name.rfind("ESBMC_srt_", 0) != 0 && name.rfind("ESBMC_mmseq_", 0) != 0))
    return nullptr;
  const nlohmann::json &value = stmt["value"];
  if (
    value.value("_type", "") == "Subscript" &&
    value["slice"].value("_type", "") == "Slice")
    return &value["value"];
  return &value;
}

// The lambda the module binds to `name` by its only assignment of it.
const nlohmann::json *
module_lambda(const nlohmann::json &body, const std::string &name)
{
  const nlohmann::json *value = nullptr;
  for (const auto &stmt : body)
    if (assigned_name(stmt) == name)
    {
      // A bare annotation, `x: int`, binds nothing.
      if (!stmt.contains("value") || stmt["value"].is_null())
        continue;
      if (value != nullptr)
        return nullptr;
      value = &stmt["value"];
    }
  return value != nullptr && value->value("_type", "") == "Lambda" ? value
                                                                   : nullptr;
}

// Whether every name `node` reads is a parameter or is not bound by the
// module, so the expression means the same wherever it is inlined.
bool reads_only_parameters(
  const nlohmann::json &node,
  const std::set<std::string> &params,
  const nlohmann::json &body)
{
  if (!node.is_object() && !node.is_array())
    return true;
  if (node.is_object() && node.value("_type", "") == "Name")
  {
    const std::string id = node.value("id", "");
    if (params.count(id) != 0)
      return true;
    for (const auto &stmt : body)
      if (assigned_name(stmt) == id)
        return false;
    return true;
  }
  for (const auto &child : node)
    if (!reads_only_parameters(child, params, body))
      return false;
  return true;
}

nlohmann::json substitute_names(
  const nlohmann::json &node,
  const std::map<std::string, nlohmann::json> &values)
{
  if (node.is_object() && node.value("_type", "") == "Name")
  {
    const auto value = values.find(node.value("id", ""));
    return value == values.end() ? node : value->second;
  }
  if (!node.is_object() && !node.is_array())
    return node;
  nlohmann::json out = node;
  for (auto it = out.begin(); it != out.end(); ++it)
    it.value() = substitute_names(it.value(), values);
  return out;
}
// `name[lo:hi:step]` with literal bounds: evaluating it again has no effect.
bool is_literal_view_slice(const nlohmann::json &arg)
{
  if (
    arg.value("_type", "") != "Subscript" ||
    arg["value"].value("_type", "") != "Name" ||
    arg["slice"].value("_type", "") != "Slice")
    return false;
  for (const char *key : {"lower", "upper", "step"})
  {
    const nlohmann::json &bound = arg["slice"][key];
    if (bound.is_null() || bound.value("_type", "") == "Constant")
      continue;
    if (
      bound.value("_type", "") != "UnaryOp" ||
      bound["operand"].value("_type", "") != "Constant")
      return false;
  }
  return true;
}
} // namespace

// Maps each lambda parameter to its argument, which must be a name, a
// constant or a literal slice so that inlining evaluates nothing twice.
// Whether some argument names a view; nullopt when the call cannot be
// inlined.
std::optional<bool> python_converter::bind_lambda_arguments(
  const nlohmann::json &signature,
  const nlohmann::json &args,
  std::map<std::string, nlohmann::json> &values) const
{
  const nlohmann::json &params = signature["args"];
  if (
    params.size() != args.size() || !signature["defaults"].empty() ||
    !signature["vararg"].is_null() || !signature["kwarg"].is_null())
    return std::nullopt;
  bool passes_view = false;
  for (std::size_t i = 0; i < params.size(); ++i)
  {
    const nlohmann::json &arg = args[i];
    const std::string type = arg.value("_type", "");
    if (type != "Name" && type != "Constant" && !is_literal_view_slice(arg))
      return std::nullopt;
    passes_view |= type == "Name" &&
                   find_numpy_pointer_view_info(arg.value("id", "")) != nullptr;
    values[params[i].value("arg", "")] = arg;
  }
  return passes_view;
}

// `f(v)` where the module binds `f` once to a lambda over its own
// parameters: the body with the arguments in place. The arguments must be
// names or constants, so inlining evaluates nothing twice.
std::optional<nlohmann::json>
python_converter::inline_numpy_view_lambda_call(const nlohmann::json &call)
{
  const nlohmann::json &body = (*ast_json)["body"];
  const nlohmann::json *lambda =
    call.value("_type", "") == "Call" &&
        call["func"].value("_type", "") == "Name"
      ? module_lambda(body, call["func"].value("id", ""))
      : nullptr;
  if (
    lambda == nullptr ||
    !call.value("keywords", nlohmann::json::array()).empty())
    return std::nullopt;
  std::map<std::string, nlohmann::json> values;
  std::optional<bool> passes_view =
    bind_lambda_arguments((*lambda)["args"], call["args"], values);
  std::set<std::string> names;
  for (const auto &[name, value] : values)
    names.insert(name);
  if (
    !passes_view ||
    (!*passes_view &&
     numpy_inlined_lambdas_.count(call["func"].value("id", "")) == 0) ||
    !reads_only_parameters((*lambda)["body"], names, body))
    return std::nullopt;
  return substitute_names((*lambda)["body"], values);
}

namespace
{
void collect_name_uses(
  const nlohmann::json &node,
  const std::string &name,
  std::size_t &loads,
  std::vector<const nlohmann::json *> &calls)
{
  if (!node.is_object() && !node.is_array())
    return;
  if (
    node.is_object() && node.value("_type", "") == "Call" &&
    node["func"].value("_type", "") == "Name" &&
    node["func"].value("id", "") == name)
    calls.push_back(&node);
  if (
    node.is_object() && node.value("_type", "") == "Name" &&
    node.value("id", "") == name &&
    node.value("ctx", nlohmann::json::object()).value("_type", "") == "Load")
    ++loads;
  for (const auto &child : node)
    collect_name_uses(child, name, loads, calls);
}

// Whether `arg` is a NumPy view by its syntax: a slice or index of a name,
// or a name the module binds once to one.
bool is_view_argument(const nlohmann::json &arg, const nlohmann::json &body)
{
  if (arg.value("_type", "") == "Subscript")
    return arg["value"].value("_type", "") == "Name";
  if (arg.value("_type", "") != "Name")
    return false;
  const nlohmann::json *value = nullptr;
  for (const auto &stmt : body)
    if (
      assigned_name(stmt) == arg.value("id", "") && stmt.contains("value") &&
      !stmt["value"].is_null())
    {
      if (value != nullptr)
        return false;
      value = &stmt["value"];
    }
  return value != nullptr && value->value("_type", "") == "Subscript";
}
} // namespace

// `f = lambda ...` whose every use is a call passing a view: each call is
// inlined, so the lambda itself, which would type its parameters without
// them, is never converted.
bool python_converter::is_inlined_numpy_view_lambda(
  const nlohmann::json &statement)
{
  const nlohmann::json &body = (*ast_json)["body"];
  const std::string name = assigned_name(statement);
  if (
    name.empty() || !current_func_name_.empty() ||
    module_lambda(body, name) != &statement["value"])
    return false;
  std::size_t loads = 0;
  std::vector<const nlohmann::json *> calls;
  collect_name_uses(body, name, loads, calls);
  std::set<std::string> params;
  for (const auto &param : statement["value"]["args"]["args"])
    params.insert(param.value("arg", ""));
  if (
    calls.empty() || loads != calls.size() ||
    !reads_only_parameters(statement["value"]["body"], params, body))
    return false;
  for (const nlohmann::json *call : calls)
  {
    const nlohmann::json &args = (*call)["args"];
    if (!std::any_of(args.begin(), args.end(), [&](const auto &arg) {
          return is_view_argument(arg, body);
        }))
      return false;
  }
  numpy_inlined_lambdas_.insert(name);
  return true;
}

// Whether `name` is the builtin of that name: the module neither defines nor
// imports a function that shadows it.
bool python_converter::names_python_builtin(const std::string &name) const
{
  const nlohmann::json &body = (*ast_json)["body"];
  if (!json_utils::try_find_function(body, name).empty())
    return false;
  for (const auto &stmt : body)
    if (stmt.value("_type", "") == "ImportFrom" && stmt.contains("names"))
      for (const auto &alias : stmt["names"])
      {
        const std::string bound =
          alias.contains("asname") && !alias["asname"].is_null()
            ? alias["asname"].get<std::string>()
            : alias.value("name", "");
        if (bound == name || bound == "*")
          return false;
      }
  return true;
}

// A call of a list-consuming builtin whose iterable is a 1-D NumPy view.
// Only the keywords whose meaning does not depend on the elements are
// accepted; a `key=` callback would run on them.
bool python_converter::consumes_numpy_view_as_list(
  const nlohmann::json &node) const
{
  if (
    !node.is_object() || node.value("_type", "") != "Call" ||
    !node.contains("func") || node["func"].value("_type", "") != "Name" ||
    list_consuming_builtins().count(node["func"].value("id", "")) == 0 ||
    node["args"].empty() ||
    !names_python_builtin(node["func"]["id"].get<std::string>()))
    return false;
  const std::string name = node["func"]["id"].get<std::string>();
  const auto &keywords = node.value("keywords", nlohmann::json::array());
  return std::all_of(
    keywords.begin(), keywords.end(), [&name](const nlohmann::json &kw) {
      return is_element_independent_keyword(name, kw.value("arg", ""));
    });
}

// A list-consuming builtin, whatever its keywords, applied to a 1-D view or
// array.
bool python_converter::is_builtin_over_numpy_view(
  const nlohmann::json &node) const
{
  return node.is_object() && node.value("_type", "") == "Call" &&
         node.contains("func") && node["func"].value("_type", "") == "Name" &&
         list_consuming_builtins().count(node["func"].value("id", "")) != 0 &&
         !node["args"].empty() &&
         names_python_builtin(node["func"]["id"].get<std::string>()) &&
         (is_rank1_numpy_view(node["args"][0]) ||
          is_rank1_numpy_array_name(node["args"][0]));
}

// Whether `node` is a 1-D NumPy view: a name bound to one, or a slice of
// such a name or of a 1-D array.
bool python_converter::is_rank1_numpy_view(const nlohmann::json &node) const
{
  const std::string type = node.value("_type", "");
  if (type == "Subscript")
    return node["slice"].value("_type", "") == "Slice" &&
           node["value"].value("_type", "") == "Name" &&
           (is_rank1_numpy_view(node["value"]) ||
            is_rank1_numpy_array_name(node["value"]));
  if (type != "Name")
    return false;
  const numpy_scalar_pointer_view_infot *info =
    find_numpy_pointer_view_info(node.value("id", ""));
  return info != nullptr && info->shape.size() <= 1;
}

bool python_converter::is_rank1_numpy_array_name(
  const nlohmann::json &node) const
{
  const std::string id = resolve_name_symbol_id(node.value("id", ""));
  const symbolt *symbol = id.empty() ? nullptr : symbol_table_.find_symbol(id);
  if (symbol == nullptr || numpy_array_symbols_.count(id) == 0)
    return false;
  const typet type = ns.follow(symbol->get_type());
  return type.is_array() && !ns.follow(type.subtype()).is_array();
}

// The builtins read the elements of a numpy array as real numbers.
void python_converter::reject_builtin_over_complex_array(
  const nlohmann::json &node) const
{
  if (
    !is_builtin_over_numpy_view(node) || node["func"].value("id", "") == "list")
    return;
  const nlohmann::json &arg = node["args"][0];
  const std::string id = arg.value("_type", "") == "Name"
                           ? resolve_name_symbol_id(arg.value("id", ""))
                           : std::string();
  const symbolt *symbol = id.empty() ? nullptr : symbol_table_.find_symbol(id);
  if (symbol == nullptr)
    return;
  const typet type = ns.follow(symbol->get_type());
  if (type.is_array() && is_complex_type(ns.follow(type.subtype())))
    throw std::runtime_error(
      "TypeError: passing a numpy complex array to " +
      node["func"]["id"].get<std::string>() + "() is not supported");
}

// Replaces, in the expressions of `node`, each view argument of a
// list-consuming builtin by the list of its elements.
bool python_converter::adapt_numpy_view_builtin_arguments(nlohmann::json &node)
{
  reject_builtin_over_complex_array(node);
  if (!node.is_object() && !node.is_array())
    return false;
  bool changed = false;
  if (node.is_object())
    if (
      std::optional<nlohmann::json> inlined =
        inline_numpy_view_lambda_call(node))
    {
      node = std::move(*inlined);
      changed = true;
    }
  if (is_builtin_over_numpy_view(node) && !consumes_numpy_view_as_list(node))
    throw std::runtime_error(
      "TypeError: passing a numpy view to " +
      node["func"]["id"].get<std::string>() +
      "() with a key or another element-dependent keyword is not supported");
  if (is_builtin_over_numpy_view(node))
  {
    node["args"][0] = tolist_call(node["args"][0]);
    changed = true;
  }
  for (auto it = node.begin(); it != node.end(); ++it)
    if (!node.is_object() || !is_block_key(it.key()))
      changed |= adapt_numpy_view_builtin_arguments(*it);
  return changed;
}

const nlohmann::json &python_converter::adapt_numpy_view_builtin_calls(
  const nlohmann::json &statement,
  nlohmann::json &rewritten)
{
  const std::string kind = statement.value("_type", "");
  if (kind == "FunctionDef" || kind == "ClassDef" || kind == "Try")
    return statement;
  if (is_inlined_numpy_view_lambda(statement))
  {
    rewritten = {{"_type", "Pass"}};
    for (const char *key :
         {"lineno", "col_offset", "end_lineno", "end_col_offset"})
      if (statement.contains(key))
        rewritten[key] = statement[key];
    return rewritten;
  }
  const nlohmann::json *scanned = key_scan_iterable(statement);
  if (
    scanned != nullptr &&
    (is_rank1_numpy_view(*scanned) || is_rank1_numpy_array_name(*scanned)))
  {
    rewritten = statement;
    rewritten["value"] = tolist_call(*scanned);
    return rewritten;
  }
  nlohmann::json copy = statement;
  if (!adapt_numpy_view_builtin_arguments(copy))
    return statement;
  rewritten = std::move(copy);
  return rewritten;
}
