#include <python-frontend/math_guard_utils.h>

#include <python-frontend/json_utils.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/type_handler.h>

namespace
{
constexpr std::size_t MAX_COMPLEX_SCAN_DEPTH = 16;

bool is_complex_annotation(const nlohmann::json &node)
{
  return node.contains("esbmc_type_annotation") &&
         node["esbmc_type_annotation"] == "complex";
}

bool json_arg_contains_or_is_complex_impl(
  const nlohmann::json &arg_json,
  const python_converter &converter,
  const type_handler &type_handler,
  const std::string &current_function,
  std::size_t depth)
{
  if (depth > MAX_COMPLEX_SCAN_DEPTH)
    return true;

  if (is_complex_annotation(arg_json))
    return true;

  if (!arg_json.is_object() || !arg_json.contains("_type"))
    return false;

  const std::string node_type = arg_json["_type"].get<std::string>();

  if (node_type == "Name" && arg_json.contains("id"))
  {
    const std::string var_name = arg_json["id"].get<std::string>();
    if (type_handler.get_var_type(var_name) == "complex")
      return true;

    const nlohmann::json var_decl_or_arg =
      json_utils::get_var_value(var_name, current_function, converter.ast());

    if (!var_decl_or_arg.empty())
    {
      if (
        var_decl_or_arg.contains("_type") &&
        var_decl_or_arg["_type"] == "arg" &&
        var_decl_or_arg.contains("annotation") &&
        var_decl_or_arg["annotation"].is_object() &&
        var_decl_or_arg["annotation"].contains("id") &&
        var_decl_or_arg["annotation"]["id"] == "complex")
      {
        return true;
      }

      if (
        var_decl_or_arg.contains("value") &&
        json_arg_contains_or_is_complex_impl(
          var_decl_or_arg["value"],
          converter,
          type_handler,
          current_function,
          depth + 1))
      {
        return true;
      }
    }
    return false;
  }

  if (
    node_type == "Call" && arg_json.contains("func") &&
    arg_json["func"].contains("_type") && arg_json["func"]["_type"] == "Name" &&
    arg_json["func"].contains("id"))
  {
    const std::string called_name = arg_json["func"]["id"].get<std::string>();
    if (called_name == "complex")
      return true;

    const nlohmann::json &func_node =
      json_utils::find_function(converter.ast()["body"], called_name);

    if (!func_node.empty())
    {
      if (
        func_node.contains("returns") && !func_node["returns"].is_null() &&
        func_node["returns"].contains("id") &&
        func_node["returns"]["id"] == "complex")
      {
        return true;
      }

      if (func_node.contains("body") && func_node["body"].is_array())
      {
        for (const auto &stmt : func_node["body"])
        {
          if (
            stmt.contains("_type") && stmt["_type"] == "Return" &&
            stmt.contains("value") && !stmt["value"].is_null() &&
            json_arg_contains_or_is_complex_impl(
              stmt["value"], converter, type_handler, called_name, depth + 1))
          {
            return true;
          }
        }
      }
    }
  }

  // Generic recursive walk so kwargs merges / BinOp / nested expressions are covered.
  for (const auto &[key, value] : arg_json.items())
  {
    if (key == "ctx" || key == "_type")
      continue;

    if (value.is_object())
    {
      if (json_arg_contains_or_is_complex_impl(
            value, converter, type_handler, current_function, depth + 1))
      {
        return true;
      }
    }
    else if (value.is_array())
    {
      for (const auto &elem : value)
      {
        if (
          elem.is_object() &&
          json_arg_contains_or_is_complex_impl(
            elem, converter, type_handler, current_function, depth + 1))
        {
          return true;
        }
      }
    }
  }

  return false;
}
} // namespace

namespace math_guard_utils
{
const std::unordered_set<std::string> &math_wrapper_function_names()
{
  static const std::unordered_set<std::string> names = {
    "__ESBMC_sin",   "__ESBMC_cos",      "__ESBMC_sqrt", "__ESBMC_exp",
    "__ESBMC_log",   "__ESBMC_acos",     "__ESBMC_atan", "__ESBMC_atan2",
    "__ESBMC_log2",  "__ESBMC_pow",      "__ESBMC_fabs", "__ESBMC_trunc",
    "__ESBMC_fmod",  "__ESBMC_copysign", "__ESBMC_tan",  "__ESBMC_asin",
    "__ESBMC_sinh",  "__ESBMC_cosh",     "__ESBMC_tanh", "__ESBMC_log10",
    "__ESBMC_expm1", "__ESBMC_log1p",    "__ESBMC_exp2", "__ESBMC_asinh",
    "__ESBMC_acosh", "__ESBMC_atanh",    "__ESBMC_hypot"};
  return names;
}

const std::unordered_set<std::string> &math_module_function_names()
{
  static const std::unordered_set<std::string> names = {
    "sin",   "cos",      "sqrt",      "exp",       "log",     "acos",
    "atan",  "atan2",    "log2",      "pow",       "fabs",    "trunc",
    "fmod",  "copysign", "tan",       "asin",      "sinh",    "cosh",
    "tanh",  "log10",    "expm1",     "log1p",     "exp2",    "asinh",
    "acosh", "atanh",    "hypot",     "floor",     "ceil",    "factorial",
    "gcd",   "lcm",      "isqrt",     "perm",      "prod",    "isclose",
    "isinf", "isnan",    "isfinite",  "degrees",   "radians", "modf",
    "cbrt",  "erf",      "erfc",      "frexp",     "fsum",    "gamma",
    "ldexp", "lgamma",   "nextafter", "remainder", "sumprod", "ulp",
    "dist"};
  return names;
}

const std::unordered_set<std::string> &math_guard_real_general_functions()
{
  static const std::unordered_set<std::string> names = {
    "floor",
    "ceil",
    "degrees",
    "radians",
    "modf",
    "cbrt",
    "erf",
    "erfc",
    "frexp",
    "gamma",
    "lgamma",
    "ulp",
    "isinf",
    "isnan",
    "isfinite"};
  return names;
}

const std::unordered_set<std::string> &math_guard_int_general_functions()
{
  static const std::unordered_set<std::string> names = {
    "factorial", "isqrt", "gcd", "lcm", "perm"};
  return names;
}

const std::unordered_set<std::string> &
math_guard_real_general_twoarg_functions()
{
  static const std::unordered_set<std::string> names = {
    "ldexp", "nextafter", "remainder", "isclose"};
  return names;
}

bool json_arg_contains_or_is_complex(
  const nlohmann::json &arg_json,
  const python_converter &converter,
  const type_handler &type_handler,
  const std::string &current_function)
{
  return json_arg_contains_or_is_complex_impl(
    arg_json, converter, type_handler, current_function, 0);
}

bool call_has_complex_in_args_or_keywords(
  const nlohmann::json &call,
  const python_converter &converter,
  const type_handler &type_handler,
  const std::string &current_function)
{
  if (call.contains("args") && call["args"].is_array())
  {
    for (const auto &arg : call["args"])
    {
      if (json_arg_contains_or_is_complex(
            arg, converter, type_handler, current_function))
      {
        return true;
      }
    }
  }

  if (call.contains("keywords") && call["keywords"].is_array())
  {
    for (const auto &kw : call["keywords"])
    {
      if (
        kw.contains("value") &&
        json_arg_contains_or_is_complex(
          kw["value"], converter, type_handler, current_function))
      {
        return true;
      }
    }
  }

  return false;
}

bool call_first_cpp_throw_in_args_or_keywords(
  const nlohmann::json &call,
  python_converter &converter,
  exprt &throw_expr)
{
  auto is_cpp_throw = [](const exprt &e) -> bool {
    return e.statement() == "cpp-throw";
  };

  if (call.contains("args") && call["args"].is_array())
  {
    for (const auto &arg_json : call["args"])
    {
      exprt arg_expr = converter.get_expr(arg_json);
      if (is_cpp_throw(arg_expr))
      {
        throw_expr = arg_expr;
        return true;
      }
    }
  }

  if (call.contains("keywords") && call["keywords"].is_array())
  {
    for (const auto &kw_json : call["keywords"])
    {
      if (!kw_json.contains("value"))
        continue;

      exprt kw_expr = converter.get_expr(kw_json["value"]);
      if (is_cpp_throw(kw_expr))
      {
        throw_expr = kw_expr;
        return true;
      }
    }
  }

  return false;
}
} // namespace math_guard_utils
