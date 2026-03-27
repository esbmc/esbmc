#include <python-frontend/json_utils.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/string_handler_utils.h>

#include <stdexcept>

namespace string_call_utils
{
keyword_valuest collect_keyword_values(
  const std::string &method_name,
  const nlohmann::json &keywords)
{
  keyword_valuest keyword_values;
  keyword_values.reserve(keywords.size());

  for (const auto &kw : keywords)
  {
    if (!kw.contains("arg") || !kw["arg"].is_string() || !kw.contains("value"))
      continue;

    const std::string keyword_name = kw["arg"].get<std::string>();
    if (!keyword_values.emplace(keyword_name, &kw["value"]).second)
    {
      throw std::runtime_error(
        method_name + "() got multiple values for keyword argument '" +
        keyword_name + "'");
    }
  }

  return keyword_values;
}

const nlohmann::json *find_keyword_value(
  const keyword_valuest &keyword_values,
  const std::string &name)
{
  auto it = keyword_values.find(name);
  return it == keyword_values.end() ? nullptr : it->second;
}

void ensure_allowed_keywords(
  const std::string &method_name,
  const keyword_valuest &keyword_values,
  std::initializer_list<const char *> allowed)
{
  for (const auto &[keyword_name, _] : keyword_values)
  {
    bool is_allowed = false;
    for (const char *allowed_name : allowed)
    {
      if (keyword_name == allowed_name)
      {
        is_allowed = true;
        break;
      }
    }

    if (!is_allowed)
    {
      throw std::runtime_error(
        method_name + "() got an unexpected keyword argument '" + keyword_name +
        "'");
    }
  }
}

const nlohmann::json *resolve_positional_or_keyword_arg(
  const std::string &method_name,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  const std::string &arg_name,
  std::size_t positional_index,
  bool required)
{
  const nlohmann::json *kw_arg = find_keyword_value(keyword_values, arg_name);
  if (args.size() > positional_index && kw_arg != nullptr)
  {
    throw std::runtime_error(
      method_name + "() got multiple values for argument '" + arg_name + "'");
  }
  if (args.size() > positional_index)
    return &args[positional_index];
  if (kw_arg != nullptr)
    return kw_arg;
  if (required)
  {
    throw std::runtime_error(
      method_name + "() missing required argument '" + arg_name + "'");
  }
  return nullptr;
}

const nlohmann::json *required_arg_node_or_throw(
  const std::string &method_name,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  const std::string &arg_name,
  std::size_t positional_index,
  const std::string &error_message)
{
  const nlohmann::json *node = resolve_positional_or_keyword_arg(
    method_name, args, keyword_values, arg_name, positional_index, false);
  if (node == nullptr)
    throw std::runtime_error(error_message);
  return node;
}

long long optional_constant_int_arg_or_default(
  const nlohmann::json *arg_node,
  long long default_value,
  python_converter &converter)
{
  if (arg_node == nullptr)
    return default_value;

  long long value = 0;
  if (!json_utils::extract_constant_integer(
        *arg_node,
        converter.get_current_func_name(),
        converter.get_ast_json(),
        value))
  {
    return default_value;
  }
  return value;
}

long long required_constant_int_arg(
  const nlohmann::json &arg_node,
  const std::string &error_message,
  python_converter &converter)
{
  long long value = 0;
  if (!json_utils::extract_constant_integer(
        arg_node,
        converter.get_current_func_name(),
        converter.get_ast_json(),
        value))
  {
    throw std::runtime_error(error_message);
  }
  return value;
}

std::string optional_constant_string_arg_or_default(
  const nlohmann::json *arg_node,
  const std::string &default_value,
  python_converter &converter,
  const std::function<bool(const nlohmann::json &, std::string &)>
    &extract_constant_string_cb,
  const std::function<bool(const nlohmann::json &)> &is_none_literal)
{
  if (arg_node == nullptr || is_none_literal(*arg_node))
    return default_value;

  std::string value;
  if (!extract_constant_string_cb(*arg_node, value))
    return default_value;
  return value;
}
} // namespace string_call_utils
