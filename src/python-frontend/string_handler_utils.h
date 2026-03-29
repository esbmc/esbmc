#ifndef PYTHON_FRONTEND_STRING_HANDLER_UTILS_H
#define PYTHON_FRONTEND_STRING_HANDLER_UTILS_H

#include <nlohmann/json.hpp>
#include <functional>
#include <initializer_list>
#include <string>
#include <unordered_map>

class python_converter;

namespace string_call_utils
{
using keyword_valuest = std::unordered_map<std::string, const nlohmann::json *>;

keyword_valuest collect_keyword_values(
  const std::string &method_name,
  const nlohmann::json &keywords);

const nlohmann::json *find_keyword_value(
  const keyword_valuest &keyword_values,
  const std::string &name);

void ensure_allowed_keywords(
  const std::string &method_name,
  const keyword_valuest &keyword_values,
  std::initializer_list<const char *> allowed);

const nlohmann::json *resolve_positional_or_keyword_arg(
  const std::string &method_name,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  const std::string &arg_name,
  std::size_t positional_index,
  bool required);

const nlohmann::json *required_arg_node_or_throw(
  const std::string &method_name,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  const std::string &arg_name,
  std::size_t positional_index,
  const std::string &error_message);

long long optional_constant_int_arg_or_default(
  const nlohmann::json *arg_node,
  long long default_value,
  python_converter &converter);

long long required_constant_int_arg(
  const nlohmann::json &arg_node,
  const std::string &error_message,
  python_converter &converter);

std::string optional_constant_string_arg_or_default(
  const nlohmann::json *arg_node,
  const std::string &default_value,
  python_converter &converter,
  const std::function<bool(const nlohmann::json &, std::string &)>
    &extract_constant_string_cb,
  const std::function<bool(const nlohmann::json &)> &is_none_literal);
} // namespace string_call_utils

#endif
