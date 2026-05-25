#ifndef PYTHON_FRONTEND_STRING_METHOD_DISPATCH_H
#define PYTHON_FRONTEND_STRING_METHOD_DISPATCH_H

#include <python-frontend/string/string_handler.h>
#include <python-frontend/string/string_handler_utils.h>

#include <functional>
#include <optional>
#include <string>

namespace string_method_dispatch
{
using keyword_valuest = string_call_utils::keyword_valuest;

std::optional<exprt> dispatch_decode_join_method(
  string_handler &self,
  const std::string &method_name,
  const nlohmann::json &call_json,
  const nlohmann::json &receiver_json,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  python_converter &converter);
std::optional<exprt> dispatch_no_arg_string_methods(
  string_handler &self,
  const std::string &method_name,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  const std::function<exprt()> &get_receiver_expr,
  const std::function<locationt()> &get_location);
std::optional<exprt> dispatch_one_arg_string_methods(
  string_handler &self,
  const std::string &method_name,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  const std::function<exprt()> &get_receiver_expr,
  const std::function<locationt()> &get_location,
  python_converter &converter);
std::optional<exprt> dispatch_search_string_methods(
  string_handler &self,
  const std::string &method_name,
  const nlohmann::json &call_json,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  const std::function<exprt()> &get_receiver_expr,
  const std::function<locationt()> &get_location,
  python_converter &converter);
std::optional<exprt> dispatch_spacing_and_padding_methods(
  string_handler &self,
  const std::string &method_name,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  const std::function<exprt()> &get_receiver_expr,
  const std::function<locationt()> &get_location,
  python_converter &converter);
std::optional<exprt> dispatch_replace_method(
  string_handler &self,
  const std::string &method_name,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  const std::function<exprt()> &get_receiver_expr,
  const std::function<locationt()> &get_location,
  python_converter &converter);
std::optional<exprt> dispatch_count_method(
  string_handler &self,
  const std::string &method_name,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  const std::function<exprt()> &get_receiver_expr,
  const std::function<locationt()> &get_location,
  python_converter &converter);
std::optional<exprt> dispatch_splitlines_method(
  string_handler &self,
  const std::string &method_name,
  const nlohmann::json &call_json,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  const std::function<exprt()> &get_receiver_expr,
  const std::function<locationt()> &get_location);
std::optional<exprt> dispatch_format_methods(
  string_handler &self,
  const std::string &method_name,
  const nlohmann::json &call_json,
  const keyword_valuest &keyword_values,
  const std::function<exprt()> &get_receiver_expr,
  const std::function<locationt()> &get_location);
std::optional<exprt> dispatch_split_method(
  const std::string &method_name,
  const nlohmann::json &receiver_json,
  const nlohmann::json &call_json,
  const nlohmann::json &args,
  const keyword_valuest &keyword_values,
  const std::function<exprt()> &get_receiver_expr,
  python_converter &converter);
} // namespace string_method_dispatch

#endif
