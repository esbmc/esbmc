#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <unordered_set>

class python_converter;
class type_handler;
class exprt;

namespace math_guard_utils
{
const std::unordered_set<std::string> &math_wrapper_function_names();

const std::unordered_set<std::string> &math_module_function_names();

const std::unordered_set<std::string> &math_guard_real_general_functions();

const std::unordered_set<std::string> &math_guard_int_general_functions();

const std::unordered_set<std::string> &
math_guard_real_general_twoarg_functions();

bool json_arg_contains_or_is_complex(
  const nlohmann::json &arg_json,
  const python_converter &converter,
  const type_handler &type_handler,
  const std::string &current_function);

bool call_has_complex_in_args_or_keywords(
  const nlohmann::json &call,
  const python_converter &converter,
  const type_handler &type_handler,
  const std::string &current_function);

bool call_first_cpp_throw_in_args_or_keywords(
  const nlohmann::json &call,
  python_converter &converter,
  exprt &throw_expr);
} // namespace math_guard_utils
