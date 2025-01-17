#pragma once

#include <util/c_types.h>
#include <nlohmann/json.hpp>

class python_converter;

class type_handler
{
public:
  type_handler(const python_converter &converter);

  bool is_constructor_call(const nlohmann::json &json) const;

  std::string type_to_string(const typet &t) const;

  std::string get_var_type(const std::string &var_name) const;

  typet build_array(const typet &sub_type, const size_t size) const;

  typet get_typet(const std::string &ast_type, size_t type_size = 0) const;

  typet get_typet(const nlohmann::json &elem) const;

  bool has_multiple_types(const nlohmann::json &container) const;

  typet get_list_type(const nlohmann::json &list_value) const;

  // Get the type of an operand in binary operations
  std::string get_operand_type(const nlohmann::json &operand) const;

private:
  const python_converter &converter_;
};
