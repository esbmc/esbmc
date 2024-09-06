#pragma once

#include <string>

class python_module
{
public:
  python_module(const std::string& module_name);

  bool is_standard_module() const;

  std::string get_function_return(const std::string &function_name) const;

private:
  std::string module_name_;
};
