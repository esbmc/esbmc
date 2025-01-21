#pragma once

#include <nlohmann/json.hpp>
#include <util/expr.h>

class python_converter;

class function_call_builder
{
public:
  function_call_builder(python_converter &converter);

  exprt build(const nlohmann::json &call) const;

private:
  python_converter& converter_;
};
