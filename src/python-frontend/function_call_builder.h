#pragma once

#include <nlohmann/json.hpp>
#include <util/expr.h>

class python_converter;
class symbol_id;

class function_call_builder
{
public:
  function_call_builder(
    python_converter &converter,
    const nlohmann::json &call);

  exprt build() const;

  /*
   * Extracts information from the call to populate the function_id attribute.
   */
  symbol_id build_function_id() const;

  /*
   * Checks if assume (__ESBMC_assume or __VERIFIER_assume) function is being invoked.
   */
  bool is_assume_call(const symbol_id &function_id) const;

  /*
   * Checks if the Python len() function is being invoked.
   */
  bool is_len_call(const symbol_id &function_id) const;

  /*
   * Checks if a NumPy function is being invoked.
   */
  bool is_numpy_call(const symbol_id &function_id) const;

private:
  python_converter &converter_;
  const nlohmann::json &call_;
};
