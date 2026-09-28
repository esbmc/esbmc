#pragma once

#include <nlohmann/json.hpp>
#include <util/irep/expr.h>

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
   * Checks if __ESBMC_assert function is being invoked.
   */
  bool is_assert_call(const symbol_id &function_id) const;

  /*
   * Checks if __ESBMC_cover function is being invoked.
   */
  bool is_cover_call(const symbol_id &function_id) const;

  /*
   * Checks if __ESBMC_unreachable function is being invoked.
   */
  bool is_unreachable_call(const symbol_id &function_id) const;

  /*
   * Checks if the Python len() function is being invoked.
   */
  bool is_len_call(const symbol_id &function_id) const;

  /*
   * len() of an operand that lowered to a bitvector: length 1 for a single
   * character, TypeError for a number, which defines no __len__ (#6261).
   */
  exprt len_of_bitvector_operand(const nlohmann::json &arg) const;

  /*
   * Checks if a NumPy function is being invoked.
   */
  bool is_numpy_call(const symbol_id &function_id) const;

private:
  /*
   * Validates a call to __ESBMC_requires / __ESBMC_ensures; a no-op for any
   * other callee.
   */
  void check_contract_call(const symbol_id &function_id) const;

  /*
   * Rejects a contract clause that does not lower to a pure expression.
   * Throws with a diagnostic naming the offending construct.
   */
  /*
   * Rejects a name a clause may not mention: the return value in a
   * precondition or in a None-returning function, or a value whose type
   * could not be determined.
   */
  void check_clause_name(const nlohmann::json &node, const std::string &clause)
    const;

  void check_contract_clause(
    const nlohmann::json &node,
    const std::string &clause) const;

  /*
   * Validates a contract-intrinsic call and, for the ones that denote a value,
   * builds it. Registers the bodiless symbol the clause intrinsics need.
   * Returns nullopt for any callee that is not one, or that denotes no value.
   */
  std::optional<exprt>
  build_contract_intrinsic(const symbol_id &function_id) const;

  /*
   * __ESBMC_old(x) as the C macro spells it: the value x held before the body
   * ran. Throws when x is not a scalar the frontend can snapshot.
   */
  exprt build_old_snapshot() const;

  /*
   * Whether the name is a parameter of the function being converted, or a
   * module-level global. Only those have a value from before the body ran.
   */
  bool names_enclosing_parameter(const std::string &name) const;
  bool names_module_global(const std::string &name) const;

  /*
   * Return type of the function currently being converted; nil when there is
   * no enclosing function.
   */
  typet enclosing_return_type() const;

  python_converter &converter_;
  const nlohmann::json &call_;
};
