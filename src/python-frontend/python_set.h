#pragma once

#include <nlohmann/json.hpp>
#include <util/type.h>
#include <util/expr.h>
#include <util/symbol.h>

class python_converter;

/**
 * @class python_set
 * @brief Handles Python set operations and set-specific functionality
 * 
 * This class is responsible for converting Python set operations into
 * ESBMC's intermediate representation. Sets are internally represented
 * as lists with unique elements.
 */
class python_set
{
public:
  python_set(python_converter &converter, const nlohmann::json &set_node)
    : converter_(converter), set_value_(set_node)
  {
  }

  /**
   * @brief Create a set from a set literal
   * @return Expression representing the set
   */
  exprt get();

  /**
   * @brief Create an empty set
   * @return Expression representing the empty set
   */
  exprt get_empty_set();

  /**
   * @brief Build set difference operation (set1 - set2)
   * @param lhs Left operand (set expression)
   * @param rhs Right operand (set expression)
   * @param element AST node for location information
   * @return Expression representing the result set
   */
  exprt build_set_difference_call(
    const exprt &lhs,
    const exprt &rhs,
    const nlohmann::json &element);

  /**
   * @brief Build set intersection operation (set1 & set2)
   * @param lhs Left operand (set expression)
   * @param rhs Right operand (set expression)
   * @param element AST node for location information
   * @return Expression representing the result set
   */
  exprt build_set_intersection_call(
    const exprt &lhs,
    const exprt &rhs,
    const nlohmann::json &element);

  /**
   * @brief Build set union operation (set1 | set2)
   * @param lhs Left operand (set expression)
   * @param rhs Right operand (set expression)
   * @param element AST node for location information
   * @return Expression representing the result set
   */
  exprt build_set_union_call(
    const exprt &lhs,
    const exprt &rhs,
    const nlohmann::json &element);

private:
  /**
   * @brief Create an internal list representation for the set
   * @return Symbol representing the underlying list structure
   */
  symbolt &create_set_list();

  python_converter &converter_;
  const nlohmann::json &set_value_;
};
