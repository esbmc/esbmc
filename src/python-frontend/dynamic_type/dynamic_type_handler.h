#pragma once

#include <util/irep/type.h>
#include <util/irep/expr.h>
#include <util/irep/std_code.h>
#include <util/irep/location.h>
#include <util/symtab/symbol.h>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class python_converter;
class type_handler;


class dynamic_type_handler
{
public:
  dynamic_type_handler(python_converter &converter, type_handler &type_handler);

  /**
   * @brief Names assigned genuinely incompatible literal types across
   * every branch of `if_node`, following nested if/else (including elif)
   * on either side; a name must be assigned in every branch to qualify
   */
  std::unordered_set<std::string>
  detect_dynamic_type_names(const nlohmann::json &if_node) const;

  /**
   * @brief Declares (or aliases) each name's tagged-object symbol before
   * either branch converts, so goto-symex's struct-merge resolves the join
   * for free
   */
  void declare_dynamic_type_names(
    const std::unordered_set<std::string> &dynamic_type_names,
    const nlohmann::json &ast_node);

  /**
   * @brief True if `name` currently resolves to a tagged-object symbol,
   * either because it is one of the current if/else's dynamic-type names, or
   * because an earlier branch join already flagged it
   */
  bool is_tagged(const std::string &name) const;

  /**
   * @brief The id of the symbol that currently holds `name`'s tagged object,
   * following the alias created for a pre-existing variable
   */
  std::string tagged_symbol_id(const std::string &name) const;

  /**
   * @brief True if `op` on two tagged operands can produce either scalar
   * type at runtime (only '+' today); an untagged assignment target then
   * needs to become tagged too
   */
  bool tagged_binop_result_may_be_tagged(const std::string &op) const;

  /**
   * @brief Fills in the tagged-object fields for `name` from an
   * already-converted value
   */
  void assign(
    const exprt &rhs,
    const locationt &location,
    const std::string &name,
    codet &target_block);

  /**
   * @brief Redirects `symbol` to its tagged-object alias, if a pre-existing
   * variable was aliased for it; no-op otherwise
   */
  void resolve_read(symbolt *&symbol) const;

  /**
   * @brief Dispatches Eq/NotEq/Lt/LtE/Gt/GtE between a tagged-scalar
   * operand and the other side
   */
  exprt
  handle_comparison(const std::string &op, const exprt &lhs, const exprt &rhs);

  /**
   * @brief Dispatches Add/Sub/Div between a tagged-scalar operand and a
   * literal
   */
  exprt handle_arithmetic(
    const std::string &op,
    const exprt &lhs,
    const exprt &rhs,
    const locationt &location);

  /**
   * @brief Builds isinstance(tagged, type_name). `type_is_user_class` says
   * whether `type_name` names a class defined in the program, which a tag can
   * never hold
   */
  exprt build_isinstance_check(
    const exprt &tagged,
    const std::string &type_name,
    bool type_is_user_class) const;

  /**
   * @brief True if a function's top-level if/elif/else chain -- including
   * an early return with no else, falling through to the next statement --
   * can return genuinely incompatible literal kinds (num vs str)
   */
  bool detect_dynamic_return_type(const nlohmann::json &function_body) const;

  /**
   * @brief Builds a tagged-object temporary from an already-converted
   * value, for use as a RETURN value
   */
  exprt build_tagged_return_value(
    const exprt &value,
    const locationt &location,
    codet &target_block);

  /**
   * @brief Whole-struct copy: fills `name`'s tagged-object symbol from a
   * value that is already tagged-object-typed
   */
  void assign_tagged_object(
    const exprt &rhs,
    const locationt &location,
    const std::string &name,
    codet &target_block);

  /**
   * @brief RAII: adds `dynamic_type_names` to the transient tagged-name set
   * for the duration of converting one if/else's branches; removes on exit
   * only what it added, keeping nested invocations independent
   */
  class scope_guard
  {
  public:
    scope_guard(
      dynamic_type_handler &handler,
      const std::unordered_set<std::string> &dynamic_type_names);
    ~scope_guard();

  private:
    dynamic_type_handler &handler_;
    std::unordered_set<std::string> added_;
  };

private:
  /**
   * @brief Builds the instructions that fill a tagged-object symbol's four
   * fields (value/type_id/size/float_idx) from an already-converted value.
   * Shared by assign() (literal or non-literal reassignment) and
   * declare_dynamic_type_names() (initializing a pre-existing alias from
   * its prior value)
   */
  std::vector<codet> build_tag_field_assigns(
    symbolt &tag_symbol,
    const exprt &rhs,
    const locationt &location);

  /**
   * @brief Resolves `name` to its tagged-object symbol, following `aliases_`
   * if it was aliased from a pre-existing variable. Shared by assign() and
   * assign_tagged_object(); asserts if the symbol isn't declared yet
   */
  symbolt *find_tag_symbol(const std::string &name) const;

  /**
   * @brief Calls the num/str helper matching `literal`'s type (dispatch and
   * argument-building shared by every tagged-vs-literal Eq/ordered compare)
   * and returns its raw int result, undecoded
   */
  exprt build_literal_compare_call(
    const std::string &num_func_name,
    const std::string &str_func_name,
    const exprt &tagged,
    const exprt &literal);

  exprt build_eq_literal(const exprt &tagged, const exprt &literal);
  exprt build_ordered_literal(
    const std::string &op,
    const exprt &tagged,
    const exprt &literal);
  exprt build_add_literal(
    const exprt &tagged,
    const exprt &literal,
    bool tagged_is_left);
  exprt build_sub_literal(
    const exprt &tagged,
    const exprt &literal,
    bool tagged_is_left);
  exprt build_div_literal(
    const exprt &tagged,
    const exprt &literal,
    bool tagged_is_left,
    const locationt &location);
  exprt build_add_tagged(const exprt &lhs, const exprt &rhs);
  exprt build_sub_tagged(const exprt &lhs, const exprt &rhs);
  exprt build_div_tagged(
    const exprt &lhs,
    const exprt &rhs,
    const locationt &location);

  /**
   * @brief Emits a catchable ZeroDivisionError raise, guarded so it only
   * fires when `type_ok` holds and `divisor` is zero
   */
  void guard_zero_division(
    const exprt &divisor,
    const exprt &type_ok,
    const locationt &location);

  python_converter &converter_;
  type_handler &type_handler_;

  // Names of variables flagged as needing the tagged-object representation
  // for the duration of converting one if/else's branches.
  std::unordered_set<std::string> tagged_names_;

  // Original symbol id -> tagged-object symbol id, for a variable that
  // already had a native-typed symbol when it was flagged. Unlike the
  // straight-line retype heuristic's alias, never reverted: the whole
  // point is surviving the join.
  std::unordered_map<std::string, std::string> aliases_;
};

