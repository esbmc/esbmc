#pragma once

// Small numeric/AST helpers shared between numpy_call_expr.cpp (the
// np.<reducer>(...) free-function forms) and function_call/expr.cpp (the
// a.<reducer>(...) method forms). Both translation units need the exact same
// literal-axis parsing, "flatten" (axis=None) detection, axis normalization
// and 1-D result construction -- keeping one copy here avoids the two
// dispatch paths drifting apart on what counts as a supported spelling.

#include <python-frontend/type/type_handler.h>
#include <nlohmann/json.hpp>
#include <python-frontend/math/convert_float_literal.h>
#include <util/irep/expr.h>
#include <util/irep/std_expr.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

struct numeric_value
{
  bool is_int = true;
  int64_t int_value = 0;
  double double_value = 0.0;
};

inline numeric_value make_int_value(int64_t value)
{
  return {true, value, static_cast<double>(value)};
}

inline numeric_value make_float_value(double value)
{
  return {false, 0, value};
}

inline double to_double(const numeric_value &value)
{
  return value.is_int ? static_cast<double>(value.int_value)
                      : value.double_value;
}

inline numeric_value extract_value(const nlohmann::json &arg)
{
  if (!arg.contains("_type"))
    throw std::runtime_error("Invalid JSON: missing _type");

  if (arg["_type"] == "UnaryOp")
  {
    if (!arg.contains("operand") || !arg["operand"].contains("value"))
      throw std::runtime_error("Invalid UnaryOp: missing operand/value");

    auto operand = arg["operand"]["value"];
    if (operand.is_number_integer())
      return make_int_value(-operand.get<int64_t>());
    if (operand.is_number_float())
      return make_float_value(-operand.get<double>());
  }

  if (!arg.contains("value"))
    throw std::runtime_error("Invalid JSON: missing value");

  auto value = arg["value"];
  if (value.is_boolean())
    return make_int_value(value.get<bool>() ? 1 : 0);
  if (value.is_number_integer())
    return make_int_value(value.get<int64_t>());
  if (value.is_number_float())
    return make_float_value(value.get<double>());

  // A non-finite literal arrives with a nulled value and a spelling tag, so
  // the number checks above all miss it (#7545).
  if (arg.contains("value_nonfinite"))
    return make_float_value(
      nonfinite_float_from_spelling(arg["value_nonfinite"].get<std::string>())
        ->to_double());

  throw std::runtime_error("Unknown numeric type in JSON");
}

// The boolean try_extract_* helpers must not depend on catching an exception
// for control flow: extract_value() raises std::runtime_error on non-numeric
// input, and relying on that as a flow-control signal is fragile. Pre-check
// that the payload is numeric and only call extract_value() when it is
// guaranteed to succeed, so a non-numeric literal (e.g. a str element in
// numpy.linalg.det's matrix) makes this helper return false cleanly instead
// of letting the internal "Unknown numeric type" error escape to the user
// (issue #5206).
inline bool
try_extract_numeric_constant(const nlohmann::json &node, numeric_value &out)
{
  if (!node.is_object() || !node.contains("_type"))
    return false;

  const std::string type = node["_type"];

  if (type == "UnaryOp")
  {
    if (
      !node.contains("operand") || !node["operand"].is_object() ||
      !node["operand"].contains("value"))
      return false;
    // extract_value() only negates integer/float operands.
    const auto &operand = node["operand"]["value"];
    if (!operand.is_number_integer() && !operand.is_number_float())
      return false;
  }
  else if (type == "Constant")
  {
    if (!node.contains("value"))
      return false;
    const auto &value = node["value"];
    if (
      !value.is_boolean() && !value.is_number_integer() &&
      !value.is_number_float())
      return false;
  }
  else
    return false;

  out = extract_value(node);
  return true;
}

// True for the JSON shape a Python `None` literal takes in this AST (a
// Constant node whose value is JSON null) -- NOT a bare JSON null, which
// never occurs at this position in a well-formed AST.
inline bool is_json_none_literal(const nlohmann::json &node)
{
  return node.is_object() && node.contains("_type") &&
         node["_type"] == "Constant" && node.contains("value") &&
         node["value"].is_null();
}

// True when `call` carries any keyword besides "axis" -- used once an axis=
// value has already been accepted (or explicitly declined as None), to
// still reject keepdims/where/out/initial/dtype alongside it.
inline bool
numpy_reducer_has_unsupported_keywords_besides_axis(const nlohmann::json &call)
{
  if (!call.contains("keywords"))
    return false;
  for (const auto &kw : call["keywords"])
    if (kw.value("arg", "") != "axis")
      return true;
  return false;
}

// Builds a 1-D array_typet value from already-converted elements.
inline exprt build_1d_numpy_array_value(
  const std::vector<exprt> &elems,
  const type_handler &th)
{
  typet result_type = th.build_array(elems.front().type(), elems.size());
  exprt value = gen_zero(result_type);
  for (std::size_t i = 0; i < elems.size(); ++i)
    value.operands().at(i) = elems[i];
  return value;
}

// Normalizes a literal axis against a known rank and validates it lands in
// range, throwing AxisError otherwise.
inline long long normalize_reducer_axis(long long axis, std::size_t rank)
{
  const long long normalized =
    axis < 0 ? axis + static_cast<long long>(rank) : axis;
  if (normalized < 0 || normalized >= static_cast<long long>(rank))
    throw std::runtime_error(
      "AxisError: axis " + std::to_string(axis) +
      " is out of bounds for array of dimension " + std::to_string(rank));
  return normalized;
}
