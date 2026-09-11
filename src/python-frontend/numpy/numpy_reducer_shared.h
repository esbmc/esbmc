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
#include <python-frontend/python_converter.h>
#include <util/irep/expr.h>
#include <util/irep/std_expr.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
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

// Cap on descriptor-materialized elements a conversion-time sort/argsort
// bubble-sort network will unroll -- shared between numpy.sort()/argsort()
// and ndarray.sort()/argsort() so both dispatch paths agree on the same
// blow-up bound (ADR-NP-003 principle 3).
constexpr std::size_t max_numpy_sort_elements = 64;

// A conversion-time-unrolled bubble sort over already-converted elements,
// swapping via if_exprt on `keys` rather than extracting a C++ comparison
// key -- the same style reduce_numpy_descriptor_values's own min/max
// branches use, so this works uniformly across every element type get_expr
// can produce (int/float/bool), concrete or symbolic alike. When `payload`
// is non-null, every swap decided by `keys` is mirrored onto it (e.g.
// tracking each element's original index for argsort).
inline void
bubble_sort_numpy_paired(std::vector<exprt> &keys, std::vector<exprt> *payload)
{
  for (std::size_t pass = 0; pass + 1 < keys.size(); ++pass)
  {
    for (std::size_t j = 0; j + pass + 1 < keys.size(); ++j)
    {
      binary_relation_exprt out_of_order(keys[j], ">", keys[j + 1]);
      exprt lo = if_exprt(out_of_order, keys[j + 1], keys[j]);
      exprt hi = if_exprt(out_of_order, keys[j], keys[j + 1]);
      keys[j] = lo;
      keys[j + 1] = hi;

      if (payload != nullptr)
      {
        exprt lo_p = if_exprt(out_of_order, (*payload)[j + 1], (*payload)[j]);
        exprt hi_p = if_exprt(out_of_order, (*payload)[j], (*payload)[j + 1]);
        (*payload)[j] = lo_p;
        (*payload)[j + 1] = hi_p;
      }
    }
  }
}

// Assembles a rank 1 or 2 array_typet value from already-converted,
// row-major flat elements -- the sort/argsort counterpart of
// build_1d_numpy_array_value above, extended to rank 2 so an axis-aware
// result can be reassembled into its original shape.
inline exprt build_numpy_shape_array_value(
  const std::vector<std::size_t> &shape,
  const std::vector<exprt> &elems,
  const type_handler &th)
{
  if (shape.size() == 1)
    return build_1d_numpy_array_value(elems, th);

  const std::size_t cols = shape[1];
  typet row_type = th.build_array(elems.front().type(), cols);
  typet result_type = th.build_array(row_type, shape[0]);
  exprt value = gen_zero(result_type);
  for (std::size_t row = 0; row < shape[0]; ++row)
    for (std::size_t col = 0; col < cols; ++col)
      value.operands().at(row).operands().at(col) = elems[(row * cols) + col];
  return value;
}

// Sorts (or argsorts) already-converted, row-major flat elements from
// build_numpy_descriptor_materialized_elements -- shared by
// numpy.sort()/numpy.argsort() (numpy_call_expr.cpp) and
// ndarray.sort()/argsort() (function_call/expr.cpp), the two dispatch paths
// that both need the exact same axis-aware sort over that descriptor
// materialization. `flatten` wins over `axis` (axis=None); otherwise axis is
// normalized against shape's rank and, for a 2-D shape, each row (axis=1) or
// column (axis=0) is sorted independently, restarting the local index at 0
// per slice -- matching argmin_argmax_axis_best_index's own per-slice
// convention. want_indices selects argsort's index-permutation result over
// sort's value-permutation one.
inline exprt build_numpy_sort_or_argsort_result(
  python_converter &converter,
  const type_handler &th,
  const std::vector<std::size_t> &shape,
  std::vector<exprt> elems,
  bool flatten,
  long long axis,
  bool want_indices)
{
  auto make_index = [&](std::size_t i) {
    nlohmann::json node{
      {"_type", "Constant"},
      {"value", static_cast<int64_t>(i)},
      {"kind", nullptr}};
    return converter.get_expr(node);
  };

  auto sort_slice = [&](std::vector<exprt> values) -> std::vector<exprt> {
    if (!want_indices)
    {
      bubble_sort_numpy_paired(values, nullptr);
      return values;
    }
    std::vector<exprt> indices;
    indices.reserve(values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
      indices.push_back(make_index(i));
    bubble_sort_numpy_paired(values, &indices);
    return indices;
  };

  if (flatten)
    return build_1d_numpy_array_value(sort_slice(std::move(elems)), th);

  const long long normalized = normalize_reducer_axis(axis, shape.size());
  if (shape.size() == 1)
    return build_1d_numpy_array_value(sort_slice(std::move(elems)), th);

  const std::size_t rows = shape[0];
  const std::size_t cols = shape[1];
  std::vector<exprt> out(elems.size());

  if (normalized == 1)
  {
    for (std::size_t r = 0; r < rows; ++r)
    {
      std::vector<exprt> row(
        elems.begin() + static_cast<std::ptrdiff_t>(r * cols),
        elems.begin() + static_cast<std::ptrdiff_t>((r + 1) * cols));
      std::vector<exprt> sorted_row = sort_slice(std::move(row));
      std::copy(
        sorted_row.begin(),
        sorted_row.end(),
        out.begin() + static_cast<std::ptrdiff_t>(r * cols));
    }
  }
  else
  {
    for (std::size_t c = 0; c < cols; ++c)
    {
      std::vector<exprt> col;
      col.reserve(rows);
      for (std::size_t r = 0; r < rows; ++r)
        col.push_back(elems[(r * cols) + c]);
      std::vector<exprt> sorted_col = sort_slice(std::move(col));
      for (std::size_t r = 0; r < rows; ++r)
        out[(r * cols) + c] = sorted_col[r];
    }
  }

  return build_numpy_shape_array_value(shape, out, th);
}
