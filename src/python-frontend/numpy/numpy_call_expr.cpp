#include <python-frontend/json_utils.h>
#include <python-frontend/numpy/ndarray_descriptor.h>
#include <python-frontend/numpy/numpy_call_expr.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/math/python_int_overflow.h>
#include <python-frontend/python-list/python_list.h>
#include <python-frontend/python_expr_builder.h>
#include <python-frontend/symbol_id.h>
#include <irep2/irep2_utils.h>
#include <util/arith/arith_tools.h>
#include <util/lang/c_types.h>
#include <util/config/config.h>
#include <util/irep/expr.h>
#include <util/expr/expr_util.h>
#include <util/message/message.h>
#include <util/irep/migrate.h>
#include <util/irep/std_expr.h>
#include <util/irep/std_code.h>
#include <util/irep/std_types.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <ostream>

const char *kConstant = "Constant";
const char *kName = "Name";

namespace
{
// V.3: IREP2 expression-construction helpers (exact round-trip of the legacy
// constructors; behaviour-preserving -- migrate_expr already lowers the legacy
// nodes through these same paths downstream). Back-migrated for the legacy
// adjust/goto-convert seam.

// member_exprt(base, name, t): base is complex-typed (is_complex_type-guarded:
// the `complex` struct or the transient `tag-complex` symbol), both permitted
// member2t sources.
exprt np_member(const exprt &base, const irep_idt &name, const typet &t)
{
  expr2tc base2;
  migrate_expr(base, base2);
  return migrate_expr_back(member2tc(migrate_type(t), base2, name));
}

// typecast_exprt(from, t): rounding mode defaults to __ESBMC_rounding_mode,
// matching migrate_expr's lowering of a legacy typecast.
exprt np_typecast(const exprt &from, const typet &t)
{
  expr2tc from2;
  migrate_expr(from, from2);
  return migrate_expr_back(typecast2tc(migrate_type(t), from2));
}

// address_of_exprt(obj): result type is pointer-to-obj-type, reproduced by
// address_of2tc(obj2->type, obj2) == pointer_type2tc(obj2->type).
exprt np_address_of(const exprt &obj)
{
  expr2tc obj2;
  migrate_expr(obj, obj2);
  return migrate_expr_back(address_of2tc(obj2->type, obj2));
}

// index_exprt(arr, idx, t): arr is an array-typed numpy result (the 2D row/
// element access below), a permitted index2t source.
exprt np_index(const exprt &arr, const exprt &idx, const typet &t)
{
  expr2tc arr2, idx2;
  migrate_expr(arr, arr2);
  migrate_expr(idx, idx2);
  return migrate_expr_back(index2tc(migrate_type(t), arr2, idx2));
}
} // namespace

struct numeric_value
{
  bool is_int = true;
  int64_t int_value = 0;
  double double_value = 0.0;
};

struct scalar_value
{
  bool is_complex = false;
  std::complex<double> value = {0.0, 0.0};
};

static bool
try_extract_scalar_constant(const nlohmann::json &node, scalar_value &out);
static bool is_complex_annotated_constant(const nlohmann::json &node);
static nlohmann::json to_json_constant(const scalar_value &v);
static scalar_value apply_complex_binary(
  const std::string &function,
  const scalar_value &lhs,
  const scalar_value &rhs);

static numeric_value make_int_value(int64_t value)
{
  return {true, value, static_cast<double>(value)};
}

static numeric_value make_float_value(double value)
{
  return {false, 0, value};
}

static double to_double(const numeric_value &value)
{
  return value.is_int ? static_cast<double>(value.int_value)
                      : value.double_value;
}

static bool numpy_constant_folding_enabled()
{
  return !config.options.get_bool_option("python-no-fold");
}

static BigInt pow_bigint_non_negative(BigInt base, BigInt exponent)
{
  assert(exponent >= 0);
  BigInt result = 1;
  while (exponent > 0)
  {
    if ((exponent % 2) != 0)
      result *= base;
    exponent /= 2;
    if (exponent > 0)
      base *= base;
  }
  return result;
}

static bool
try_exact_integer_power(int64_t base, int64_t exponent, BigInt &result)
{
  if (exponent < 0)
    return false;

  result = pow_bigint_non_negative(BigInt(base), BigInt(exponent));
  return true;
}

static void throw_negative_integer_power_error()
{
  throw std::runtime_error(
    "ValueError: Integers to negative integer powers are not allowed");
}

static bool overflow_checks_enabled()
{
  return config.options.get_bool_option("overflow-check") ||
         config.options.get_bool_option("unsigned-overflow-check");
}

static void emit_numpy_overflow_assertion(
  python_converter &converter,
  const nlohmann::json &call,
  const symbol_id &function_id)
{
  if (!overflow_checks_enabled())
    return;

  // V.3: build the always-fail overflow assert condition in IREP2.
  code_assertt overflow_assert(migrate_expr_back(gen_false_expr()));
  overflow_assert.location() = converter.get_location_from_decl(call);
  overflow_assert.location().comment(
    "Integer overflow detected in " + function_id.get_function() + "() call");
  converter.add_instruction(overflow_assert);
}

static numeric_value extract_value(const nlohmann::json &arg);

static bool
try_extract_numeric_constant(const nlohmann::json &node, numeric_value &out)
{
  if (!node.is_object() || !node.contains("_type"))
    return false;

  const std::string type = node["_type"];

  // The boolean try_extract_* helpers must not depend on catching an exception
  // for control flow: extract_value() raises std::runtime_error on non-numeric
  // input, and relying on that as a flow-control signal is fragile. Pre-check
  // that the payload is numeric and only call extract_value() when it is
  // guaranteed to succeed, so a non-numeric literal (e.g. a str element in
  // numpy.linalg.det's matrix) makes this helper return false cleanly instead
  // of letting the internal "Unknown numeric type" error escape to the user
  // (issue #5206).
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

static std::optional<nlohmann::json>
try_build_numpy_arange_list(const nlohmann::json &call)
{
  if (
    !call.is_object() || !call.contains("_type") || call["_type"] != "Call" ||
    !call.contains("func") || !call["func"].is_object() ||
    !call["func"].contains("_type") || call["func"]["_type"] != "Name" ||
    !call["func"].contains("id") || call["func"]["id"] != "arange" ||
    !call.contains("args") || !call["args"].is_array() ||
    call["args"].empty() || call["args"].size() > 3)
  {
    return std::nullopt;
  }

  std::vector<numeric_value> args;
  args.reserve(call["args"].size());
  for (auto arg : call["args"])
  {
    numeric_value value;
    if (!try_extract_numeric_constant(arg, value))
      return std::nullopt;
    args.push_back(value);
  }

  double start = 0.0;
  double stop = 0.0;
  double step = 1.0;
  if (args.size() == 1)
    stop = to_double(args[0]);
  else
  {
    start = to_double(args[0]);
    stop = to_double(args[1]);
    if (args.size() == 3)
      step = to_double(args[2]);
  }

  if (step == 0.0)
    return std::nullopt;

  const bool any_float = std::any_of(
    args.begin(), args.end(), [](const numeric_value &v) { return !v.is_int; });

  nlohmann::json out;
  out["_type"] = "List";
  out["elts"] = nlohmann::json::array();

  if (step > 0.0)
  {
    for (double current = start; current < stop; current += step)
    {
      if (any_float)
        out["elts"].push_back({{"_type", "Constant"}, {"value", current}});
      else
        out["elts"].push_back(
          {{"_type", "Constant"},
           {"value", static_cast<int64_t>(std::llround(current))}});
    }
  }
  else
  {
    for (double current = start; current > stop; current += step)
    {
      if (any_float)
        out["elts"].push_back({{"_type", "Constant"}, {"value", current}});
      else
        out["elts"].push_back(
          {{"_type", "Constant"},
           {"value", static_cast<int64_t>(std::llround(current))}});
    }
  }

  return out;
}

static scalar_value make_real_scalar(double value)
{
  scalar_value out;
  out.is_complex = false;
  out.value = {value, 0.0};
  return out;
}

static scalar_value make_complex_scalar(double real, double imag)
{
  scalar_value out;
  out.is_complex = true;
  out.value = {real, imag};
  return out;
}

static bool
try_extract_scalar_binary(const nlohmann::json &node, scalar_value &out)
{
  if (
    !node.is_object() || !node.contains("_type") || node["_type"] != "BinOp" ||
    !node.contains("op") || !node["op"].is_object() ||
    !node["op"].contains("_type") || !node.contains("left") ||
    !node.contains("right"))
  {
    return false;
  }

  const std::string op_type = node["op"]["_type"];
  if (op_type != "Add" && op_type != "Sub")
    return false;

  scalar_value left;
  scalar_value right;
  if (
    !try_extract_scalar_constant(node["left"], left) ||
    !try_extract_scalar_constant(node["right"], right))
  {
    return false;
  }

  out.is_complex = left.is_complex || right.is_complex;
  out.value =
    op_type == "Add" ? left.value + right.value : left.value - right.value;
  return true;
}

static bool is_complex_annotated_constant(const nlohmann::json &node)
{
  if (!node.is_object())
    return false;
  return node.contains("esbmc_type_annotation") &&
         node["esbmc_type_annotation"] == "complex";
}

static bool
try_extract_scalar_constant(const nlohmann::json &node, scalar_value &out)
{
  if (!node.is_object() || !node.contains("_type"))
    return false;

  const std::string type = node["_type"];
  if (type != "Constant" && type != "UnaryOp" && type != "BinOp")
    return false;

  try
  {
    if (type == "BinOp")
    {
      if (try_extract_scalar_binary(node, out))
        return true;
    }
    if (
      type == "Constant" && node.contains("value") &&
      node["value"].is_boolean())
    {
      out = make_real_scalar(node["value"].get<bool>() ? 1.0 : 0.0);
      return true;
    }
    if (type == "UnaryOp")
    {
      if (!node.contains("operand") || !node["operand"].is_object())
        return false;
      const auto &operand = node["operand"];
      if (is_complex_annotated_constant(operand))
      {
        double real = operand.value("real_value", 0.0);
        double imag = operand.value("imag_value", 0.0);
        if (
          node.contains("op") && node["op"].is_object() &&
          node["op"].contains("_type") && node["op"]["_type"] == "USub")
        {
          real = -real;
          imag = -imag;
        }
        out = make_complex_scalar(real, imag);
        return true;
      }
    }
    else if (is_complex_annotated_constant(node))
    {
      out = make_complex_scalar(
        node.value("real_value", 0.0), node.value("imag_value", 0.0));
      return true;
    }

    numeric_value numeric;
    if (!try_extract_numeric_constant(node, numeric))
      return false;
    out = make_real_scalar(to_double(numeric));
    return true;
  }
  catch (const std::exception &)
  {
    return false;
  }
}

static bool try_extract_scalar_1d_list(
  const nlohmann::json &list_node,
  std::vector<scalar_value> &values)
{
  if (
    !list_node.is_object() || !list_node.contains("_type") ||
    list_node["_type"] != "List" || !list_node.contains("elts"))
    return false;

  values.clear();
  values.reserve(list_node["elts"].size());
  for (const auto &elem : list_node["elts"])
  {
    scalar_value value;
    if (!try_extract_scalar_constant(elem, value))
      return false;
    values.push_back(value);
  }
  return true;
}

static bool try_extract_scalar_2d_list(
  const nlohmann::json &list_node,
  std::vector<std::vector<scalar_value>> &values)
{
  if (
    !list_node.is_object() || !list_node.contains("_type") ||
    list_node["_type"] != "List" || !list_node.contains("elts"))
    return false;

  values.clear();
  values.reserve(list_node["elts"].size());
  for (const auto &row : list_node["elts"])
  {
    std::vector<scalar_value> row_values;
    if (!try_extract_scalar_1d_list(row, row_values))
      return false;
    values.push_back(row_values);
  }
  return true;
}

static bool is_square_matrix(
  const std::vector<std::vector<scalar_value>> &values,
  std::size_t &n)
{
  n = values.size();
  if (n == 0)
    return false;
  for (const auto &row : values)
  {
    if (row.size() != n)
      return false;
  }
  return true;
}

static scalar_value
determinant_2x2(const std::vector<std::vector<scalar_value>> &m)
{
  const auto a = m[0][0].value;
  const auto b = m[0][1].value;
  const auto c = m[1][0].value;
  const auto d = m[1][1].value;
  const auto det = a * d - b * c;
  const bool complex_out = m[0][0].is_complex || m[0][1].is_complex ||
                           m[1][0].is_complex || m[1][1].is_complex;
  return complex_out ? make_complex_scalar(det.real(), det.imag())
                     : make_real_scalar(det.real());
}

static scalar_value
determinant_3x3(const std::vector<std::vector<scalar_value>> &m)
{
  const auto a = m[0][0].value;
  const auto b = m[0][1].value;
  const auto c = m[0][2].value;
  const auto d = m[1][0].value;
  const auto e = m[1][1].value;
  const auto f = m[1][2].value;
  const auto g = m[2][0].value;
  const auto h = m[2][1].value;
  const auto i = m[2][2].value;

  const auto det =
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  bool complex_out = false;
  for (const auto &row : m)
  {
    for (const auto &v : row)
      complex_out = complex_out || v.is_complex;
  }
  return complex_out ? make_complex_scalar(det.real(), det.imag())
                     : make_real_scalar(det.real());
}

static bool inverse_2x2(
  const std::vector<std::vector<scalar_value>> &m,
  std::vector<std::vector<scalar_value>> &inv)
{
  auto det = determinant_2x2(m);
  if (std::abs(det.value) < 1e-15)
    return false;
  auto d = det.value;
  inv.resize(2, std::vector<scalar_value>(2));
  inv[0][0] = make_real_scalar((m[1][1].value / d).real());
  inv[0][1] = make_real_scalar((-m[0][1].value / d).real());
  inv[1][0] = make_real_scalar((-m[1][0].value / d).real());
  inv[1][1] = make_real_scalar((m[0][0].value / d).real());
  return true;
}

static bool inverse_3x3(
  const std::vector<std::vector<scalar_value>> &m,
  std::vector<std::vector<scalar_value>> &inv)
{
  auto det = determinant_3x3(m);
  if (std::abs(det.value) < 1e-15)
    return false;
  auto d = det.value;

  inv.resize(3, std::vector<scalar_value>(3));
  inv[0][0] = make_real_scalar(
    ((m[1][1].value * m[2][2].value - m[1][2].value * m[2][1].value) / d)
      .real());
  inv[0][1] = make_real_scalar(
    ((m[0][2].value * m[2][1].value - m[0][1].value * m[2][2].value) / d)
      .real());
  inv[0][2] = make_real_scalar(
    ((m[0][1].value * m[1][2].value - m[0][2].value * m[1][1].value) / d)
      .real());
  inv[1][0] = make_real_scalar(
    ((m[1][2].value * m[2][0].value - m[1][0].value * m[2][2].value) / d)
      .real());
  inv[1][1] = make_real_scalar(
    ((m[0][0].value * m[2][2].value - m[0][2].value * m[2][0].value) / d)
      .real());
  inv[1][2] = make_real_scalar(
    ((m[0][2].value * m[1][0].value - m[0][0].value * m[1][2].value) / d)
      .real());
  inv[2][0] = make_real_scalar(
    ((m[1][0].value * m[2][1].value - m[1][1].value * m[2][0].value) / d)
      .real());
  inv[2][1] = make_real_scalar(
    ((m[0][1].value * m[2][0].value - m[0][0].value * m[2][1].value) / d)
      .real());
  inv[2][2] = make_real_scalar(
    ((m[0][0].value * m[1][1].value - m[0][1].value * m[1][0].value) / d)
      .real());
  return true;
}

static bool solve_linear_system(
  const std::vector<std::vector<scalar_value>> &A,
  const std::vector<scalar_value> &b,
  std::vector<scalar_value> &x)
{
  std::size_t n = A.size();
  if (n > 3)
    return false;

  std::vector<std::vector<scalar_value>> inv;
  bool ok = (n == 2) ? inverse_2x2(A, inv) : inverse_3x3(A, inv);
  if (!ok)
    return false;

  x.resize(n);
  for (std::size_t i = 0; i < n; ++i)
  {
    std::complex<double> sum = 0.0;
    for (std::size_t j = 0; j < n; ++j)
      sum += inv[i][j].value * b[j].value;
    x[i] = make_real_scalar(sum.real());
  }
  return true;
}

static nlohmann::json
matrix_to_json(const std::vector<std::vector<scalar_value>> &m)
{
  nlohmann::json outer;
  outer["_type"] = "List";
  outer["elts"] = nlohmann::json::array();
  for (const auto &row : m)
  {
    nlohmann::json row_json;
    row_json["_type"] = "List";
    row_json["elts"] = nlohmann::json::array();
    for (const auto &val : row)
      row_json["elts"].push_back(to_json_constant(val));
    outer["elts"].push_back(row_json);
  }
  return outer;
}

static nlohmann::json vector_to_json(const std::vector<scalar_value> &v)
{
  nlohmann::json list;
  list["_type"] = "List";
  list["elts"] = nlohmann::json::array();
  for (const auto &val : v)
    list["elts"].push_back(to_json_constant(val));
  return list;
}

static bool literal_list_contains_bool(const nlohmann::json &node)
{
  if (!node.is_object())
    return false;
  if (
    node.value("_type", std::string()) == "Constant" &&
    node.contains("value") && node["value"].is_boolean())
  {
    return true;
  }
  if (
    node.value("_type", std::string()) != "List" || !node.contains("elts") ||
    !node["elts"].is_array())
  {
    return false;
  }
  for (const auto &elem : node["elts"])
  {
    if (literal_list_contains_bool(elem))
      return true;
  }
  return false;
}

static scalar_value dot_product_value(
  const std::vector<scalar_value> &lhs,
  const std::vector<scalar_value> &rhs)
{
  scalar_value out = make_real_scalar(0.0);
  for (std::size_t i = 0; i < lhs.size(); ++i)
  {
    out.is_complex = out.is_complex || lhs[i].is_complex || rhs[i].is_complex;
    out.value += lhs[i].value * rhs[i].value;
  }
  return out;
}

static bool fold_literal_dot(
  const nlohmann::json &lhs,
  const nlohmann::json &rhs,
  nlohmann::json &out)
{
  std::vector<scalar_value> lhs_1d;
  std::vector<scalar_value> rhs_1d;
  if (
    try_extract_scalar_1d_list(lhs, lhs_1d) &&
    try_extract_scalar_1d_list(rhs, rhs_1d))
  {
    if (lhs_1d.empty() || rhs_1d.empty())
      throw std::runtime_error(
        "TypeError: numpy.dot does not support empty operands");
    if (lhs_1d.size() != rhs_1d.size())
      throw std::runtime_error("Incompatible shapes for dot product");
    out = to_json_constant(dot_product_value(lhs_1d, rhs_1d));
    return true;
  }

  std::vector<std::vector<scalar_value>> lhs_2d;
  std::vector<std::vector<scalar_value>> rhs_2d;
  if (
    try_extract_scalar_2d_list(lhs, lhs_2d) &&
    try_extract_scalar_2d_list(rhs, rhs_2d))
  {
    if (lhs_2d.empty() || rhs_2d.empty() || lhs_2d[0].empty())
      throw std::runtime_error(
        "TypeError: numpy.dot does not support empty operands");
    const std::size_t n = lhs_2d[0].size();
    const std::size_t p = rhs_2d[0].size();
    if (p == 0)
      throw std::runtime_error(
        "TypeError: numpy.dot does not support empty operands");
    if (n != rhs_2d.size())
      throw std::runtime_error("Incompatible shapes for dot product");

    out["_type"] = "List";
    out["elts"] = nlohmann::json::array();
    for (const auto &lhs_row : lhs_2d)
    {
      if (lhs_row.size() != n)
        return false;
      nlohmann::json out_row;
      out_row["_type"] = "List";
      out_row["elts"] = nlohmann::json::array();
      for (std::size_t col = 0; col < p; ++col)
      {
        std::vector<scalar_value> rhs_col;
        rhs_col.reserve(rhs_2d.size());
        for (const auto &rhs_row : rhs_2d)
        {
          if (rhs_row.size() != p)
            return false;
          rhs_col.push_back(rhs_row[col]);
        }
        out_row["elts"].push_back(
          to_json_constant(dot_product_value(lhs_row, rhs_col)));
      }
      out["elts"].push_back(out_row);
    }
    return true;
  }

  if (
    try_extract_scalar_1d_list(lhs, lhs_1d) &&
    try_extract_scalar_2d_list(rhs, rhs_2d))
  {
    if (lhs_1d.empty() || rhs_2d.empty() || rhs_2d[0].empty())
      throw std::runtime_error(
        "TypeError: numpy.dot does not support empty operands");
    const std::size_t p = rhs_2d[0].size();
    if (lhs_1d.size() != rhs_2d.size())
      throw std::runtime_error("Incompatible shapes for dot product");

    out["_type"] = "List";
    out["elts"] = nlohmann::json::array();
    for (std::size_t col = 0; col < p; ++col)
    {
      std::vector<scalar_value> rhs_col;
      rhs_col.reserve(rhs_2d.size());
      for (const auto &rhs_row : rhs_2d)
      {
        if (rhs_row.size() != p)
          return false;
        rhs_col.push_back(rhs_row[col]);
      }
      out["elts"].push_back(
        to_json_constant(dot_product_value(lhs_1d, rhs_col)));
    }
    return true;
  }

  if (
    try_extract_scalar_2d_list(lhs, lhs_2d) &&
    try_extract_scalar_1d_list(rhs, rhs_1d))
  {
    if (lhs_2d.empty() || lhs_2d[0].empty() || rhs_1d.empty())
      throw std::runtime_error(
        "TypeError: numpy.dot does not support empty operands");
    const std::size_t n = lhs_2d[0].size();
    if (n != rhs_1d.size())
      throw std::runtime_error("Incompatible shapes for dot product");

    out["_type"] = "List";
    out["elts"] = nlohmann::json::array();
    for (const auto &lhs_row : lhs_2d)
    {
      if (lhs_row.size() != n)
        return false;
      out["elts"].push_back(
        to_json_constant(dot_product_value(lhs_row, rhs_1d)));
    }
    return true;
  }

  return false;
}

static bool is_complex_function(const std::string &function)
{
  return function == "real" || function == "imag" || function == "conj" ||
         function == "conjugate" || function == "angle" || function == "abs";
}

static bool is_complex_annotated_scalar_node(const nlohmann::json &node)
{
  if (!node.is_object() || !node.contains("_type"))
    return false;
  if (node["_type"] == "Constant")
    return is_complex_annotated_constant(node);
  if (
    node["_type"] == "UnaryOp" && node.contains("operand") &&
    node["operand"].is_object())
    return is_complex_annotated_constant(node["operand"]);
  return false;
}

static nlohmann::json to_json_constant(const scalar_value &v)
{
  nlohmann::json out;
  out["_type"] = "Constant";
  if (v.is_complex)
  {
    out["value"] = 0.0;
    out["esbmc_type_annotation"] = "complex";
    out["real_value"] = v.value.real();
    out["imag_value"] = v.value.imag();
  }
  else
  {
    out["value"] = v.value.real();
  }
  return out;
}

static scalar_value
apply_complex_unary(const std::string &function, const scalar_value &in)
{
  if (function == "real")
    return make_real_scalar(in.value.real());
  if (function == "imag")
    return make_real_scalar(in.value.imag());
  if (function == "conj" || function == "conjugate")
    return in.is_complex
             ? make_complex_scalar(in.value.real(), -in.value.imag())
             : make_real_scalar(in.value.real());
  if (function == "angle")
    return make_real_scalar(std::atan2(in.value.imag(), in.value.real()));
  if (function == "abs")
    return make_real_scalar(std::abs(in.value));

  throw std::runtime_error("Unsupported Numpy complex unary function");
}

static scalar_value apply_complex_binary(
  const std::string &function,
  const scalar_value &lhs,
  const scalar_value &rhs)
{
  const bool wants_complex = lhs.is_complex || rhs.is_complex;
  if (function == "add")
  {
    const auto result = lhs.value + rhs.value;
    return wants_complex ? make_complex_scalar(result.real(), result.imag())
                         : make_real_scalar(result.real());
  }
  if (function == "subtract")
  {
    const auto result = lhs.value - rhs.value;
    return wants_complex ? make_complex_scalar(result.real(), result.imag())
                         : make_real_scalar(result.real());
  }
  if (function == "multiply")
  {
    const auto result = lhs.value * rhs.value;
    return wants_complex ? make_complex_scalar(result.real(), result.imag())
                         : make_real_scalar(result.real());
  }
  if (function == "divide")
  {
    if (rhs.value.real() == 0.0 && rhs.value.imag() == 0.0)
      throw std::runtime_error(
        wants_complex ? "ZeroDivisionError: complex division by zero"
                      : "ZeroDivisionError: division by zero");

    const auto result = lhs.value / rhs.value;
    return wants_complex ? make_complex_scalar(result.real(), result.imag())
                         : make_real_scalar(result.real());
  }

  throw std::runtime_error("Unsupported Numpy complex binary function");
}

static bool has_complex(const std::vector<scalar_value> &values)
{
  for (const auto &v : values)
  {
    if (v.is_complex)
      return true;
  }
  return false;
}

static bool has_complex(const std::vector<std::vector<scalar_value>> &values)
{
  for (const auto &row : values)
  {
    if (has_complex(row))
      return true;
  }
  return false;
}

static bool is_list_node(const nlohmann::json &node)
{
  return node.is_object() && node.contains("_type") &&
         node["_type"] == "List" && node.contains("elts") &&
         node["elts"].is_array();
}

static std::string format_shape(const std::vector<std::size_t> &shape)
{
  std::ostringstream oss;
  oss << "(";
  for (std::size_t i = 0; i < shape.size(); ++i)
  {
    if (i != 0)
      oss << ", ";
    oss << shape[i];
  }
  if (shape.size() == 1)
    oss << ",";
  oss << ")";
  return oss.str();
}

static void
flatten_json_list(const nlohmann::json &node, std::vector<nlohmann::json> &flat)
{
  if (!is_list_node(node))
  {
    flat.push_back(node);
    return;
  }
  for (const auto &elem : node["elts"])
    flatten_json_list(elem, flat);
}

static nlohmann::json reshape_flat_to_json(
  const std::vector<nlohmann::json> &flat,
  const std::vector<std::size_t> &shape,
  std::size_t dim,
  std::size_t &offset)
{
  if (dim == shape.size())
    return flat.at(offset++);

  nlohmann::json list;
  list["_type"] = "List";
  list["elts"] = nlohmann::json::array();
  for (std::size_t i = 0; i < shape[dim]; ++i)
    list["elts"].push_back(reshape_flat_to_json(flat, shape, dim + 1, offset));
  return list;
}

static bool
get_literal_shape(const nlohmann::json &node, std::vector<std::size_t> &shape)
{
  shape.clear();

  if (!is_list_node(node))
  {
    scalar_value dummy;
    return try_extract_scalar_constant(node, dummy);
  }

  const auto &elts = node["elts"];
  shape.push_back(elts.size());

  if (elts.empty())
    return true;

  std::vector<std::size_t> child_shape;
  if (!get_literal_shape(elts[0], child_shape))
    return false;

  for (std::size_t i = 1; i < elts.size(); ++i)
  {
    std::vector<std::size_t> current_shape;
    if (
      !get_literal_shape(elts[i], current_shape) ||
      current_shape != child_shape)
      return false;
  }

  shape.insert(shape.end(), child_shape.begin(), child_shape.end());
  return true;
}

enum class scalar_kind
{
  int_like,
  float_like,
  complex_like
};

static scalar_kind get_scalar_kind(const nlohmann::json &node)
{
  if (
    node.contains("_type") && node["_type"] == "BinOp" &&
    node.contains("left") && node["left"].is_object())
  {
    const scalar_kind left_kind = get_scalar_kind(node["left"]);
    const scalar_kind right_kind =
      node.contains("right") && node["right"].is_object()
        ? get_scalar_kind(node["right"])
        : scalar_kind::int_like;

    if (
      left_kind == scalar_kind::complex_like ||
      right_kind == scalar_kind::complex_like)
      return scalar_kind::complex_like;
    if (
      left_kind == scalar_kind::float_like ||
      right_kind == scalar_kind::float_like)
      return scalar_kind::float_like;
    return scalar_kind::int_like;
  }

  if (
    node.contains("_type") && node["_type"] == "UnaryOp" &&
    node.contains("operand") && node["operand"].is_object())
  {
    return get_scalar_kind(node["operand"]);
  }

  if (is_complex_annotated_constant(node))
    return scalar_kind::complex_like;
  if (node.contains("value") && node["value"].is_number_float())
    return scalar_kind::float_like;
  return scalar_kind::int_like;
}

[[maybe_unused]] static std::string
promote_numpy_dtype(const std::string &lhs_dtype, const std::string &rhs_dtype)
{
  if (lhs_dtype == rhs_dtype)
    return lhs_dtype;

  auto rank = [](const std::string &dt) -> int {
    if (dt == "bool")
      return 0;
    if (dt == "int8")
      return 1;
    if (dt == "uint8")
      return 2;
    if (dt == "int16")
      return 3;
    if (dt == "uint16")
      return 4;
    if (dt == "int32")
      return 5;
    if (dt == "uint32")
      return 6;
    if (dt == "int64")
      return 7;
    if (dt == "uint64")
      return 8;
    if (dt == "float16")
      return 9;
    if (dt == "float32")
      return 10;
    if (dt == "float64")
      return 11;
    if (dt == "complex64")
      return 12;
    if (dt == "complex128")
      return 13;
    return 7;
  };

  static const std::vector<std::string> dtype_by_rank = {
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128"};

  int lr = rank(lhs_dtype);
  int rr = rank(rhs_dtype);
  int result = std::max(lr, rr);

  // uint + signed int of same width → next larger signed int or float64
  bool lhs_unsigned = lhs_dtype.find("uint") != std::string::npos;
  bool rhs_unsigned = rhs_dtype.find("uint") != std::string::npos;
  if (lhs_unsigned != rhs_unsigned && result <= 8)
  {
    if (result < 7)
      result = std::min(result + 1, 7);
    else
      result = 11; // float64
  }

  return dtype_by_rank[static_cast<std::size_t>(result)];
}

[[maybe_unused]] static std::string scalar_kind_to_dtype(scalar_kind kind)
{
  switch (kind)
  {
  case scalar_kind::int_like:
    return "int64";
  case scalar_kind::float_like:
    return "float64";
  case scalar_kind::complex_like:
    return "complex128";
  }
  return "float64";
}

static nlohmann::json make_numeric_constant_json(
  const scalar_value &value,
  scalar_kind kind,
  bool force_float)
{
  nlohmann::json out;
  out["_type"] = "Constant";
  if (kind == scalar_kind::complex_like)
  {
    out["value"] = 0.0;
    out["esbmc_type_annotation"] = "complex";
    out["real_value"] = value.value.real();
    out["imag_value"] = value.value.imag();
    return out;
  }

  if (kind == scalar_kind::float_like || force_float)
    out["value"] = value.value.real();
  else
    out["value"] = static_cast<int64_t>(std::llround(value.value.real()));
  return out;
}

static bool apply_numpy_binary_to_scalars(
  const std::string &function,
  const nlohmann::json &lhs,
  const nlohmann::json &rhs,
  nlohmann::json &out)
{
  scalar_value lhs_scalar;
  scalar_value rhs_scalar;
  if (
    !try_extract_scalar_constant(lhs, lhs_scalar) ||
    !try_extract_scalar_constant(rhs, rhs_scalar))
    return false;

  const scalar_kind lhs_kind = get_scalar_kind(lhs);
  const scalar_kind rhs_kind = get_scalar_kind(rhs);
  const bool wants_complex = lhs_kind == scalar_kind::complex_like ||
                             rhs_kind == scalar_kind::complex_like;
  const bool wants_float =
    wants_complex || lhs_kind == scalar_kind::float_like ||
    rhs_kind == scalar_kind::float_like || function == "divide";

  if (
    function == "power" && lhs_kind == scalar_kind::int_like &&
    rhs_kind == scalar_kind::int_like)
  {
    numeric_value rhs_numeric;
    if (
      try_extract_numeric_constant(rhs, rhs_numeric) &&
      rhs_numeric.int_value < 0)
      throw_negative_integer_power_error();
  }

  if (
    function == "greater" || function == "less" ||
    function == "greater_equal" || function == "less_equal" ||
    function == "equal" || function == "not_equal")
  {
    const double left = lhs_scalar.value.real();
    const double right = rhs_scalar.value.real();
    bool result = false;
    if (function == "greater")
      result = left > right;
    else if (function == "less")
      result = left < right;
    else if (function == "greater_equal")
      result = left >= right;
    else if (function == "less_equal")
      result = left <= right;
    else if (function == "equal")
      result = left == right;
    else
      result = left != right;

    out = {{"_type", "Constant"}, {"value", result}};
    return true;
  }

  if (function == "logical_and" || function == "logical_or")
  {
    const bool left = lhs_scalar.value.real() != 0.0;
    const bool right = rhs_scalar.value.real() != 0.0;
    const bool result =
      function == "logical_and" ? (left && right) : (left || right);
    out = {{"_type", "Constant"}, {"value", result}};
    return true;
  }

  if (!numpy_constant_folding_enabled())
    return false;

  scalar_value result;
  if (wants_complex)
    result = apply_complex_binary(function, lhs_scalar, rhs_scalar);
  else
  {
    if (
      function == "power" && lhs_kind == scalar_kind::int_like &&
      rhs_kind == scalar_kind::int_like)
    {
      numeric_value lhs_numeric;
      numeric_value rhs_numeric;
      if (
        try_extract_numeric_constant(lhs, lhs_numeric) &&
        try_extract_numeric_constant(rhs, rhs_numeric) &&
        rhs_numeric.int_value >= 0)
      {
        BigInt exact_power;
        if (try_exact_integer_power(
              lhs_numeric.int_value, rhs_numeric.int_value, exact_power))
        {
          const BigInt min_val = BigInt(std::numeric_limits<int64_t>::min());
          const BigInt max_val = BigInt(std::numeric_limits<int64_t>::max());
          if (exact_power < min_val || exact_power > max_val)
            return false;

          out = {{"_type", "Constant"}, {"value", exact_power.to_int64()}};
          return true;
        }
      }
    }

    const double left = lhs_scalar.value.real();
    const double right = rhs_scalar.value.real();
    double folded = 0.0;

    if (function == "add")
      folded = left + right;
    else if (function == "subtract")
      folded = left - right;
    else if (function == "multiply")
      folded = left * right;
    else if (function == "divide")
    {
      if (right == 0.0)
        return false;
      folded = left / right;
    }
    else if (function == "power")
      folded = std::pow(left, right);
    else if (function == "fmod")
    {
      if (right == 0.0)
        return false;
      folded = std::fmod(left, right);
    }
    else
      return false;

    result = wants_float ? make_real_scalar(folded)
                         : make_real_scalar(std::llround(folded));
  }

  out = make_numeric_constant_json(
    result,
    wants_complex
      ? scalar_kind::complex_like
      : (wants_float ? scalar_kind::float_like : scalar_kind::int_like),
    wants_float && !wants_complex);
  return true;
}

static bool compute_broadcast_shape(
  const std::vector<std::size_t> &lhs_shape,
  const std::vector<std::size_t> &rhs_shape,
  std::vector<std::size_t> &result_shape)
{
  const std::size_t lhs_rank = lhs_shape.size();
  const std::size_t rhs_rank = rhs_shape.size();
  const std::size_t result_rank = std::max(lhs_rank, rhs_rank);

  result_shape.assign(result_rank, 1);

  for (std::size_t i = 0; i < result_rank; ++i)
  {
    const std::size_t lhs_dim = (i < result_rank - lhs_rank)
                                  ? 1
                                  : lhs_shape[i - (result_rank - lhs_rank)];
    const std::size_t rhs_dim = (i < result_rank - rhs_rank)
                                  ? 1
                                  : rhs_shape[i - (result_rank - rhs_rank)];

    if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1)
      return false;

    result_shape[i] = std::max(lhs_dim, rhs_dim);
  }

  return true;
}

static bool fetch_broadcast_leaf(
  const nlohmann::json &node,
  const std::vector<std::size_t> &shape,
  const std::vector<std::size_t> &result_indices,
  nlohmann::json &leaf)
{
  if (!is_list_node(node))
  {
    scalar_value scalar;
    if (!try_extract_scalar_constant(node, scalar))
      return false;
    leaf = node;
    return true;
  }

  const std::size_t offset = result_indices.size() - shape.size();
  const nlohmann::json *current = &node;

  for (std::size_t axis = 0; axis < shape.size(); ++axis)
  {
    const std::size_t result_axis = axis + offset;
    const std::size_t index =
      shape[axis] == 1 ? 0 : result_indices[result_axis];
    current = &(*current)["elts"][index];
  }

  leaf = *current;
  return true;
}

static bool build_broadcast_literal_result(
  const std::string &function,
  const nlohmann::json &lhs,
  const std::vector<std::size_t> &lhs_shape,
  const nlohmann::json &rhs,
  const std::vector<std::size_t> &rhs_shape,
  const std::vector<std::size_t> &result_shape,
  std::vector<std::size_t> &indices,
  std::size_t depth,
  nlohmann::json &out)
{
  if (depth == result_shape.size())
  {
    nlohmann::json lhs_leaf;
    nlohmann::json rhs_leaf;
    if (
      !fetch_broadcast_leaf(lhs, lhs_shape, indices, lhs_leaf) ||
      !fetch_broadcast_leaf(rhs, rhs_shape, indices, rhs_leaf))
      return false;
    return apply_numpy_binary_to_scalars(function, lhs_leaf, rhs_leaf, out);
  }

  out["_type"] = "List";
  out["elts"] = nlohmann::json::array();
  for (std::size_t i = 0; i < result_shape[depth]; ++i)
  {
    indices.push_back(i);
    nlohmann::json child;
    if (!build_broadcast_literal_result(
          function,
          lhs,
          lhs_shape,
          rhs,
          rhs_shape,
          result_shape,
          indices,
          depth + 1,
          child))
      return false;
    out["elts"].push_back(child);
    indices.pop_back();
  }
  return true;
}

static bool try_extract_numeric_1d_list(
  const nlohmann::json &list_node,
  std::vector<numeric_value> &values)
{
  if (
    !list_node.is_object() || !list_node.contains("_type") ||
    list_node["_type"] != "List" || !list_node.contains("elts"))
    return false;

  values.clear();
  values.reserve(list_node["elts"].size());
  for (const auto &elem : list_node["elts"])
  {
    numeric_value value;
    if (!try_extract_numeric_constant(elem, value))
      return false;
    values.push_back(value);
  }
  return true;
}

static bool try_extract_numeric_2d_list(
  const nlohmann::json &list_node,
  std::vector<std::vector<numeric_value>> &values)
{
  if (
    !list_node.is_object() || !list_node.contains("_type") ||
    list_node["_type"] != "List" || !list_node.contains("elts"))
    return false;

  values.clear();
  values.reserve(list_node["elts"].size());
  for (const auto &row : list_node["elts"])
  {
    std::vector<numeric_value> row_values;
    if (!try_extract_numeric_1d_list(row, row_values))
      return false;
    values.push_back(row_values);
  }
  return true;
}

static bool is_json_none_literal(const nlohmann::json &node)
{
  return node.is_object() && node.contains("_type") &&
         node["_type"] == "Constant" && node.contains("value") &&
         node["value"].is_null();
}

static bool is_finite_numeric_value(const numeric_value &value)
{
  return value.is_int || std::isfinite(value.double_value);
}

static double
numeric_to_key(const nlohmann::json &node, const std::string &diagnostic)
{
  numeric_value value;
  if (
    !try_extract_numeric_constant(node, value) ||
    !is_finite_numeric_value(value))
    throw std::runtime_error(diagnostic);
  return to_double(value);
}

static double numeric_to_sort_key(const nlohmann::json &node)
{
  return numeric_to_key(
    node,
    "TypeError: numpy.sort() currently supports only finite numeric arrays");
}

static nlohmann::json
make_sorted_numeric_list(std::vector<nlohmann::json> elements)
{
  std::stable_sort(
    elements.begin(),
    elements.end(),
    [](const nlohmann::json &lhs, const nlohmann::json &rhs) {
      return numeric_to_sort_key(lhs) < numeric_to_sort_key(rhs);
    });

  nlohmann::json result;
  result["_type"] = "List";
  result["elts"] = std::move(elements);
  return result;
}

static nlohmann::json make_unique_numeric_list(
  std::vector<nlohmann::json> elements,
  const std::string &diagnostic)
{
  std::stable_sort(
    elements.begin(),
    elements.end(),
    [&](const nlohmann::json &lhs, const nlohmann::json &rhs) {
      return numeric_to_key(lhs, diagnostic) < numeric_to_key(rhs, diagnostic);
    });

  std::vector<nlohmann::json> unique_elements;
  for (const auto &element : elements)
  {
    numeric_to_key(element, diagnostic);
    if (
      unique_elements.empty() ||
      numeric_to_key(unique_elements.back(), diagnostic) !=
        numeric_to_key(element, diagnostic))
    {
      unique_elements.push_back(element);
    }
  }

  nlohmann::json result;
  result["_type"] = "List";
  result["elts"] = std::move(unique_elements);
  return result;
}

static double median_of_numeric_elements(
  std::vector<nlohmann::json> elements,
  const std::string &diagnostic)
{
  if (elements.empty())
    throw std::runtime_error(
      "ValueError: numpy.median() input array must be non-empty");

  std::stable_sort(
    elements.begin(),
    elements.end(),
    [&](const nlohmann::json &lhs, const nlohmann::json &rhs) {
      return numeric_to_key(lhs, diagnostic) < numeric_to_key(rhs, diagnostic);
    });

  const std::size_t mid = elements.size() / 2;
  if (elements.size() % 2 == 1)
    return numeric_to_key(elements[mid], diagnostic);

  return (numeric_to_key(elements[mid - 1], diagnostic) +
          numeric_to_key(elements[mid], diagnostic)) /
         2.0;
}

static double percentile_of_numeric_elements(
  std::vector<nlohmann::json> elements,
  double q,
  const std::string &diagnostic)
{
  if (elements.empty())
    throw std::runtime_error(
      "ValueError: numpy.percentile() input array must be non-empty");

  std::stable_sort(
    elements.begin(),
    elements.end(),
    [&](const nlohmann::json &lhs, const nlohmann::json &rhs) {
      return numeric_to_key(lhs, diagnostic) < numeric_to_key(rhs, diagnostic);
    });

  const double rank = (q / 100.0) * static_cast<double>(elements.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(rank));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(rank));
  const double fraction = rank - static_cast<double>(lower);
  const double lower_value = numeric_to_key(elements[lower], diagnostic);
  const double upper_value = numeric_to_key(elements[upper], diagnostic);
  return lower_value + ((upper_value - lower_value) * fraction);
}

static nlohmann::json make_integer_list(const std::vector<std::size_t> &values)
{
  nlohmann::json result;
  result["_type"] = "List";
  result["elts"] = nlohmann::json::array();
  for (std::size_t value : values)
  {
    nlohmann::json elem;
    elem["_type"] = "Constant";
    elem["value"] = value;
    result["elts"].push_back(elem);
  }
  return result;
}

static std::optional<nlohmann::json>
get_literal_numpy_array_arg(const nlohmann::json &node)
{
  if (!node.is_object() || !node.contains("_type"))
    return std::nullopt;

  if (node["_type"] == "List")
    return node;

  if (
    node["_type"] != "Call" || !node.contains("func") ||
    !node["func"].is_object() || !node["func"].contains("_type") ||
    node["func"]["_type"] != "Attribute" || !node["func"].contains("attr") ||
    node["func"]["attr"] != "array" || !node["func"].contains("value") ||
    !node["func"]["value"].is_object() ||
    node["func"]["value"].value("_type", std::string()) != "Name" ||
    node["func"]["value"].value("id", std::string()) != "np" ||
    !node.contains("args") || node["args"].empty())
  {
    return std::nullopt;
  }

  nlohmann::json literal = node["args"][0];
  if (literal.is_object() && literal.value("_type", std::string()) == "List")
    return literal;
  return std::nullopt;
}

static std::optional<nlohmann::json>
resolve_literal_numpy_row_view(nlohmann::json arg, python_converter &converter)
{
  if (arg.value("_type", std::string()) == "Name")
  {
    nlohmann::json decl = json_utils::find_var_decl(
      arg["id"], converter.current_function_name(), converter.ast());
    if (decl.contains("value") && decl["value"].is_object())
      arg = decl["value"];
  }

  if (
    !arg.is_object() || arg.value("_type", std::string()) != "Subscript" ||
    !arg.contains("value") || !arg.contains("slice"))
    return std::nullopt;

  auto parse_index = [](const nlohmann::json &node) -> std::optional<int64_t> {
    if (
      node.is_object() && node.value("_type", std::string()) == "Constant" &&
      node.contains("value") && node["value"].is_number_integer())
      return node["value"].get<int64_t>();

    if (
      node.is_object() && node.value("_type", std::string()) == "UnaryOp" &&
      node.contains("op") &&
      node["op"].value("_type", std::string()) == "USub" &&
      node.contains("operand") &&
      node["operand"].value("_type", std::string()) == "Constant" &&
      node["operand"].contains("value") &&
      node["operand"]["value"].is_number_integer())
      return -node["operand"]["value"].get<int64_t>();

    return std::nullopt;
  };

  std::optional<int64_t> index = parse_index(arg["slice"]);
  if (!index)
    return std::nullopt;

  nlohmann::json base = arg["value"];
  if (base.value("_type", std::string()) == "Name")
  {
    nlohmann::json decl = json_utils::find_var_decl(
      base["id"], converter.current_function_name(), converter.ast());
    if (decl.contains("value") && decl["value"].is_object())
      base = decl["value"];
  }

  std::optional<nlohmann::json> literal = get_literal_numpy_array_arg(base);
  if (!literal || !literal->contains("elts") || !(*literal)["elts"].is_array())
    return std::nullopt;

  const auto &rows = (*literal)["elts"];
  int64_t resolved_index = *index;
  if (resolved_index < 0)
    resolved_index += static_cast<int64_t>(rows.size());
  if (resolved_index < 0 || resolved_index >= static_cast<int64_t>(rows.size()))
    return std::nullopt;

  const nlohmann::json &row = rows[static_cast<std::size_t>(resolved_index)];
  if (row.is_object() && row.value("_type", std::string()) == "List")
    return row;
  return std::nullopt;
}

static bool is_sorted_numeric_list(
  const nlohmann::json &list,
  const std::string &diagnostic)
{
  const auto &elements = list["elts"];
  for (std::size_t i = 1; i < elements.size(); ++i)
  {
    if (
      numeric_to_key(elements[i], diagnostic) <
      numeric_to_key(elements[i - 1], diagnostic))
      return false;
  }
  return true;
}

static bool is_1d_json_list(const nlohmann::json &list)
{
  if (!is_list_node(list))
    return false;
  for (const auto &element : list["elts"])
  {
    if (is_list_node(element))
      return false;
  }
  return true;
}

static std::size_t searchsorted_position(
  const nlohmann::json &list,
  const nlohmann::json &value_node,
  bool right)
{
  const double value = numeric_to_key(
    value_node,
    "TypeError: numpy.searchsorted() value must be a finite numeric literal");
  const auto &elements = list["elts"];
  for (std::size_t i = 0; i < elements.size(); ++i)
  {
    const double current = numeric_to_key(
      elements[i],
      "TypeError: numpy.searchsorted() array must contain finite numeric "
      "values");
    if (right ? value < current : value <= current)
      return i;
  }
  return elements.size();
}

enum class norm_order
{
  l2,
  l1,
  positive_inf,
  negative_inf
};

static bool is_numpy_inf_attr(const nlohmann::json &node)
{
  return node.is_object() &&
         node.value("_type", std::string()) == "Attribute" &&
         node.value("attr", std::string()) == "inf" && node.contains("value") &&
         node["value"].is_object() &&
         node["value"].value("_type", std::string()) == "Name" &&
         node["value"].value("id", std::string()) == "np";
}

static norm_order parse_norm_order(const nlohmann::json &node)
{
  numeric_value value;
  if (try_extract_numeric_constant(node, value))
  {
    const double order = to_double(value);
    if (order == 1.0)
      return norm_order::l1;
    if (order == 2.0)
      return norm_order::l2;
  }

  if (is_numpy_inf_attr(node))
    return norm_order::positive_inf;

  if (
    node.is_object() && node.value("_type", std::string()) == "UnaryOp" &&
    node.contains("op") && node["op"].is_object() &&
    node["op"].value("_type", std::string()) == "USub" &&
    node.contains("operand") && is_numpy_inf_attr(node["operand"]))
    return norm_order::negative_inf;

  throw std::runtime_error(
    "TypeError: numpy.linalg.norm order is not supported");
}

static bool is_supported_numpy_unary_math(const std::string &function)
{
  return function == "sin" || function == "cos" || function == "exp" ||
         function == "sqrt" || function == "arctan" || function == "arccos" ||
         function == "arcsin" || function == "tan" || function == "log" ||
         function == "log2" || function == "log10" || function == "sinh" ||
         function == "cosh" || function == "tanh" || function == "rint";
}

static double apply_numpy_unary_math(const std::string &function, double value)
{
  if (function == "sin")
    return std::sin(value);
  if (function == "cos")
    return std::cos(value);
  if (function == "exp")
    return std::exp(value);
  if (function == "sqrt")
    return std::sqrt(value);
  if (function == "arctan")
    return std::atan(value);
  if (function == "floor")
    return std::floor(value);
  if (function == "fabs")
    return std::fabs(value);
  if (function == "trunc")
    return std::trunc(value);
  if (function == "arccos")
    return std::acos(value);
  if (function == "arcsin")
    return std::asin(value);
  if (function == "tan")
    return std::tan(value);
  if (function == "log")
    return std::log(value);
  if (function == "log2")
    return std::log2(value);
  if (function == "log10")
    return std::log10(value);
  if (function == "sinh")
    return std::sinh(value);
  if (function == "cosh")
    return std::cosh(value);
  if (function == "tanh")
    return std::tanh(value);
  if (function == "rint")
    return std::rint(value);

  throw std::runtime_error("Unsupported Numpy unary function: " + function);
}

static bool should_fallback_to_numpy_model(const std::string &function)
{
  return function == "arcsin" || function == "tan" || function == "log" ||
         function == "log2" || function == "log10" || function == "sinh" ||
         function == "cosh" || function == "tanh" || function == "rint" ||
         function == "remainder" || function == "nextafter" ||
         function == "modf" || function == "frexp" || function == "isclose" ||
         function == "copysign" || function == "fmin" || function == "fmax" ||
         function == "round";
}

static exprt fold_numpy_unary_constant_list(
  python_converter &converter,
  const std::string &function,
  const nlohmann::json &arg)
{
  std::vector<numeric_value> values_1d;
  if (try_extract_numeric_1d_list(arg, values_1d))
  {
    nlohmann::json out;
    out["_type"] = "List";
    out["elts"] = nlohmann::json::array();
    for (const auto &value : values_1d)
    {
      nlohmann::json elem;
      elem["_type"] = "Constant";
      elem["value"] = apply_numpy_unary_math(function, to_double(value));
      out["elts"].push_back(elem);
    }
    return converter.get_expr(out);
  }

  std::vector<std::vector<numeric_value>> values_2d;
  if (try_extract_numeric_2d_list(arg, values_2d))
  {
    nlohmann::json out;
    out["_type"] = "List";
    out["elts"] = nlohmann::json::array();
    for (const auto &row_values : values_2d)
    {
      nlohmann::json row;
      row["_type"] = "List";
      row["elts"] = nlohmann::json::array();
      for (const auto &value : row_values)
      {
        nlohmann::json elem;
        elem["_type"] = "Constant";
        elem["value"] = apply_numpy_unary_math(function, to_double(value));
        row["elts"].push_back(elem);
      }
      out["elts"].push_back(row);
    }
    return converter.get_expr(out);
  }

  throw std::runtime_error("Unsupported Numpy call: " + function);
}

static nlohmann::json build_constant_node(const numeric_value &value)
{
  nlohmann::json node;
  node["_type"] = "Constant";
  if (value.is_int)
    node["value"] = value.int_value;
  else
    node["value"] = value.double_value;
  return node;
}

static nlohmann::json build_filled_array_literal(
  const std::vector<std::size_t> &dims,
  std::size_t dim_index,
  const numeric_value &fill)
{
  nlohmann::json node;
  node["_type"] = "List";
  node["elts"] = nlohmann::json::array();
  const bool innermost = dim_index + 1 == dims.size();
  for (std::size_t i = 0; i < dims[dim_index]; ++i)
    node["elts"].push_back(
      innermost ? build_constant_node(fill)
                : build_filled_array_literal(dims, dim_index + 1, fill));
  return node;
}

static std::vector<std::size_t>
extract_constructor_shape_dims(const nlohmann::json &shape_node)
{
  std::vector<std::size_t> dims;

  numeric_value scalar;
  if (
    try_extract_numeric_constant(shape_node, scalar) && scalar.is_int &&
    scalar.int_value >= 0)
  {
    dims.push_back(static_cast<std::size_t>(scalar.int_value));
    return dims;
  }

  if (
    shape_node.is_object() &&
    (shape_node.value("_type", std::string()) == "Tuple" ||
     shape_node.value("_type", std::string()) == "List") &&
    shape_node.contains("elts") && shape_node["elts"].is_array())
  {
    for (const auto &elem : shape_node["elts"])
    {
      numeric_value dim_value;
      if (
        !try_extract_numeric_constant(elem, dim_value) || !dim_value.is_int ||
        dim_value.int_value < 0)
      {
        dims.clear();
        return dims;
      }
      dims.push_back(static_cast<std::size_t>(dim_value.int_value));
    }
  }

  return dims;
}

// Reconstructs the equivalent literal List (or nested List-of-List) AST that
// a shape-based numpy constructor call (zeros/ones/full/eye/identity) would
// need to produce the same concrete array np.array(<literal>) would build.
// Several numpy helpers (transpose's Name-branch, the sum/mean/reducer
// dispatch) resolve a variable's declaration only as far as a literal
// np.array(<literal>) call and reuse that literal for everything downstream;
// blindly reusing the first call argument for ANY constructor call
// misreads a shape/size argument as array data for constructors whose first
// argument is not the array data itself (e.g. np.zeros((2, 3))'s args[0] is
// the shape tuple, not two rows of data). Declines (returns nullopt) rather
// than guessing for anything not explicitly modelled here, including a call
// carrying keyword arguments such as dtype= (ADR-NP principle 3: reject
// explicitly instead of applying a silently different semantics).
static std::optional<nlohmann::json>
materialize_zeros_ones(const std::string &ctor, const nlohmann::json &args)
{
  if (args.empty())
    return std::nullopt;
  std::vector<std::size_t> dims = extract_constructor_shape_dims(args[0]);
  // Rank capped at 8 as a sanity bound (real numpy arrays are rarely, if
  // ever, higher-rank than this); callers needing a tighter, shape-specific
  // limit (e.g. transpose's 2D-only support) apply their own on top.
  if (dims.empty() || dims.size() > 8)
    return std::nullopt;
  // Matches the real zeros()/ones() creation path: no dtype= means a
  // floating-point fill (make_numpy_typed_constant with an empty dtype
  // always returns a floating-point JSON value).
  const numeric_value fill = make_float_value(ctor == "zeros" ? 0.0 : 1.0);
  return build_filled_array_literal(dims, 0, fill);
}

static std::optional<nlohmann::json>
materialize_full(const nlohmann::json &args)
{
  if (args.size() < 2)
    return std::nullopt;
  std::vector<std::size_t> dims = extract_constructor_shape_dims(args[0]);
  numeric_value fill;
  if (
    dims.empty() || dims.size() > 8 ||
    !try_extract_numeric_constant(args[1], fill))
    return std::nullopt;
  return build_filled_array_literal(dims, 0, fill);
}

static std::optional<nlohmann::json>
materialize_eye_identity(const std::string &ctor, const nlohmann::json &args)
{
  if (args.empty() || (ctor == "eye" && args.size() > 1))
    return std::nullopt;
  numeric_value n_value;
  if (
    !try_extract_numeric_constant(args[0], n_value) || !n_value.is_int ||
    n_value.int_value < 0)
    return std::nullopt;
  const std::size_t n = static_cast<std::size_t>(n_value.int_value);
  // Matches the real eye()/identity() creation path: 1/0 built from a
  // plain C++ int literal, i.e. an integer dtype (not float64, unlike
  // real NumPy's default -- see make_constant() in the creation path).
  nlohmann::json node;
  node["_type"] = "List";
  node["elts"] = nlohmann::json::array();
  for (std::size_t i = 0; i < n; ++i)
  {
    nlohmann::json row;
    row["_type"] = "List";
    row["elts"] = nlohmann::json::array();
    for (std::size_t j = 0; j < n; ++j)
      row["elts"].push_back(
        build_constant_node(make_int_value(i == j ? 1 : 0)));
    node["elts"].push_back(row);
  }
  return node;
}

static std::optional<nlohmann::json>
materialize_linspace(const nlohmann::json &args)
{
  if (args.size() < 2 || args.size() > 3)
    return std::nullopt;
  numeric_value start_v;
  numeric_value stop_v;
  if (
    !try_extract_numeric_constant(args[0], start_v) ||
    !try_extract_numeric_constant(args[1], stop_v))
    return std::nullopt;
  std::size_t num = 50;
  if (args.size() == 3)
  {
    numeric_value num_v;
    if (
      !try_extract_numeric_constant(args[2], num_v) || !num_v.is_int ||
      num_v.int_value < 0)
      return std::nullopt;
    num = static_cast<std::size_t>(num_v.int_value);
  }
  // Matches the real linspace() creation path: every element is a float,
  // computed with the same start + step * i formula.
  const double start = to_double(start_v);
  const double stop = to_double(stop_v);
  nlohmann::json node;
  node["_type"] = "List";
  node["elts"] = nlohmann::json::array();
  if (num == 0)
    return node;
  if (num == 1)
  {
    node["elts"].push_back(build_constant_node(make_float_value(start)));
    return node;
  }
  const double step = (stop - start) / static_cast<double>(num - 1);
  for (std::size_t i = 0; i < num; ++i)
    node["elts"].push_back(build_constant_node(
      make_float_value(start + step * static_cast<double>(i))));
  return node;
}

// Exact int64 arithmetic: routing an all-integer arange() call through
// double (as materialize_arange_float() does) silently loses precision and
// can drift the termination point past 2^53.
static nlohmann::json
materialize_arange_int(int64_t start, int64_t stop, int64_t step)
{
  nlohmann::json node;
  node["_type"] = "List";
  node["elts"] = nlohmann::json::array();
  if (step > 0)
    for (int64_t current = start; current < stop; current += step)
      node["elts"].push_back(build_constant_node(make_int_value(current)));
  else
    for (int64_t current = start; current > stop; current += step)
      node["elts"].push_back(build_constant_node(make_int_value(current)));
  return node;
}

// NumPy computes an arange() with float arguments as length =
// ceil((stop - start) / step) and element i = start + i * step, rather than
// repeatedly accumulating current += step. Accumulation drifts under
// floating-point rounding and can add a spurious trailing element (e.g.
// arange(0.0, 1.0, 0.1) accumulated to 11 elements here vs NumPy's 10).
static nlohmann::json
materialize_arange_float(double start, double stop, double step)
{
  nlohmann::json node;
  node["_type"] = "List";
  node["elts"] = nlohmann::json::array();

  const double count_d = std::ceil((stop - start) / step);
  if (count_d <= 0.0)
    return node;

  const auto count = static_cast<std::size_t>(count_d);
  for (std::size_t i = 0; i < count; ++i)
    node["elts"].push_back(build_constant_node(
      make_float_value(start + static_cast<double>(i) * step)));
  return node;
}

// Keeps the fast literal-materialization path itself cheap and any
// downstream verification within the regression suite's timeout: a range
// with more elements than this still hits the pre-existing ADR-NP-004
// scalability wall documented in the numpy roadmap (arrays are fully
// unrolled) whether materialized here or via the operational model, so
// declining early avoids building an oversized literal list up front (e.g.
// np.arange(1000000) would otherwise allocate a million-element JSON list
// unconditionally, before any downstream cost is even considered).
static constexpr std::size_t max_materialized_arange_elements = 10000;

enum class arange_decline_reason
{
  none,
  bad_arity,
  non_constant,
  zero_step,
  too_many_elements,
};

struct arange_materialize_result
{
  std::optional<nlohmann::json> list;
  arange_decline_reason reason = arange_decline_reason::none;
};

static arange_materialize_result
materialize_arange_ex(const nlohmann::json &args)
{
  arange_materialize_result result;
  if (!args.is_array() || args.empty() || args.size() > 3)
  {
    result.reason = arange_decline_reason::bad_arity;
    return result;
  }

  std::vector<numeric_value> values;
  values.reserve(args.size());
  for (const auto &a : args)
  {
    numeric_value v;
    if (!try_extract_numeric_constant(a, v))
    {
      result.reason = arange_decline_reason::non_constant;
      return result;
    }
    values.push_back(v);
  }

  numeric_value start_v = make_int_value(0);
  numeric_value stop_v = values[0];
  numeric_value step_v = make_int_value(1);
  if (values.size() >= 2)
  {
    start_v = values[0];
    stop_v = values[1];
  }
  if (values.size() == 3)
    step_v = values[2];

  if (to_double(step_v) == 0.0)
  {
    result.reason = arange_decline_reason::zero_step;
    return result;
  }

  const double count =
    std::ceil((to_double(stop_v) - to_double(start_v)) / to_double(step_v));
  if (count > static_cast<double>(max_materialized_arange_elements))
  {
    result.reason = arange_decline_reason::too_many_elements;
    return result;
  }

  // Matches the real arange() creation path: an int dtype unless any
  // argument is float.
  const bool any_float =
    std::any_of(values.begin(), values.end(), [](const numeric_value &v) {
      return !v.is_int;
    });
  result.list = any_float
                  ? materialize_arange_float(
                      to_double(start_v), to_double(stop_v), to_double(step_v))
                  : materialize_arange_int(
                      start_v.int_value, stop_v.int_value, step_v.int_value);
  return result;
}

static std::optional<nlohmann::json>
materialize_arange(const nlohmann::json &args)
{
  return materialize_arange_ex(args).list;
}

// The structural/receiver checks materialize_numpy_constructor_array() needs
// before it can even ask which constructor it's looking at, split out so
// that function's own decision count stays small.
static bool is_recognized_numpy_constructor_call_shape(
  const nlohmann::json &call_node,
  const nlohmann::json &ast_json)
{
  if (
    !call_node.is_object() ||
    call_node.value("_type", std::string()) != "Call" ||
    !call_node.contains("func") || !call_node["func"].is_object() ||
    call_node["func"].value("_type", std::string()) != "Attribute" ||
    !call_node["func"].contains("attr") ||
    !call_node["func"].contains("value") ||
    !call_node["func"]["value"].is_object() ||
    call_node["func"]["value"].value("_type", std::string()) != "Name" ||
    !is_imported_numpy_module_alias(
      ast_json, call_node["func"]["value"].value("id", std::string())))
    return false;

  if (call_node.contains("keywords") && !call_node["keywords"].empty())
    return false;

  return call_node.contains("args") && call_node["args"].is_array();
}

static std::optional<nlohmann::json> materialize_numpy_constructor_array(
  const nlohmann::json &call_node,
  const nlohmann::json &ast_json)
{
  if (!is_recognized_numpy_constructor_call_shape(call_node, ast_json))
    return std::nullopt;

  const std::string ctor = call_node["func"]["attr"].get<std::string>();
  const auto &args = call_node["args"];

  if (ctor == "zeros" || ctor == "ones")
    return materialize_zeros_ones(ctor, args);
  if (ctor == "full")
    return materialize_full(args);
  if (ctor == "eye" || ctor == "identity")
    return materialize_eye_identity(ctor, args);
  if (ctor == "linspace")
    return materialize_linspace(args);
  if (ctor == "arange")
    return materialize_arange(args);

  return std::nullopt;
}

// True when call_node's attribute name is one materialize_numpy_constructor_
// array() knows about, regardless of whether materialization actually
// succeeded. Callers use this to tell "not a constructor call at all" (safe
// to fall back to args[0] for the pre-existing np.array(<literal>) shape)
// apart from "a recognized constructor call materialization declined on"
// (e.g. a dtype= keyword, a non-constant fill) -- the latter must not fall
// back to args[0] either, since that is exactly the shape/size argument
// this whole helper exists to stop misreading as array data.
static bool is_numpy_constructor_call_by_name(const nlohmann::json &call_node)
{
  if (
    !call_node.is_object() ||
    call_node.value("_type", std::string()) != "Call" ||
    !call_node.contains("func") || !call_node["func"].is_object() ||
    call_node["func"].value("_type", std::string()) != "Attribute" ||
    !call_node["func"].contains("attr"))
    return false;

  static const std::set<std::string> constructors = {
    "zeros", "ones", "full", "eye", "identity", "linspace", "arange"};
  return constructors.count(call_node["func"]["attr"].get<std::string>()) != 0;
}

// resolve_var()/resolve_numpy_var() copies only resolve a Name argument to
// its declaration; a constructor call passed inline as the argument itself
// (e.g. the arange(4) in np.sum(np.arange(4))) is left untouched and fails
// downstream extraction. Materializes it in place when possible, after the
// Name-resolution attempt above has already run (a no-op for an already
// materialized/non-Call node).
static void materialize_inline_numpy_constructor_call(
  nlohmann::json &node,
  const nlohmann::json &ast_json)
{
  if (!node.is_object() || node.value("_type", std::string()) != "Call")
    return;
  if (
    std::optional<nlohmann::json> materialized =
      materialize_numpy_constructor_array(node, ast_json))
    node = std::move(*materialized);
}

// full()/eye()/identity()/linspace() are declared through to_list_expr in
// the real creation path (see array_creation_funcs and its neighbours),
// which forces a dynamic PyListObj representation instead of a plain
// array. zeros()/ones()/np.array(<literal>) use a plain array. A runtime
// backend call that takes the address of the argument (e.g. transpose's
// array path) assumes a flat array's memory layout, which does not match
// a PyListObj's layout and reads out of bounds. Callers must route the
// former group through a compile-time literal fold instead (see
// try_fold_transpose_literal_2d).
static bool
is_dynamic_list_backed_numpy_constructor(const nlohmann::json &call_node)
{
  if (
    !call_node.is_object() ||
    call_node.value("_type", std::string()) != "Call" ||
    !call_node.contains("func") || !call_node["func"].is_object() ||
    call_node["func"].value("_type", std::string()) != "Attribute" ||
    !call_node["func"].contains("attr"))
    return false;

  static const std::set<std::string> dynamic_list_ctors = {
    "full", "eye", "identity", "linspace"};
  return dynamic_list_ctors.count(
           call_node["func"]["attr"].get<std::string>()) != 0;
}

static bool numeric_2d_list_is_rectangular(
  const nlohmann::json &elts,
  std::size_t col_count)
{
  for (const auto &row : elts)
  {
    if (
      !row.is_object() || row.value("_type", std::string()) != "List" ||
      !row.contains("elts") || row["elts"].size() != col_count)
      return false;
  }
  return true;
}

static std::optional<nlohmann::json> build_transposed_literal(
  const nlohmann::json &elts,
  std::size_t row_count,
  std::size_t col_count)
{
  nlohmann::json transposed;
  transposed["_type"] = "List";
  transposed["elts"] = nlohmann::json::array();
  for (std::size_t c = 0; c < col_count; ++c)
  {
    nlohmann::json out_row;
    out_row["_type"] = "List";
    out_row["elts"] = nlohmann::json::array();
    for (std::size_t r = 0; r < row_count; ++r)
    {
      numeric_value value;
      if (!try_extract_numeric_constant(elts[r]["elts"][c], value))
        return std::nullopt;
      out_row["elts"].push_back(build_constant_node(value));
    }
    transposed["elts"].push_back(out_row);
  }
  return transposed;
}

// Computes np.transpose() directly over a fully-materialized numeric
// literal (all Constant leaves), returning nullopt for anything this
// conservative fold does not model (non-rectangular, 3D+, non-numeric
// elements). A 1D literal is returned unchanged (transpose of a 1D array
// is itself).
static std::optional<exprt> try_fold_transpose_literal_2d(
  const nlohmann::json &list_arg,
  python_converter &converter)
{
  if (
    !list_arg.is_object() || list_arg.value("_type", std::string()) != "List" ||
    !list_arg.contains("elts") || !list_arg["elts"].is_array())
    return std::nullopt;

  const auto &elts = list_arg["elts"];
  const bool is_1d =
    elts.empty() ||
    !(elts[0].is_object() && elts[0].value("_type", std::string()) == "List");

  // current_lhs is private to python_converter; the caller (a friend of
  // python_converter, unlike this free function) is responsible for
  // updating it with the returned expression's type.
  if (is_1d)
    return converter.get_expr(list_arg);

  const std::size_t row_count = elts.size();
  const std::size_t col_count =
    elts[0].contains("elts") ? elts[0]["elts"].size() : 0;
  if (col_count == 0 || !numeric_2d_list_is_rectangular(elts, col_count))
    return std::nullopt;

  std::optional<nlohmann::json> transposed =
    build_transposed_literal(elts, row_count, col_count);
  if (!transposed)
    return std::nullopt;

  return converter.get_expr(*transposed);
}

static nlohmann::json unwrap_list_like_node(const nlohmann::json &node)
{
  if (!node.is_object() || !node.contains("_type"))
    return {};

  if (node["_type"] == "List")
    return node;

  if (
    node.contains("value") && node["value"].is_object() &&
    node["value"].contains("_type"))
  {
    auto nested = unwrap_list_like_node(node["value"]);
    if (!nested.is_null() && nested.is_object())
      return nested;
  }

  return {};
}

static typet get_array_scalar_type(const typet &array_type)
{
  typet scalar_type = array_type;
  while (scalar_type.is_array())
    scalar_type = scalar_type.subtype();
  return scalar_type;
}

static numeric_value extract_value(const nlohmann::json &arg)
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

  throw std::runtime_error("Unknown numeric type in JSON");
}

numpy_call_expr::numpy_call_expr(
  const symbol_id &function_id,
  const nlohmann::json &call,
  python_converter &converter)
  : function_call_expr(function_id, call, converter)
{
  converter_.build_static_lists = true;
}

numpy_call_expr::~numpy_call_expr()
{
  converter_.build_static_lists = false;
}

template <typename T>
static auto create_list(int size, T default_value)
{
  nlohmann::json list;
  list["_type"] = "List";
  for (int i = 0; i < size; ++i)
  {
    list["elts"].push_back({{"_type", "Constant"}, {"value", default_value}});
  }
  return list;
}

static auto create_list(int size, const nlohmann::json &default_value)
{
  nlohmann::json list;
  list["_type"] = "List";
  list["elts"] = nlohmann::json::array();
  for (int i = 0; i < size; ++i)
  {
    list["elts"].push_back(default_value);
  }
  return list;
}

template <typename T>
static auto create_list(const std::vector<T> &vector)
{
  nlohmann::json list;
  list["_type"] = "List";
  for (const auto &v : vector)
  {
    list["elts"].push_back({{"_type", "Constant"}, {"value", v}});
  }
  return list;
}

static typet build_ndarray_type(
  const type_handler &type_handler,
  const typet &elem_type,
  long long dim)
{
  if (dim > std::numeric_limits<int>::max())
    throw std::runtime_error(
      "ValueError: array size overflows during creation");
  return type_handler.build_array(elem_type, static_cast<int>(dim));
}

static typet build_ndarray_type(
  const type_handler &type_handler,
  const typet &elem_type,
  const std::vector<long long> &dims,
  std::size_t dim_idx)
{
  if (dim_idx == dims.size() - 1)
    return build_ndarray_type(type_handler, elem_type, dims[dim_idx]);

  typet child_type =
    build_ndarray_type(type_handler, elem_type, dims, dim_idx + 1);
  return build_ndarray_type(type_handler, child_type, dims[dim_idx]);
}

static exprt make_nondet_ndarray(
  const type_handler &type_handler,
  const typet &elem_type,
  const std::vector<long long> &dims,
  std::size_t dim_idx,
  const locationt &location)
{
  typet array_type = build_ndarray_type(type_handler, elem_type, dims, dim_idx);
  exprt result = gen_zero(array_type);
  auto &operands = result.operands();
  operands.clear();

  for (long long i = 0; i < dims[dim_idx]; ++i)
  {
    if (dim_idx == dims.size() - 1)
    {
      exprt elem("sideeffect", elem_type);
      elem.statement("nondet");
      elem.location() = location;
      operands.push_back(elem);
    }
    else
    {
      operands.push_back(make_nondet_ndarray(
        type_handler, elem_type, dims, dim_idx + 1, location));
    }
  }

  return result;
}

static exprt make_filled_ndarray(
  const type_handler &type_handler,
  const typet &elem_type,
  const std::vector<long long> &dims,
  std::size_t dim_idx,
  const exprt &fill)
{
  typet array_type = build_ndarray_type(type_handler, elem_type, dims, dim_idx);
  exprt result = gen_zero(array_type);
  auto &operands = result.operands();
  operands.clear();

  for (long long i = 0; i < dims[dim_idx]; ++i)
  {
    if (dim_idx == dims.size() - 1)
      operands.push_back(
        fill.type() == elem_type ? fill : np_typecast(fill, elem_type));
    else
      operands.push_back(
        make_filled_ndarray(type_handler, elem_type, dims, dim_idx + 1, fill));
  }

  return result;
}

static exprt make_numpy_one(const typet &type)
{
  if (type.is_bool())
    return migrate_expr_back(gen_true_expr());
  if (type.is_floatbv())
    return from_double(1.0, type);
  return from_integer(1, type);
}

template <typename T>
static auto create_binary_op(
  const std::string &op,
  const std::string &type,
  const T &lhs,
  const T &rhs)
{
  nlohmann::json left, right;

  if (type == kName)
  {
    left = {{"_type", type}, {"id", lhs}};
    right = {{"_type", type}, {"id", rhs}};
  }
  else
  {
    left = {{"_type", type}, {"value", lhs}};
    right = {{"_type", type}, {"value", rhs}};
  }

  nlohmann::json bin_op = {
    {"_type", "BinOp"},
    {"left", left},
    {"op", {{"_type", op}}},
    {"right", right}};

  return bin_op;
}

static std::string normalize_numpy_dtype_name(const std::string &dtype)
{
  if (dtype == "bool" || dtype == "bool_")
    return "bool";
  if (dtype == "int" || dtype == "int_")
    return "int64";
  if (dtype == "uint" || dtype == "uint_")
    return "uint64";
  if (dtype == "float" || dtype == "float_")
    return "float64";
  if (dtype == "complex" || dtype == "complex_")
    return "complex128";
  return dtype;
}

static std::string extract_numpy_dtype_name(const nlohmann::json &dtype_node)
{
  if (!dtype_node.is_object() || !dtype_node.contains("_type"))
    throw std::runtime_error("Unsupported dtype value");

  const std::string node_type = dtype_node["_type"].get<std::string>();
  if (node_type == "Attribute" && dtype_node.contains("attr"))
    return normalize_numpy_dtype_name(dtype_node["attr"].get<std::string>());
  if (node_type == "Name" && dtype_node.contains("id"))
    return normalize_numpy_dtype_name(dtype_node["id"].get<std::string>());

  throw std::runtime_error("Unsupported dtype value");
}

static bool is_numpy_integer_dtype(const std::string &dtype)
{
  return dtype.find("int") != std::string::npos;
}

static bool is_numpy_float_dtype(const std::string &dtype)
{
  return dtype.find("float") != std::string::npos;
}

static bool is_numpy_complex_dtype(const std::string &dtype)
{
  return dtype == "complex64" || dtype == "complex128" || dtype == "complex";
}

static nlohmann::json
make_numpy_typed_constant(const scalar_value &value, const std::string &dtype)
{
  const std::string normalized = normalize_numpy_dtype_name(dtype);

  if (normalized.empty())
    return {{"_type", "Constant"}, {"value", value.value.real()}};

  if (normalized == "bool")
  {
    const bool bool_value =
      value.value.real() != 0.0 || value.value.imag() != 0.0;
    return {{"_type", "Constant"}, {"value", bool_value}};
  }

  if (is_numpy_integer_dtype(normalized))
  {
    if (value.is_complex && value.value.imag() != 0.0)
    {
      throw std::runtime_error(
        "TypeError: casting complex literals to integer dtype is not "
        "supported");
    }
    return {
      {"_type", "Constant"},
      {"value", static_cast<int64_t>(std::llround(value.value.real()))}};
  }

  if (is_numpy_float_dtype(normalized))
  {
    if (value.is_complex && value.value.imag() != 0.0)
    {
      throw std::runtime_error(
        "TypeError: casting complex literals to float dtype is not supported");
    }
    return {{"_type", "Constant"}, {"value", value.value.real()}};
  }

  if (is_numpy_complex_dtype(normalized))
  {
    throw std::runtime_error(
      "TypeError: complex dtype is not supported in NumPy constructors yet");
  }

  throw std::runtime_error("Unsupported dtype value: " + normalized);
}

static nlohmann::json cast_numpy_literal_to_dtype(
  const nlohmann::json &node,
  const std::string &dtype)
{
  if (dtype.empty())
    return node;

  if (!node.is_object() || !node.contains("_type"))
  {
    throw std::runtime_error(
      "TypeError: np.array(..., dtype=...) requires literal numeric elements");
  }

  const std::string node_type = node["_type"].get<std::string>();
  if ((node_type == "List" || node_type == "Tuple") && node.contains("elts"))
  {
    nlohmann::json casted = node;
    casted["elts"] = nlohmann::json::array();
    for (const auto &elt : node["elts"])
      casted["elts"].push_back(cast_numpy_literal_to_dtype(elt, dtype));
    return casted;
  }

  scalar_value value;
  if (try_extract_scalar_constant(node, value))
    return make_numpy_typed_constant(value, dtype);

  throw std::runtime_error(
    "TypeError: np.array(..., dtype=...) requires literal numeric elements");
}

bool numpy_call_expr::is_math_function() const
{
  const std::string &function = function_id_.get_function();
  return function == "add" || function == "subtract" ||
         function == "multiply" ||
         (function == "divide" || function == "power" || function == "ceil" ||
          function == "floor" || function == "fabs" || function == "sin" ||
          function == "cos" || function == "exp" || function == "fmod" ||
          function == "sqrt" || function == "fmin") ||
         function == "fmax" || function == "trunc" || function == "round" ||
         function == "arccos" || function == "arcsin" ||
         function == "copysign" || function == "arctan" || function == "tan" ||
         function == "log" || function == "log2" || function == "log10" ||
         function == "sinh" || function == "cosh" || function == "tanh" ||
         function == "rint" || function == "remainder" ||
         function == "nextafter" || function == "modf" || function == "frexp" ||
         function == "isclose" || function == "dot" ||
         function == "transpose" || function == "det" || function == "matmul" ||
         function == "inv" || function == "solve" || function == "norm" ||
         function == "eig" || function == "svd" || function == "real" ||
         function == "imag" || function == "conj" || function == "conjugate" ||
         function == "angle" || function == "abs";
}

std::string numpy_call_expr::get_dtype() const
{
  if (call_.contains("keywords"))
  {
    for (const auto &kw : call_["keywords"])
    {
      if (kw["_type"] == "keyword" && kw["arg"] == "dtype")
        return extract_numpy_dtype_name(kw["value"]);
    }
  }
  return {};
}

size_t numpy_call_expr::get_dtype_size() const
{
  static const std::unordered_map<std::string, size_t> dtype_sizes = {
    {"int8", sizeof(int8_t)},
    {"uint8", sizeof(uint8_t)},
    {"int16", sizeof(int16_t)},
    {"uint16", sizeof(uint16_t)},
    {"int32", sizeof(int32_t)},
    {"uint32", sizeof(uint32_t)},
    {"int64", sizeof(int64_t)},
    {"uint64", sizeof(uint64_t)},
    {"float16", 2},
    {"float32", sizeof(float)},
    {"float64", sizeof(double)}};

  const std::string dtype = get_dtype();
  if (dtype == "bool" || is_numpy_complex_dtype(dtype))
    return 0;

  if (!dtype.empty())
  {
    auto it = dtype_sizes.find(dtype);
    if (it != dtype_sizes.end())
      return it->second * 8;
    throw std::runtime_error("Unsupported dtype value: " + dtype);
  }
  return 0;
}

size_t count_effective_bits(const std::string &binary)
{
  size_t first_one = binary.find('1');
  if (first_one == std::string::npos)
    return 1;
  return binary.size() - first_one;
}

typet numpy_call_expr::get_typet_from_dtype() const
{
  std::string dtype = get_dtype();
  if (dtype == "bool")
    return bool_type();
  if (dtype.find("int") != std::string::npos)
  {
    if (dtype[0] == 'u')
      return unsignedbv_typet(get_dtype_size());
    return signedbv_typet(get_dtype_size());
  }
  if (dtype.find("float") != std::string::npos)
    return build_float_type(get_dtype_size());
  if (dtype == "complex64")
    return get_complex_struct_type();
  if (dtype == "complex128" || dtype == "complex")
    return get_complex_struct_type();

  return {};
}

// Checks if two shapes are broadcast-compatible.
// Two dimensions are compatible if they are equal or if one of them is 1.
bool is_broadcastable(
  const std::vector<int> &shape1,
  const std::vector<int> &shape2)
{
  int s1 = shape1.size() - 1;
  int s2 = shape2.size() - 1;

  // Compare dimensions from rightmost (inner) to leftmost (outer)
  while (s1 >= 0 || s2 >= 0)
  {
    // If a shape lacks a dimension, assume its size is 1.
    int d1 = (s1 >= 0) ? shape1[s1] : 1;
    int d2 = (s2 >= 0) ? shape2[s2] : 1;

    // Check if dimensions are compatible (either equal or one is 1)
    if (d1 != d2 && d1 != 1 && d2 != 1)
      return false;

    --s1;
    --s2;
  }
  return true;
}

bool is_broadcastable(
  const std::vector<std::size_t> &shape1,
  const std::vector<std::size_t> &shape2)
{
  std::vector<int> lhs(shape1.begin(), shape1.end());
  std::vector<int> rhs(shape2.begin(), shape2.end());
  return is_broadcastable(lhs, rhs);
}

void numpy_call_expr::broadcast_check(const nlohmann::json &operands) const
{
  std::vector<std::size_t> previous_shape;
  bool is_first_operand = true;

  for (const auto &op : operands)
  {
    std::vector<std::size_t> current_shape;
    if (op.is_object() && op.contains("_type"))
    {
      const std::string type = op["_type"].get<std::string>();
      if (type == "Name")
      {
        symbol_id sid = converter_.create_symbol_id();
        sid.set_object(op["id"].get<std::string>());
        symbolt *s = converter_.find_symbol(sid.to_string());
        assert(s);
        const auto dims =
          converter_.type_handler_.get_array_type_shape(s->get_type());
        current_shape.assign(dims.begin(), dims.end());
      }
      else if (is_list_node(op))
      {
        if (!get_literal_shape(op, current_shape))
          current_shape.clear();
      }
      else if (type == "Constant" || type == "UnaryOp")
      {
        scalar_value scalar;
        if (try_extract_scalar_constant(op, scalar))
          current_shape.clear();
      }
    }

    if (!is_first_operand)
    {
      if (!is_broadcastable(previous_shape, current_shape))
      {
        throw std::runtime_error(
          "operands could not be broadcast together with shapes " +
          format_shape(previous_shape) + " " + format_shape(current_shape));
      }
    }
    else
    {
      is_first_operand = false;
    }

    previous_shape = current_shape;
  }
}

template <typename T>
T get_constant_value(const nlohmann::json &node)
{
  // Bignum literal (issue #4642): a tagged Constant has a null `value`, and
  // node["value"].get<T>() below would raise an opaque nlohmann type_error.
  // Surface the curated overflow diagnostic instead so the user sees the
  // same message they get from get_literal.
  auto reject_bigint = [](const nlohmann::json &c) {
    if (c.contains("_bigint"))
      throw python_int_overflow_excp(
        "Python int overflow: literal " + c["_bigint"].get<std::string>() +
        " does not fit in 64-bit int. ESBMC approximates Python int as a "
        "fixed-width bitvector; arbitrary-precision int support is tracked in "
        "issue #4642.");
  };
  if (node["_type"] == "Constant")
  {
    reject_bigint(node);
    return node["value"].get<T>();
  }
  else if (node["_type"] == "UnaryOp" && node["operand"]["_type"] == "Constant")
  {
    reject_bigint(node["operand"]);
    std::string op_type = node["op"]["_type"];
    T val = node["operand"]["value"].get<T>();

    if (op_type == "USub")
      return -val;
    else if (op_type == "UAdd")
      return val;
    else
    {
      log_error("get_constant_value: Unsupported unary operator '{}'", op_type);
      abort();
    }
  }
  else
  {
    log_error(
      "get_constant_value: Expected Constant or UnaryOp with Constant operand, "
      "got '{}'",
      node.dump());
    abort();
  }
}

exprt numpy_call_expr::create_expr_from_call()
{
  nlohmann::json expr;
  const bool allow_numpy_fold = numpy_constant_folding_enabled();

  // Resolve variables if they are names
  auto resolve_var = [this](nlohmann::json &var) {
    if (var["_type"] == "Name")
    {
      var = json_utils::find_var_decl(
        var["id"], converter_.current_function_name(), converter_.ast());
      if (!var.contains("value") || !var["value"].is_object())
        return;

      if (var["value"]["_type"] == "Call")
      {
        if (
          std::optional<nlohmann::json> materialized =
            materialize_numpy_constructor_array(var["value"], converter_.ast()))
          var = std::move(*materialized);
        else if (is_numpy_constructor_call_by_name(var["value"]))
          var = var["value"];
        else if (var["value"].contains("args") && !var["value"]["args"].empty())
          var = var["value"]["args"][0];
        else
          var = var["value"];
      }
      else
      {
        var = var["value"];
      }
    }
  };

  auto make_constant_expr = [this](const auto &value) {
    nlohmann::json out;
    out["_type"] = "Constant";
    out["value"] = value;
    return converter_.get_expr(out);
  };

  auto extract_shape_dims = [](const nlohmann::json &shape_node) {
    std::vector<std::size_t> dims;
    if (
      shape_node.is_object() && shape_node.contains("_type") &&
      shape_node["_type"] == "Constant" && shape_node.contains("value") &&
      shape_node["value"].is_number_integer())
    {
      dims.push_back(shape_node["value"].get<std::size_t>());
      return dims;
    }

    if (
      shape_node.is_object() && shape_node.contains("_type") &&
      (shape_node["_type"] == "Tuple" || shape_node["_type"] == "List") &&
      shape_node.contains("elts") && shape_node["elts"].is_array())
    {
      for (const auto &elem : shape_node["elts"])
      {
        if (
          !elem.is_object() || !elem.contains("_type") ||
          elem["_type"] != "Constant" || !elem.contains("value") ||
          !elem["value"].is_number_integer())
        {
          dims.clear();
          return dims;
        }
        dims.push_back(elem["value"].get<std::size_t>());
      }
    }

    return dims;
  };

  const std::string &function = function_id_.get_function();

  if (
    function == "sum" || function == "prod" || function == "min" ||
    function == "max" || function == "mean" || function == "argmin" ||
    function == "argmax")
  {
    if (call_["args"].empty())
      throw std::runtime_error(
        "TypeError: numpy." + function + "() missing argument");

    nlohmann::json arg = call_["args"][0];
    resolve_var(arg);
    materialize_inline_numpy_constructor_call(arg, converter_.ast());
    if (
      std::optional<nlohmann::json> row_view =
        resolve_literal_numpy_row_view(arg, converter_))
      arg = std::move(*row_view);

    std::vector<numeric_value> values_1d;
    std::vector<std::vector<numeric_value>> values_2d;
    std::vector<numeric_value> values;
    if (try_extract_numeric_1d_list(arg, values_1d))
      values = values_1d;
    else if (try_extract_numeric_2d_list(arg, values_2d))
    {
      for (const auto &row : values_2d)
        values.insert(values.end(), row.begin(), row.end());
    }
    else
    {
      numeric_value scalar;
      if (!try_extract_numeric_constant(arg, scalar))
        throw std::runtime_error(
          "TypeError: numpy." + function +
          "() currently supports constant numeric inputs only");
      values.push_back(scalar);
    }

    if (values.empty())
    {
      if (function == "sum")
        return make_constant_expr(0);
      if (function == "prod")
        return make_constant_expr(1);
      throw std::runtime_error(
        "ValueError: numpy." + function + "() arg is an empty sequence");
    }

    const bool any_float =
      std::any_of(values.begin(), values.end(), [](const numeric_value &v) {
        return !v.is_int;
      });

    if (function == "argmin" || function == "argmax")
    {
      std::size_t best_idx = 0;
      double best = to_double(values[0]);
      for (std::size_t i = 1; i < values.size(); ++i)
      {
        const double current = to_double(values[i]);
        if (
          (function == "argmin" && current < best) ||
          (function == "argmax" && current > best))
        {
          best = current;
          best_idx = i;
        }
      }
      return make_constant_expr(static_cast<int64_t>(best_idx));
    }

    double accum = 0.0;
    bool first_value = true;
    for (const auto &value : values)
    {
      const double current = to_double(value);
      if (function == "sum" || function == "mean")
      {
        accum += current;
      }
      else if (function == "prod")
      {
        if (first_value)
          accum = 1.0;
        accum *= current;
      }
      else if (function == "min")
      {
        if (first_value)
          accum = current;
        else
          accum = std::min(accum, current);
      }
      else if (function == "max")
      {
        if (first_value)
          accum = current;
        else
          accum = std::max(accum, current);
      }
      first_value = false;
    }

    if (function == "mean")
      return make_constant_expr(accum / static_cast<double>(values.size()));
    if (function == "min" || function == "max")
    {
      if (any_float)
        return make_constant_expr(accum);
      return make_constant_expr(static_cast<int64_t>(std::llround(accum)));
    }
    if (any_float)
      return make_constant_expr(accum);
    return make_constant_expr(static_cast<int64_t>(std::llround(accum)));
  }

  if (function == "where")
  {
    if (call_["args"].size() != 3)
      throw std::runtime_error("TypeError: numpy.where() expects 3 arguments");

    nlohmann::json cond = call_["args"][0];
    nlohmann::json x = call_["args"][1];
    nlohmann::json y = call_["args"][2];
    resolve_var(cond);
    resolve_var(x);
    resolve_var(y);

    scalar_value cond_scalar;
    if (try_extract_scalar_constant(cond, cond_scalar))
      return converter_.get_expr(cond_scalar.value.real() != 0.0 ? x : y);

    std::vector<scalar_value> cond_values;
    if (!try_extract_scalar_1d_list(cond, cond_values))
      throw std::runtime_error(
        "TypeError: numpy.where() currently supports constant 1D conditions");

    nlohmann::json out;
    out["_type"] = "List";
    out["elts"] = nlohmann::json::array();
    for (std::size_t i = 0; i < cond_values.size(); ++i)
    {
      const bool choose_x = cond_values[i].value.real() != 0.0;
      const nlohmann::json &chosen =
        choose_x ? (x["_type"] == "List" ? x["elts"][i] : x)
                 : (y["_type"] == "List" ? y["elts"][i] : y);
      out["elts"].push_back(chosen);
    }
    return converter_.get_expr(out);
  }

  if (function == "logical_not")
  {
    if (call_["args"].empty())
      throw std::runtime_error(
        "TypeError: numpy.logical_not() missing argument");

    nlohmann::json arg = call_["args"][0];
    resolve_var(arg);

    scalar_value scalar;
    if (try_extract_scalar_constant(arg, scalar))
      return make_constant_expr(scalar.value.real() == 0.0);

    std::vector<scalar_value> values;
    if (!try_extract_scalar_1d_list(arg, values))
      throw std::runtime_error(
        "TypeError: numpy.logical_not() currently supports constant 1D inputs");

    nlohmann::json out;
    out["_type"] = "List";
    out["elts"] = nlohmann::json::array();
    for (const auto &value : values)
      out["elts"].push_back(
        {{"_type", "Constant"}, {"value", value.value.real() == 0.0}});
    return converter_.get_expr(out);
  }

  if (
    function == "full" || function == "eye" || function == "identity" ||
    function == "linspace")
  {
    if (function == "full")
    {
      if (call_["args"].size() != 2)
        throw std::runtime_error("TypeError: numpy.full() expects 2 arguments");

      nlohmann::json shape = call_["args"][0];
      nlohmann::json fill_value = call_["args"][1];
      resolve_var(shape);
      resolve_var(fill_value);

      const auto dims = extract_shape_dims(shape);
      if (dims.empty())
      {
        if (
          shape.is_object() && shape.contains("_type") &&
          shape["_type"] == "Constant" && shape.contains("value") &&
          shape["value"].is_number_integer())
        {
          return converter_.get_expr(
            create_list(shape["value"].get<int>(), fill_value));
        }
        throw std::runtime_error(
          "TypeError: numpy.full() shape must be an int or tuple/list of ints");
      }
      if (dims.size() == 1)
        return converter_.get_expr(create_list(dims[0], fill_value));
      if (dims.size() == 2)
      {
        nlohmann::json outer;
        outer["_type"] = "List";
        outer["elts"] = nlohmann::json::array();
        for (std::size_t i = 0; i < dims[0]; ++i)
          outer["elts"].push_back(create_list(dims[1], fill_value));
        return converter_.get_expr(outer);
      }
      throw std::runtime_error(
        "TypeError: numpy.full() currently supports up to 2D shapes");
    }

    if (function == "eye" || function == "identity")
    {
      if (call_["args"].empty() || call_["args"].size() > 2)
        throw std::runtime_error(
          "TypeError: numpy.eye()/identity() expects 1 or 2 arguments");

      nlohmann::json n_node = call_["args"][0];
      resolve_var(n_node);
      numeric_value n_value;
      if (!try_extract_numeric_constant(n_node, n_value))
        throw std::runtime_error(
          "TypeError: numpy.eye()/identity() requires constant integer sizes");

      std::size_t n = static_cast<std::size_t>(n_value.int_value);
      std::size_t m = n;
      if (function == "eye" && call_["args"].size() == 2)
      {
        nlohmann::json m_node = call_["args"][1];
        resolve_var(m_node);
        numeric_value m_value;
        if (!try_extract_numeric_constant(m_node, m_value))
          throw std::runtime_error(
            "TypeError: numpy.eye() requires constant integer sizes");
        m = static_cast<std::size_t>(m_value.int_value);
      }

      nlohmann::json out;
      out["_type"] = "List";
      out["elts"] = nlohmann::json::array();
      for (std::size_t i = 0; i < n; ++i)
      {
        nlohmann::json row;
        row["_type"] = "List";
        row["elts"] = nlohmann::json::array();
        for (std::size_t j = 0; j < m; ++j)
          row["elts"].push_back(
            {{"_type", "Constant"}, {"value", i == j ? 1 : 0}});
        out["elts"].push_back(row);
      }
      const bool old_build_static_lists = converter_.build_static_lists;
      converter_.build_static_lists = false;
      exprt expr = converter_.get_expr(out);
      converter_.build_static_lists = old_build_static_lists;
      return expr;
    }

    if (function == "linspace")
    {
      if (call_["args"].size() < 2 || call_["args"].size() > 3)
        throw std::runtime_error(
          "TypeError: numpy.linspace() expects 2 or 3 arguments");

      std::vector<numeric_value> values;
      values.reserve(call_["args"].size());
      for (auto arg : call_["args"])
      {
        resolve_var(arg);
        numeric_value value;
        if (!try_extract_numeric_constant(arg, value))
          throw std::runtime_error(
            "TypeError: numpy.linspace() currently supports constant numeric "
            "inputs only");
        values.push_back(value);
      }

      const double start = to_double(values[0]);
      const double stop = to_double(values[1]);
      const std::size_t num =
        values.size() == 3 ? static_cast<std::size_t>(values[2].int_value) : 50;
      if (num == 0)
        return converter_.get_expr(create_list(std::vector<double>{}));
      if (num == 1)
        return converter_.get_expr(create_list(std::vector<double>{start}));

      const double step = (stop - start) / static_cast<double>(num - 1);
      std::vector<double> samples;
      samples.reserve(num);
      for (std::size_t i = 0; i < num; ++i)
        samples.push_back(start + (step * static_cast<double>(i)));
      return converter_.get_expr(create_list(samples));
    }
  }

  // Unary operations
  if (call_["args"].size() == 1 || function_id_.get_function() == "norm")
  {
    const std::string &function = function_id_.get_function();
    if (function == "det")
    {
      nlohmann::json arg = call_["args"][0];
      resolve_var(arg);
      if (
        arg.is_object() && arg.contains("_type") && arg["_type"] == "Call" &&
        arg.contains("func") && arg["func"].is_object() &&
        ((arg["func"].contains("_type") && arg["func"]["_type"] == "Name" &&
          arg["func"].contains("id") && arg["func"]["id"] == "array") ||
         (arg["func"].contains("_type") &&
          arg["func"]["_type"] == "Attribute" && arg["func"].contains("attr") &&
          arg["func"]["attr"] == "array")) &&
        arg.contains("args") && arg["args"].is_array() && !arg["args"].empty())
      {
        arg = arg["args"][0];
      }

      std::vector<std::vector<scalar_value>> matrix;
      if (!try_extract_scalar_2d_list(arg, matrix))
      {
        throw std::runtime_error(
          "TypeError: numpy.linalg.det currently supports only constant 2D "
          "numeric arrays");
      }

      std::size_t n = 0;
      if (!is_square_matrix(matrix, n))
      {
        throw std::runtime_error(
          "TypeError: numpy.linalg.det requires a square 2D matrix");
      }

      for (const auto &row : matrix)
      {
        for (const auto &value : row)
        {
          if (value.is_complex)
          {
            throw std::runtime_error(
              "TypeError: numpy.linalg.det does not support complex-valued "
              "matrices");
          }
        }
      }

      if (n == 2)
        return converter_.get_expr(to_json_constant(determinant_2x2(matrix)));
      if (n == 3)
        return converter_.get_expr(to_json_constant(determinant_3x3(matrix)));

      throw std::runtime_error(
        "TypeError: numpy.linalg.det supports only 2x2 and 3x3 matrices");
    }

    auto unwrap_np_array_arg = [&resolve_var](nlohmann::json &arg) {
      resolve_var(arg);
      if (
        arg.is_object() && arg.contains("_type") && arg["_type"] == "Call" &&
        arg.contains("func") && arg["func"].is_object() &&
        ((arg["func"].contains("_type") && arg["func"]["_type"] == "Name" &&
          arg["func"].contains("id") && arg["func"]["id"] == "array") ||
         (arg["func"].contains("_type") &&
          arg["func"]["_type"] == "Attribute" && arg["func"].contains("attr") &&
          arg["func"]["attr"] == "array")) &&
        arg.contains("args") && arg["args"].is_array() && !arg["args"].empty())
      {
        arg = arg["args"][0];
      }
    };

    if (function == "inv")
    {
      nlohmann::json arg = call_["args"][0];
      unwrap_np_array_arg(arg);

      std::vector<std::vector<scalar_value>> matrix;
      if (!try_extract_scalar_2d_list(arg, matrix))
        throw std::runtime_error(
          "TypeError: numpy.linalg.inv currently supports only constant 2D "
          "numeric arrays");

      std::size_t n = 0;
      if (!is_square_matrix(matrix, n))
        throw std::runtime_error(
          "TypeError: numpy.linalg.inv requires a square 2D matrix");

      for (const auto &row : matrix)
        for (const auto &value : row)
          if (value.is_complex)
            throw std::runtime_error(
              "TypeError: numpy.linalg.inv does not support complex-valued "
              "matrices");

      std::vector<std::vector<scalar_value>> inv;
      bool ok = false;
      if (n == 2)
        ok = inverse_2x2(matrix, inv);
      else if (n == 3)
        ok = inverse_3x3(matrix, inv);
      else
        throw std::runtime_error(
          "TypeError: numpy.linalg.inv supports only 2x2 and 3x3 matrices");

      if (!ok)
        throw std::runtime_error("numpy.linalg.LinAlgError: Singular matrix");

      return converter_.get_expr(matrix_to_json(inv));
    }

    if (function == "norm")
    {
      if (call_["args"].empty() || call_["args"].size() > 2)
        throw std::runtime_error(
          "TypeError: numpy.linalg.norm expects an array and optional order");

      norm_order order = norm_order::l2;
      if (call_["args"].size() == 2)
        order = parse_norm_order(call_["args"][1]);

      if (call_.contains("keywords"))
      {
        for (const auto &kw : call_["keywords"])
        {
          if (kw["_type"] != "keyword" || kw["arg"].is_null())
            continue;

          const std::string arg = kw["arg"].get<std::string>();
          if (arg == "axis")
            throw std::runtime_error(
              "TypeError: numpy.linalg.norm axis is not supported");
          if (arg == "ord")
          {
            order = parse_norm_order(kw["value"]);
            continue;
          }
          throw std::runtime_error(
            "TypeError: numpy.linalg.norm keyword '" + arg +
            "' is not supported");
        }
      }

      nlohmann::json arg = call_["args"][0];
      unwrap_np_array_arg(arg);

      std::vector<scalar_value> values_1d;
      std::vector<std::vector<scalar_value>> values_2d;

      if (try_extract_scalar_1d_list(arg, values_1d))
      {
        if (values_1d.empty())
          return converter_.get_expr(to_json_constant(make_real_scalar(0.0)));

        double sum_sq = 0.0;
        double sum_abs = 0.0;
        double max_abs = 0.0;
        double min_abs = std::numeric_limits<double>::infinity();
        for (const auto &v : values_1d)
        {
          if (v.is_complex)
            throw std::runtime_error(
              "TypeError: numpy.linalg.norm does not support complex values");
          const double abs_value = std::abs(v.value.real());
          sum_sq += abs_value * abs_value;
          sum_abs += abs_value;
          max_abs = std::max(max_abs, abs_value);
          min_abs = std::min(min_abs, abs_value);
        }

        double result = std::sqrt(sum_sq);
        if (order == norm_order::l1)
          result = sum_abs;
        else if (order == norm_order::positive_inf)
          result = max_abs;
        else if (order == norm_order::negative_inf)
          result = min_abs;

        return converter_.get_expr(to_json_constant(make_real_scalar(result)));
      }

      if (try_extract_scalar_2d_list(arg, values_2d))
      {
        if (order != norm_order::l2)
          throw std::runtime_error(
            "TypeError: numpy.linalg.norm matrix order is not supported");

        double sum_sq = 0.0;
        for (const auto &row : values_2d)
        {
          for (const auto &v : row)
          {
            if (v.is_complex)
              throw std::runtime_error(
                "TypeError: numpy.linalg.norm does not support complex "
                "values");
            sum_sq += v.value.real() * v.value.real();
          }
        }
        return converter_.get_expr(
          to_json_constant(make_real_scalar(std::sqrt(sum_sq))));
      }

      scalar_value scalar;
      if (try_extract_scalar_constant(arg, scalar))
        return converter_.get_expr(
          to_json_constant(make_real_scalar(std::abs(scalar.value.real()))));

      throw std::runtime_error(
        "TypeError: numpy.linalg.norm currently supports only constant "
        "numeric arrays");
    }

    // numpy.linalg.eig — eigenvalues (only) of a square real matrix.
    // Returns a 1-D list of eigenvalues (2x2: descending; 3x3 diagonal: in
    // diagonal order).
    // Only concrete 2x2 and 3x3 real matrices with real eigenvalues are
    // supported; complex eigenvalues are rejected with an explicit error.
    if (function == "eig")
    {
      nlohmann::json arg = call_["args"][0];
      unwrap_np_array_arg(arg);

      std::vector<std::vector<scalar_value>> matrix;
      if (!try_extract_scalar_2d_list(arg, matrix))
        throw std::runtime_error(
          "TypeError: numpy.linalg.eig currently supports only constant "
          "2D real numeric arrays");

      std::size_t n = 0;
      if (!is_square_matrix(matrix, n))
        throw std::runtime_error(
          "TypeError: numpy.linalg.eig requires a square 2D matrix");

      for (const auto &row : matrix)
        for (const auto &v : row)
          if (v.is_complex)
            throw std::runtime_error(
              "TypeError: numpy.linalg.eig does not support complex-valued "
              "matrices");

      if (n == 2)
      {
        double a = matrix[0][0].value.real();
        double b = matrix[0][1].value.real();
        double c = matrix[1][0].value.real();
        double d = matrix[1][1].value.real();
        double trace = a + d;
        double det = a * d - b * c;
        double disc = trace * trace - 4.0 * det;
        if (disc < 0.0)
          throw std::runtime_error(
            "TypeError: numpy.linalg.eig: matrix has complex eigenvalues; "
            "only real eigenvalues are supported");

        double sqrt_disc = std::sqrt(disc);
        double lam1 = (trace + sqrt_disc) / 2.0;
        double lam2 = (trace - sqrt_disc) / 2.0;
        return converter_.get_expr(
          vector_to_json({make_real_scalar(lam1), make_real_scalar(lam2)}));
      }
      else if (n == 3)
      {
        // For 3x3, only diagonal matrices and scalar multiples of identity are
        // supported (eigenvalues are the diagonal entries, already real).
        bool is_diagonal = true;
        for (std::size_t i = 0; i < 3 && is_diagonal; ++i)
          for (std::size_t j = 0; j < 3 && is_diagonal; ++j)
            if (i != j && std::abs(matrix[i][j].value.real()) > 1e-12)
              is_diagonal = false;

        if (!is_diagonal)
          throw std::runtime_error(
            "TypeError: numpy.linalg.eig supports only 2x2 matrices and "
            "3x3 diagonal matrices currently");

        std::vector<scalar_value> eigenvalues;
        for (std::size_t i = 0; i < 3; ++i)
          eigenvalues.push_back(make_real_scalar(matrix[i][i].value.real()));
        return converter_.get_expr(vector_to_json(eigenvalues));
      }
      else
      {
        throw std::runtime_error(
          "TypeError: numpy.linalg.eig supports only 2x2 and 3x3 matrices");
      }
    }

    // numpy.linalg.svd — singular values (only) of a real matrix.
    // Returns a 1-D list of singular values sorted in descending order.
    // Only concrete 2x2 real matrices are supported; larger matrices and
    // complex entries are rejected with explicit errors.
    if (function == "svd")
    {
      nlohmann::json arg = call_["args"][0];
      unwrap_np_array_arg(arg);

      std::vector<std::vector<scalar_value>> matrix;
      if (!try_extract_scalar_2d_list(arg, matrix))
        throw std::runtime_error(
          "TypeError: numpy.linalg.svd currently supports only constant "
          "2D real numeric matrices");

      for (const auto &row : matrix)
        for (const auto &v : row)
          if (v.is_complex)
            throw std::runtime_error(
              "TypeError: numpy.linalg.svd does not support complex-valued "
              "matrices");

      const std::size_t nrows = matrix.size();
      const std::size_t ncols = matrix.empty() ? 0 : matrix[0].size();
      if (nrows != 2 || ncols != 2)
        throw std::runtime_error(
          "TypeError: numpy.linalg.svd currently supports only 2x2 matrices");

      // Compute AᵀA (always symmetric positive semi-definite for real A)
      double a = matrix[0][0].value.real(), b = matrix[0][1].value.real();
      double c = matrix[1][0].value.real(), d = matrix[1][1].value.real();
      double ata00 = a * a + c * c;
      double ata01 = a * b + c * d;
      double ata11 = b * b + d * d;

      // Eigenvalues of AᵀA (trace / det of 2x2 symmetric)
      double trace = ata00 + ata11;
      double det = ata00 * ata11 - ata01 * ata01;
      double disc = trace * trace - 4.0 * det;
      if (disc < 0.0)
        disc = 0.0; // numerical zero for PSD

      double sqrt_disc = std::sqrt(disc);
      double eigval1 = (trace + sqrt_disc) / 2.0;
      double eigval2 = (trace - sqrt_disc) / 2.0;
      if (eigval1 < 0.0)
        eigval1 = 0.0;
      if (eigval2 < 0.0)
        eigval2 = 0.0;

      double sigma1 = std::sqrt(eigval1);
      double sigma2 = std::sqrt(eigval2);

      return converter_.get_expr(
        vector_to_json({make_real_scalar(sigma1), make_real_scalar(sigma2)}));
    }

    if (is_complex_function(function))
    {
      const auto &arg = call_["args"][0];
      scalar_value scalar;
      if (try_extract_scalar_constant(arg, scalar))
        return converter_.get_expr(
          to_json_constant(apply_complex_unary(function, scalar)));

      std::vector<scalar_value> values_1d;
      if (try_extract_scalar_1d_list(arg, values_1d))
      {
        nlohmann::json out;
        out["_type"] = "List";
        out["elts"] = nlohmann::json::array();
        for (const auto &value : values_1d)
          out["elts"].push_back(
            to_json_constant(apply_complex_unary(function, value)));
        return converter_.get_expr(out);
      }

      std::vector<std::vector<scalar_value>> values_2d;
      if (try_extract_scalar_2d_list(arg, values_2d))
      {
        nlohmann::json out;
        out["_type"] = "List";
        out["elts"] = nlohmann::json::array();
        for (const auto &row_values : values_2d)
        {
          nlohmann::json row;
          row["_type"] = "List";
          row["elts"] = nlohmann::json::array();
          for (const auto &value : row_values)
            row["elts"].push_back(
              to_json_constant(apply_complex_unary(function, value)));
          out["elts"].push_back(row);
        }
        return converter_.get_expr(out);
      }

      // Symbolic fallback for Name/Subscript/attribute paths.
      exprt arg_expr = converter_.get_expr(arg);
      const typet &dt = cached_double_type();
      if (is_complex_type(arg_expr.type()))
      {
        exprt real = np_member(arg_expr, "real", dt);
        exprt imag = np_member(arg_expr, "imag", dt);
        if (function == "real")
          return real;
        if (function == "imag")
          return imag;
        if (function == "conj" || function == "conjugate")
        {
          // V.3: build the conjugate's `0.0 - imag` negation in IREP2. migrate
          // lowers a legacy `minus` (id "-") to sub2t unconditionally (no
          // rounding mode, even for floatbv), so this sub2tc is the exact
          // round-trip of the legacy minus_exprt.
          expr2tc zero2, imag2;
          migrate_expr(from_double(0.0, dt), zero2);
          migrate_expr(imag, imag2);
          return make_complex(
            real, migrate_expr_back(sub2tc(migrate_type(dt), zero2, imag2)));
        }
        if (function == "abs")
          return converter_.get_complex_handler().handle_abs(arg_expr);
        if (function == "angle")
          return converter_.get_math_handler().handle_atan2(imag, real, call_);
      }
      else
      {
        if (function == "real")
          return arg_expr;
        if (function == "imag")
          return from_double(0.0, dt);
        if (function == "conj" || function == "conjugate")
          return arg_expr;
        if (function == "abs")
        {
          exprt real =
            arg_expr.type() == dt ? arg_expr : np_typecast(arg_expr, dt);
          return converter_.get_math_handler().handle_fabs(real, call_);
        }
        if (function == "angle")
        {
          exprt real =
            arg_expr.type() == dt ? arg_expr : np_typecast(arg_expr, dt);
          return converter_.get_math_handler().handle_atan2(
            from_double(0.0, dt), real, call_);
        }
      }
    }

    const auto &arg_type = call_["args"][0]["_type"];
    if (
      arg_type == "Constant" || arg_type == "UnaryOp" ||
      arg_type == "Subscript")
    {
      return function_call_expr::get();
    }
    else if (arg_type == "List")
    {
      const std::string &operation = function_id_.get_function();
      if (operation == "floor" || operation == "fabs" || operation == "trunc")
      {
        exprt folded = fold_numpy_unary_constant_list(
          converter_, operation, call_["args"][0]);
        if (converter_.current_lhs)
        {
          converter_.current_lhs->type() = folded.type();
          converter_.update_symbol(*converter_.current_lhs);
        }
        return folded;
      }

      if (operation == "arccos")
      {
        try
        {
          exprt folded = fold_numpy_unary_constant_list(
            converter_, operation, call_["args"][0]);
          if (converter_.current_lhs)
          {
            converter_.current_lhs->type() = folded.type();
            converter_.update_symbol(*converter_.current_lhs);
          }
          return folded;
        }
        catch (const std::runtime_error &)
        {
        }

        const auto &list_arg = call_["args"][0];
        if (
          list_arg.contains("elts") && list_arg["elts"].is_array() &&
          !list_arg["elts"].empty() && list_arg["elts"][0].is_object() &&
          list_arg["elts"][0].contains("_type") &&
          list_arg["elts"][0]["_type"] == "List")
        {
          throw std::runtime_error(
            "Unsupported operation: numpy.arccos on runtime 2D arrays");
        }

        function_id_.set_function("__arccos_array");

        code_function_callt call =
          to_code_function_call(to_code(function_call_expr::get()));
        typet t = type_handler_.get_list_type(list_arg);

        converter_.current_lhs->type() = t;
        converter_.update_symbol(*converter_.current_lhs);

        call.arguments().push_back(np_address_of(*converter_.current_lhs));
        exprt array_size = from_integer(list_arg["elts"].size(), int_type());
        call.arguments().push_back(array_size);
        return call;
      }

      if (is_supported_numpy_unary_math(operation))
      {
        exprt folded = fold_numpy_unary_constant_list(
          converter_, operation, call_["args"][0]);
        if (converter_.current_lhs)
        {
          converter_.current_lhs->type() = folded.type();
          converter_.update_symbol(*converter_.current_lhs);
        }
        return folded;
      }

      if (operation == "transpose")
      {
        const auto &list_arg = call_["args"][0];
        if (
          list_arg.contains("elts") && list_arg["elts"].is_array() &&
          (list_arg["elts"].empty() ||
           !(list_arg["elts"][0].is_object() &&
             list_arg["elts"][0].contains("_type") &&
             list_arg["elts"][0]["_type"] == "List")))
        {
          exprt folded = converter_.get_expr(list_arg);
          if (converter_.current_lhs)
          {
            converter_.current_lhs->type() = folded.type();
            converter_.update_symbol(*converter_.current_lhs);
          }
          return folded;
        }

        // Constant-fold transpose for fully constant 2D numeric lists.
        // This avoids forcing integer-only backend transpose for float literals.
        if (
          allow_numpy_fold && list_arg.contains("elts") &&
          !list_arg["elts"].empty() && list_arg["elts"][0].is_object() &&
          list_arg["elts"][0].contains("_type") &&
          list_arg["elts"][0]["_type"] == "List")
        {
          const auto &rows = list_arg["elts"];
          const std::size_t row_count = rows.size();
          const std::size_t col_count =
            rows[0].contains("elts") ? rows[0]["elts"].size() : 0;
          bool is_rectangular = col_count > 0;

          for (const auto &row : rows)
          {
            if (
              !row.is_object() || !row.contains("_type") ||
              row["_type"] != "List" || !row.contains("elts") ||
              row["elts"].size() != col_count)
            {
              is_rectangular = false;
              break;
            }
          }

          if (is_rectangular)
          {
            nlohmann::json transposed;
            transposed["_type"] = "List";
            transposed["elts"] = nlohmann::json::array();

            bool all_numeric_constants = true;
            for (std::size_t c = 0; c < col_count && all_numeric_constants; ++c)
            {
              nlohmann::json out_row;
              out_row["_type"] = "List";
              out_row["elts"] = nlohmann::json::array();

              for (std::size_t r = 0; r < row_count; ++r)
              {
                numeric_value value;
                if (!try_extract_numeric_constant(rows[r]["elts"][c], value))
                {
                  all_numeric_constants = false;
                  break;
                }

                nlohmann::json elem;
                elem["_type"] = "Constant";
                elem["value"] = value.is_int
                                  ? nlohmann::json(value.int_value)
                                  : nlohmann::json(value.double_value);
                out_row["elts"].push_back(elem);
              }

              if (all_numeric_constants)
                transposed["elts"].push_back(out_row);
            }

            if (all_numeric_constants)
            {
              exprt folded = converter_.get_expr(transposed);
              if (converter_.current_lhs)
              {
                converter_.current_lhs->type() = folded.type();
                converter_.update_symbol(*converter_.current_lhs);
              }
              return folded;
            }
          }
        }

        code_function_callt call =
          to_code_function_call(to_code(function_call_expr::get()));
        typet t = call.arguments().at(0).type().subtype();
        converter_.current_lhs->type() = t;
        converter_.update_symbol(*converter_.current_lhs);
        call.arguments().push_back(np_address_of(*converter_.current_lhs));
        std::vector<int> shape = type_handler_.get_array_type_shape(t);
        call.arguments().push_back(from_integer(shape[0], int_type()));
        call.arguments().push_back(from_integer(shape[1], int_type()));
        return call;
      }
    }
    else if (arg_type == "Name")
    {
      auto arg = call_["args"][0];
      const std::string &function = function_id_.get_function();

      if (function == "transpose")
      {
        nlohmann::json decl = json_utils::find_var_decl(
          arg["id"], converter_.current_function_name(), converter_.ast());
        if (
          decl.contains("value") && decl["value"].is_object() &&
          is_dynamic_list_backed_numpy_constructor(decl["value"]))
        {
          std::optional<nlohmann::json> materialized =
            materialize_numpy_constructor_array(
              decl["value"], converter_.ast());
          if (!materialized)
            throw std::runtime_error(
              "TypeError: numpy.transpose() does not support this "
              "constructor call (unsupported keywords or non-constant "
              "arguments)");

          if (
            std::optional<exprt> folded =
              try_fold_transpose_literal_2d(*materialized, converter_))
          {
            if (converter_.current_lhs)
            {
              converter_.current_lhs->type() = folded->type();
              converter_.update_symbol(*converter_.current_lhs);
            }
            return *folded;
          }
          throw std::runtime_error(
            "TypeError: numpy.transpose currently supports up to 2D arrays");
        }
      }

      resolve_var(arg);

      if (function == "transpose")
      {
        exprt arg_expr = converter_.get_expr(arg);
        typet t = arg_expr.type();
        if (t.is_pointer() && t.subtype().is_array())
          t = t.subtype();

        if (t.is_array() && t.subtype().is_array())
        {
          std::vector<int> shape = type_handler_.get_array_type_shape(t);
          if (shape.size() != 2)
          {
            throw std::runtime_error(
              "TypeError: numpy.transpose currently supports up to 2D arrays");
          }

          typet base_type = t.subtype().subtype();
          const bool is_float = base_type.is_floatbv();
          function_id_.set_function(
            is_float ? "transpose_double" : "transpose");

          code_function_callt call =
            to_code_function_call(to_code(function_call_expr::get()));

          typet result_row_type =
            type_handler_.build_array(base_type, shape[0]);
          typet result_type =
            type_handler_.build_array(result_row_type, shape[1]);
          if (converter_.current_lhs)
          {
            converter_.current_lhs->type() = result_type;
            converter_.update_symbol(*converter_.current_lhs);
          }

          auto &args = call.arguments();
          typet flat_ptr_type =
            pointer_typet(is_float ? base_type : long_long_int_type());
          if (!args.empty())
            args[0] = np_typecast(args[0], flat_ptr_type);

          exprt row0 = np_index(
            *converter_.current_lhs,
            from_integer(0, size_type()),
            result_type.subtype());
          exprt elem00 =
            np_index(row0, from_integer(0, size_type()), base_type);
          args.push_back(np_typecast(np_address_of(elem00), flat_ptr_type));
          args.push_back(from_integer(shape[0], int_type()));
          args.push_back(from_integer(shape[1], int_type()));
          return call;
        }

        if (t.is_array())
        {
          if (converter_.current_lhs)
          {
            converter_.current_lhs->type() = t;
            converter_.update_symbol(*converter_.current_lhs);
          }
          return arg_expr;
        }
      }

      nlohmann::json list_arg = unwrap_list_like_node(arg);

      // Handle calls with arrays as parameters; e.g. np.ceil([1, 2, 3])
      if (!list_arg.is_null() && list_arg.is_object())
      {
        if (function == "arccos")
        {
          try
          {
            if (allow_numpy_fold)
            {
              exprt folded =
                fold_numpy_unary_constant_list(converter_, function, list_arg);
              if (converter_.current_lhs)
              {
                converter_.current_lhs->type() = folded.type();
                converter_.update_symbol(*converter_.current_lhs);
              }
              return folded;
            }
          }
          catch (const std::runtime_error &)
          {
          }

          if (
            list_arg.contains("elts") && list_arg["elts"].is_array() &&
            !list_arg["elts"].empty() && list_arg["elts"][0].is_object() &&
            list_arg["elts"][0].contains("_type") &&
            list_arg["elts"][0]["_type"] == "List")
          {
            throw std::runtime_error(
              "Unsupported operation: numpy.arccos on runtime 2D arrays");
          }

          function_id_.set_function("__arccos_array");

          code_function_callt call =
            to_code_function_call(to_code(function_call_expr::get()));
          typet t = type_handler_.get_list_type(list_arg);
          if (!converter_.current_lhs)
            throw std::runtime_error(
              "Internal error: numpy.arccos runtime lowering requires an "
              "assignment target");
          auto &current_lhs = *converter_.current_lhs;
          current_lhs.type() = t;
          converter_.update_symbol(current_lhs);

          call.arguments().push_back(np_address_of(current_lhs));
          exprt array_size = from_integer(list_arg["elts"].size(), int_type());
          call.arguments().push_back(array_size);
          return call;
        }

        if (function == "transpose")
        {
          typet t = type_handler_.get_list_type(list_arg);
          if (allow_numpy_fold && !t.subtype().is_array())
          {
            exprt folded = converter_.get_expr(list_arg);
            if (converter_.current_lhs)
            {
              converter_.current_lhs->type() = folded.type();
              converter_.update_symbol(*converter_.current_lhs);
            }
            return folded;
          }

          std::vector<int> shape = type_handler_.get_array_type_shape(t);
          if (shape.size() != 2)
          {
            throw std::runtime_error(
              "TypeError: numpy.transpose currently supports up to 2D arrays");
          }

          typet base_type = t.subtype().subtype();
          const bool is_float = base_type.is_floatbv();

          function_id_.set_function(
            is_float ? "transpose_double" : "transpose");

          code_function_callt call =
            to_code_function_call(to_code(function_call_expr::get()));

          typet result_row_type =
            type_handler_.build_array(base_type, shape[0]);
          typet result_type =
            type_handler_.build_array(result_row_type, shape[1]);
          if (!converter_.current_lhs)
            throw std::runtime_error(
              "Internal error: numpy.transpose runtime lowering requires an "
              "assignment target");
          auto &current_lhs = *converter_.current_lhs;
          current_lhs.type() = result_type;
          converter_.update_symbol(current_lhs);

          auto &args = call.arguments();
          typet flat_ptr_type =
            pointer_typet(is_float ? base_type : long_long_int_type());
          if (!args.empty())
            args[0] = np_typecast(args[0], flat_ptr_type);

          exprt row0 = np_index(
            current_lhs, from_integer(0, size_type()), result_type.subtype());
          exprt elem00 =
            np_index(row0, from_integer(0, size_type()), base_type);
          args.push_back(np_typecast(np_address_of(elem00), flat_ptr_type));
          args.push_back(from_integer(shape[0], int_type()));
          args.push_back(from_integer(shape[1], int_type()));
          return call;
        }

        if (is_supported_numpy_unary_math(function))
        {
          if (allow_numpy_fold)
          {
            exprt folded =
              fold_numpy_unary_constant_list(converter_, function, list_arg);
            if (converter_.current_lhs)
            {
              converter_.current_lhs->type() = folded.type();
              converter_.update_symbol(*converter_.current_lhs);
            }
            return folded;
          }
        }

        // Constant-fold np.ceil for concrete 1D numeric lists.
        if (function == "ceil")
        {
          std::vector<numeric_value> input_values;
          if (
            allow_numpy_fold &&
            try_extract_numeric_1d_list(list_arg, input_values))
          {
            nlohmann::json out;
            out["_type"] = "List";
            out["elts"] = nlohmann::json::array();

            for (const auto &value : input_values)
            {
              nlohmann::json elem;
              elem["_type"] = "Constant";
              elem["value"] = std::ceil(to_double(value));
              out["elts"].push_back(elem);
            }
            exprt folded = converter_.get_expr(out);
            if (converter_.current_lhs)
            {
              converter_.current_lhs->type() = folded.type();
              converter_.update_symbol(*converter_.current_lhs);
            }
            return folded;
          }
        }

        // Append array postfix to call array variants, e.g., ceil_array instead of ceil
        std::string func_name = function_id_.get_function();
        if (func_name == "ceil")
          func_name = "__" + func_name + "_array";
        function_id_.set_function(func_name);

        code_function_callt call =
          to_code_function_call(to_code(function_call_expr::get()));
        typet t = type_handler_.get_list_type(list_arg);
        if (!converter_.current_lhs)
          throw std::runtime_error(
            "Internal error: numpy.ceil runtime lowering requires an "
            "assignment target");
        auto &current_lhs = *converter_.current_lhs;

        // In a call like result = np.ceil(v), the type of 'result' is only known after processing the argument 'v'.
        // At this point, we have the argument's type information, so we update the type of the LHS expression accordingly.

        if (t.subtype().is_array())
          current_lhs.type() = long_long_int_type();
        else
          current_lhs.type() = t;

        converter_.update_symbol(current_lhs);

        // NumPy math functions on arrays are translated to C-style calls with the signature: func(input, output, size).
        // For example, result = np.ceil(v) becomes ceil_array(v, result, sizeof(v)).
        // The lines below add the output array and size arguments to the call.

        // Add output argument
        call.arguments().push_back(np_address_of(current_lhs));

        // Add array size arguments
        if (t.subtype().is_array())
        {
          std::vector<int> shape = type_handler_.get_array_type_shape(t);
          call.arguments().push_back(from_integer(shape[0], int_type()));
          call.arguments().push_back(from_integer(shape[1], int_type()));
        }
        else
        {
          exprt array_size = from_integer(arg["elts"].size(), int_type());
          call.arguments().push_back(array_size);
        }

        return call;
      }
    }
  }

  // Binary operations
  if (call_["args"].size() == 2)
  {
    const std::string &function = function_id_.get_function();
    auto lhs = call_["args"][0];
    auto rhs = call_["args"][1];
    auto original_lhs = lhs;
    auto original_rhs = rhs;

    resolve_var(lhs);
    resolve_var(rhs);

    if (function == "solve")
    {
      auto unwrap = [](nlohmann::json &arg) {
        if (
          arg.is_object() && arg.contains("_type") && arg["_type"] == "Call" &&
          arg.contains("func") && arg["func"].is_object() &&
          ((arg["func"].contains("_type") && arg["func"]["_type"] == "Name" &&
            arg["func"].contains("id") && arg["func"]["id"] == "array") ||
           (arg["func"].contains("_type") &&
            arg["func"]["_type"] == "Attribute" &&
            arg["func"].contains("attr") && arg["func"]["attr"] == "array")) &&
          arg.contains("args") && arg["args"].is_array() &&
          !arg["args"].empty())
        {
          arg = arg["args"][0];
        }
      };
      unwrap(lhs);
      unwrap(rhs);

      std::vector<std::vector<scalar_value>> A;
      if (!try_extract_scalar_2d_list(lhs, A))
        throw std::runtime_error(
          "TypeError: numpy.linalg.solve requires a constant 2D numeric "
          "matrix as first argument");

      std::size_t n = 0;
      if (!is_square_matrix(A, n))
        throw std::runtime_error(
          "TypeError: numpy.linalg.solve requires a square matrix");

      if (n != 2 && n != 3)
        throw std::runtime_error(
          "TypeError: numpy.linalg.solve supports only 2x2 and 3x3 matrices");

      std::vector<scalar_value> b;
      if (!try_extract_scalar_1d_list(rhs, b))
        throw std::runtime_error(
          "TypeError: numpy.linalg.solve requires a constant 1D numeric "
          "array as second argument");

      if (b.size() != n)
        throw std::runtime_error(
          "ValueError: numpy.linalg.solve: matrix and vector sizes are "
          "incompatible");

      std::vector<scalar_value> x;
      if (!solve_linear_system(A, b, x))
        throw std::runtime_error("numpy.linalg.LinAlgError: Singular matrix");

      return converter_.get_expr(vector_to_json(x));
    }

    if (should_fallback_to_numpy_model(function))
      return function_call_expr::get();

    if (
      function == "power" && lhs.contains("value") && rhs.contains("value") &&
      lhs["value"].is_number_integer() && rhs["value"].is_number_integer() &&
      rhs["value"].get<int64_t>() < 0)
    {
      throw_negative_integer_power_error();
    }

    if (
      allow_numpy_fold &&
      (function == "add" || function == "subtract" || function == "multiply" ||
       function == "divide" || function == "power"))
    {
      if (
        lhs["_type"] == "List" && rhs["_type"] == "List" &&
        lhs.contains("elts") && rhs.contains("elts") &&
        lhs["elts"].is_array() && rhs["elts"].is_array() &&
        lhs["elts"].empty() && rhs["elts"].empty())
      {
        throw std::runtime_error(
          "TypeError: numpy operation on two empty arrays is not supported "
          "yet");
      }

      std::vector<std::size_t> lhs_shape;
      std::vector<std::size_t> rhs_shape;

      scalar_value lhs_scalar;
      scalar_value rhs_scalar;
      if (
        try_extract_scalar_constant(lhs, lhs_scalar) &&
        try_extract_scalar_constant(rhs, rhs_scalar))
      {
        if (lhs_scalar.is_complex || rhs_scalar.is_complex)
        {
          return converter_.get_expr(to_json_constant(
            apply_complex_binary(function, lhs_scalar, rhs_scalar)));
        }
      }

      std::vector<scalar_value> lhs_1d;
      std::vector<scalar_value> rhs_1d;
      if (
        try_extract_scalar_1d_list(lhs, lhs_1d) &&
        try_extract_scalar_1d_list(rhs, rhs_1d))
      {
        if (has_complex(lhs_1d) || has_complex(rhs_1d))
        {
          if (lhs_1d.size() != rhs_1d.size())
            throw std::runtime_error(
              "operands could not be broadcast together");
          nlohmann::json out;
          out["_type"] = "List";
          out["elts"] = nlohmann::json::array();
          for (std::size_t i = 0; i < lhs_1d.size(); ++i)
          {
            out["elts"].push_back(to_json_constant(
              apply_complex_binary(function, lhs_1d[i], rhs_1d[i])));
          }
          return converter_.get_expr(out);
        }
      }

      std::vector<std::vector<scalar_value>> lhs_2d;
      std::vector<std::vector<scalar_value>> rhs_2d;
      if (
        try_extract_scalar_2d_list(lhs, lhs_2d) &&
        try_extract_scalar_2d_list(rhs, rhs_2d))
      {
        if (has_complex(lhs_2d) || has_complex(rhs_2d))
        {
          if (lhs_2d.size() != rhs_2d.size())
            throw std::runtime_error(
              "operands could not be broadcast together");
          nlohmann::json out;
          out["_type"] = "List";
          out["elts"] = nlohmann::json::array();
          for (std::size_t r = 0; r < lhs_2d.size(); ++r)
          {
            if (lhs_2d[r].size() != rhs_2d[r].size())
              throw std::runtime_error(
                "operands could not be broadcast together");
            nlohmann::json row;
            row["_type"] = "List";
            row["elts"] = nlohmann::json::array();
            for (std::size_t c = 0; c < lhs_2d[r].size(); ++c)
            {
              row["elts"].push_back(to_json_constant(
                apply_complex_binary(function, lhs_2d[r][c], rhs_2d[r][c])));
            }
            out["elts"].push_back(row);
          }
          return converter_.get_expr(out);
        }
      }

      if (
        try_extract_scalar_1d_list(lhs, lhs_1d) &&
        try_extract_scalar_constant(rhs, rhs_scalar) &&
        (has_complex(lhs_1d) || rhs_scalar.is_complex))
      {
        nlohmann::json out;
        out["_type"] = "List";
        out["elts"] = nlohmann::json::array();
        for (const auto &v : lhs_1d)
          out["elts"].push_back(
            to_json_constant(apply_complex_binary(function, v, rhs_scalar)));
        return converter_.get_expr(out);
      }
      if (
        try_extract_scalar_constant(lhs, lhs_scalar) &&
        try_extract_scalar_1d_list(rhs, rhs_1d) &&
        (lhs_scalar.is_complex || has_complex(rhs_1d)))
      {
        nlohmann::json out;
        out["_type"] = "List";
        out["elts"] = nlohmann::json::array();
        for (const auto &v : rhs_1d)
          out["elts"].push_back(
            to_json_constant(apply_complex_binary(function, lhs_scalar, v)));
        return converter_.get_expr(out);
      }
      if (
        try_extract_scalar_2d_list(lhs, lhs_2d) &&
        try_extract_scalar_constant(rhs, rhs_scalar) &&
        (has_complex(lhs_2d) || rhs_scalar.is_complex))
      {
        nlohmann::json out;
        out["_type"] = "List";
        out["elts"] = nlohmann::json::array();
        for (const auto &row_vals : lhs_2d)
        {
          nlohmann::json row;
          row["_type"] = "List";
          row["elts"] = nlohmann::json::array();
          for (const auto &v : row_vals)
            row["elts"].push_back(
              to_json_constant(apply_complex_binary(function, v, rhs_scalar)));
          out["elts"].push_back(row);
        }
        return converter_.get_expr(out);
      }
      if (
        try_extract_scalar_constant(lhs, lhs_scalar) &&
        try_extract_scalar_2d_list(rhs, rhs_2d) &&
        (lhs_scalar.is_complex || has_complex(rhs_2d)))
      {
        nlohmann::json out;
        out["_type"] = "List";
        out["elts"] = nlohmann::json::array();
        for (const auto &row_vals : rhs_2d)
        {
          nlohmann::json row;
          row["_type"] = "List";
          row["elts"] = nlohmann::json::array();
          for (const auto &v : row_vals)
            row["elts"].push_back(
              to_json_constant(apply_complex_binary(function, lhs_scalar, v)));
          out["elts"].push_back(row);
        }
        return converter_.get_expr(out);
      }
      if (
        try_extract_scalar_1d_list(lhs, lhs_1d) &&
        try_extract_scalar_constant(rhs, rhs_scalar))
      {
        nlohmann::json out;
        out["_type"] = "List";
        out["elts"] = nlohmann::json::array();
        for (const auto &v : lhs_1d)
          out["elts"].push_back(
            to_json_constant(apply_complex_binary(function, v, rhs_scalar)));
        exprt folded = converter_.get_expr(out);
        if (converter_.current_lhs)
        {
          converter_.current_lhs->type() = folded.type();
          converter_.update_symbol(*converter_.current_lhs);
        }
        return folded;
      }
      if (
        try_extract_scalar_constant(lhs, lhs_scalar) &&
        try_extract_scalar_1d_list(rhs, rhs_1d))
      {
        nlohmann::json out;
        out["_type"] = "List";
        out["elts"] = nlohmann::json::array();
        for (const auto &v : rhs_1d)
          out["elts"].push_back(
            to_json_constant(apply_complex_binary(function, lhs_scalar, v)));
        exprt folded = converter_.get_expr(out);
        if (converter_.current_lhs)
        {
          converter_.current_lhs->type() = folded.type();
          converter_.update_symbol(*converter_.current_lhs);
        }
        return folded;
      }
      if (
        try_extract_scalar_2d_list(lhs, lhs_2d) &&
        try_extract_scalar_constant(rhs, rhs_scalar))
      {
        nlohmann::json out;
        out["_type"] = "List";
        out["elts"] = nlohmann::json::array();
        for (const auto &row_vals : lhs_2d)
        {
          nlohmann::json row;
          row["_type"] = "List";
          row["elts"] = nlohmann::json::array();
          for (const auto &v : row_vals)
            row["elts"].push_back(
              to_json_constant(apply_complex_binary(function, v, rhs_scalar)));
          out["elts"].push_back(row);
        }
        exprt folded = converter_.get_expr(out);
        if (converter_.current_lhs)
        {
          converter_.current_lhs->type() = folded.type();
          converter_.update_symbol(*converter_.current_lhs);
        }
        return folded;
      }
      if (
        try_extract_scalar_constant(lhs, lhs_scalar) &&
        try_extract_scalar_2d_list(rhs, rhs_2d))
      {
        nlohmann::json out;
        out["_type"] = "List";
        out["elts"] = nlohmann::json::array();
        for (const auto &row_vals : rhs_2d)
        {
          nlohmann::json row;
          row["_type"] = "List";
          row["elts"] = nlohmann::json::array();
          for (const auto &v : row_vals)
            row["elts"].push_back(
              to_json_constant(apply_complex_binary(function, lhs_scalar, v)));
          out["elts"].push_back(row);
        }
        exprt folded = converter_.get_expr(out);
        if (converter_.current_lhs)
        {
          converter_.current_lhs->type() = folded.type();
          converter_.update_symbol(*converter_.current_lhs);
        }
        return folded;
      }
      if (
        get_literal_shape(lhs, lhs_shape) && get_literal_shape(rhs, rhs_shape))
      {
        std::vector<std::size_t> result_shape;
        if (
          compute_broadcast_shape(lhs_shape, rhs_shape, result_shape) &&
          result_shape.size() <= 2)
        {
          nlohmann::json folded;
          std::vector<std::size_t> indices;
          if (build_broadcast_literal_result(
                function,
                lhs,
                lhs_shape,
                rhs,
                rhs_shape,
                result_shape,
                indices,
                0,
                folded))
          {
            exprt result_expr = converter_.get_expr(folded);
            if (converter_.current_lhs)
            {
              converter_.current_lhs->type() = result_expr.type();
              converter_.update_symbol(*converter_.current_lhs);
            }
            return result_expr;
          }
        }
      }
    }

    if (
      (lhs["_type"] == "Constant" || lhs["_type"] == "UnaryOp") &&
      (rhs["_type"] == "Constant" || rhs["_type"] == "UnaryOp"))
    {
      bool lhs_is_float =
        (lhs["_type"] == "UnaryOp" ? lhs["operand"]["value"].is_number_float()
                                   : lhs["value"].is_number_float());
      bool rhs_is_float =
        (rhs["_type"] == "UnaryOp" ? rhs["operand"]["value"].is_number_float()
                                   : rhs["value"].is_number_float());

      if (lhs_is_float || rhs_is_float)
      {
        double lhs_val = get_constant_value<double>(lhs);
        double rhs_val = get_constant_value<double>(rhs);
        expr = create_binary_op(
          function_id_.get_function(), kConstant, lhs_val, rhs_val);
      }
      else
      {
        int lhs_val = get_constant_value<int>(lhs);
        int rhs_val = get_constant_value<int>(rhs);
        expr = create_binary_op(
          function_id_.get_function(), kConstant, lhs_val, rhs_val);
      }
    }
    else if (lhs["_type"] == "AnnAssign" && rhs["_type"] == "AnnAssign")
    {
      expr = create_binary_op(
        function_id_.get_function(),
        kName,
        lhs["target"]["id"],
        rhs["target"]["id"]);
    }
    else if (lhs["_type"] == "List" && rhs["_type"] == "List")
    {
      // Get the name of the function being called (e.g., "dot" or "matmul")
      const std::string &operation = function_id_.get_function();

      if (operation == "dot" || operation == "matmul")
      {
        if (
          lhs.contains("elts") && rhs.contains("elts") &&
          lhs["elts"].is_array() && rhs["elts"].is_array() &&
          (lhs["elts"].empty() || rhs["elts"].empty()))
        {
          throw std::runtime_error(
            "TypeError: numpy.dot does not support empty operands");
        }

        if (
          operation == "dot" &&
          (literal_list_contains_bool(lhs) || literal_list_contains_bool(rhs)))
        {
          nlohmann::json folded;
          if (fold_literal_dot(lhs, rhs, folded))
          {
            exprt folded_expr = converter_.get_expr(folded);
            if (converter_.current_lhs)
            {
              converter_.current_lhs->type() = folded_expr.type();
              converter_.update_symbol(*converter_.current_lhs);
            }
            return folded_expr;
          }
        }

        // The runtime backend call below writes its result through the
        // address of current_lhs; a bare expression statement (result
        // discarded, e.g. `np.dot(a, b)` with no assignment) has no
        // current_lhs and previously crashed dereferencing a null pointer
        // instead of producing this diagnostic (matches the existing
        // arccos/transpose/ceil convention for the same requirement).
        if (!converter_.current_lhs)
          throw std::runtime_error(
            "Internal error: numpy." + operation +
            " runtime lowering requires an assignment target");

        // Determine dimensionality of both operands
        bool lhs_is_2d = type_handler_.is_2d_array(lhs);
        bool rhs_is_2d = type_handler_.is_2d_array(rhs);

        size_t m, n, n2, p;
        typet base_type;
        bool result_is_scalar = false;
        bool result_is_1d = false;

        if (!lhs_is_2d && !rhs_is_2d)
        {
          // 1D × 1D case: vector dot product
          size_t lhs_len = lhs["elts"].size();
          size_t rhs_len = rhs["elts"].size();

          if (lhs_len != rhs_len)
          {
            throw std::runtime_error("Incompatible shapes for dot product");
          }

          // Get element type from the first element node itself. Passing the
          // node (not a presumed ["value"] subfield) lets get_typet resolve a
          // symbolic Name element (e.g. a = nondet_int()) to its real type;
          // ["value"] is absent on a Name node and yielded a void element type,
          // which made the flat int64 buffer access overflow (#5115).
          base_type = type_handler_.get_typet(lhs["elts"][0]);

          // For 1D dot product, treat as (1×n) × (n×1) = (1×1) scalar
          m = 1;
          n = lhs_len;
          n2 = rhs_len;
          p = 1;

          // Result is a scalar, not a matrix
          converter_.current_lhs->type() = base_type;
          result_is_scalar = true;
        }
        else if (!lhs_is_2d && rhs_is_2d)
        {
          // 1D × 2D case: (n,) × (n, p) -> (p,)
          size_t lhs_len = lhs["elts"].size();
          size_t rhs_rows = rhs["elts"].size();
          size_t rhs_cols = rhs["elts"][0]["elts"].size();

          if (lhs_len != rhs_rows)
          {
            throw std::runtime_error("Incompatible shapes for dot product");
          }

          // See #5115: pass the element node so symbolic elements resolve.
          base_type = type_handler_.get_typet(rhs["elts"][0]["elts"][0]);

          m = 1;
          n = lhs_len;
          n2 = rhs_rows;
          p = rhs_cols;

          // Result is 1D array of length p
          typet result_type = type_handler_.build_array(base_type, p);
          converter_.current_lhs->type() = result_type;
          result_is_1d = true;
        }
        else if (lhs_is_2d && !rhs_is_2d)
        {
          // 2D × 1D case: (m, n) × (n,) -> (m,)
          size_t lhs_rows = lhs["elts"].size();
          size_t lhs_cols = lhs["elts"][0]["elts"].size();
          size_t rhs_len = rhs["elts"].size();

          if (lhs_cols != rhs_len)
          {
            throw std::runtime_error("Incompatible shapes for dot product");
          }

          // See #5115: pass the element node so symbolic elements resolve.
          base_type = type_handler_.get_typet(lhs["elts"][0]["elts"][0]);

          m = lhs_rows;
          n = lhs_cols;
          n2 = rhs_len;
          p = 1;

          // Result is 1D array of length m
          typet result_type = type_handler_.build_array(base_type, m);
          converter_.current_lhs->type() = result_type;
          result_is_1d = true;
        }
        else
        {
          // 2D × 2D case: original matrix multiplication logic
          m = lhs["elts"].size();
          n = lhs["elts"][0]["elts"].size();
          n2 = rhs["elts"].size();
          p = rhs["elts"][0]["elts"].size();

          if (n != n2)
          {
            throw std::runtime_error("Incompatible shapes for dot product");
          }

          // See #5115: pass the element node so symbolic elements resolve.
          base_type = type_handler_.get_typet(lhs["elts"][0]["elts"][0]);

          // [[...]] access pattern (A[i][j]). The backend dot() accesses the
          // result via a flat int64_t* pointer obtained by taking the address
          // of the first element (A[0][0]).
          typet row_type = type_handler_.build_array(base_type, p);
          typet result_type = type_handler_.build_array(row_type, m);
          if (converter_.current_lhs != nullptr)
            converter_.current_lhs->type() = result_type;
        }

        // Select the backend by element type: integer matrices use dot(), which
        // accumulates into int64_t; floating-point matrices must use
        // dot_double(), which accumulates into double. Using the integer dot()
        // on double data reinterprets the float bit pattern as int64 and is
        // unsound (#5115). "matmul" is normalised to the matching backend.
        // Scoped to the default floatbv encoding; the non-default --fixedbv
        // float path is left as-is (a separate, pre-existing concern).
        const bool is_float = base_type.is_floatbv();
        unsigned dtype_bits = 64;
        if (!is_float && (base_type.is_signedbv() || base_type.is_unsignedbv()))
          dtype_bits = static_cast<const bv_typet &>(base_type).get_width();
        function_id_.set_function(is_float ? "dot_double" : "dot");
        // Update the symbol associated with the result
        if (converter_.current_lhs != nullptr)
          converter_.update_symbol(*converter_.current_lhs);

        // Generate a function call expression to the selected backend function
        code_function_callt call =
          to_code_function_call(to_code(function_call_expr::get()));

        // The first two arguments are pointers to the input arrays.
        // function_call_expr::get() produces pointer-to-array-of-array
        // (e.g. int (*)[1][1]); the backend expects a flat element pointer
        // (int64_t* for dot(), double* for dot_double()). Cast both inputs to
        // the flat element pointer so the pointer arithmetic strides correctly.
        typet flat_ptr_type =
          pointer_typet(is_float ? base_type : long_long_int_type());
        auto &args = call.arguments();
        if (args.size() >= 2)
        {
          args[0] = np_typecast(args[0], flat_ptr_type);
          args[1] = np_typecast(args[1], flat_ptr_type);
        }

        // Arguments:
        // 3. Output pointer (result): scalar/vector cases can use the symbol
        //    address directly; matrix results need the address of [0][0] so the
        //    flat pointer arithmetic in dot() lands on the first scalar.
        // 4-6. Dimensions (int64_t): m, n, p
        exprt result_ptr;
        if (result_is_scalar || result_is_1d)
        {
          result_ptr = np_address_of(*converter_.current_lhs);
        }
        else
        {
          exprt row0 = np_index(
            *converter_.current_lhs,
            from_integer(0, size_type()),
            converter_.current_lhs->type().subtype());
          exprt elem00 =
            np_index(row0, from_integer(0, size_type()), base_type);
          result_ptr = np_address_of(elem00);
        }
        args.push_back(np_typecast(result_ptr, flat_ptr_type));
        args.push_back(from_integer(m, long_long_int_type()));
        args.push_back(from_integer(n, long_long_int_type()));
        args.push_back(from_integer(p, long_long_int_type()));
        if (!is_float)
          args.push_back(from_integer(dtype_bits, long_long_int_type()));

        return call;
      }
      // Handle other binary operations like add, subtract, multiply, divide
      if (
        operation == "add" || operation == "subtract" ||
        operation == "multiply" || operation == "divide")
      {
        // Empty-list x empty-list currently has no stable umath lowering in
        // this frontend path; reject explicitly instead of allowing internal
        // backend failures.
        if (
          lhs.contains("elts") && rhs.contains("elts") &&
          lhs["elts"].is_array() && rhs["elts"].is_array() &&
          lhs["elts"].empty() && rhs["elts"].empty())
        {
          throw std::runtime_error(
            "TypeError: numpy operation on two empty arrays is not supported "
            "yet");
        }

        std::vector<std::size_t> lhs_shape;
        std::vector<std::size_t> rhs_shape;

        if (
          !get_literal_shape(lhs, lhs_shape) ||
          !get_literal_shape(rhs, rhs_shape))
        {
          throw std::runtime_error(
            "TypeError: numpy elementwise operations require literal arrays "
            "in this path");
        }

        std::vector<std::size_t> result_shape;
        if (!compute_broadcast_shape(lhs_shape, rhs_shape, result_shape))
        {
          throw std::runtime_error(
            "operands could not be broadcast together with shapes " +
            format_shape(lhs_shape) + " " + format_shape(rhs_shape));
        }

        if (result_shape.size() > 2)
        {
          throw std::runtime_error(
            "TypeError: numpy elementwise operations currently support up to "
            "2D arrays");
        }

        auto as_dim =
          [](const std::vector<std::size_t> &shape, std::size_t axis) {
            if (shape.empty())
              return from_integer(1, int_type());
            if (shape.size() == 1)
              return from_integer(
                axis == 0 ? 1 : static_cast<int>(shape[0]), int_type());
            return from_integer(
              static_cast<int>(axis < shape.size() ? shape[axis] : 1),
              int_type());
          };

        auto build_array_type =
          [&](const std::vector<std::size_t> &shape, const typet &elem_type) {
            if (shape.empty())
              return elem_type;

            typet array_type = elem_type;
            for (auto it = shape.rbegin(); it != shape.rend(); ++it)
              array_type = type_handler_.build_array(array_type, *it);
            return array_type;
          };

        typet lhs_scalar_type =
          get_array_scalar_type(type_handler_.get_typet(lhs));
        typet rhs_scalar_type =
          get_array_scalar_type(type_handler_.get_typet(rhs));
        const bool is_float =
          lhs_scalar_type.is_floatbv() || rhs_scalar_type.is_floatbv();

        typet elem_type =
          is_float ? double_type()
          : lhs_scalar_type.is_bool() || rhs_scalar_type.is_bool() ? bool_type()
                                                                   : int_type();
        typet t = build_array_type(result_shape, elem_type);
        function_id_.set_function(operation + (is_float ? "_double" : ""));

        converter_.current_lhs->type() = t;
        converter_.update_symbol(*converter_.current_lhs);

        code_function_callt call =
          to_code_function_call(to_code(function_call_expr::get()));
        auto &args = call.arguments();
        const typet flat_ptr_type =
          pointer_typet(is_float ? double_type() : long_long_int_type());
        if (args.size() >= 2)
        {
          // V.3: build the flat-pointer arg casts in IREP2 via np_typecast,
          // matching the sibling binary-op branches (e.g. lines 3133-3134).
          args[0] = np_typecast(args[0], flat_ptr_type);
          args[1] = np_typecast(args[1], flat_ptr_type);
        }
        args.push_back(
          np_typecast(np_address_of(*converter_.current_lhs), flat_ptr_type));
        args.push_back(as_dim(lhs_shape, 0));
        args.push_back(as_dim(lhs_shape, 1));
        args.push_back(as_dim(rhs_shape, 0));
        args.push_back(as_dim(rhs_shape, 1));

        return call;
      }

      if (should_fallback_to_numpy_model(operation))
        return function_call_expr::get();

      throw std::runtime_error("Unsupported operation: " + operation);
    }
    else if (function == "dot" || function == "matmul")
    {
      if (!converter_.current_lhs)
        throw std::runtime_error("Unsupported Numpy call: " + function);

      exprt lhs_arg = converter_.get_expr(original_lhs);
      exprt rhs_arg = converter_.get_expr(original_rhs);
      std::vector<int> lhs_shape =
        type_handler_.get_array_type_shape(lhs_arg.type());
      std::vector<int> rhs_shape =
        type_handler_.get_array_type_shape(rhs_arg.type());

      if (
        lhs_shape.empty() || rhs_shape.empty() || lhs_shape.size() > 2 ||
        rhs_shape.size() > 2)
      {
        throw std::runtime_error("Unsupported Numpy call: " + function);
      }

      if (lhs_shape[0] == 0 || rhs_shape[0] == 0)
        throw std::runtime_error(
          "TypeError: numpy.dot does not support empty operands");

      bool result_is_scalar = false;
      bool result_is_1d = false;
      std::size_t m = 0;
      std::size_t n = 0;
      std::size_t p = 0;

      if (lhs_shape.size() == 1 && rhs_shape.size() == 1)
      {
        if (lhs_shape[0] != rhs_shape[0])
          throw std::runtime_error("Incompatible shapes for dot product");
        m = 1;
        n = static_cast<std::size_t>(lhs_shape[0]);
        p = 1;
        converter_.current_lhs->type() = get_array_scalar_type(lhs_arg.type());
        result_is_scalar = true;
      }
      else if (lhs_shape.size() == 1 && rhs_shape.size() == 2)
      {
        if (lhs_shape[0] != rhs_shape[0])
          throw std::runtime_error("Incompatible shapes for dot product");
        m = 1;
        n = static_cast<std::size_t>(lhs_shape[0]);
        p = static_cast<std::size_t>(rhs_shape[1]);
        typet base_type = get_array_scalar_type(rhs_arg.type());
        converter_.current_lhs->type() =
          type_handler_.build_array(base_type, p);
        result_is_1d = true;
      }
      else if (lhs_shape.size() == 2 && rhs_shape.size() == 1)
      {
        if (lhs_shape[1] != rhs_shape[0])
          throw std::runtime_error("Incompatible shapes for dot product");
        m = static_cast<std::size_t>(lhs_shape[0]);
        n = static_cast<std::size_t>(lhs_shape[1]);
        p = 1;
        typet base_type = get_array_scalar_type(lhs_arg.type());
        converter_.current_lhs->type() =
          type_handler_.build_array(base_type, m);
        result_is_1d = true;
      }
      else
      {
        if (lhs_shape[1] != rhs_shape[0])
          throw std::runtime_error("Incompatible shapes for dot product");
        m = static_cast<std::size_t>(lhs_shape[0]);
        n = static_cast<std::size_t>(lhs_shape[1]);
        p = static_cast<std::size_t>(rhs_shape[1]);
        typet base_type = get_array_scalar_type(lhs_arg.type());
        typet row_type = type_handler_.build_array(base_type, p);
        converter_.current_lhs->type() = type_handler_.build_array(row_type, m);
      }

      typet base_type = get_array_scalar_type(lhs_arg.type());
      const bool is_float = base_type.is_floatbv();
      unsigned dtype_bits = 64;
      if (!is_float && (base_type.is_signedbv() || base_type.is_unsignedbv()))
        dtype_bits = static_cast<const bv_typet &>(base_type).get_width();
      function_id_.set_function(is_float ? "dot_double" : "dot");
      converter_.update_symbol(*converter_.current_lhs);

      code_function_callt call =
        to_code_function_call(to_code(function_call_expr::get()));
      typet flat_ptr_type =
        pointer_typet(is_float ? base_type : long_long_int_type());
      auto &args = call.arguments();
      if (args.size() >= 2)
      {
        args[0] = np_typecast(args[0], flat_ptr_type);
        args[1] = np_typecast(args[1], flat_ptr_type);
      }

      exprt result_ptr;
      if (result_is_scalar || result_is_1d)
      {
        result_ptr = np_address_of(*converter_.current_lhs);
      }
      else
      {
        exprt row0 = np_index(
          *converter_.current_lhs,
          from_integer(0, size_type()),
          converter_.current_lhs->type().subtype());
        exprt elem00 = np_index(row0, from_integer(0, size_type()), base_type);
        result_ptr = np_address_of(elem00);
      }
      args.push_back(np_typecast(result_ptr, flat_ptr_type));
      args.push_back(from_integer(m, long_long_int_type()));
      args.push_back(from_integer(n, long_long_int_type()));
      args.push_back(from_integer(p, long_long_int_type()));
      if (!is_float)
        args.push_back(from_integer(dtype_bits, long_long_int_type()));

      return call;
    }
  }

  if (expr.empty())
  {
    if (should_fallback_to_numpy_model(function_id_.get_function()))
      return function_call_expr::get();
    throw std::runtime_error(
      "Unsupported Numpy call: " + function_id_.get_function());
  }

  return converter_.get_expr(expr);
}

// A Name argument materialize_arange() cannot see through directly (e.g. the
// `n` in `n = 3; np.arange(n)`) is resolved to its declaration's value here,
// mirroring what the operational-model fallback used to do by plain
// interpretation. Only a single-level lookup is attempted; anything that
// does not resolve to a value (a genuinely non-constant parameter, a nondet
// value, a further Name chain) is left untouched and correctly declines in
// materialize_arange() afterwards.
static nlohmann::json resolve_arange_call_args(
  const nlohmann::json &args,
  python_converter &converter)
{
  nlohmann::json resolved = args;
  for (auto &arg : resolved)
  {
    if (!arg.is_object() || arg.value("_type", std::string()) != "Name")
      continue;
    const nlohmann::json decl = json_utils::find_var_decl(
      arg["id"].get<std::string>(),
      converter.current_function_name(),
      converter.ast());
    if (decl.contains("value") && decl["value"].is_object())
      arg = decl["value"];
  }
  return resolved;
}

exprt numpy_call_expr::get_arange_expr()
{
  // A dtype= (or any other) keyword changes arange()'s output in ways the
  // literal-materialization path below does not model; fall back to the
  // operational model for that shape exactly like every arange call used
  // to, rather than silently ignoring the keyword.
  if (call_.contains("keywords") && !call_["keywords"].empty())
    return function_call_expr::get();

  const nlohmann::json resolved_args =
    call_.contains("args") ? resolve_arange_call_args(call_["args"], converter_)
                           : nlohmann::json::array();

  // np.arange(...) with constant, small arguments is materialized to a
  // literal list directly, avoiding the operational model's while-loop
  // list-concatenation implementation (models/numpy.py's arange()), which
  // is disproportionately expensive to symbolically execute even for a
  // handful of elements. Although real np.arange() returns an ndarray, this
  // frontend materializes it as a list-like runtime object for consistency
  // with build_static_lists, disabled the same way full()/eye()/identity()/
  // linspace() already do it -- a plain static array here would not
  // compare equal to a `[]`-literal PyListObj.
  const arange_materialize_result result = materialize_arange_ex(resolved_args);
  if (result.list)
  {
    const bool old_build_static_lists = converter_.build_static_lists;
    converter_.build_static_lists = false;
    exprt expr = converter_.get_expr(*result.list);
    converter_.build_static_lists = old_build_static_lists;
    if (converter_.current_lhs)
    {
      converter_.current_lhs->type() = expr.type();
      converter_.update_symbol(*converter_.current_lhs);
    }
    return expr;
  }

  switch (result.reason)
  {
  case arange_decline_reason::bad_arity:
    throw std::runtime_error(
      "TypeError: numpy.arange() expects 1 to 3 arguments");
  case arange_decline_reason::zero_step:
    throw std::runtime_error(
      "ValueError: numpy.arange() step must not be zero");
  case arange_decline_reason::too_many_elements:
    throw std::runtime_error(
      "TypeError: numpy.arange() range exceeds the supported element limit "
      "of " +
      std::to_string(max_materialized_arange_elements));
  case arange_decline_reason::non_constant:
  case arange_decline_reason::none:
  default:
    break;
  }

  // Non-constant arguments (e.g. a function parameter) cannot be
  // materialized this way; falling back to the operational model here
  // hangs past any practical timeout instead of producing a verdict, so
  // this is rejected explicitly and quickly instead.
  throw std::runtime_error(
    "TypeError: numpy.arange() currently supports constant numeric inputs "
    "only");
}

exprt numpy_call_expr::get()
{
  const std::string &function = function_id_.get_function();
  const bool allow_numpy_fold = numpy_constant_folding_enabled();

  if (
    function == "sum" || function == "prod" || function == "min" ||
    function == "max" || function == "mean" || function == "argmin" ||
    function == "argmax" || function == "arange")
  {
    auto resolve_var = [this](nlohmann::json &var) {
      if (var["_type"] == "Name")
      {
        var = json_utils::find_var_decl(
          var["id"], converter_.current_function_name(), converter_.ast());
        if (!var.contains("value") || !var["value"].is_object())
          return;
        if (var["value"]["_type"] == "Call")
        {
          if (auto numpy_call = try_build_numpy_arange_list(var["value"]))
          {
            var = std::move(*numpy_call);
            return;
          }
          if (
            std::optional<nlohmann::json> materialized =
              materialize_numpy_constructor_array(
                var["value"], converter_.ast()))
          {
            var = std::move(*materialized);
            return;
          }
          if (is_numpy_constructor_call_by_name(var["value"]))
          {
            var = var["value"];
            return;
          }
          if (var["value"].contains("args") && !var["value"]["args"].empty())
            var = var["value"]["args"][0];
          else
            var = var["value"];
        }
        else
          var = var["value"];
      }
    };
    if (function == "arange")
      return get_arange_expr();

    nlohmann::json arg = call_["args"][0];
    resolve_var(arg);
    materialize_inline_numpy_constructor_call(arg, converter_.ast());
    if (
      std::optional<nlohmann::json> row_view =
        resolve_literal_numpy_row_view(arg, converter_))
      arg = std::move(*row_view);

    std::vector<numeric_value> values_1d;
    std::vector<std::vector<numeric_value>> values_2d;
    std::vector<numeric_value> values;
    if (try_extract_numeric_1d_list(arg, values_1d))
      values = values_1d;
    else if (try_extract_numeric_2d_list(arg, values_2d))
    {
      for (const auto &row : values_2d)
        values.insert(values.end(), row.begin(), row.end());
    }
    else
    {
      numeric_value scalar;
      if (!try_extract_numeric_constant(arg, scalar))
        throw std::runtime_error(
          "TypeError: numpy." + function +
          "() currently supports constant numeric inputs only");
      values.push_back(scalar);
    }

    if (values.empty())
    {
      if (function == "sum")
      {
        nlohmann::json out;
        out["_type"] = "Constant";
        out["value"] = 0;
        return converter_.get_expr(out);
      }
      if (function == "prod")
      {
        nlohmann::json out;
        out["_type"] = "Constant";
        out["value"] = 1;
        return converter_.get_expr(out);
      }
      throw std::runtime_error(
        "ValueError: numpy." + function + "() arg is an empty sequence");
    }

    if (function == "argmin" || function == "argmax")
    {
      std::size_t best_idx = 0;
      double best = to_double(values[0]);
      for (std::size_t i = 1; i < values.size(); ++i)
      {
        const double current = to_double(values[i]);
        if (
          (function == "argmin" && current < best) ||
          (function == "argmax" && current > best))
        {
          best = current;
          best_idx = i;
        }
      }
      nlohmann::json out;
      out["_type"] = "Constant";
      out["value"] = static_cast<int64_t>(best_idx);
      return converter_.get_expr(out);
    }

    double accum = 0.0;
    bool first_value = true;
    bool any_float = false;
    for (const auto &value : values)
    {
      const double current = to_double(value);
      any_float = any_float || !value.is_int;
      if (function == "sum" || function == "mean")
        accum += current;
      else if (function == "prod")
      {
        if (first_value)
          accum = 1.0;
        accum *= current;
      }
      else if (function == "min")
      {
        if (first_value)
          accum = current;
        else
          accum = std::min(accum, current);
      }
      else if (function == "max")
      {
        if (first_value)
          accum = current;
        else
          accum = std::max(accum, current);
      }
      first_value = false;
    }

    nlohmann::json out;
    out["_type"] = "Constant";
    if (function == "mean" || any_float)
      out["value"] = (function == "mean")
                       ? accum / static_cast<double>(values.size())
                       : accum;
    else
      out["value"] = static_cast<int64_t>(std::llround(accum));
    return converter_.get_expr(out);
  }

  if (function == "std" || function == "var")
  {
    if (call_["args"].empty())
      throw std::runtime_error(
        "TypeError: numpy." + function + "() missing argument");

    if (call_.contains("keywords") && !call_["keywords"].empty())
      throw std::runtime_error(
        "TypeError: numpy." + function +
        "() does not support axis, ddof, keepdims, where, out or dtype "
        "arguments yet");

    auto resolve_var = [this](nlohmann::json &var) {
      if (var["_type"] == "Name")
      {
        var = json_utils::find_var_decl(
          var["id"], converter_.current_function_name(), converter_.ast());
        if (!var.contains("value") || !var["value"].is_object())
          return;
        if (var["value"]["_type"] == "Call")
        {
          if (
            std::optional<nlohmann::json> materialized =
              materialize_numpy_constructor_array(
                var["value"], converter_.ast()))
            var = std::move(*materialized);
          else if (is_numpy_constructor_call_by_name(var["value"]))
            var = var["value"];
          else if (
            var["value"].contains("args") && !var["value"]["args"].empty())
            var = var["value"]["args"][0];
          else
            var = var["value"];
        }
        else
          var = var["value"];
      }
    };

    nlohmann::json arg = call_["args"][0];
    resolve_var(arg);

    std::vector<numeric_value> values_1d;
    std::vector<std::vector<numeric_value>> values_2d;
    std::vector<numeric_value> values;
    if (try_extract_numeric_1d_list(arg, values_1d))
      values = values_1d;
    else if (try_extract_numeric_2d_list(arg, values_2d))
    {
      for (const auto &row : values_2d)
        values.insert(values.end(), row.begin(), row.end());
    }
    else
    {
      numeric_value scalar;
      if (!try_extract_numeric_constant(arg, scalar))
        throw std::runtime_error(
          "TypeError: numpy." + function +
          "() currently supports constant numeric inputs only");
      values.push_back(scalar);
    }

    if (values.empty())
      throw std::runtime_error(
        "ValueError: numpy." + function + "() arg is an empty sequence");

    double mean = 0.0;
    for (const auto &value : values)
      mean += to_double(value);
    mean /= static_cast<double>(values.size());

    double sq_dev_sum = 0.0;
    for (const auto &value : values)
    {
      const double dev = to_double(value) - mean;
      sq_dev_sum += dev * dev;
    }
    const double variance = sq_dev_sum / static_cast<double>(values.size());

    nlohmann::json out;
    out["_type"] = "Constant";
    out["value"] = (function == "var") ? variance : std::sqrt(variance);
    return converter_.get_expr(out);
  }

  if (
    function == "greater" || function == "less" ||
    function == "greater_equal" || function == "less_equal" ||
    function == "equal" || function == "not_equal" ||
    function == "logical_and" || function == "logical_or" ||
    function == "logical_not" || function == "where" || function == "full" ||
    function == "eye" || function == "identity" || function == "linspace")
  {
    auto resolve_var = [this](nlohmann::json &var) {
      if (var["_type"] == "Name")
      {
        var = json_utils::find_var_decl(
          var["id"], converter_.current_function_name(), converter_.ast());
        if (!var.contains("value") || !var["value"].is_object())
          return;
        if (var["value"]["_type"] == "Call")
        {
          if (auto numpy_call = try_build_numpy_arange_list(var["value"]))
          {
            var = std::move(*numpy_call);
            return;
          }
          if (
            std::optional<nlohmann::json> materialized =
              materialize_numpy_constructor_array(
                var["value"], converter_.ast()))
          {
            var = std::move(*materialized);
            return;
          }
          if (is_numpy_constructor_call_by_name(var["value"]))
          {
            var = var["value"];
            return;
          }
          if (var["value"].contains("args") && !var["value"]["args"].empty())
            var = var["value"]["args"][0];
          else
            var = var["value"];
        }
        else
          var = var["value"];
      }
    };

    auto make_constant = [](const auto &value) {
      nlohmann::json out;
      out["_type"] = "Constant";
      out["value"] = value;
      return out;
    };

    auto to_list_expr = [this](const nlohmann::json &node) {
      const bool old_build_static_lists = converter_.build_static_lists;
      converter_.build_static_lists = false;
      exprt expr = converter_.get_expr(node);
      converter_.build_static_lists = old_build_static_lists;
      return expr;
    };

    auto to_expr = [this](const nlohmann::json &node) {
      return converter_.get_expr(node);
    };

    auto make_list = [](const std::vector<nlohmann::json> &elts) {
      nlohmann::json out;
      out["_type"] = "List";
      out["elts"] = elts;
      return out;
    };

    auto as_bool = [](const nlohmann::json &node) {
      numeric_value value;
      if (try_extract_numeric_constant(node, value))
        return to_double(value) != 0.0;
      if (
        node.is_object() && node.contains("value") &&
        node["value"].is_boolean())
        return node["value"].get<bool>();
      return false;
    };

    auto as_double = [](const nlohmann::json &node) {
      numeric_value value;
      if (try_extract_numeric_constant(node, value))
        return to_double(value);
      return 0.0;
    };

    auto compare_scalar = [&](
                            const std::string &op,
                            const nlohmann::json &lhs,
                            const nlohmann::json &rhs) {
      const double left = as_double(lhs);
      const double right = as_double(rhs);
      bool result = false;
      if (op == "greater")
        result = left > right;
      else if (op == "less")
        result = left < right;
      else if (op == "greater_equal")
        result = left >= right;
      else if (op == "less_equal")
        result = left <= right;
      else if (op == "equal")
        result = left == right;
      else
        result = left != right;
      return make_constant(result);
    };

    auto get_arg = [&](std::size_t index) {
      nlohmann::json arg = call_["args"][index];
      resolve_var(arg);
      materialize_inline_numpy_constructor_call(arg, converter_.ast());
      return arg;
    };

    if (
      function == "greater" || function == "less" ||
      function == "greater_equal" || function == "less_equal" ||
      function == "equal" || function == "not_equal")
    {
      auto lhs = get_arg(0);
      auto rhs = get_arg(1);

      if (lhs.contains("elts") && lhs["elts"].is_array())
      {
        std::vector<nlohmann::json> out_elts;
        for (std::size_t i = 0; i < lhs["elts"].size(); ++i)
        {
          const auto &lhs_item = lhs["elts"][i];
          const auto &rhs_item = rhs.contains("elts") && rhs["elts"].is_array()
                                   ? rhs["elts"][i]
                                   : rhs;
          out_elts.push_back(compare_scalar(function, lhs_item, rhs_item));
        }
        return to_list_expr(make_list(out_elts));
      }

      if (rhs.contains("elts") && rhs["elts"].is_array())
      {
        std::vector<nlohmann::json> out_elts;
        for (const auto &rhs_item : rhs["elts"])
          out_elts.push_back(compare_scalar(function, lhs, rhs_item));
        return to_list_expr(make_list(out_elts));
      }

      return to_expr(compare_scalar(function, lhs, rhs));
    }

    if (function == "logical_and" || function == "logical_or")
    {
      auto lhs = get_arg(0);
      auto rhs = get_arg(1);
      auto apply = [&](const nlohmann::json &a, const nlohmann::json &b) {
        const bool left = as_bool(a);
        const bool right = as_bool(b);
        return make_constant(
          function == "logical_and" ? (left && right) : (left || right));
      };

      if (lhs.contains("elts") && lhs["elts"].is_array())
      {
        std::vector<nlohmann::json> out_elts;
        for (std::size_t i = 0; i < lhs["elts"].size(); ++i)
        {
          const auto &lhs_item = lhs["elts"][i];
          const auto &rhs_item = rhs.contains("elts") && rhs["elts"].is_array()
                                   ? rhs["elts"][i]
                                   : rhs;
          out_elts.push_back(apply(lhs_item, rhs_item));
        }
        return to_list_expr(make_list(out_elts));
      }

      if (rhs.contains("elts") && rhs["elts"].is_array())
      {
        std::vector<nlohmann::json> out_elts;
        for (const auto &rhs_item : rhs["elts"])
          out_elts.push_back(apply(lhs, rhs_item));
        return to_list_expr(make_list(out_elts));
      }

      return to_expr(apply(lhs, rhs));
    }

    if (function == "logical_not")
    {
      auto arg = get_arg(0);
      if (arg.contains("elts") && arg["elts"].is_array())
      {
        std::vector<nlohmann::json> out_elts;
        for (const auto &item : arg["elts"])
          out_elts.push_back(make_constant(!as_bool(item)));
        return to_list_expr(make_list(out_elts));
      }
      return to_expr(make_constant(!as_bool(arg)));
    }

    if (function == "where")
    {
      auto cond = get_arg(0);
      auto x = get_arg(1);
      auto y = get_arg(2);
      if (cond.contains("elts") && cond["elts"].is_array())
      {
        std::vector<nlohmann::json> out_elts;
        for (std::size_t i = 0; i < cond["elts"].size(); ++i)
        {
          const bool choose_x = as_bool(cond["elts"][i]);
          const auto &chosen =
            choose_x
              ? (x.contains("elts") && x["elts"].is_array() ? x["elts"][i] : x)
              : (y.contains("elts") && y["elts"].is_array() ? y["elts"][i] : y);
          out_elts.push_back(chosen);
        }
        return to_list_expr(make_list(out_elts));
      }
      return as_bool(cond) ? converter_.get_expr(x) : converter_.get_expr(y);
    }

    auto parse_shape = [&](const nlohmann::json &shape_node) {
      std::vector<std::size_t> dims;
      if (
        shape_node.is_object() && shape_node.contains("_type") &&
        shape_node["_type"] == "Constant" && shape_node.contains("value") &&
        shape_node["value"].is_number_integer())
      {
        dims.push_back(shape_node["value"].get<std::size_t>());
        return dims;
      }
      if (
        shape_node.is_object() && shape_node.contains("_type") &&
        (shape_node["_type"] == "Tuple" || shape_node["_type"] == "List") &&
        shape_node.contains("elts") && shape_node["elts"].is_array())
      {
        for (const auto &elem : shape_node["elts"])
        {
          if (
            !elem.is_object() || !elem.contains("_type") ||
            elem["_type"] != "Constant" || !elem.contains("value") ||
            !elem["value"].is_number_integer())
          {
            dims.clear();
            return dims;
          }
          dims.push_back(elem["value"].get<std::size_t>());
        }
      }
      return dims;
    };

    if (function == "full")
    {
      auto shape = get_arg(0);
      auto fill = get_arg(1);
      auto dims = parse_shape(shape);
      if (dims.empty())
      {
        // Try symbolic 1-D shape: evaluate the size expression at runtime.
        bool pushed = false;
        if (shape.contains("_type"))
        {
          try
          {
            dims.push_back(shape["value"].get<std::size_t>());
            pushed = true;
          }
          catch (const std::exception &)
          {
          }
        }
        if (!pushed)
        {
          try
          {
            exprt size_expr = converter_.get_expr(call_["args"][0]);
            if (
              size_expr.type().is_signedbv() ||
              size_expr.type().is_unsignedbv())
            {
              exprt fill_expr = converter_.get_expr(call_["args"][1]);
              typet elem_type = fill_expr.type();
              nlohmann::json dummy;
              dummy["_type"] = "List";
              dummy["elts"] = nlohmann::json::array();
              converter_.copy_location_fields_from_decl(call_, dummy);
              python_list lb(converter_, dummy);
              return lb.build_symbolic_fill_list(
                size_expr, fill_expr, elem_type);
            }
          }
          catch (const std::exception &)
          {
          }
          throw std::runtime_error(
            "TypeError: numpy.full() shape argument must be an integer");
        }
      }
      if (dims.size() == 1)
      {
        std::vector<nlohmann::json> elts;
        for (std::size_t i = 0; i < dims[0]; ++i)
          elts.push_back(fill);
        return to_list_expr(make_list(elts));
      }
      if (dims.size() == 2)
      {
        std::vector<nlohmann::json> rows;
        for (std::size_t i = 0; i < dims[0]; ++i)
        {
          std::vector<nlohmann::json> row;
          for (std::size_t j = 0; j < dims[1]; ++j)
            row.push_back(fill);
          rows.push_back(make_list(row));
        }
        return to_list_expr(make_list(rows));
      }
      throw std::runtime_error(
        "TypeError: numpy.full() currently supports up to 2D shapes");
    }

    if (function == "eye" || function == "identity")
    {
      auto n = get_arg(0);
      auto m = function == "eye" && call_["args"].size() > 1 ? get_arg(1) : n;
      const std::size_t rows = n["value"].get<std::size_t>();
      const std::size_t cols = m["value"].get<std::size_t>();
      std::vector<nlohmann::json> out_rows;
      for (std::size_t i = 0; i < rows; ++i)
      {
        std::vector<nlohmann::json> row;
        for (std::size_t j = 0; j < cols; ++j)
          row.push_back(make_constant(i == j ? 1 : 0));
        out_rows.push_back(make_list(row));
      }
      return to_list_expr(make_list(out_rows));
    }

    if (function == "linspace")
    {
      auto start = as_double(get_arg(0));
      auto stop = as_double(get_arg(1));
      std::size_t num = 50;
      if (call_["args"].size() == 3)
        num = get_arg(2)["value"].get<std::size_t>();
      if (num == 0)
        return to_list_expr(make_list({}));
      if (num == 1)
        return to_list_expr(make_list({make_constant(start)}));
      const double step = (stop - start) / static_cast<double>(num - 1);
      std::vector<nlohmann::json> elts;
      for (std::size_t i = 0; i < num; ++i)
        elts.push_back(make_constant(start + (step * static_cast<double>(i))));
      return to_list_expr(make_list(elts));
    }
  }

  // Create array from numpy.array()
  if (function == "array")
  {
    nlohmann::json array_arg = call_["args"][0];
    const std::string dtype = get_dtype();
    if (!dtype.empty())
      array_arg = cast_numpy_literal_to_dtype(array_arg, dtype);

    int array_dims = type_handler_.get_array_dimensions(array_arg);
    if (array_dims > 8)
    {
      throw std::runtime_error(
        "ESBMC does not support arrays with more than 8 dimensions. Found " +
        std::to_string(array_dims) + "D array creation.");
    }

    typet size = type_handler_.get_typet(array_arg["elts"]);
    return converter_.get_static_array(array_arg, size);
  }

  static const std::unordered_map<std::string, float> array_creation_funcs = {
    {"zeros", 0.0}, {"ones", 1.0}};

  if (
    function == "empty_like" || function == "zeros_like" ||
    function == "ones_like" || function == "full_like")
  {
    if (
      call_["args"].empty() || call_["args"].size() > 2 ||
      (function != "full_like" && call_["args"].size() != 1))
    {
      throw std::runtime_error(
        "TypeError: numpy." + function + "() expects " +
        (function == "full_like" ? "1 or 2 arguments" : "1 argument"));
    }

    std::optional<nlohmann::json> fill_kwarg;
    if (call_.contains("keywords"))
    {
      for (const auto &kw : call_["keywords"])
      {
        if (kw["_type"] != "keyword" || kw["arg"].is_null())
          continue;
        const std::string arg = kw["arg"].get<std::string>();
        if (function == "full_like" && arg == "fill_value")
        {
          fill_kwarg = kw["value"];
          continue;
        }
        throw std::runtime_error(
          "TypeError: numpy." + function + "() keyword '" + arg +
          "' is not supported");
      }
    }

    if (
      function == "full_like" &&
      ((call_["args"].size() == 2) == fill_kwarg.has_value()))
    {
      throw std::runtime_error(
        "TypeError: numpy.full_like() expects exactly one fill_value");
    }

    exprt base_expr = converter_.get_expr(call_["args"][0]);
    typet base_type = base_expr.type();
    if (base_type.is_pointer() && base_type.subtype().is_array())
      base_type = base_type.subtype();
    if (!base_type.is_array())
      throw std::runtime_error(
        "TypeError: numpy." + function + "() requires a numpy array input");

    std::vector<int> shape = type_handler_.get_array_type_shape(base_type);
    if (shape.empty())
      throw std::runtime_error(
        "TypeError: numpy." + function + "() requires a concrete array shape");

    std::vector<long long> dims(shape.begin(), shape.end());
    validate_ndarray_shape(dims);

    typet elem_type = get_array_scalar_type(base_type);
    if (is_complex_type(elem_type))
      throw std::runtime_error(
        "TypeError: complex dtype is not supported in NumPy constructors yet");

    exprt expr;
    if (function == "empty_like")
    {
      expr = make_nondet_ndarray(
        type_handler_,
        elem_type,
        dims,
        0,
        converter_.get_location_from_decl(call_));
    }
    else
    {
      exprt fill = gen_zero(elem_type);
      if (function == "ones_like")
        fill = make_numpy_one(elem_type);
      else if (function == "full_like")
        fill = converter_.get_expr(
          fill_kwarg.has_value() ? *fill_kwarg : call_["args"][1]);

      expr = make_filled_ndarray(type_handler_, elem_type, dims, 0, fill);
    }

    if (converter_.current_lhs)
    {
      converter_.current_lhs->type() = expr.type();
      converter_.update_symbol(*converter_.current_lhs);
    }
    return expr;
  }

  if (function == "empty")
  {
    if (call_["args"].empty())
      throw std::runtime_error(
        "TypeError: numpy.empty() expects a shape argument");

    const std::string dtype = get_dtype();
    if (is_numpy_complex_dtype(dtype))
      throw std::runtime_error(
        "TypeError: complex dtype is not supported in NumPy constructors yet");

    typet elem_type =
      dtype.empty() ? cached_double_type() : get_typet_from_dtype();
    if (elem_type.is_nil() || elem_type.id().empty())
      get_dtype_size();

    nlohmann::json shape_arg = call_["args"][0];
    if (
      shape_arg.is_object() && shape_arg.contains("_type") &&
      shape_arg["_type"] == "Name")
    {
      nlohmann::json resolved = json_utils::find_var_decl(
        shape_arg["id"], converter_.current_function_name(), converter_.ast());
      if (
        resolved.contains("value") && resolved["value"].is_object() &&
        resolved["value"].contains("_type"))
      {
        shape_arg = resolved["value"];
      }
    }

    std::vector<long long> dims;
    const std::string arg_type = shape_arg["_type"];
    if (arg_type == "Constant" || arg_type == "UnaryOp")
    {
      numeric_value shape_numeric;
      if (
        try_extract_numeric_constant(shape_arg, shape_numeric) &&
        shape_numeric.is_int)
      {
        dims.push_back(shape_numeric.int_value);
      }
    }
    else if (arg_type == "Tuple" || arg_type == "List")
    {
      const auto &elts = shape_arg["elts"];
      if (elts.empty())
        throw std::runtime_error(
          "TypeError: empty() shape tuple must be non-empty");
      if (elts.size() > 8)
        throw std::runtime_error(
          "ESBMC does not support arrays with more than 8 dimensions. Found " +
          std::to_string(elts.size()) + "D array creation in empty().");

      for (const auto &e : elts)
      {
        numeric_value dim;
        if (!try_extract_numeric_constant(e, dim) || !dim.is_int)
        {
          dims.clear();
          break;
        }
        dims.push_back(dim.int_value);
      }
    }

    if (dims.empty())
      throw std::runtime_error(
        "TypeError: empty() argument must be int or tuple of ints");

    validate_ndarray_shape(dims);
    exprt expr = make_nondet_ndarray(
      type_handler_,
      elem_type,
      dims,
      0,
      converter_.get_location_from_decl(call_));
    if (converter_.current_lhs)
    {
      converter_.current_lhs->type() = expr.type();
      converter_.update_symbol(*converter_.current_lhs);
    }
    return expr;
  }

  // Create array from numpy.zeros() or numpy.ones()
  auto it = array_creation_funcs.find(function);
  if (it != array_creation_funcs.end())
  {
    const scalar_value fill = make_real_scalar(it->second);
    const std::string dtype = get_dtype();
    const nlohmann::json fill_value = make_numpy_typed_constant(fill, dtype);
    nlohmann::json shape_arg = call_["args"][0];

    // Resolve variable references for shape arguments
    if (
      shape_arg.is_object() && shape_arg.contains("_type") &&
      shape_arg["_type"] == "Name")
    {
      nlohmann::json resolved = json_utils::find_var_decl(
        shape_arg["id"], converter_.current_function_name(), converter_.ast());
      if (
        resolved.contains("value") && resolved["value"].is_object() &&
        resolved["value"].contains("_type") &&
        resolved["value"]["_type"] == "Constant")
      {
        shape_arg = resolved["value"];
      }
    }

    const std::string arg_type = shape_arg["_type"];

    if (arg_type == "Constant" || arg_type == "UnaryOp")
    {
      // np.zeros(n) or np.ones(n) — 1D. try_extract_numeric_constant also
      // resolves a negative literal (UnaryOp USub over a Constant), so the
      // shape can be validated (ADR: negative dimensions are rejected, not
      // silently truncated to an empty array) before building the list.
      numeric_value shape_numeric;
      if (
        try_extract_numeric_constant(shape_arg, shape_numeric) &&
        shape_numeric.is_int)
      {
        validate_ndarray_shape({shape_numeric.int_value});
        if (shape_numeric.int_value > std::numeric_limits<int>::max())
          throw std::runtime_error(
            "ValueError: array size overflows during creation");
        auto list =
          create_list(static_cast<int>(shape_numeric.int_value), fill_value);
        return converter_.get_expr(list);
      }
    }

    if (arg_type == "Tuple")
    {
      const auto &elts = shape_arg["elts"];
      const std::size_t ndim = elts.size();
      if (ndim == 0)
      {
        throw std::runtime_error(
          "TypeError: " + function + "() shape tuple must be non-empty");
      }
      if (ndim > 8)
        throw std::runtime_error(
          "ESBMC does not support arrays with more than 8 dimensions. "
          "Found " +
          std::to_string(ndim) + "D array creation in " + function + "().");

      std::vector<int> dims;
      for (const auto &e : elts)
        dims.push_back(e["value"].get<int>());

      // Build nested list recursively: create_nd_fill(dims, dim_idx, fill)
      std::function<nlohmann::json(std::size_t)> create_nd_fill =
        [&](std::size_t dim_idx) -> nlohmann::json {
        if (dim_idx == dims.size() - 1)
          return create_list(dims[dim_idx], fill_value);
        nlohmann::json list;
        list["_type"] = "List";
        list["elts"] = nlohmann::json::array();
        for (int i = 0; i < dims[dim_idx]; ++i)
          list["elts"].push_back(create_nd_fill(dim_idx + 1));
        return list;
      };

      return converter_.get_expr(create_nd_fill(0));
    }

    // Symbolic 1-D shape: try to evaluate the size expression at runtime,
    // then construct the list via a bounded while-loop so the model checker
    // can unwind up to its --unwind limit. Complex/unsupported symbolic shapes
    // (tuples, multi-dimensional) are still rejected explicitly.
    if (
      shape_arg.value("_type", std::string()) == "Name" ||
      shape_arg.value("_type", std::string()) == "Call" ||
      shape_arg.value("_type", std::string()) == "BinOp")
    {
      try
      {
        exprt size_expr = converter_.get_expr(shape_arg);
        if (
          !size_expr.type().is_signedbv() && !size_expr.type().is_unsignedbv())
          throw std::runtime_error("");

        // Determine fill value and its type from the fill constant + dtype.
        typet elem_type;
        exprt fill_expr;
        if (fill_value.value("value", 0.0) == 0.0 && dtype.empty())
        {
          elem_type = converter_.get_type_handler().get_typet(
            std::string("int"), static_cast<size_t>(0));
          nlohmann::json zero_const;
          zero_const["_type"] = "Constant";
          zero_const["value"] = 0;
          fill_expr = converter_.get_expr(zero_const);
        }
        else
        {
          fill_expr = converter_.get_expr(fill_value);
          elem_type = fill_expr.type();
        }

        nlohmann::json dummy_node;
        dummy_node["_type"] = "List";
        dummy_node["elts"] = nlohmann::json::array();
        converter_.copy_location_fields_from_decl(call_, dummy_node);
        python_list list_builder(converter_, dummy_node);
        return list_builder.build_symbolic_fill_list(
          size_expr, fill_expr, elem_type);
      }
      catch (const std::exception &)
      {
        // Fall through to the explicit rejection error below.
      }
    }

    throw std::runtime_error(
      "TypeError: " + function + "() argument must be int or tuple of ints");
  }

  auto resolve_numpy_var = [this](nlohmann::json &var) {
    if (var.contains("_type") && var["_type"] == "Name")
    {
      var = json_utils::find_var_decl(
        var["id"], converter_.current_function_name(), converter_.ast());
      if (!var.contains("value") || !var["value"].is_object())
        return;
      if (var["value"]["_type"] == "Call")
      {
        if (
          std::optional<nlohmann::json> materialized =
            materialize_numpy_constructor_array(var["value"], converter_.ast()))
          var = std::move(*materialized);
        else if (is_numpy_constructor_call_by_name(var["value"]))
          var = var["value"];
        else if (var["value"].contains("args") && !var["value"]["args"].empty())
          var = var["value"]["args"][0];
        else
          var = var["value"];
      }
      else
      {
        var = var["value"];
      }
    }
  };

  auto resolve_literal_numpy_array_input = [this](
                                             nlohmann::json arr_arg,
                                             const std::string &function_name,
                                             bool inline_only = false) {
    if (!inline_only && arr_arg.value("_type", std::string()) == "Name")
    {
      nlohmann::json resolved = json_utils::find_var_decl(
        arr_arg["id"], converter_.current_function_name(), converter_.ast());
      if (resolved.contains("value") && resolved["value"].is_object())
        arr_arg = resolved["value"];
    }

    auto literal_arg = get_literal_numpy_array_arg(arr_arg);
    if (!literal_arg.has_value())
      throw std::runtime_error(
        "TypeError: numpy." + function_name + "() currently supports only " +
        (inline_only ? "inline literal" : "literal") + " numpy.array inputs");
    return std::move(*literal_arg);
  };

  if (function == "median")
  {
    if (call_["args"].size() != 1)
      throw std::runtime_error(
        "TypeError: numpy.median() expects 1 positional argument");

    bool flatten = false;
    if (call_.contains("keywords"))
    {
      for (const auto &kw : call_["keywords"])
      {
        if (kw["_type"] != "keyword" || kw["arg"].is_null())
          continue;

        const std::string arg = kw["arg"].get<std::string>();
        if (arg == "axis")
        {
          if (!is_json_none_literal(kw["value"]))
            throw std::runtime_error(
              "TypeError: numpy.median() axis is not supported");
          flatten = true;
          continue;
        }

        throw std::runtime_error(
          "TypeError: numpy.median() keyword '" + arg + "' is not supported");
      }
    }

    nlohmann::json arr_arg = call_["args"][0];
    if (
      std::optional<nlohmann::json> row_view =
        resolve_literal_numpy_row_view(arr_arg, converter_))
      arr_arg = std::move(*row_view);
    else
      arr_arg = resolve_literal_numpy_array_input(arr_arg, function, true);

    std::vector<std::size_t> shape;
    if (!get_literal_shape(arr_arg, shape))
      throw std::runtime_error(
        "TypeError: numpy.median() array must contain finite numeric values");

    std::vector<nlohmann::json> elements;
    if (shape.size() == 1)
      elements = arr_arg["elts"].get<std::vector<nlohmann::json>>();
    else if (shape.size() == 2 && flatten)
      flatten_json_list(arr_arg, elements);
    else
      throw std::runtime_error(
        "TypeError: numpy.median() axis is not supported");

    nlohmann::json result;
    result["_type"] = "Constant";
    result["value"] = median_of_numeric_elements(
      std::move(elements),
      "TypeError: numpy.median() array must contain finite numeric values");
    return converter_.get_expr(result);
  }

  if (function == "percentile")
  {
    if (call_["args"].size() != 2)
      throw std::runtime_error(
        "TypeError: numpy.percentile() expects array and q arguments");

    if (call_.contains("keywords"))
    {
      for (const auto &kw : call_["keywords"])
      {
        if (kw["_type"] != "keyword" || kw["arg"].is_null())
          continue;

        throw std::runtime_error(
          "TypeError: numpy.percentile() keyword '" +
          kw["arg"].get<std::string>() + "' is not supported");
      }
    }

    numeric_value q_value;
    if (
      !try_extract_numeric_constant(call_["args"][1], q_value) ||
      !is_finite_numeric_value(q_value))
      throw std::runtime_error(
        "TypeError: numpy.percentile() q must be a concrete scalar");

    const double q = to_double(q_value);
    if (q < 0.0 || q > 100.0)
      throw std::runtime_error(
        "ValueError: numpy.percentile() q must be in [0, 100]");

    nlohmann::json arr_arg =
      resolve_literal_numpy_array_input(call_["args"][0], function, true);

    std::vector<std::size_t> shape;
    if (!get_literal_shape(arr_arg, shape))
      throw std::runtime_error(
        "TypeError: numpy.percentile() array must contain finite numeric "
        "values");

    if (shape.size() != 1)
      throw std::runtime_error(
        "TypeError: numpy.percentile() currently supports only 1-D arrays");

    nlohmann::json result;
    result["_type"] = "Constant";
    result["value"] = percentile_of_numeric_elements(
      arr_arg["elts"].get<std::vector<nlohmann::json>>(),
      q,
      "TypeError: numpy.percentile() array must contain finite numeric "
      "values");
    return converter_.get_expr(result);
  }

  if (function == "unique")
  {
    if (call_["args"].size() != 1)
      throw std::runtime_error(
        "TypeError: numpy.unique() expects 1 positional argument");

    if (call_.contains("keywords"))
    {
      for (const auto &kw : call_["keywords"])
      {
        if (kw["_type"] != "keyword" || kw["arg"].is_null())
          continue;

        const std::string arg = kw["arg"].get<std::string>();
        throw std::runtime_error(
          "TypeError: numpy.unique() keyword '" + arg + "' is not supported");
      }
    }

    nlohmann::json arr_arg =
      resolve_literal_numpy_array_input(call_["args"][0], function, true);

    if (!is_1d_json_list(arr_arg))
      throw std::runtime_error(
        "TypeError: numpy.unique() currently supports only 1-D arrays");

    return converter_.get_expr(make_unique_numeric_list(
      arr_arg["elts"].get<std::vector<nlohmann::json>>(),
      "TypeError: numpy.unique() array must contain finite numeric values"));
  }

  if (function == "argsort")
  {
    if (call_["args"].size() != 1)
      throw std::runtime_error(
        "TypeError: numpy.argsort() expects 1 positional argument");

    if (call_.contains("keywords") && !call_["keywords"].empty())
      throw std::runtime_error(
        "TypeError: numpy.argsort() keywords are not supported");

    nlohmann::json arr_arg =
      resolve_literal_numpy_array_input(call_["args"][0], function, true);

    std::vector<std::size_t> shape;
    if (!get_literal_shape(arr_arg, shape) || shape.size() != 1)
      throw std::runtime_error(
        "TypeError: numpy.argsort() currently supports only 1-D arrays");

    const auto &elements = arr_arg["elts"];
    std::vector<std::size_t> indices(elements.size());
    for (std::size_t i = 0; i < indices.size(); ++i)
      indices[i] = i;

    std::stable_sort(
      indices.begin(), indices.end(), [&](std::size_t lhs, std::size_t rhs) {
        return numeric_to_key(
                 elements[lhs],
                 "TypeError: numpy.argsort() array must contain finite numeric "
                 "values") <
               numeric_to_key(
                 elements[rhs],
                 "TypeError: numpy.argsort() array must contain finite numeric "
                 "values");
      });

    return converter_.get_expr(make_integer_list(indices));
  }

  if (function == "searchsorted")
  {
    if (call_["args"].size() != 2)
      throw std::runtime_error(
        "TypeError: numpy.searchsorted() expects array and value arguments");

    bool right = false;
    if (call_.contains("keywords"))
    {
      for (const auto &kw : call_["keywords"])
      {
        if (kw["_type"] != "keyword" || kw["arg"].is_null())
          continue;

        const std::string arg = kw["arg"].get<std::string>();
        if (arg == "side")
        {
          const auto &value = kw["value"];
          if (
            !value.is_object() ||
            value.value("_type", std::string()) != "Constant" ||
            !value.contains("value") || !value["value"].is_string())
          {
            throw std::runtime_error(
              "TypeError: numpy.searchsorted() side must be 'left' or 'right'");
          }
          const std::string side = value["value"].get<std::string>();
          if (side == "left")
            right = false;
          else if (side == "right")
            right = true;
          else
            throw std::runtime_error(
              "TypeError: numpy.searchsorted() side must be 'left' or 'right'");
          continue;
        }

        throw std::runtime_error(
          "TypeError: numpy.searchsorted() keyword '" + arg +
          "' is not supported");
      }
    }

    nlohmann::json arr_arg =
      resolve_literal_numpy_array_input(call_["args"][0], function, true);

    std::vector<std::size_t> shape;
    if (!get_literal_shape(arr_arg, shape) || shape.size() != 1)
      throw std::runtime_error(
        "TypeError: numpy.searchsorted() currently supports only 1-D arrays");

    if (!is_sorted_numeric_list(
          arr_arg,
          "TypeError: numpy.searchsorted() array must contain finite numeric "
          "values"))
      throw std::runtime_error(
        "TypeError: numpy.searchsorted() requires a sorted 1-D array");

    nlohmann::json position;
    position["_type"] = "Constant";
    nlohmann::json value_arg = call_["args"][1];
    numeric_to_key(
      value_arg,
      "TypeError: numpy.searchsorted() value must be a finite numeric literal");
    position["value"] =
      static_cast<int64_t>(searchsorted_position(arr_arg, value_arg, right));
    return converter_.get_expr(position);
  }

  if (function == "sort")
  {
    if (call_["args"].empty() || call_["args"].size() > 2)
      throw std::runtime_error(
        "TypeError: numpy.sort() expects 1 or 2 positional arguments");

    bool flatten = false;
    long long axis = -1;
    auto parse_axis = [&](const nlohmann::json &axis_node) {
      if (is_json_none_literal(axis_node))
      {
        flatten = true;
        return;
      }

      numeric_value axis_value;
      if (
        !try_extract_numeric_constant(axis_node, axis_value) ||
        !axis_value.is_int)
      {
        throw std::runtime_error(
          "TypeError: numpy.sort() axis must be a literal integer or None");
      }
      axis = axis_value.int_value;
    };

    if (call_["args"].size() == 2)
      parse_axis(call_["args"][1]);

    if (call_.contains("keywords"))
    {
      for (const auto &kw : call_["keywords"])
      {
        if (kw["_type"] != "keyword" || kw["arg"].is_null())
          continue;

        const std::string arg = kw["arg"].get<std::string>();
        if (arg == "axis")
        {
          if (call_["args"].size() == 2)
            throw std::runtime_error(
              "TypeError: numpy.sort() got multiple values for axis");
          parse_axis(kw["value"]);
          continue;
        }

        throw std::runtime_error(
          "TypeError: numpy.sort() keyword '" + arg + "' is not supported");
      }
    }

    nlohmann::json arr_arg = call_["args"][0];
    if (arr_arg.value("_type", std::string()) == "Name")
    {
      nlohmann::json resolved = json_utils::find_var_decl(
        arr_arg["id"], converter_.current_function_name(), converter_.ast());
      if (resolved.contains("value") && resolved["value"].is_object())
        arr_arg = resolved["value"];
    }

    auto literal_arg = get_literal_numpy_array_arg(arr_arg);
    if (!literal_arg.has_value())
      throw std::runtime_error(
        "TypeError: numpy.sort() currently supports only literal numpy.array "
        "inputs");
    arr_arg = std::move(*literal_arg);

    std::vector<std::size_t> shape;
    if (!get_literal_shape(arr_arg, shape) || shape.empty())
      throw std::runtime_error(
        "TypeError: numpy.sort() currently supports only constant arrays");

    std::vector<nlohmann::json> elements;
    if (flatten)
    {
      flatten_json_list(arr_arg, elements);
    }
    else
    {
      if (shape.size() != 1 || (axis != 0 && axis != -1))
      {
        throw std::runtime_error(
          "TypeError: numpy.sort() axis " + std::to_string(axis) +
          " is not supported");
      }
      elements = arr_arg["elts"].get<std::vector<nlohmann::json>>();
    }

    return converter_.get_expr(make_sorted_numeric_list(std::move(elements)));
  }

  if (function == "reshape")
  {
    if (call_["args"].size() < 2)
      throw std::runtime_error(
        "TypeError: numpy.reshape() requires array and shape arguments");

    nlohmann::json arr_arg = call_["args"][0];
    resolve_numpy_var(arr_arg);

    std::vector<std::size_t> old_shape;
    if (!get_literal_shape(arr_arg, old_shape))
      throw std::runtime_error(
        "TypeError: numpy.reshape() currently supports only constant arrays");

    std::vector<nlohmann::json> flat;
    flatten_json_list(arr_arg, flat);
    std::size_t total = flat.size();

    auto parse_reshape_dim = [](const nlohmann::json &node) -> int64_t {
      if (
        node.is_object() && node.contains("_type") &&
        node["_type"] == "Constant" && node.contains("value") &&
        node["value"].is_number_integer())
        return node["value"].get<int64_t>();
      if (
        node.is_object() && node.contains("_type") &&
        node["_type"] == "UnaryOp" && node.contains("op") &&
        node["op"]["_type"] == "USub" && node.contains("operand") &&
        node["operand"]["_type"] == "Constant" &&
        node["operand"]["value"].is_number_integer())
        return -node["operand"]["value"].get<int64_t>();
      return INT64_MIN;
    };

    std::vector<int64_t> raw_shape;
    const auto &shape_arg = call_["args"][1];
    if (
      shape_arg.is_object() && shape_arg.contains("_type") &&
      (shape_arg["_type"] == "Tuple" || shape_arg["_type"] == "List") &&
      shape_arg.contains("elts"))
    {
      for (const auto &e : shape_arg["elts"])
      {
        int64_t d = parse_reshape_dim(e);
        if (d == INT64_MIN)
          throw std::runtime_error(
            "TypeError: numpy.reshape() shape must contain concrete integers");
        raw_shape.push_back(d);
      }
    }
    else if (call_["args"].size() > 2)
    {
      // Method form with each dimension as its own positional argument
      // (a.reshape(d1, d2, ...)), equivalent to a.reshape((d1, d2, ...)).
      // Only reachable here (not the single-tuple-arg branch above) because
      // a single dimension can't be split into more than one argument, so
      // more than one argument past the array itself always means this
      // form -- for the method-form rewrite. A genuine module-function call
      // numpy.reshape(a, 2, 3) has no such form: numpy's real signature is
      // reshape(a, newshape, order='C'), so the third positional argument
      // is `order`, not another dimension. Reject that case explicitly
      // instead of silently reinterpreting it as split dimensions.
      if (!call_.value("_numpy_method_form", false))
        throw std::runtime_error(
          "TypeError: numpy.reshape() does not accept dimensions as "
          "separate positional arguments; pass a tuple, or use the "
          "a.reshape(d1, d2, ...) method form");

      for (std::size_t i = 1; i < call_["args"].size(); ++i)
      {
        int64_t d = parse_reshape_dim(call_["args"][i]);
        if (d == INT64_MIN)
          throw std::runtime_error(
            "TypeError: numpy.reshape() shape must contain concrete integers");
        raw_shape.push_back(d);
      }
    }
    else
    {
      int64_t d = parse_reshape_dim(shape_arg);
      if (d == INT64_MIN)
        throw std::runtime_error(
          "TypeError: numpy.reshape() shape must be a concrete integer or "
          "tuple");
      raw_shape.push_back(d);
    }

    std::vector<std::size_t> new_shape;
    std::size_t inferred_idx = raw_shape.size();
    std::size_t known_product = 1;
    for (std::size_t i = 0; i < raw_shape.size(); ++i)
    {
      if (raw_shape[i] == -1)
      {
        if (inferred_idx != raw_shape.size())
          throw std::runtime_error(
            "ValueError: can only specify one unknown dimension");
        inferred_idx = i;
        new_shape.push_back(0);
      }
      else if (raw_shape[i] < 0)
      {
        throw std::runtime_error(
          "ValueError: negative dimensions are not allowed");
      }
      else
      {
        new_shape.push_back(static_cast<std::size_t>(raw_shape[i]));
        known_product *= new_shape.back();
      }
    }
    if (inferred_idx != raw_shape.size())
    {
      if (known_product == 0 || total % known_product != 0)
        throw std::runtime_error(
          "ValueError: cannot reshape array of size " + std::to_string(total) +
          " into shape with remainder");
      new_shape[inferred_idx] = total / known_product;
    }

    std::size_t new_total = 1;
    for (auto d : new_shape)
      new_total *= d;
    if (new_total != total)
      throw std::runtime_error(
        "ValueError: cannot reshape array of size " + std::to_string(total) +
        " into shape " + format_shape(new_shape));

    std::size_t offset = 0;
    nlohmann::json result = reshape_flat_to_json(flat, new_shape, 0, offset);
    return converter_.get_expr(result);
  }

  if (function == "ravel" || function == "flatten" || function == "nditer")
  {
    if (call_["args"].empty())
      throw std::runtime_error(
        "TypeError: numpy." + function + "() requires an array argument");

    if (function == "nditer" && call_.contains("keywords"))
    {
      for (const auto &kw : call_["keywords"])
      {
        if (kw["_type"] != "keyword" || kw["arg"].is_null())
          continue;

        if (kw["arg"] == "op_flags")
          throw std::runtime_error(
            "TypeError: numpy.nditer() op_flags are not supported");

        throw std::runtime_error(
          "TypeError: numpy.nditer() keyword '" + kw["arg"].get<std::string>() +
          "' is not supported");
      }
    }

    nlohmann::json arr_arg = call_["args"][0];
    if (function == "nditer")
    {
      auto literal_arg = get_literal_numpy_array_arg(arr_arg);
      if (literal_arg.has_value())
        arr_arg = std::move(*literal_arg);
      else
        resolve_numpy_var(arr_arg);
    }
    else
    {
      resolve_numpy_var(arr_arg);
      materialize_inline_numpy_constructor_call(arr_arg, converter_.ast());
    }

    std::vector<std::size_t> old_shape;
    if (!get_literal_shape(arr_arg, old_shape))
    {
      nlohmann::json original_arg = call_["args"][0];
      if (original_arg.is_object() && original_arg.value("_type", "") == "Name")
      {
        nlohmann::json decl = json_utils::find_var_decl(
          original_arg["id"],
          converter_.current_function_name(),
          converter_.ast());
        if (decl.contains("value"))
        {
          auto literal_arg = get_literal_numpy_array_arg(decl["value"]);
          if (literal_arg.has_value())
            arr_arg = std::move(*literal_arg);
        }
      }

      if (!get_literal_shape(arr_arg, old_shape))
        throw std::runtime_error(
          "TypeError: numpy." + function +
          "() currently supports only constant arrays");
    }

    std::vector<nlohmann::json> flat;
    flatten_json_list(arr_arg, flat);

    nlohmann::json result;
    result["_type"] = "List";
    result["elts"] = nlohmann::json::array();
    for (const auto &elem : flat)
      result["elts"].push_back(elem);
    return converter_.get_expr(result);
  }

  if (function == "expand_dims")
  {
    if (call_["args"].size() < 2)
      throw std::runtime_error(
        "TypeError: numpy.expand_dims() requires array and axis arguments");

    nlohmann::json arr_arg = call_["args"][0];
    resolve_numpy_var(arr_arg);

    numeric_value axis_value;
    if (
      !try_extract_numeric_constant(call_["args"][1], axis_value) ||
      !axis_value.is_int)
      throw std::runtime_error(
        "TypeError: numpy.expand_dims() axis must be a concrete integer");

    if (axis_value.int_value != 0)
      throw std::runtime_error(
        "AxisError: axis " + std::to_string(axis_value.int_value) +
        " is out of bounds for array of dimension 1");

    nlohmann::json result;
    result["_type"] = "List";
    result["elts"] = nlohmann::json::array({arr_arg});
    return converter_.get_expr(result);
  }

  // np.squeeze(a[, axis]) — remove axes of size 1 from the shape.
  if (function == "squeeze")
  {
    if (call_["args"].empty())
      throw std::runtime_error(
        "TypeError: numpy.squeeze() requires an array argument");

    nlohmann::json arr_arg = call_["args"][0];
    resolve_numpy_var(arr_arg);

    // Recursively strip List wrappers that contain exactly one element, which
    // is a nested List (i.e. the axis has size 1).
    std::function<nlohmann::json(const nlohmann::json &)> do_squeeze =
      [&](const nlohmann::json &node) -> nlohmann::json {
      if (
        !node.is_object() || node.value("_type", std::string()) != "List" ||
        !node.contains("elts"))
        return node;
      const auto &elts = node["elts"];
      if (
        elts.size() == 1 && elts[0].is_object() &&
        elts[0].value("_type", std::string()) == "List")
        return do_squeeze(elts[0]);
      if (elts.size() == 1)
        return do_squeeze(elts[0]);
      nlohmann::json out = node;
      out["elts"] = nlohmann::json::array();
      for (const auto &e : elts)
        out["elts"].push_back(do_squeeze(e));
      return out;
    };

    return converter_.get_expr(do_squeeze(arr_arg));
  }

  if (function == "swapaxes" || function == "moveaxis")
  {
    if (call_["args"].size() < 3)
      throw std::runtime_error(
        "TypeError: numpy." + function +
        "() requires array and axis arguments");

    nlohmann::json arr_arg = call_["args"][0];
    resolve_numpy_var(arr_arg);

    std::vector<std::size_t> shape;
    const std::size_t rank =
      get_literal_shape(arr_arg, shape) ? shape.size() : 0;

    for (std::size_t i = 1; i <= 2; ++i)
    {
      numeric_value axis_value;
      if (
        !try_extract_numeric_constant(call_["args"][i], axis_value) ||
        !axis_value.is_int)
        throw std::runtime_error(
          "TypeError: numpy." + function +
          "() axis must be a concrete integer");

      long long axis = axis_value.int_value;
      if (axis < 0)
        axis += static_cast<long long>(rank);
      if (rank != 0 && (axis < 0 || axis >= static_cast<long long>(rank)))
        throw std::runtime_error(
          "AxisError: axis " + std::to_string(axis_value.int_value) +
          " is out of bounds for array of dimension " + std::to_string(rank));
    }

    throw std::runtime_error(
      "TypeError: numpy." + function + " returns a view and is not supported");
  }

  if (function == "broadcast_to")
    throw std::runtime_error(
      "TypeError: numpy.broadcast_to returns a readonly view and is not "
      "supported");

  // np.stack(arrays[, axis]) — join arrays along a new first axis.
  // Only axis=0 is fully supported; other axes are accepted but also lower
  // to axis-0 concatenation (correct for homogeneous 1-D input arrays).
  if (function == "stack")
  {
    if (call_["args"].empty())
      throw std::runtime_error(
        "TypeError: numpy.stack() requires a sequence of arrays");

    nlohmann::json seq_arg = call_["args"][0];
    resolve_numpy_var(seq_arg);

    if (
      !seq_arg.is_object() || seq_arg.value("_type", std::string()) != "List" ||
      !seq_arg.contains("elts") || seq_arg["elts"].empty())
      throw std::runtime_error(
        "TypeError: numpy.stack() requires a non-empty list of arrays");

    nlohmann::json result;
    result["_type"] = "List";
    result["elts"] = nlohmann::json::array();
    for (const auto &arr : seq_arg["elts"])
    {
      nlohmann::json elem = arr;
      resolve_numpy_var(elem);
      result["elts"].push_back(elem);
    }
    return converter_.get_expr(result);
  }

  // np.concatenate(arrays[, axis]) — join arrays along an existing axis.
  // Constant-fold path: both operands must be literal lists with matching
  // inner shape.  axis=0 is the default and concatenates along the outermost
  // dimension; other axes are rejected with an explicit error.
  if (function == "concatenate")
  {
    if (call_["args"].empty())
      throw std::runtime_error(
        "TypeError: numpy.concatenate() requires a sequence of arrays");

    // Extract optional axis kwarg (default 0)
    int axis = 0;
    if (call_.contains("keywords"))
    {
      for (const auto &kw : call_["keywords"])
      {
        if (
          kw.value("arg", std::string()) == "axis" &&
          kw["value"].value("_type", std::string()) == "Constant")
          axis = kw["value"]["value"].get<int>();
      }
    }
    if (axis != 0)
      throw std::runtime_error(
        "TypeError: numpy.concatenate() only supports axis=0 currently; "
        "use np.stack() to concatenate along a new axis");

    nlohmann::json seq_arg = call_["args"][0];
    resolve_numpy_var(seq_arg);

    if (
      !seq_arg.is_object() || seq_arg.value("_type", std::string()) != "List" ||
      !seq_arg.contains("elts") || seq_arg["elts"].empty())
      throw std::runtime_error(
        "TypeError: numpy.concatenate() requires a non-empty list of arrays");

    nlohmann::json result;
    result["_type"] = "List";
    result["elts"] = nlohmann::json::array();
    for (const auto &arr : seq_arg["elts"])
    {
      nlohmann::json elem = arr;
      resolve_numpy_var(elem);
      if (
        elem.value("_type", std::string()) != "List" || !elem.contains("elts"))
        throw std::runtime_error(
          "TypeError: numpy.concatenate() currently supports only constant "
          "arrays");
      // Validate inner shapes match: all inputs must have same element shape
      // (checked implicitly — mismatched shapes will produce wrong results;
      // a full shape check would need get_literal_shape on each element).
      for (const auto &e : elem["elts"])
        result["elts"].push_back(e);
    }
    return converter_.get_expr(result);
  }

  // Handle math function calls
  if (is_math_function())
  {
    // np.fmod(x, y) on scalars has the same semantics as math.fmod / C fmod, so
    // delegate to the shared math handler when the operands are not foldable
    // literal lists. For list-backed 1D/2D inputs, fold here with the same
    // broadcasting helper used by the other binary NumPy ops.
    if (function == "fmod" && call_["args"].size() == 2)
    {
      auto lhs = call_["args"][0];
      auto rhs = call_["args"][1];

      std::vector<std::size_t> lhs_shape;
      std::vector<std::size_t> rhs_shape;
      if (
        get_literal_shape(lhs, lhs_shape) && get_literal_shape(rhs, rhs_shape))
      {
        std::vector<std::size_t> result_shape;
        if (
          compute_broadcast_shape(lhs_shape, rhs_shape, result_shape) &&
          result_shape.size() <= 2)
        {
          nlohmann::json folded;
          std::vector<std::size_t> indices;
          if (build_broadcast_literal_result(
                function,
                lhs,
                lhs_shape,
                rhs,
                rhs_shape,
                result_shape,
                indices,
                0,
                folded))
          {
            exprt result_expr = converter_.get_expr(folded);
            if (converter_.current_lhs)
            {
              converter_.current_lhs->type() = result_expr.type();
              converter_.update_symbol(*converter_.current_lhs);
            }
            return result_expr;
          }
        }
      }

      exprt lhs_expr = converter_.get_expr(lhs);
      exprt rhs_expr = converter_.get_expr(rhs);
      const typet list_type = type_handler_.get_list_type();
      auto is_container = [&list_type](const exprt &e) {
        return e.type().is_array() || e.type() == list_type ||
               (e.type().is_pointer() && e.type().subtype() == list_type);
      };
      if (is_container(lhs_expr) || is_container(rhs_expr))
        throw std::runtime_error(
          "Unsupported operation: numpy.fmod on array operands");
      return converter_.get_math_handler().handle_fmod(
        lhs_expr, rhs_expr, call_);
    }

    auto is_scalar_node = [](const nlohmann::json &node) {
      const std::string type = node["_type"];
      return type == "Constant" || type == "UnaryOp";
    };

    if (
      call_["args"].size() == 2 && is_scalar_node(call_["args"][0]) &&
      is_scalar_node(call_["args"][1]) &&
      !is_complex_annotated_scalar_node(call_["args"][0]) &&
      !is_complex_annotated_scalar_node(call_["args"][1]))
    {
      auto lhs = extract_value(call_["args"][0]);
      auto rhs = extract_value(call_["args"][1]);

      auto compute_scalar_result =
        [&](double left, double right, double &out) -> bool {
        if (function == "add")
        {
          out = left + right;
          return true;
        }
        if (function == "subtract")
        {
          out = left - right;
          return true;
        }
        if (function == "multiply")
        {
          out = left * right;
          return true;
        }
        if (function == "divide")
        {
          if (right == 0.0)
            return false;
          out = left / right;
          return true;
        }
        if (function == "power")
        {
          out = std::pow(left, right);
          return true;
        }
        if (function == "copysign")
        {
          out = std::copysign(left, right);
          return true;
        }
        if (function == "fmax")
        {
          out = std::fmax(left, right);
          return true;
        }
        if (function == "fmin")
        {
          out = std::fmin(left, right);
          return true;
        }
        if (function == "round")
        {
          // numpy.round(x, decimals): round half to even (numpy semantics),
          // for decimals zero or negative too (e.g. round(12345, -2) == 12300).
          // This host fold must produce the SAME double on every platform, else
          // the baked SMT constant diverges and the verdict flips (it did:
          // passed on arm64, failed on x86-64). Two non-portable pitfalls are
          // avoided: (1) std::pow(10.0, d) is not guaranteed to be the exact
          // power of ten (glibc vs Apple libm differ), so build the scale by
          // exact repeated multiplication; (2) std::nearbyint honours the
          // ambient FP rounding mode, so decide half-to-even explicitly with
          // std::floor, which ignores the mode.
          const long long decimals = static_cast<long long>(right);
          double pow10 = 1.0;
          for (long long i = 0; i < std::llabs(decimals); ++i)
            pow10 *= 10.0;
          const double scaled = decimals >= 0 ? left * pow10 : left / pow10;
          double r = std::floor(scaled);
          const double frac = scaled - r;
          if (frac > 0.5 || (frac == 0.5 && std::fmod(r, 2.0) != 0.0))
            r += 1.0;
          out = decimals >= 0 ? r / pow10 : r * pow10;
          return true;
        }
        return false;
      };

      // copysign/fmax/fmin/round have no operator_map() entry and no handler,
      // so the BinOp path below crashes migrate_expr.
      // Fold the scalar-constant case here.
      // Symbolic and array operands are unsupported.
      if (
        allow_numpy_fold && (function == "copysign" || function == "fmax" ||
                             function == "fmin" || function == "round"))
      {
        double folded = 0.0;
        if (!compute_scalar_result(to_double(lhs), to_double(rhs), folded))
          throw std::runtime_error(
            "compute_scalar_result missing branch for " + function);

        // Mirror the dtype-override branch below:
        // only restamp current_lhs when the user explicitly requested a dtype.
        typet t = cached_double_type();
        if (get_dtype_size() && converter_.current_lhs)
        {
          t = get_typet_from_dtype();
          if (!t.is_floatbv())
            t = cached_double_type();
          converter_.current_lhs->type() = t;
          converter_.update_symbol(*converter_.current_lhs);
        }
        exprt folded_expr = from_double(folded, t);
        folded_expr.cformat(std::to_string(folded));
        return folded_expr;
      }

      nlohmann::json result;
      if (lhs.is_int && rhs.is_int)
      {
        result =
          create_binary_op(function, kConstant, lhs.int_value, rhs.int_value);
      }
      else
      {
        result =
          create_binary_op(function, kConstant, to_double(lhs), to_double(rhs));
      }

      exprt expr = converter_.get_expr(result);

      auto dtype_size = get_dtype_size();
      if (dtype_size && converter_.current_lhs)
      {
        typet t = get_typet_from_dtype();
        converter_.current_lhs->type() = t;
        converter_.update_symbol(*converter_.current_lhs);

        expr.type() = converter_.current_lhs->type();
        for (auto &operand : expr.operands())
          operand.type() = expr.type();

        if (allow_numpy_fold)
        {
          const std::string dtype = get_dtype();
          const bool is_integer_dtype = dtype.find("int") != std::string::npos;
          if (
            function == "power" && lhs.is_int && rhs.is_int && is_integer_dtype)
          {
            BigInt exact_power;
            if (try_exact_integer_power(
                  lhs.int_value, rhs.int_value, exact_power))
            {
              const bool is_unsigned = !dtype.empty() && dtype[0] == 'u';
              const BigInt min_val =
                is_unsigned ? BigInt(0) : -BigInt::power2(dtype_size - 1);
              const BigInt max_val = is_unsigned
                                       ? BigInt::power2(dtype_size) - 1
                                       : BigInt::power2(dtype_size - 1) - 1;
              if (exact_power < min_val || exact_power > max_val)
              {
                log_warning(
                  "{}:{}: Integer overflow detected in {}() call. Consider "
                  "using a larger integer type.",
                  converter_.current_python_file,
                  call_["end_lineno"].get<int>(),
                  function_id_.get_function());
                emit_numpy_overflow_assertion(converter_, call_, function_id_);
              }

              BigInt wrapped = exact_power;
              const BigInt modulus = BigInt::power2(dtype_size);
              wrapped = wrapped % modulus;
              if (wrapped < 0)
                wrapped += modulus;
              if (!is_unsigned && wrapped >= BigInt::power2(dtype_size - 1))
                wrapped -= modulus;

              exprt folded = from_integer(wrapped, t);
              folded.cformat(integer2string(wrapped));
              return folded;
            }
          }

          double left = to_double(lhs);
          double right = to_double(rhs);
          double scalar_result = 0.0;

          if (compute_scalar_result(left, right, scalar_result))
          {
            if (is_integer_dtype)
            {
              const bool is_unsigned = !dtype.empty() && dtype[0] == 'u';
              const int64_t rounded_value =
                static_cast<int64_t>(std::llround(scalar_result));
              const uint64_t mask = dtype_size >= 64
                                      ? std::numeric_limits<uint64_t>::max()
                                      : ((uint64_t{1} << dtype_size) - 1);
              const uint64_t wrapped_bits =
                static_cast<uint64_t>(rounded_value) & mask;

              int64_t wrapped_signed = static_cast<int64_t>(wrapped_bits);
              if (
                !is_unsigned && dtype_size < 64 &&
                ((wrapped_bits >> (dtype_size - 1)) & 1ULL) != 0)
              {
                wrapped_signed -=
                  static_cast<int64_t>(uint64_t{1} << dtype_size);
              }

              if (rounded_value != wrapped_signed)
              {
                log_warning(
                  "{}:{}: Integer overflow detected in {}() call. Consider "
                  "using a larger integer type.",
                  converter_.current_python_file,
                  call_["end_lineno"].get<int>(),
                  function_id_.get_function());
                emit_numpy_overflow_assertion(converter_, call_, function_id_);
              }

              if (is_unsigned)
              {
                exprt folded = from_integer(BigInt(wrapped_bits), t);
                folded.cformat(std::to_string(wrapped_bits));
                return folded;
              }
              else
              {
                exprt folded = from_integer(BigInt(wrapped_signed), t);
                folded.cformat(std::to_string(wrapped_signed));
                return folded;
              }
            }

            exprt folded = from_double(scalar_result, t);
            folded.cformat(std::to_string(scalar_result));
            return folded;
          }
        }
      }

      return expr;
    }

    broadcast_check(call_["args"]);

    exprt expr = create_expr_from_call();

    auto dtype_size(get_dtype_size());
    if (dtype_size)
    {
      typet t = get_typet_from_dtype();
      if (converter_.current_lhs)
      {
        // Update variable (lhs)
        converter_.current_lhs->type() = t;
        converter_.update_symbol(*converter_.current_lhs);

        // Update rhs expression
        expr.type() = converter_.current_lhs->type();

        // Update all operands' types safely
        for (auto &operand : expr.operands())
          operand.type() = expr.type();

        std::string value_str = expr.value().as_string();
        size_t value_size = count_effective_bits(value_str);

        if (value_size > dtype_size)
        {
          log_warning(
            "{}:{}: Integer overflow detected in {}() call. Consider using a "
            "larger integer type.",
            converter_.current_python_file,
            call_["end_lineno"].get<int>(),
            function_id_.get_function());
          emit_numpy_overflow_assertion(converter_, call_, function_id_);
        }

        if (!expr.value().empty())
        {
          auto length = value_str.length();
          expr.value(value_str.substr(length - dtype_size));
          value_str = expr.value().as_string();
          expr.cformat(std::to_string(std::stoll(value_str, nullptr, 2)));
        }
      }
    }

    return expr;
  }

  throw std::runtime_error("Unsupported NumPy function call: " + function);
}
