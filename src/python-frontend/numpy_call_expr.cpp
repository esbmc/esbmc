#include <python-frontend/json_utils.h>
#include <python-frontend/numpy_call_expr.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_int_overflow.h>
#include <python-frontend/symbol_id.h>
#include <irep2/irep2_utils.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/config.h>
#include <util/expr.h>
#include <util/expr_util.h>
#include <util/message.h>
#include <util/migrate.h>
#include <util/std_expr.h>
#include <util/std_code.h>
#include <util/std_types.h>

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
         function == "real" || function == "imag" || function == "conj" ||
         function == "conjugate" || function == "angle" || function == "abs";
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
        if (var["value"].contains("args") && !var["value"]["args"].empty())
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

  if (function == "arange")
  {
    if (call_["args"].empty() || call_["args"].size() > 3)
      throw std::runtime_error(
        "TypeError: numpy.arange() expects 1 to 3 arguments");

    std::vector<numeric_value> args;
    args.reserve(call_["args"].size());
    for (auto arg : call_["args"])
    {
      resolve_var(arg);
      numeric_value value;
      if (!try_extract_numeric_constant(arg, value))
        throw std::runtime_error(
          "TypeError: numpy.arange() currently supports constant numeric "
          "inputs only");
      args.push_back(value);
    }

    double start = 0.0;
    double stop = 0.0;
    double step = 1.0;
    if (args.size() == 1)
    {
      stop = to_double(args[0]);
    }
    else
    {
      start = to_double(args[0]);
      stop = to_double(args[1]);
      if (args.size() == 3)
        step = to_double(args[2]);
    }

    if (step == 0.0)
      throw std::runtime_error(
        "ValueError: numpy.arange() step must not be zero");

    const bool any_float =
      std::any_of(args.begin(), args.end(), [](const numeric_value &v) {
        return !v.is_int;
      });

    std::vector<double> float_values;
    std::vector<int64_t> int_values;
    if (step > 0.0)
    {
      for (double current = start; current < stop; current += step)
      {
        if (any_float)
          float_values.push_back(current);
        else
          int_values.push_back(static_cast<int64_t>(std::llround(current)));
      }
    }
    else
    {
      for (double current = start; current > stop; current += step)
      {
        if (any_float)
          float_values.push_back(current);
        else
          int_values.push_back(static_cast<int64_t>(std::llround(current)));
      }
    }
    if (any_float)
      return converter_.get_expr(create_list(float_values));
    return converter_.get_expr(create_list(int_values));
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
  if (call_["args"].size() == 1)
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
      nlohmann::json arg = call_["args"][0];
      unwrap_np_array_arg(arg);

      std::vector<scalar_value> values_1d;
      std::vector<std::vector<scalar_value>> values_2d;

      if (try_extract_scalar_1d_list(arg, values_1d))
      {
        double sum_sq = 0.0;
        for (const auto &v : values_1d)
        {
          if (v.is_complex)
            throw std::runtime_error(
              "TypeError: numpy.linalg.norm does not support complex values");
          sum_sq += v.value.real() * v.value.real();
        }
        return converter_.get_expr(
          to_json_constant(make_real_scalar(std::sqrt(sum_sq))));
      }

      if (try_extract_scalar_2d_list(arg, values_2d))
      {
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
          return make_complex(real, minus_exprt(from_double(0.0, dt), imag));
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
            bool has_float_literal = false;
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
                has_float_literal = has_float_literal || !value.is_int;

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

            if (all_numeric_constants && has_float_literal)
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
      resolve_var(arg);
      const std::string &function = function_id_.get_function();

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
    {
      return function_call_expr::get();
    }

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
        dims.push_back(shape["value"].get<std::size_t>());
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

    if (arg_type == "Constant")
    {
      // np.zeros(n) or np.ones(n) — 1D
      auto list = create_list(shape_arg["value"].get<int>(), fill_value);
      return converter_.get_expr(list);
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
        if (var["value"].contains("args") && !var["value"]["args"].empty())
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

  if (function == "ravel" || function == "flatten")
  {
    if (call_["args"].empty())
      throw std::runtime_error(
        "TypeError: numpy." + function + "() requires an array argument");

    nlohmann::json arr_arg = call_["args"][0];
    resolve_numpy_var(arr_arg);

    std::vector<std::size_t> old_shape;
    if (!get_literal_shape(arr_arg, old_shape))
      throw std::runtime_error(
        "TypeError: numpy." + function +
        "() currently supports only constant arrays");

    std::vector<nlohmann::json> flat;
    flatten_json_list(arr_arg, flat);

    nlohmann::json result;
    result["_type"] = "List";
    result["elts"] = nlohmann::json::array();
    for (const auto &elem : flat)
      result["elts"].push_back(elem);
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
