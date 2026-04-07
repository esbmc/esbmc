#include <python-frontend/complex_handler.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_exception_handler.h>
#include <python-frontend/python_math.h>
#include <python-frontend/type_handler.h>

#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/std_code.h>
#include <util/std_expr.h>

#include <limits>

// -----------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------

complex_handler::complex_handler(
  python_converter &converter,
  contextt &symbol_table,
  type_handler &type_handler_ref)
  : converter_(converter),
    symbol_table_(symbol_table),
    type_handler_(type_handler_ref)
{
}

// -----------------------------------------------------------------------
// Cache helpers
// -----------------------------------------------------------------------

void complex_handler::clear_cache() const
{
  symbol_cache_.clear();
}

const symbolt *complex_handler::find_cached_symbol(const std::string &id) const
{
  auto it = symbol_cache_.find(id);
  if (it != symbol_cache_.end())
    return it->second;
  const symbolt *s = symbol_table_.find_symbol(id);
  symbol_cache_[id] = s;
  return s;
}

// -----------------------------------------------------------------------
// Shared IEEE / complex arithmetic helpers
// -----------------------------------------------------------------------

exprt complex_handler::ieee_binop(
  const irep_idt &id,
  const exprt &x,
  const exprt &y)
{
  exprt out(id, cached_double_type());
  out.copy_to_operands(x, y);
  return out;
}

exprt complex_handler::complex_mul(const exprt &x, const exprt &y) const
{
  const typet &dt = cached_double_type();
  exprt xr = member_exprt(x, "real", dt);
  exprt xi = member_exprt(x, "imag", dt);
  exprt yr = member_exprt(y, "real", dt);
  exprt yi = member_exprt(y, "imag", dt);

  exprt ac = ieee_binop("ieee_mul", xr, yr);
  exprt bd = ieee_binop("ieee_mul", xi, yi);
  exprt ad = ieee_binop("ieee_mul", xr, yi);
  exprt bc = ieee_binop("ieee_mul", xi, yr);

  return make_complex(
    ieee_binop("ieee_sub", ac, bd), ieee_binop("ieee_add", ad, bc));
}

exprt complex_handler::complex_div(
  const exprt &x,
  const exprt &y,
  const nlohmann::json &loc_source) const
{
  const typet &dt = cached_double_type();
  exprt xr = member_exprt(x, "real", dt);
  exprt xi = member_exprt(x, "imag", dt);
  exprt yr = member_exprt(y, "real", dt);
  exprt yi = member_exprt(y, "imag", dt);
  exprt zero = from_double(0.0, dt);

  exprt ac = ieee_binop("ieee_mul", xr, yr);
  exprt bd = ieee_binop("ieee_mul", xi, yi);
  exprt bc = ieee_binop("ieee_mul", xi, yr);
  exprt ad = ieee_binop("ieee_mul", xr, yi);
  exprt c2 = ieee_binop("ieee_mul", yr, yr);
  exprt d2 = ieee_binop("ieee_mul", yi, yi);

  exprt numer_real = ieee_binop("ieee_add", ac, bd);
  exprt numer_imag = ieee_binop("ieee_sub", bc, ad);
  exprt denom = ieee_binop("ieee_add", c2, d2);

  exprt normal_result = make_complex(
    ieee_binop("ieee_div", numer_real, denom),
    ieee_binop("ieee_div", numer_imag, denom));

  // Runtime ZeroDivisionError guard — denom==0 iff yr==0 AND yi==0.
  exprt yr_zero = equality_exprt(yr, zero);
  exprt yi_zero = equality_exprt(yi, zero);
  exprt denom_is_zero = and_exprt(yr_zero, yi_zero);

  exprt raise_zdiv = converter_.get_exception_handler().gen_exception_raise(
    "ZeroDivisionError", "complex division by zero");

  locationt loc = converter_.get_location_from_decl(loc_source);
  raise_zdiv.location() = loc;
  raise_zdiv.location().user_provided(true);

  code_expressiont raise_code(raise_zdiv);
  raise_code.location() = loc;

  code_ifthenelset guard;
  guard.cond() = denom_is_zero;
  guard.then_case() = raise_code;
  guard.location() = loc;

  converter_.current_block->copy_to_operands(guard);
  return normal_result;
}

exprt complex_handler::complex_log(
  const exprt &z,
  const nlohmann::json &loc_source) const
{
  const typet &dt = cached_double_type();
  exprt zr = member_exprt(z, "real", dt);
  exprt zi = member_exprt(z, "imag", dt);

  exprt zr2 = ieee_binop("ieee_mul", zr, zr);
  exprt zi2 = ieee_binop("ieee_mul", zi, zi);
  exprt abs2 = ieee_binop("ieee_add", zr2, zi2);

  exprt abs_z = converter_.get_math_handler().handle_sqrt(abs2, loc_source);
  if (abs_z.statement() == "cpp-throw")
    return abs_z;

  exprt ln_abs = converter_.get_math_handler().handle_log(abs_z, loc_source);
  if (ln_abs.statement() == "cpp-throw")
    return ln_abs;

  exprt arg_z = converter_.get_math_handler().handle_atan2(zi, zr, loc_source);
  if (arg_z.statement() == "cpp-throw")
    return arg_z;

  return make_complex(ln_abs, arg_z);
}

exprt complex_handler::complex_exp(
  const exprt &z,
  const nlohmann::json &loc_source) const
{
  const typet &dt = cached_double_type();
  exprt zr = member_exprt(z, "real", dt);
  exprt zi = member_exprt(z, "imag", dt);

  exprt exp_real = converter_.get_math_handler().handle_exp(zr, loc_source);
  if (exp_real.statement() == "cpp-throw")
    return exp_real;

  exprt cos_imag = converter_.get_math_handler().handle_cos(zi, loc_source);
  if (cos_imag.statement() == "cpp-throw")
    return cos_imag;

  exprt sin_imag = converter_.get_math_handler().handle_sin(zi, loc_source);
  if (sin_imag.statement() == "cpp-throw")
    return sin_imag;

  exprt real = ieee_binop("ieee_mul", exp_real, cos_imag);
  exprt imag = ieee_binop("ieee_mul", exp_real, sin_imag);
  return make_complex(real, imag);
}

// -----------------------------------------------------------------------
// Numeric normalisation helpers
// -----------------------------------------------------------------------

exprt complex_handler::promote_int_arith_to_double(
  const exprt &input_expr,
  std::size_t depth) const
{
  if (depth > 64)
    return typecast_exprt(input_expr, cached_double_type());

  if (input_expr.statement() == "cpp-throw")
    return input_expr;

  const irep_idt id = input_expr.id();
  if (
    (id == "+" || id == "-" || id == "*" || id == "/") &&
    input_expr.operands().size() == 2)
  {
    exprt lhs = promote_int_arith_to_double(input_expr.op0(), depth + 1);
    if (lhs.statement() == "cpp-throw")
      return lhs;
    exprt rhs = promote_int_arith_to_double(input_expr.op1(), depth + 1);
    if (rhs.statement() == "cpp-throw")
      return rhs;

    const typet &dt = cached_double_type();
    exprt op_expr;
    if (id == "+")
      op_expr = exprt("ieee_add", dt);
    else if (id == "-")
      op_expr = exprt("ieee_sub", dt);
    else if (id == "*")
      op_expr = exprt("ieee_mul", dt);
    else
      op_expr = exprt("ieee_div", dt);

    op_expr.copy_to_operands(lhs, rhs);
    return op_expr;
  }

  const typet &dt = cached_double_type();
  if (input_expr.type() == dt)
    return input_expr;

  const typet &expr_type = input_expr.type();
  const bool numeric_like = expr_type.is_floatbv() || expr_type.is_signedbv() ||
                            expr_type.is_unsignedbv() || expr_type.is_bool();
  if (!numeric_like)
    return input_expr;

  return typecast_exprt(input_expr, dt);
}

exprt complex_handler::normalize_numeric_expr(const exprt &value) const
{
  if (value.statement() == "cpp-throw")
    return value;
  if (is_complex_type(value.type()))
    return value;

  if (
    value.type().is_signedbv() || value.type().is_unsignedbv() ||
    value.type().is_bool())
    return promote_int_arith_to_double(value, 0);

  return value;
}

// -----------------------------------------------------------------------
// Public entry points
// -----------------------------------------------------------------------

exprt complex_handler::handle_binary_op(
  const std::string &op,
  const exprt &lhs,
  const exprt &rhs,
  const nlohmann::json &element) const
{
  clear_cache();

  // ---- helper lambdas (lightweight, capturing *this) ----

  auto op_symbol = [](const std::string &op_name) -> std::string {
    if (op_name == "Add")
      return "+";
    if (op_name == "Sub")
      return "-";
    if (op_name == "Mult")
      return "*";
    if (op_name == "Div")
      return "/";
    if (op_name == "Pow")
      return "**";
    return op_name;
  };

  auto expr_python_type_name = [&](const exprt &e) -> std::string {
    const typet &t = e.type();
    if (is_complex_type(t))
      return "complex";
    if (t.is_bool())
      return "bool";
    if (t.is_floatbv())
      return "float";
    if (t.is_signedbv() || t.is_unsignedbv())
      return "int";
    if (t.is_array() && t.subtype() == char_type())
      return "str";
    if (t.is_array())
      return "bytes";
    return type_handler_.type_to_string(t);
  };

  auto raise_complex_type_error = [&](const std::string &msg) -> exprt {
    return converter_.get_exception_handler().gen_exception_raise(
      "TypeError", msg);
  };

  auto is_real_numeric = [](const typet &t) -> bool {
    return t.is_floatbv() || t.is_signedbv() || t.is_unsignedbv() ||
           t.is_bool();
  };

  auto is_compatible_numeric = [&](const exprt &e) -> bool {
    return is_complex_type(e.type()) || is_real_numeric(e.type());
  };

  // ---- reject unsupported operations ----

  if (op == "Lt" || op == "LtE" || op == "Gt" || op == "GtE")
    return raise_complex_type_error(
      "no ordering relation is defined for complex numbers");

  if (op == "FloorDiv" || op == "Mod")
    return raise_complex_type_error(
      "can't take floor or mod of complex number");

  // ---- Pow ----

  if (op == "Pow")
  {
    if (!is_compatible_numeric(lhs) || !is_compatible_numeric(rhs))
      return raise_complex_type_error(
        "unsupported operand type(s) for " + op_symbol(op) + ": '" +
        expr_python_type_name(lhs) + "' and '" + expr_python_type_name(rhs) +
        "'");

    exprt lhs_complex = promote_to_complex(lhs);
    if (lhs_complex.statement() == "cpp-throw")
      return lhs_complex;

    exprt resolved_rhs = rhs;
    if (rhs.is_symbol())
    {
      const symbolt *s = find_cached_symbol(rhs.identifier().as_string());
      if (s && !s->value.is_nil())
        resolved_rhs = s->value;
    }
    else if (
      rhs.id() == "+" || rhs.id() == "-" || rhs.id() == "*" || rhs.id() == "/")
    {
      resolved_rhs = converter_.get_math_handler().compute_expr(rhs);
    }
    if (resolved_rhs.statement() == "cpp-throw")
      return resolved_rhs;

    BigInt exponent_big;
    bool has_integer_exponent = false;

    if (resolved_rhs.is_true())
    {
      exponent_big = BigInt(1);
      has_integer_exponent = true;
    }
    else if (resolved_rhs.is_false())
    {
      exponent_big = BigInt(0);
      has_integer_exponent = true;
    }
    else if (
      resolved_rhs.id() == "unary-" && resolved_rhs.operands().size() == 1)
    {
      BigInt inner;
      if (!to_integer(resolved_rhs.op0(), inner))
      {
        exponent_big = -inner;
        has_integer_exponent = true;
      }
    }
    else if (
      resolved_rhs.id() == "unary+" && resolved_rhs.operands().size() == 1)
    {
      BigInt inner;
      if (!to_integer(resolved_rhs.op0(), inner))
      {
        exponent_big = inner;
        has_integer_exponent = true;
      }
    }
    else if (!to_integer(resolved_rhs, exponent_big))
    {
      has_integer_exponent = true;
    }

    if (!has_integer_exponent)
    {
      exprt rhs_complex = promote_to_complex(resolved_rhs);
      if (rhs_complex.statement() == "cpp-throw")
        return rhs_complex;

      exprt lhs_log = complex_log(lhs_complex, element);
      if (lhs_log.statement() == "cpp-throw")
        return lhs_log;

      exprt product = complex_mul(rhs_complex, lhs_log);
      return complex_exp(product, element);
    }

    static const BigInt min_long = BigInt(std::numeric_limits<long>::min());
    static const BigInt max_long = BigInt(std::numeric_limits<long>::max());
    if (exponent_big < min_long || exponent_big > max_long)
    {
      return raise_complex_type_error(
        "complex exponent out of supported integer range");
    }

    auto pow_nonnegative = [&](unsigned long long exponent_abs) -> exprt {
      const typet &dt = cached_double_type();
      exprt acc = make_complex(from_double(1.0, dt), from_double(0.0, dt));
      exprt base = lhs_complex;

      while (exponent_abs > 0)
      {
        if ((exponent_abs & 1ULL) != 0ULL)
          acc = complex_mul(acc, base);

        exponent_abs >>= 1U;
        if (exponent_abs > 0)
          base = complex_mul(base, base);
      }

      return acc;
    };

    const long exponent = exponent_big.to_int64();
    if (exponent >= 0)
      return pow_nonnegative(static_cast<unsigned long long>(exponent));

    if (exponent == std::numeric_limits<long>::min())
    {
      return raise_complex_type_error(
        "complex exponent out of supported integer range");
    }

    const unsigned long long exponent_abs =
      static_cast<unsigned long long>(-exponent);
    exprt positive_power = pow_nonnegative(exponent_abs);
    const typet &dt = cached_double_type();
    exprt one = make_complex(from_double(1.0, dt), from_double(0.0, dt));
    return complex_div(one, positive_power, element);
  }

  // ---- Add / Sub / Mult / Div / Eq / NotEq ----

  if (
    op == "Add" || op == "Sub" || op == "Mult" || op == "Div" || op == "Eq" ||
    op == "NotEq")
  {
    if (op == "Eq" || op == "NotEq")
    {
      if (!is_compatible_numeric(lhs) || !is_compatible_numeric(rhs))
        return gen_boolean(op == "NotEq");
    }
    else
    {
      if (!is_compatible_numeric(lhs) || !is_compatible_numeric(rhs))
        return raise_complex_type_error(
          "unsupported operand type(s) for " + op_symbol(op) + ": '" +
          expr_python_type_name(lhs) + "' and '" + expr_python_type_name(rhs) +
          "'");
    }

    exprt lhs_complex = promote_to_complex(lhs);
    exprt rhs_complex = promote_to_complex(rhs);
    if (lhs_complex.statement() == "cpp-throw")
      return lhs_complex;
    if (rhs_complex.statement() == "cpp-throw")
      return rhs_complex;

    const typet &dt = cached_double_type();
    const exprt a = member_exprt(lhs_complex, "real", dt);
    const exprt b = member_exprt(lhs_complex, "imag", dt);
    const exprt c = member_exprt(rhs_complex, "real", dt);
    const exprt d = member_exprt(rhs_complex, "imag", dt);

    if (op == "Eq")
      return and_exprt(equality_exprt(a, c), equality_exprt(b, d));
    if (op == "NotEq")
      return or_exprt(
        not_exprt(equality_exprt(a, c)), not_exprt(equality_exprt(b, d)));

    if (op == "Add")
      return make_complex(
        ieee_binop("ieee_add", a, c), ieee_binop("ieee_add", b, d));

    if (op == "Sub")
      return make_complex(
        ieee_binop("ieee_sub", a, c), ieee_binop("ieee_sub", b, d));

    if (op == "Mult")
    {
      exprt ac = ieee_binop("ieee_mul", a, c);
      exprt bd = ieee_binop("ieee_mul", b, d);
      exprt ad = ieee_binop("ieee_mul", a, d);
      exprt bc = ieee_binop("ieee_mul", b, c);
      return make_complex(
        ieee_binop("ieee_sub", ac, bd), ieee_binop("ieee_add", ad, bc));
    }

    // Div — delegate to complex_div which includes ZeroDivisionError guard.
    return complex_div(lhs_complex, rhs_complex, element);
  }

  return raise_complex_type_error("unsupported operation for complex operands");
}

// -----------------------------------------------------------------------

exprt complex_handler::handle_unary_op(
  const std::string &op,
  const exprt &operand) const
{
  if (op == "UAdd")
    return operand;

  assert(op == "USub" && "handle_unary_op: unexpected operator");

  // USub: negate both components.
  const typet &dt = cached_double_type();
  exprt real = member_exprt(operand, "real", dt);
  exprt imag = member_exprt(operand, "imag", dt);
  exprt zero = from_double(0.0, dt);

  exprt neg_real("ieee_sub", dt);
  neg_real.copy_to_operands(zero, real);
  exprt neg_imag("ieee_sub", dt);
  neg_imag.copy_to_operands(zero, imag);

  return make_complex(neg_real, neg_imag);
}

// -----------------------------------------------------------------------

exprt complex_handler::handle_attribute(const nlohmann::json &element) const
{
  if (!(element.contains("func") && element["func"].contains("attr")))
    return nil_exprt();

  const std::string &method_name = element["func"]["attr"].get<std::string>();

  if (method_name != "conjugate")
    return nil_exprt();

  const auto &args =
    element.contains("args") ? element["args"] : nlohmann::json::array();
  const auto &keywords = element.contains("keywords") ? element["keywords"]
                                                      : nlohmann::json::array();

  if (!args.empty() || !keywords.empty())
    return nil_exprt();

  exprt obj_expr = converter_.get_expr(element["func"]["value"]);
  if (obj_expr.statement() == "cpp-throw")
    return obj_expr;

  if (!is_complex_type(obj_expr.type()))
    return nil_exprt();

  const typet &dt = cached_double_type();
  exprt real = member_exprt(obj_expr, "real", dt);
  exprt imag = member_exprt(obj_expr, "imag", dt);
  exprt zero = from_double(0.0, dt);

  exprt neg_imag("ieee_sub", dt);
  neg_imag.copy_to_operands(zero, imag);
  return make_complex(real, neg_imag);
}

// -----------------------------------------------------------------------

exprt complex_handler::handle_attribute_access(
  const exprt &obj,
  const std::string &attr) const
{
  if (attr == "real")
    return member_exprt(obj, "real", cached_double_type());

  if (attr == "imag")
    return member_exprt(obj, "imag", cached_double_type());

  return nil_exprt();
}

// -----------------------------------------------------------------------

exprt complex_handler::handle_abs(const exprt &z) const
{
  const typet &dt = cached_double_type();
  exprt real = member_exprt(z, "real", dt);
  exprt imag = member_exprt(z, "imag", dt);

  exprt real_sq("ieee_mul", dt);
  real_sq.copy_to_operands(real, real);
  exprt imag_sq("ieee_mul", dt);
  imag_sq.copy_to_operands(imag, imag);

  exprt sum("ieee_add", dt);
  sum.copy_to_operands(real_sq, imag_sq);

  exprt sqrt_expr("ieee_sqrt", dt);
  sqrt_expr.copy_to_operands(sum);
  sqrt_expr.add("rounding_mode") =
    symbol_exprt("c:@__ESBMC_rounding_mode", int_type());
  return sqrt_expr;
}

// -----------------------------------------------------------------------

exprt complex_handler::handle_cmath_log(
  const std::string &func_name,
  const nlohmann::json &call,
  const nlohmann::json &args,
  const nlohmann::json &keywords) const
{
  clear_cache();

  auto raise_type_error = [&](const std::string &msg) -> exprt {
    return converter_.get_exception_handler().gen_exception_raise(
      "TypeError", msg);
  };

  auto as_complex_or_throw = [&](const nlohmann::json &node) -> exprt {
    exprt value = converter_.get_expr(node);
    if (value.statement() == "cpp-throw")
      return value;
    return promote_to_complex(value);
  };

  if (func_name == "log10")
  {
    if (!keywords.empty())
      return raise_type_error("cmath.log10() takes no keyword arguments");
    if (args.size() != 1)
      return raise_type_error("log10() takes exactly 1 argument");

    exprt z = as_complex_or_throw(args[0]);
    if (z.statement() == "cpp-throw")
      return z;
    exprt ln_z = complex_log(z, call);
    if (ln_z.statement() == "cpp-throw")
      return ln_z;

    const typet &dt = cached_double_type();
    exprt ln10 =
      make_complex(from_double(2.302585092994046, dt), from_double(0.0, dt));
    return complex_div(ln_z, ln10, call);
  }

  // func_name == "log"
  if (args.empty() || args.size() > 2)
    return raise_type_error("log() takes from 1 to 2 positional arguments");
  if (!keywords.empty())
    return raise_type_error("cmath.log() takes no keyword arguments");

  const nlohmann::json *base_json = nullptr;
  if (args.size() == 2)
    base_json = &args[1];

  exprt z = as_complex_or_throw(args[0]);
  if (z.statement() == "cpp-throw")
    return z;
  exprt ln_z = complex_log(z, call);
  if (ln_z.statement() == "cpp-throw")
    return ln_z;

  if (base_json == nullptr)
    return ln_z;

  exprt base = as_complex_or_throw(*base_json);
  if (base.statement() == "cpp-throw")
    return base;
  exprt ln_base = complex_log(base, call);
  if (ln_base.statement() == "cpp-throw")
    return ln_base;

  return complex_div(ln_z, ln_base, call);
}
