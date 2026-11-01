#include <python-frontend/math/bounded_math.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/exception/python_exception_handler.h>
#include <irep2/irep2_utils.h>
#include <util/arith/arith_tools.h>
#include <util/arith/transcendental.h>
#include <util/irep/migrate.h>
#include <util/lang/c_types.h>
#include <util/irep/std_code.h>
#include <util/irep/std_expr.h>

#include <cmath>
#include <set>

namespace
{
const double inf = INFINITY;

exprt between(const exprt &value, double low, double high)
{
  return and_exprt(
    binary_relation_exprt(value, ">=", from_double(low, double_type())),
    binary_relation_exprt(value, "<=", from_double(high, double_type())));
}

// An exact value at a signed zero, where the bounds cannot keep the sign.
exprt at_zero(const std::string &function, const exprt &x)
{
  static const std::set<std::string> one = {"exp", "cos", "cosh"};
  static const std::set<std::string> pole = {"log", "log2", "log10"};
  if (one.count(function) != 0)
    return from_double(1.0, double_type());
  if (pole.count(function) != 0)
    return from_double(-inf, double_type());
  if (transcendental::is_odd(function))
    return x;
  return nil_exprt();
}

exprt is_nan(const exprt &x)
{
  expr2tc x2;
  migrate_expr(x, x2);
  return migrate_expr_back(isnan2tc(x2));
}

exprt is_infinite(const exprt &x)
{
  expr2tc x2;
  migrate_expr(x, x2);
  return migrate_expr_back(isinf2tc(x2));
}

// Where math.* raises "math domain error": outside the domain, at the pole
// of a logarithm, or at an infinity of a periodic function.
exprt domain_error(const std::string &function, const exprt &x)
{
  const transcendental::domain domain = transcendental::domain_of(function);
  exprt::operandst outside = {not_exprt(between(x, domain.low, domain.high))};
  if (domain.low == 0.0)
    outside.push_back(equality_exprt(x, from_double(0.0, double_type())));
  if (domain.periodic)
    outside.push_back(is_infinite(x));
  return and_exprt(not_exprt(is_nan(x)), or_exprt(outside));
}
} // namespace

bounded_math::bounded_math(
  python_converter &converter,
  const nlohmann::json &call)
  : converter_(converter), call_(call)
{
}

bool bounded_math::is_bounded(const std::string &function)
{
  return transcendental::is_bounded(function);
}

bool bounded_math::has_runtime_model(const std::string &function)
{
  return is_bounded(function) || function == "sqrt";
}

double bounded_math::host_value(const std::string &function, double x)
{
  return transcendental::host_value(function, x);
}

exprt bounded_math::literal(const std::string &function, double x)
{
  const transcendental::interval result = transcendental::at(function, x);
  if (result.low == result.high || std::isnan(result.low))
    return from_double(result.low, double_type());
  return within(result.low, result.high);
}

exprt bounded_math::runtime(
  const std::string &function,
  const exprt &x,
  math_module module)
{
  const exprt argument = read_once(x);
  if (function == "sqrt")
    return converter_.get_math_handler().handle_sqrt(argument, call_);
  const exprt value =
    special_cases(function, argument, bounded_call(function, argument, module));
  if (module == math_module::math)
    raise_math_errors(function, argument, value);
  return value;
}

// Symex gives a constant argument the module's literal result (the C
// library's exact value for math.*) and otherwise the bounds of its piece.
exprt bounded_math::bounded_call(
  const std::string &function,
  const exprt &x,
  math_module module)
{
  const std::string name = "__builtin_bounded_" + function;
  const std::string id = "c:@F@" + name;
  code_typet type;
  type.return_type() = double_type();
  type.arguments().push_back(code_typet::argumentt(double_type()));
  type.arguments().push_back(code_typet::argumentt(bool_type()));
  if (!converter_.symbol_table_.find_symbol(id))
    converter_.add_symbol(converter_.create_symbol(
      converter_.python_file(),
      name,
      id,
      converter_.get_location_from_decl(call_),
      type));
  code_function_callt call;
  call.function() = symbol_exprt(id, type);
  call.arguments().push_back(x);
  call.arguments().push_back(
    module == math_module::math ? exprt(true_exprt()) : false_exprt());
  call.type() = double_type();
  call.location() = converter_.get_location_from_decl(call_);
  return fresh("$math$", call);
}

// NaN propagates, an argument outside the domain gives NaN, and a signed
// zero or an infinity has an exact result.
exprt bounded_math::special_cases(
  const std::string &function,
  const exprt &x,
  exprt result)
{
  const transcendental::domain domain = transcendental::domain_of(function);
  const exprt zero = at_zero(function, x);
  if (!zero.is_nil())
    result = if_exprt(
      equality_exprt(x, from_double(0.0, double_type())), zero, result);
  const exprt positive =
    binary_relation_exprt(x, ">", from_double(0.0, double_type()));
  result = if_exprt(
    is_infinite(x),
    if_exprt(positive, literal(function, inf), literal(function, -inf)),
    result);
  result = if_exprt(
    not_exprt(between(x, domain.low, domain.high)),
    from_double(NAN, double_type()),
    result);
  return if_exprt(is_nan(x), x, result);
}

exprt bounded_math::fresh(const std::string &name, const exprt &value)
{
  const typet type = value.is_nil() ? double_type() : value.type();
  symbolt &tmp = converter_.create_tmp_symbol(call_, name, type, exprt());
  const exprt symbol = symbol_expr(tmp);
  const locationt location = converter_.get_location_from_decl(call_);
  code_declt decl(symbol);
  decl.location() = location;
  converter_.add_instruction(decl);
  if (value.id() == "code" && value.get("statement") == "function_call")
  {
    code_function_callt call = to_code_function_call(to_code(value));
    call.lhs() = symbol;
    converter_.add_instruction(call);
    return symbol;
  }
  code_assignt assign(
    symbol, value.is_nil() ? side_effect_expr_nondett(type) : value);
  assign.location() = location;
  converter_.add_instruction(assign);
  return symbol;
}

exprt bounded_math::stored(const exprt &value)
{
  return value.is_symbol() || value.is_constant() ? value
                                                  : fresh("$math_arg$", value);
}

exprt bounded_math::read_once(const exprt &x)
{
  const exprt value = stored(x);
  if (value.type() == double_type())
    return value;
  return typecast_exprt(value, double_type());
}

exprt bounded_math::within(double low, double high)
{
  const exprt result = fresh("$math$", nil_exprt());
  assume(between(result, low, high));
  return result;
}

void bounded_math::raise_math_errors(
  const std::string &function,
  const exprt &x,
  const exprt &value)
{
  raise_if(domain_error(function, x), "ValueError", "math domain error");
  raise_overflow(x, value);
}

void bounded_math::raise_overflow(const exprt &x, const exprt &value)
{
  raise_if(
    and_exprt(
      not_exprt(or_exprt(is_nan(x), is_infinite(x))), is_infinite(value)),
    "OverflowError",
    "math range error");
}

exprt bounded_math::raise_on_overflow(const exprt &x, const exprt &value)
{
  const exprt result = fresh("$math$", value);
  raise_overflow(x, result);
  return result;
}

void bounded_math::raise_if(
  const exprt &condition,
  const std::string &exception,
  const std::string &message)
{
  const locationt location = converter_.get_location_from_decl(call_);
  exprt raise =
    converter_.get_exception_handler().gen_exception_raise(exception, message);
  raise.location() = location;
  raise.location().user_provided(true);
  code_expressiont raise_code(raise);
  raise_code.location() = location;
  code_ifthenelset guard;
  guard.cond() = condition;
  guard.then_case() = raise_code;
  guard.location() = location;
  guard.location().property("skipped");
  converter_.add_instruction(guard);
}

void bounded_math::assume(const exprt &condition)
{
  code_assumet assumption(condition);
  assumption.location() = converter_.get_location_from_decl(call_);
  converter_.add_instruction(assumption);
}
