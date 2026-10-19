#include <python-frontend/numpy/numpy_random.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/numpy/numpy_call_expr.h>
#include <irep2/irep2_utils.h>
#include <util/arith/arith_tools.h>
#include <util/expr/expr_util.h>
#include <util/irep/migrate.h>
#include <util/irep/std_expr.h>
#include <util/lang/c_types.h>

#include <algorithm>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

namespace
{
// Parameters in NumPy's declaration order. "size" and "dtype" are read from
// the source; every other parameter is a scalar value.
const std::map<std::string, std::vector<std::string>> &signatures()
{
  static const std::map<std::string, std::vector<std::string>> table = {
    {"random", {"size"}},
    {"standard_normal", {"size"}},
    {"normal", {"loc", "scale", "size"}},
    {"uniform", {"low", "high", "size"}},
    {"exponential", {"scale", "size"}},
    {"poisson", {"lam", "size"}},
    {"binomial", {"n", "p", "size"}},
    {"geometric", {"p", "size"}},
    {"lognormal", {"mean", "sigma", "size"}},
    {"beta", {"a", "b", "size"}},
    {"gamma", {"shape", "scale", "size"}},
    {"weibull", {"a", "size"}},
    {"chisquare", {"df", "size"}},
    {"randint", {"low", "high", "size", "dtype"}},
    {"choice", {"a", "size", "replace", "p"}}};
  return table;
}

const std::map<std::string, std::map<std::string, double>> &defaults()
{
  static const std::map<std::string, std::map<std::string, double>> table = {
    {"normal", {{"loc", 0.0}, {"scale", 1.0}}},
    {"uniform", {{"low", 0.0}, {"high", 1.0}}},
    {"exponential", {{"scale", 1.0}}},
    {"poisson", {{"lam", 1.0}}},
    {"lognormal", {{"mean", 0.0}, {"sigma", 1.0}}},
    {"gamma", {{"scale", 1.0}}}};
  return table;
}

bool is_integer_param(const std::string &name, const std::string &param)
{
  return (name == "binomial" && param == "n") ||
         (name == "randint" && (param == "low" || param == "high"));
}

bool contains_call(const nlohmann::json &node)
{
  if (node.is_object() && node.value("_type", "") == "Call")
    return true;
  if (!node.is_object() && !node.is_array())
    return false;
  for (const auto &child : node)
    if (contains_call(child))
      return true;
  return false;
}

bool is_bool_literal(const nlohmann::json &node)
{
  return node.value("_type", "") == "Constant" && node["value"].is_boolean();
}

bool is_integral_result(const std::string &name)
{
  return name == "poisson" || name == "binomial" || name == "geometric" ||
         name == "randint";
}

/// The distribution an alias names: legacy spellings of the same sampler.
std::string canonical_name(const std::string &name)
{
  if (name == "random_sample" || name == "ranf" || name == "sample")
    return "random";
  if (name == "rand")
    return "random";
  if (name == "randn")
    return "standard_normal";
  return name;
}

bool is_none(const nlohmann::json &node)
{
  return node.value("_type", "") == "Constant" && node.contains("value") &&
         node["value"].is_null();
}

/// An integer literal, negated or not; nullopt for anything else.
std::optional<long long> literal_dimension(const nlohmann::json &node)
{
  const bool negated = node.value("_type", "") == "UnaryOp" &&
                       node["op"].value("_type", "") == "USub";
  const nlohmann::json &operand = negated ? node["operand"] : node;
  if (
    operand.value("_type", "") != "Constant" ||
    !operand["value"].is_number_integer())
    return std::nullopt;
  const long long value = operand["value"].get<long long>();
  return negated ? -value : value;
}

exprt real(double value)
{
  return from_double(value, double_type());
}

exprt finite(const exprt &value)
{
  expr2tc v;
  migrate_expr(value, v);
  return migrate_expr_back(not2tc(or2tc(isinf2tc(v), isnan2tc(v))));
}

exprt implies(const exprt &premise, const exprt &conclusion)
{
  return or_exprt(not_exprt(premise), conclusion);
}

exprt arith(const exprt &lhs, const irep_idt &op, const exprt &rhs)
{
  return binary_exprt(lhs, op, rhs, double_type());
}

exprt rel(const exprt &lhs, const irep_idt &op, const exprt &rhs)
{
  return binary_relation_exprt(lhs, op, rhs);
}
} // namespace

numpy_random::numpy_random(
  python_converter &converter,
  const nlohmann::json &call)
  : converter_(converter),
    call_(call),
    function_(call["func"]["attr"].get<std::string>()),
    name_(canonical_name(function_)),
    location_(converter.get_location_from_decl(call))
{
}

bool numpy_random::is_random_call(
  const nlohmann::json &call,
  const nlohmann::json &ast)
{
  const auto &func = call["func"];
  if (
    func.value("_type", "") != "Attribute" ||
    func["value"].value("_type", "") != "Attribute" ||
    func["value"].value("attr", "") != "random")
    return false;
  const auto &module = func["value"]["value"];
  if (
    module.value("_type", "") != "Name" ||
    !is_imported_numpy_module_alias(ast, module["id"]))
    return false;
  return signatures().count(canonical_name(func["attr"].get<std::string>()));
}

exprt numpy_random::get()
{
  code_blockt *const saved_block = converter_.current_block;
  converter_.current_block = &block_;
  bind_arguments();
  validate();
  check_shape();
  exprt result = sized_ ? sample_array() : sample();
  if (!result.is_symbol())
  {
    const exprt sampled = fresh(result.type(), result);
    result = sampled;
  }
  converter_.current_block = saved_block;
  return finish(result);
}

void numpy_random::bind_arguments()
{
  bind_positional();
  bind_keywords();
  if (error_.empty())
    error_ = missing_argument();
  if (!error_.empty())
    raise_if(true_exprt(), "TypeError", error_);
}

void numpy_random::bind(const std::string &param, const nlohmann::json &arg)
{
  if (param == "dtype")
    dtype_ = extract_numpy_dtype_name(arg, converter_.ast());
  else if (param == "a" && name_ == "choice")
    population_ = read_population(arg);
  else if (param == "replace")
    replace_ = read_replace(arg);
  else if (param == "p" && name_ == "choice")
    has_probabilities_ = read_probabilities(arg);
  else if (param == "size")
    read_shape(arg);
  else
    values_[param] = evaluate_once(arg, param);
  bound_.insert(param);
}

void numpy_random::reject_unpacked() const
{
  throw std::runtime_error(
    "TypeError: numpy.random." + function_ +
    "() with unpacked arguments is not supported");
}

// An argument that binds to nothing is still evaluated, as CPython does
// before raising the TypeError.
void numpy_random::bind_positional()
{
  if (takes_dimensions())
    return bind_dimensions();
  const std::vector<std::string> &params = signatures().at(name_);
  const auto &args = call_["args"];
  for (std::size_t i = 0; i < args.size(); ++i)
  {
    if (args[i].value("_type", "") == "Starred")
      reject_unpacked();
    if (i < params.size())
    {
      bind(params[i], args[i]);
      continue;
    }
    evaluate_once(args[i], "");
    if (error_.empty())
      error_ = function_ + "() takes at most " + std::to_string(params.size()) +
               " positional arguments (" + std::to_string(args.size()) +
               " given)";
  }
}

void numpy_random::bind_keywords()
{
  if (!call_.contains("keywords"))
    return;
  static const std::vector<std::string> none;
  const std::vector<std::string> &params =
    takes_dimensions() ? none : signatures().at(name_);
  for (const auto &kw : call_["keywords"])
  {
    if (!kw["arg"].is_string())
      reject_unpacked();
    const std::string param = kw["arg"].get<std::string>();
    const bool known =
      std::find(params.begin(), params.end(), param) != params.end();
    if (known && !bound_.count(param))
    {
      bind(param, kw["value"]);
      continue;
    }
    evaluate_once(kw["value"], "");
    if (error_.empty())
      error_ =
        known
          ? function_ + "() got multiple values for argument '" + param + "'"
          : function_ + "() got an unexpected keyword argument '" + param + "'";
  }
}

// The population is copied into a numpy array once, as NumPy does, so a
// list, a tuple, an array and a view are read alike.
exprt numpy_random::read_population(const nlohmann::json &arg)
{
  const std::string kind = arg.value("_type", "");
  const bool literal = kind == "List" || kind == "Tuple";
  if (!literal && !holds_numpy_array(arg))
    reject_population();
  if (literal && contains_call(arg))
    return materialize_population(arg["elts"]);
  nlohmann::json elements = arg;
  if (kind == "Tuple")
    elements["_type"] = "List";
  nlohmann::json copy = call_;
  copy["func"]["value"] = call_["func"]["value"]["value"];
  copy["func"]["attr"] = "array";
  copy["args"] = nlohmann::json::array({elements});
  copy["keywords"] = nlohmann::json::array();
  const exprt population = converter_.get_expr(copy);
  if (
    !population.type().is_array() ||
    !to_array_type(population.type()).size().is_constant())
    reject_population();
  return population;
}

// Each element is read once, in order, before any sample is drawn.
exprt numpy_random::materialize_population(const nlohmann::json &elements)
{
  std::vector<exprt> values;
  bool real = false;
  for (const auto &element : elements)
  {
    exprt value = evaluate_once(element, "");
    const typet &type = value.type();
    if (
      !type.is_signedbv() && !type.is_unsignedbv() && !type.is_floatbv() &&
      !type.is_bool())
      reject_population();
    real = real || type.is_floatbv();
    values.push_back(value.is_constant() ? value : fresh(type, value));
  }
  const typet element = real ? double_type() : long_long_int_type();
  const exprt population =
    fresh(converter_.get_type_handler().build_array(element, values.size()));
  for (std::size_t i = 0; i < values.size(); ++i)
  {
    code_assignt assign(
      index_exprt(population, from_integer(i, index_type()), element),
      typecast_exprt(values[i], element));
    assign.location() = location_;
    converter_.add_instruction(assign);
  }
  return population;
}

[[noreturn]] void numpy_random::reject_population() const
{
  throw std::runtime_error(
    "TypeError: numpy.random.choice() population must be a literal "
    "sequence or a numpy array");
}

bool numpy_random::holds_numpy_array(const nlohmann::json &arg) const
{
  if (arg.value("_type", "") == "Subscript")
    return holds_numpy_array(arg["value"]);
  if (arg.value("_type", "") != "Name")
    return false;
  const std::string id = converter_.resolve_name_symbol_id(arg["id"]);
  return converter_.numpy_array_symbols_.count(id) != 0 ||
         converter_.numpy_pointer_view_info_.count(id) != 0;
}

bool numpy_random::read_replace(const nlohmann::json &arg) const
{
  if (!is_bool_literal(arg))
    throw std::runtime_error(
      "TypeError: numpy.random.choice() replace must be a boolean literal");
  return arg["value"].get<bool>();
}

// p= is evaluated like any argument; its value is never read.
bool numpy_random::read_probabilities(const nlohmann::json &arg)
{
  if (is_none(arg))
    return false;
  evaluate_once(arg, "");
  return true;
}

bool numpy_random::takes_dimensions() const
{
  return function_ == "rand" || function_ == "randn";
}

// rand(d0, d1, ...) and randn(d0, d1, ...) spell the shape as arguments.
void numpy_random::bind_dimensions()
{
  const auto &args = call_["args"];
  if (args.empty())
    return;
  for (const auto &arg : args)
  {
    std::optional<long long> dimension = literal_dimension(arg);
    if (!dimension)
      reject_shape();
    shape_.push_back(*dimension);
  }
  sized_ = true;
}

void numpy_random::read_shape(const nlohmann::json &size)
{
  if (is_none(size))
    return;
  const bool tuple = size.value("_type", "") == "Tuple";
  for (const auto &dim : tuple ? size["elts"] : nlohmann::json::array({size}))
  {
    std::optional<long long> dimension = literal_dimension(dim);
    if (!dimension)
      reject_shape();
    shape_.push_back(*dimension);
  }
  sized_ = true;
}

[[noreturn]] void numpy_random::reject_shape() const
{
  throw std::runtime_error(
    "TypeError: numpy.random." + function_ +
    "() size must be an integer literal or a tuple of them");
}

// The output is unrolled one sample per element, so its size is capped.
void numpy_random::check_shape()
{
  long long samples = 1;
  for (long long dimension : shape_)
  {
    if (dimension < 0)
      return raise_if(
        true_exprt(), "ValueError", "negative dimensions are not allowed");
    samples *= dimension;
    if (samples > max_samples)
      throw std::runtime_error(
        "TypeError: numpy.random." + function_ + "() with more than " +
        std::to_string(max_samples) + " samples is not supported");
  }
}

exprt numpy_random::sample_array()
{
  element_type_ = name_ == "choice"           ? population_.type().subtype()
                  : is_integral_result(name_) ? long_long_int_type()
                                              : double_type();
  typet type = element_type_;
  for (auto it = shape_.rbegin(); it != shape_.rend(); ++it)
    type = converter_.get_type_handler().build_array(
      type, static_cast<std::size_t>(std::max(*it, 0LL)));
  symbolt &array =
    converter_.create_tmp_symbol(call_, "$np_random_array$", type, exprt());
  const exprt result = symbol_expr(array);
  code_declt decl(result);
  decl.location() = location_;
  converter_.add_instruction(decl);
  fill(result);
  return result;
}

// Each element is its own sample.
void numpy_random::fill(const exprt &target)
{
  if (target.type() == element_type_)
  {
    code_assignt assign(target, sample());
    assign.location() = location_;
    converter_.add_instruction(assign);
    return;
  }
  const array_typet &type = to_array_type(target.type());
  const long long size =
    binary2integer(type.size().value().c_str(), false).to_int64();
  for (long long i = 0; i < size; ++i)
    fill(index_exprt(target, from_integer(i, index_type()), type.subtype()));
}

std::string numpy_random::missing_argument() const
{
  for (const std::string &param : signatures().at(name_))
  {
    const bool has_default =
      param == "size" || param == "dtype" || param == "replace" ||
      param == "p" || (name_ == "randint" && param == "high") ||
      (defaults().count(name_) && defaults().at(name_).count(param));
    if (!bound_.count(param) && !has_default)
      return function_ + "() missing required argument '" + param + "'";
  }
  return {};
}

exprt numpy_random::evaluate_once(
  const nlohmann::json &arg,
  const std::string &param)
{
  exprt value = converter_.store_call_result(
    converter_.get_expr(arg), location_, "np_random_arg");
  if (param.empty())
    return value;

  const typet &type = value.type();
  const bool integral = type.is_signedbv() || type.is_unsignedbv();
  if (is_integer_param(name_, param))
  {
    if (!integral)
      throw std::runtime_error(
        "TypeError: numpy.random." + function_ + "() argument '" + param +
        "' must be an integer");
  }
  else if (!integral && !type.is_floatbv() && !type.is_bool())
    throw std::runtime_error(
      "TypeError: numpy.random." + function_ + "() argument '" + param +
      "' must be a real number");
  else if (type != double_type())
    value = typecast_exprt(value, double_type());

  if (value.is_constant())
    return value;
  const typet stored =
    is_integer_param(name_, param) ? long_long_int_type() : double_type();
  symbolt &tmp =
    converter_.create_tmp_symbol(call_, "$np_random_arg$", stored, exprt());
  code_declt decl(symbol_expr(tmp));
  decl.location() = location_;
  converter_.add_instruction(decl);
  code_assignt assign(
    symbol_expr(tmp),
    value.type() == stored ? value : typecast_exprt(value, stored));
  assign.location() = location_;
  converter_.add_instruction(assign);
  return symbol_expr(tmp);
}

exprt numpy_random::value(const std::string &param) const
{
  auto it = values_.find(param);
  if (it != values_.end())
  {
    if (
      is_integer_param(name_, param) &&
      it->second.type() != long_long_int_type())
      return typecast_exprt(it->second, long_long_int_type());
    return it->second;
  }
  return real(defaults().at(name_).at(param));
}

// NumPy checks the parameters before drawing, so a call that draws nothing
// still raises.
void numpy_random::validate()
{
  if (name_ == "randint")
    return validate_randint();
  if (name_ == "choice")
    return validate_choice();
  const exprt zero = real(0.0);
  auto below = [&](const std::string &param, const exprt &bound) {
    // `!(x >= bound)` also rejects NaN.
    return not_exprt(rel(value(param), ">=", bound));
  };
  auto not_above = [&](const std::string &param, const exprt &bound) {
    return not_exprt(rel(value(param), ">", bound));
  };

  if (name_ == "normal" || name_ == "exponential")
    raise_if(below("scale", zero), "ValueError", "scale < 0");
  else if (name_ == "lognormal")
    raise_if(below("sigma", zero), "ValueError", "sigma < 0");
  else if (name_ == "weibull")
    raise_if(below("a", zero), "ValueError", "a < 0");
  else if (name_ == "chisquare")
    raise_if(not_above("df", zero), "ValueError", "df <= 0");
  else if (name_ == "poisson")
  {
    raise_if(below("lam", zero), "ValueError", "lam < 0");
    raise_if(
      rel(value("lam"), ">", real(9.223372006484771e18)),
      "ValueError",
      "lam value too large");
  }
  else if (name_ == "gamma")
  {
    raise_if(below("shape", zero), "ValueError", "shape < 0");
    raise_if(below("scale", zero), "ValueError", "scale < 0");
  }
  else if (name_ == "beta")
  {
    raise_if(not_above("a", zero), "ValueError", "a <= 0");
    raise_if(not_above("b", zero), "ValueError", "b <= 0");
  }
  else if (name_ == "binomial")
  {
    raise_if(
      rel(value("n"), "<", gen_zero(long_long_int_type())),
      "ValueError",
      "n < 0");
    raise_if(below("p", zero), "ValueError", "p < 0");
    raise_if(rel(value("p"), ">", real(1.0)), "ValueError", "p > 1");
  }
  else if (name_ == "geometric")
  {
    raise_if(not_above("p", zero), "ValueError", "p <= 0");
    raise_if(rel(value("p"), ">", real(1.0)), "ValueError", "p > 1");
  }
  else if (name_ == "uniform")
    raise_if(
      not_exprt(finite(arith(value("high"), exprt::minus, value("low")))),
      "OverflowError",
      "Range exceeds valid bounds");
}

void numpy_random::validate_randint()
{
  const exprt zero = gen_zero(long_long_int_type());
  if (!values_.count("high"))
    raise_if(rel(value("low"), "<=", zero), "ValueError", "high <= 0");
  else
    raise_if(
      rel(value("low"), ">=", value("high")), "ValueError", "low >= high");

  if (dtype_.empty() || dtype_ == "int64")
    return;
  if (dtype_.find("float") == 0 || dtype_.find("complex") == 0)
    return raise_if(
      true_exprt(),
      "TypeError",
      "Unsupported dtype dtype('" + dtype_ + "') for randint");
  if (dtype_.find("int") == std::string::npos || dtype_ == "uint64")
    throw std::runtime_error(
      "TypeError: numpy.random.randint() dtype " + dtype_ +
      " is not supported");
  const int bits = std::stoi(dtype_.substr(dtype_.find("int") + 3));
  const bool is_unsigned = dtype_[0] == 'u';
  const long long min = is_unsigned ? 0 : -(1LL << (bits - 1));
  const long long max = is_unsigned ? (1LL << bits) : (1LL << (bits - 1));
  const exprt low = values_.count("high") ? value("low") : zero;
  const exprt high = values_.count("high") ? value("high") : value("low");
  raise_if(
    rel(low, "<", from_integer(min, long_long_int_type())),
    "ValueError",
    "low is out of bounds for " + dtype_);
  raise_if(
    rel(high, ">", from_integer(max, long_long_int_type())),
    "ValueError",
    "high is out of bounds for " + dtype_);
}

long long numpy_random::population_size() const
{
  return binary2integer(
           to_array_type(population_.type()).size().value().c_str(), false)
    .to_int64();
}

long long numpy_random::sample_count() const
{
  long long samples = 1;
  for (long long dimension : shape_)
    samples *= std::max(dimension, 0LL);
  return samples;
}

void numpy_random::validate_choice()
{
  const typet &element = population_.type().subtype();
  if (element.is_array() && element.subtype() != char_type())
    return raise_if(true_exprt(), "ValueError", "a must be 1-dimensional");
  const long long samples = sample_count();
  if (population_size() == 0 && samples != 0)
    return raise_if(
      true_exprt(),
      "ValueError",
      "a cannot be empty unless no samples are taken");
  if (has_probabilities_)
    return raise_if(
      true_exprt(),
      "TypeError",
      "numpy.random.choice() probabilities are not supported");
  if (!replace_ && samples > population_size())
    raise_if(
      true_exprt(),
      "ValueError",
      "Cannot take a larger sample than population when replace is False");
}

// An index into the population, distinct from the earlier ones of this call
// without replacement; never a range between the smallest and largest value.
exprt numpy_random::sample_choice()
{
  const typet &type = long_long_int_type();
  const exprt index = fresh(type);
  assume(and_exprt(
    rel(index, ">=", gen_zero(type)),
    rel(index, "<", from_integer(population_size(), type))));
  if (!replace_)
    for (const exprt &drawn : drawn_indices_)
      assume(not_exprt(equality_exprt(index, drawn)));
  drawn_indices_.push_back(index);
  const typet &element = population_.type().subtype();
  if (!element.is_array())
    return index_exprt(population_, index, element);
  // A symbolic read of a string row is not encoded by the solver backends,
  // so the row is picked by comparing the index with each constant one.
  exprt picked = index_exprt(population_, gen_zero(index_type()), element);
  for (long long i = 1; i < population_size(); ++i)
    picked = if_exprt(
      equality_exprt(index, from_integer(i, type)),
      index_exprt(population_, from_integer(i, index_type()), element),
      picked);
  return picked;
}

exprt numpy_random::sample()
{
  if (name_ == "choice")
    return sample_choice();
  if (is_integral_result(name_))
    return sample_integer();

  const exprt zero = real(0.0);
  const exprt x = fresh(double_type());
  if (name_ == "random" || name_ == "uniform")
  {
    assume(and_exprt(rel(x, ">=", zero), rel(x, "<", real(1.0))));
    if (name_ == "random")
      return x;
    // NumPy computes low + (high - low) * u, which may round up to high.
    const exprt low = value("low");
    return arith(
      low,
      exprt::plus,
      arith(arith(value("high"), exprt::minus, low), exprt::mult, x));
  }
  if (name_ == "beta")
  {
    assume(and_exprt(rel(x, ">=", zero), rel(x, "<=", real(1.0))));
    return x;
  }
  if (name_ == "standard_normal" || name_ == "normal")
  {
    assume(finite(x));
    if (name_ == "standard_normal")
      return x;
    return arith(
      value("loc"), exprt::plus, arith(value("scale"), exprt::mult, x));
  }
  // lognormal is exp() of a normal sample, which may overflow to infinity.
  assume(rel(x, ">=", zero));
  if (name_ == "lognormal")
    return x;
  assume(finite(x));
  if (name_ == "weibull")
    assume(implies(equality_exprt(value("a"), zero), equality_exprt(x, zero)));
  if (name_ == "gamma")
    assume(
      implies(equality_exprt(value("shape"), zero), equality_exprt(x, zero)));
  if (name_ == "exponential" || name_ == "gamma")
    return arith(value("scale"), exprt::mult, x);
  return x;
}

exprt numpy_random::sample_integer()
{
  const typet &type = long_long_int_type();
  const exprt k = fresh(type);
  const exprt zero = gen_zero(type);
  if (name_ == "randint")
  {
    const bool has_high = values_.count("high");
    assume(and_exprt(
      rel(k, ">=", has_high ? value("low") : zero),
      rel(k, "<", has_high ? value("high") : value("low"))));
    return k;
  }
  if (name_ == "poisson")
  {
    assume(rel(k, ">=", zero));
    assume(implies(
      equality_exprt(value("lam"), real(0.0)), equality_exprt(k, zero)));
    return k;
  }
  const exprt one = from_integer(1, type);
  if (name_ == "geometric")
  {
    assume(rel(k, ">=", one));
    assume(
      implies(equality_exprt(value("p"), real(1.0)), equality_exprt(k, one)));
    return k;
  }
  const exprt n = value("n");
  assume(and_exprt(rel(k, ">=", zero), rel(k, "<=", n)));
  assume(
    implies(equality_exprt(value("p"), real(0.0)), equality_exprt(k, zero)));
  assume(implies(equality_exprt(value("p"), real(1.0)), equality_exprt(k, n)));
  return k;
}

exprt numpy_random::fresh(const typet &type, const exprt &value)
{
  symbolt &tmp =
    converter_.create_tmp_symbol(call_, "$np_random$", type, exprt());
  const exprt symbol = symbol_expr(tmp);
  code_declt decl(symbol);
  decl.location() = location_;
  converter_.add_instruction(decl);
  code_assignt assign(
    symbol, value.is_nil() ? side_effect_expr_nondett(type) : value);
  assign.location() = location_;
  converter_.add_instruction(assign);
  return symbol;
}

void numpy_random::assume(const exprt &cond)
{
  code_assumet assumption(cond);
  assumption.location() = location_;
  converter_.add_instruction(assumption);
}

void numpy_random::raise_if(
  const exprt &cond,
  const std::string &exc,
  const std::string &msg)
{
  converter_.emit_guarded_raise(cond, exc, msg, location_);
}

// Statements run where the call is evaluated: spliced ahead of the enclosing
// statement when it runs once, unconditionally; otherwise kept with the
// value in a statement expression (an and/or tail, a while test).
exprt numpy_random::finish(const exprt &result)
{
  if (converter_.can_emit_runtime_guard())
  {
    for (const auto &statement : block_.operands())
      converter_.add_instruction(statement);
    return result;
  }
  code_expressiont value(result);
  value.location() = location_;
  block_.copy_to_operands(value);
  side_effect_exprt expr("statement_expression", result.type());
  expr.copy_to_operands(block_);
  expr.location() = location_;
  return expr;
}
