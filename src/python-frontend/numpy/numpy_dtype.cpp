#include <python-frontend/numpy/numpy_dtype.h>

#include <set>

#include <python-frontend/type/type_handler.h>
#include <python-frontend/type/type_utils.h>
#include <util/lang/c_types.h>
#include <util/irep/std_types.h>

#include <algorithm>
#include <map>

std::string numpy_dtypet::name() const
{
  switch (kind)
  {
  case kindt::boolean:
    return "bool";
  case kindt::signed_integer:
    return "int" + std::to_string(bits);
  case kindt::unsigned_integer:
    return "uint" + std::to_string(bits);
  case kindt::floating:
    return "float" + std::to_string(bits);
  case kindt::complex:
    return "complex" + std::to_string(bits);
  case kindt::string:
    return bits == 0 ? "str" : "str" + std::to_string(bits);
  }
  return {};
}

typet numpy_dtypet::element_type() const
{
  switch (kind)
  {
  case kindt::boolean:
    return bool_type();
  case kindt::signed_integer:
    return type_utils::numpy_integer_type(signedbv_typet(bits));
  case kindt::unsigned_integer:
    return type_utils::numpy_integer_type(unsignedbv_typet(bits));
  case kindt::floating:
    return build_float_type(bits);
  case kindt::complex:
    return bits == 64 ? typet(get_complex64_struct_type())
                      : typet(get_complex_struct_type());
  case kindt::string:
    return typet();
  }
  return typet();
}

std::size_t numpy_dtypet::itemsize() const
{
  return kind == kindt::boolean ? 1 : bits / 8;
}

std::optional<numpy_dtypet> numpy_dtype_from_name(const std::string &name)
{
  using kindt = numpy_dtypet::kindt;
  static const std::map<std::string, numpy_dtypet> dtypes = {
    {"bool", {kindt::boolean, 8}},
    {"int8", {kindt::signed_integer, 8}},
    {"int16", {kindt::signed_integer, 16}},
    {"int32", {kindt::signed_integer, 32}},
    {"int64", {kindt::signed_integer, 64}},
    {"uint8", {kindt::unsigned_integer, 8}},
    {"uint16", {kindt::unsigned_integer, 16}},
    {"uint32", {kindt::unsigned_integer, 32}},
    {"uint64", {kindt::unsigned_integer, 64}},
    {"float16", {kindt::floating, 16}},
    {"float32", {kindt::floating, 32}},
    {"float64", {kindt::floating, 64}},
    {"complex64", {kindt::complex, 64}},
    {"complex128", {kindt::complex, 128}},
    {"str", {kindt::string, 0}}};
  const auto it = dtypes.find(name);
  if (it != dtypes.end())
    return it->second;
  // `U<n>`: four bytes a character, as `.itemsize` reports them.
  if (name.size() > 1 && name[0] == 'U')
    return numpy_dtypet{
      kindt::string, static_cast<unsigned>(std::stoul(name.substr(1))) * 32};
  return std::nullopt;
}

std::optional<numpy_dtypet> numpy_dtype_of_element(const typet &element)
{
  using kindt = numpy_dtypet::kindt;
  if (element.is_bool())
    return numpy_dtypet{kindt::boolean, 8};
  if (element.is_signedbv())
    return numpy_dtypet{
      kindt::signed_integer, to_signedbv_type(element).get_width()};
  if (element.is_unsignedbv())
    return numpy_dtypet{
      kindt::unsigned_integer, to_unsignedbv_type(element).get_width()};
  if (element.is_floatbv())
    return numpy_dtypet{kindt::floating, to_floatbv_type(element).get_width()};
  if (is_complex_type(element))
    return numpy_dtypet{
      kindt::complex, is_complex64_type(element) ? 64u : 128u};
  return std::nullopt;
}

static const numpy_dtypet numpy_int64{numpy_dtypet::kindt::signed_integer, 64};
static const numpy_dtypet numpy_float64{numpy_dtypet::kindt::floating, 64};

// np.result_type of two integers: a mixed pair needs a signed type above the
// unsigned one, and nothing signed holds uint64.
static numpy_dtypet
promote_integers(const numpy_dtypet &a, const numpy_dtypet &b)
{
  if (a.kind == b.kind)
    return a.bits >= b.bits ? a : b;
  const numpy_dtypet &signed_side =
    a.kind == numpy_dtypet::kindt::signed_integer ? a : b;
  const numpy_dtypet &unsigned_side =
    a.kind == numpy_dtypet::kindt::signed_integer ? b : a;
  if (unsigned_side.bits == 64)
    return numpy_float64;
  return {
    numpy_dtypet::kindt::signed_integer,
    std::max(signed_side.bits, unsigned_side.bits * 2)};
}

// The smallest float that holds every value of an integer dtype, as NumPy's
// promotion table has it: int8 fits float16, int16 float32, wider float64.
static unsigned float_bits_holding(const numpy_dtypet &integer)
{
  return integer.bits <= 8 ? 16 : integer.bits == 16 ? 32 : 64;
}

static numpy_dtypet
promote_complex(const numpy_dtypet &a, const numpy_dtypet &b)
{
  using kindt = numpy_dtypet::kindt;
  const numpy_dtypet &other = a.kind == kindt::complex ? b : a;
  const numpy_dtypet &complex_side = a.kind == kindt::complex ? a : b;
  const bool single = complex_side.bits == 64 &&
                      (other.kind == kindt::boolean ||
                       (other.kind == kindt::complex && other.bits == 64) ||
                       (other.kind == kindt::floating && other.bits <= 32) ||
                       (other.is_integer() && float_bits_holding(other) <= 32));
  return {kindt::complex, single ? 64u : 128u};
}

static numpy_dtypet promote_strong(const numpy_dtypet &a, const numpy_dtypet &b)
{
  using kindt = numpy_dtypet::kindt;
  if (a.kind == kindt::boolean)
    return b;
  if (b.kind == kindt::boolean)
    return a;
  if (a.kind == kindt::complex || b.kind == kindt::complex)
    return promote_complex(a, b);
  if (a.is_integer() && b.is_integer())
    return promote_integers(a, b);
  if (a.kind == kindt::floating && b.kind == kindt::floating)
    return a.bits >= b.bits ? a : b;
  const numpy_dtypet &real = a.kind == kindt::floating ? a : b;
  const numpy_dtypet &integer = a.kind == kindt::floating ? b : a;
  return {kindt::floating, std::max(real.bits, float_bits_holding(integer))};
}

// NEP 50: the weak number keeps the strong operand's dtype while its kind does
// not exceed that dtype's, and otherwise gives the kind's default.
static numpy_dtypet
promote_weak(const numpy_dtypet &weak, const numpy_dtypet &strong)
{
  using kindt = numpy_dtypet::kindt;
  switch (weak.kind)
  {
  case kindt::complex:
    if (strong.kind == kindt::complex)
      return strong;
    return {
      kindt::complex,
      strong.kind == kindt::floating && strong.bits <= 32 ? 64u : 128u};
  case kindt::floating:
    return strong.kind == kindt::floating || strong.kind == kindt::complex
             ? strong
             : numpy_float64;
  default:
    return strong.kind == kindt::boolean ? numpy_int64 : strong;
  }
}

numpy_dtypet numpy_promote(const numpy_operandt &a, const numpy_operandt &b)
{
  if (a.weak && !b.weak)
    return promote_weak(a.dtype, b.dtype);
  if (b.weak && !a.weak)
    return promote_weak(b.dtype, a.dtype);
  return promote_strong(a.dtype, b.dtype);
}

bool numpy_consumer_is_modelled(
  numpy_dtypet::kindt kind,
  const std::string &name)
{
  static const std::set<std::string> over_strings = {
    "array",
    "copy",
    "full",
    "tolist",
    "shape",
    "size",
    "ndim",
    "itemsize",
    "nbytes",
    "dtype",
    "choice"};
  static const std::set<std::string> over_complex = {
    "array",  "copy",        "full",     "tolist",  "shape",       "size",
    "ndim",   "itemsize",    "nbytes",   "dtype",   "sum",         "prod",
    "mean",   "any",         "all",      "add",     "subtract",    "multiply",
    "divide", "true_divide", "negative", "astype",  "transpose",   "reshape",
    "ravel",  "swapaxes",    "moveaxis", "squeeze", "expand_dims", "T"};
  if (kind == numpy_dtypet::kindt::string)
    return over_strings.count(name) != 0;
  return kind != numpy_dtypet::kindt::complex || over_complex.count(name) != 0;
}

std::string numpy_kind_name(numpy_dtypet::kindt kind)
{
  return kind == numpy_dtypet::kindt::string ? "string" : "complex";
}
