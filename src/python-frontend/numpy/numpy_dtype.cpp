#include <python-frontend/numpy/numpy_dtype.h>

#include <python-frontend/type/type_handler.h>
#include <python-frontend/type/type_utils.h>
#include <util/lang/c_types.h>
#include <util/irep/std_types.h>

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
    return "str";
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
    return get_complex_struct_type();
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
  if (it == dtypes.end())
    return std::nullopt;
  return it->second;
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
    return numpy_dtypet{
      kindt::floating, to_floatbv_type(element).get_width()};
  if (is_complex_type(element))
    return numpy_dtypet{kindt::complex, 128};
  return std::nullopt;
}
