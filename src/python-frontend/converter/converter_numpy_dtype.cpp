#include <python-frontend/json_utils.h>
#include <python-frontend/numpy/numpy_call_expr.h>
#include <python-frontend/numpy/numpy_dtype.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/type/type_utils.h>

#include <stdexcept>

static bool is_dtype_attribute(const nlohmann::json &node)
{
  return node.is_object() && node.value("_type", "") == "Attribute" &&
         node.value("attr", "") == "dtype";
}

// The dtype of the array or view @p base denotes. Only a name or a subscript is
// read, since reading it twice must not repeat a call.
std::optional<numpy_dtypet>
python_converter::numpy_array_dtype(const nlohmann::json &base)
{
  const std::string kind = base.value("_type", "");
  if (
    (kind != "Name" && kind != "Subscript") ||
    json_utils::ast_contains_call(base))
    return std::nullopt;
  const exprt array = numpy_param_as_array(get_expr(base));
  const typet type = ns.follow(array.type());
  const bool is_view =
    type.is_pointer() && array.is_symbol() &&
    numpy_pointer_view_info_.count(array.identifier().as_string()) != 0;
  if (!type.is_array() && !is_view)
    return std::nullopt;
  return numpy_dtype_of_element(
    type_utils::array_scalar_type(is_view ? type.subtype() : type));
}

// `.itemsize`, `.nbytes` and `.dtype.name` of an array or view; nbytes follows
// the logical shape, so a view reports its own bytes.
std::optional<exprt>
python_converter::try_get_numpy_storage_attr(const nlohmann::json &element)
{
  if (element.value("_type", "") != "Attribute")
    return std::nullopt;
  const std::string attr = element.value("attr", "");
  const nlohmann::json &base = element["value"];
  if (attr == "name" && is_dtype_attribute(base))
  {
    const auto dtype = numpy_array_dtype(base["value"]);
    if (!dtype)
      return std::nullopt;
    nlohmann::json name = {{"_type", "Constant"}, {"value", dtype->name()}};
    return get_expr(name);
  }
  if (attr != "itemsize" && attr != "nbytes" && attr != "dtype")
    return std::nullopt;
  const auto dtype = numpy_array_dtype(base);
  if (!dtype)
    return std::nullopt;
  if (attr == "dtype")
    throw std::runtime_error(
      "TypeError: a numpy dtype is only supported in a comparison or as "
      ".dtype.name");
  const exprt itemsize = from_integer(dtype->itemsize(), int_type());
  if (attr == "itemsize")
    return itemsize;
  nlohmann::json size = element;
  size["attr"] = "size";
  const exprt count = typecast_exprt(get_expr(size), int_type());
  exprt nbytes("*", int_type());
  nbytes.copy_to_operands(count, itemsize);
  return nbytes;
}

// `a.dtype == np.int8`: decided here, as NumPy compares any spelling of a dtype
// and not the text of its name.
std::optional<exprt>
python_converter::try_numpy_dtype_compare(const nlohmann::json &element)
{
  if (element.value("_type", "") != "Compare" || element["ops"].size() != 1)
    return std::nullopt;
  const std::string op = element["ops"][0].value("_type", "");
  if (op != "Eq" && op != "NotEq")
    return std::nullopt;
  const nlohmann::json &left = element["left"];
  const nlohmann::json &right = element["comparators"][0];
  const auto side_name = [&](const nlohmann::json &side) -> std::string {
    if (!is_dtype_attribute(side))
      return extract_numpy_dtype_name(side, *ast_json);
    const auto dtype = numpy_array_dtype(side["value"]);
    if (!dtype)
      throw std::runtime_error(
        "TypeError: .dtype of this value is not supported");
    return dtype->name();
  };
  if (!is_dtype_attribute(left) && !is_dtype_attribute(right))
    return std::nullopt;
  const bool same = side_name(left) == side_name(right);
  return gen_boolean(same == (op == "Eq"));
}
