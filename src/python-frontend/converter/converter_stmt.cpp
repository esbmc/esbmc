#include <boost/algorithm/string/predicate.hpp>
#include <python-frontend/converter/converter_internal.h>
#include <python-frontend/math/convert_float_literal.h>
#include <python-frontend/function_call/expr.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/consteval/python_consteval.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_expr_builder.h>
#include <python-frontend/python-dict/python_dict_handler.h>
#include <python-frontend/python_annotation/python_annotation.h>
#include <python-frontend/exception/python_exception_handler.h>
#include <python-frontend/lambda/python_lambda.h>
#include <python-frontend/python-list/python_list.h>
#include <python-frontend/type/python_typechecking.h>
#include <python-frontend/math/python_math.h>
#include <python-frontend/string/string_builder.h>
#include <python-frontend/string/string_handler.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/tuple/tuple_handler.h>
#include <python-frontend/dynamic_type/dynamic_type_handler.h>
#include <python-frontend/type/type_handler.h>
#include <python-frontend/type/type_utils.h>
#include <irep2/irep2_utils.h>
#include <util/arith/arith_tools.h>
#include <util/expr/base_type.h>
#include <util/arith/bitvector.h>
#include <util/lang/c_typecast.h>
#include <util/lang/c_types.h>
#include <util/config/config.h>
#include <util/base/encoding.h>
#include <util/expr/expr_util.h>
#include <util/irep/irep.h>
#include <util/message/message.h>
#include <util/irep/migrate.h>
#include <util/lang/python_types.h>
#include <util/irep/std_code.h>
#include <util/expr/string_constant.h>
#include <util/expr/symbolic_types.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>

using namespace json_utils;

namespace
{
// True when `expr` (or any of its operands) is a `cpp-throw` marker, i.e. a
// probe build hit an error path instead of producing a usable value.
bool contains_cpp_throw(const exprt &expr)
{
  if (expr.statement() == "cpp-throw")
    return true;

  for (const auto &op : expr.operands())
  {
    if (contains_cpp_throw(op))
      return true;
  }

  return false;
}

// True when reassigning a value of type rhs to a variable currently typed lhs
// crosses the numeric<->string boundary, which a single GOTO symbol cannot
// represent in place.
bool is_incompatible_scalar_string_retype(
  const type_handler &th,
  const typet &lhs,
  const typet &rhs)
{
  return (th.is_numeric_scalar_type(lhs) && th.is_string_type(rhs)) ||
         (th.is_string_type(lhs) && th.is_numeric_scalar_type(rhs));
}

// True if the AST subtree contains a function-call node. Used to gate
// constant-folding of assertion tests to expressions that actually invoke a
// (potentially pure) function — plain symbolic asserts stay on the solver path.
bool ast_contains_call(const nlohmann::json &n)
{
  if (n.is_object())
  {
    if (n.contains("_type") && n["_type"] == "Call")
      return true;
    for (auto it = n.begin(); it != n.end(); ++it)
      if (ast_contains_call(it.value()))
        return true;
  }
  else if (n.is_array())
  {
    for (const auto &e : n)
      if (ast_contains_call(e))
        return true;
  }
  return false;
}

bool is_literal_int_node(const nlohmann::json &node)
{
  if (
    node.value("_type", "") == "Constant" && node.contains("value") &&
    node["value"].is_number_integer())
    return true;

  return node.value("_type", "") == "UnaryOp" && node.contains("op") &&
         node["op"].value("_type", "") == "USub" && node.contains("operand") &&
         node["operand"].value("_type", "") == "Constant" &&
         node["operand"].contains("value") &&
         node["operand"]["value"].is_number_integer();
}

std::optional<long long> literal_int_value(const nlohmann::json &node)
{
  if (!is_literal_int_node(node))
    return std::nullopt;

  if (node.value("_type", "") == "Constant")
    return node["value"].get<long long>();

  return -node["operand"]["value"].get<long long>();
}

std::vector<long long>
subscript_indices_from_root(const nlohmann::json &node, std::string &root_name)
{
  if (!node.is_object())
    return {};

  if (node.value("_type", "") == "Name" && node.contains("id"))
  {
    root_name = node["id"].get<std::string>();
    return {};
  }

  if (
    node.value("_type", "") != "Subscript" || !node.contains("value") ||
    !node.contains("slice"))
    return {};

  std::vector<long long> indices =
    subscript_indices_from_root(node["value"], root_name);
  if (root_name.empty())
    return {};

  std::optional<long long> index = literal_int_value(node["slice"]);
  if (!index)
  {
    root_name.clear();
    return {};
  }

  indices.push_back(*index);
  return indices;
}

exprt build_numpy_array_cell(
  const namespacet &ns,
  const symbolt &symbol,
  const std::vector<long long> &indices)
{
  exprt cell = symbol_expr(symbol);
  typet cell_type = ns.follow(symbol.get_type());

  for (long long index : indices)
  {
    if (cell_type.is_pointer())
      cell_type = ns.follow(cell_type.subtype());
    if (!cell_type.is_array())
      return exprt();

    const typet elem_type = ns.follow(to_array_type(cell_type).subtype());
    cell = python_expr::build_index(cell, from_integer(index, size_type()));
    cell.type() = elem_type;
    cell_type = elem_type;
  }

  return cell;
}

bool is_numpy_transpose_call_node(const nlohmann::json &node)
{
  return node.value("_type", "") == "Call" && node.contains("func") &&
         node["func"].is_object() &&
         node["func"].value("_type", "") == "Attribute" &&
         node["func"].value("attr", "") == "transpose";
}

bool is_numpy_axis_permutation_call_node(const nlohmann::json &node)
{
  if (
    node.value("_type", "") != "Call" || !node.contains("func") ||
    !node["func"].is_object() || node["func"].value("_type", "") != "Attribute")
    return false;

  const std::string attr = node["func"].value("attr", "");
  return attr == "swapaxes" || attr == "moveaxis";
}

bool is_numpy_transpose_view_call_node(const nlohmann::json &node)
{
  return is_numpy_transpose_call_node(node) ||
         is_numpy_axis_permutation_call_node(node);
}

const nlohmann::json *
numpy_transpose_source_node(const nlohmann::json &node, bool &swaps_axes)
{
  if (
    !is_numpy_transpose_view_call_node(node) || !node.contains("args") ||
    !node["args"].is_array() || node["args"].empty())
    return nullptr;

  const nlohmann::json *source = &node["args"][0];
  swaps_axes = true;
  if (
    is_numpy_transpose_call_node(*source) && source->contains("args") &&
    (*source)["args"].is_array() && !(*source)["args"].empty())
  {
    swaps_axes = false;
    source = &(*source)["args"][0];
  }

  return source;
}

std::optional<bool> numpy_axis_permutation_swaps_axes(
  const nlohmann::json &node,
  std::size_t rank,
  bool default_swaps_axes)
{
  if (!is_numpy_axis_permutation_call_node(node))
    return default_swaps_axes;

  if (rank > 2 || node["args"].size() < 3)
    return std::nullopt;

  std::array<long long, 2> axes{};
  for (std::size_t i = 0; i < axes.size(); ++i)
  {
    std::optional<long long> axis = literal_int_value(node["args"][i + 1]);
    if (!axis)
      return std::nullopt;

    axes[i] = *axis;
    if (axes[i] < 0)
      axes[i] += static_cast<long long>(rank);
    if (axes[i] < 0 || axes[i] >= static_cast<long long>(rank))
      return std::nullopt;
  }

  return axes[0] != axes[1];
}

bool is_numpy_reshape_call_node(const nlohmann::json &node)
{
  return node.value("_type", "") == "Call" && node.contains("func") &&
         node["func"].is_object() &&
         node["func"].value("_type", "") == "Attribute" &&
         node["func"].value("attr", "") == "reshape";
}

bool is_numpy_squeeze_call_node(const nlohmann::json &node)
{
  return node.value("_type", "") == "Call" && node.contains("func") &&
         node["func"].is_object() &&
         node["func"].value("_type", "") == "Attribute" &&
         node["func"].value("attr", "") == "squeeze";
}

bool is_numpy_expand_dims_call_node(const nlohmann::json &node)
{
  return node.value("_type", "") == "Call" && node.contains("func") &&
         node["func"].is_object() &&
         node["func"].value("_type", "") == "Attribute" &&
         node["func"].value("attr", "") == "expand_dims";
}

bool is_numpy_broadcast_to_call_node(const nlohmann::json &node)
{
  return node.value("_type", "") == "Call" && node.contains("func") &&
         node["func"].is_object() &&
         node["func"].value("_type", "") == "Attribute" &&
         node["func"].value("attr", "") == "broadcast_to";
}

bool is_numpy_shape_only_view_call_node(const nlohmann::json &node)
{
  const std::string attr = node.contains("func") && node["func"].is_object()
                             ? node["func"].value("attr", "")
                             : "";
  return attr == "reshape" || attr == "squeeze" || attr == "expand_dims" ||
         attr == "broadcast_to";
}

std::optional<std::size_t>
normalize_numpy_axis(long long axis, std::size_t rank, bool insertion_axis)
{
  const long long upper =
    static_cast<long long>(rank) + (insertion_axis ? 1 : 0);
  if (axis < 0)
    axis += upper;
  if (axis < 0 || axis >= upper)
    return std::nullopt;
  return static_cast<std::size_t>(axis);
}

std::vector<std::size_t>
numpy_shape_from_type(const namespacet &ns, typet source_type)
{
  if (source_type.is_pointer())
    source_type = ns.follow(source_type.subtype());

  std::vector<std::size_t> shape;
  while (source_type.is_array())
  {
    const array_typet &array_type = to_array_type(source_type);
    const exprt &size = array_type.size();
    if (!size.is_constant())
      return {};
    shape.push_back(static_cast<std::size_t>(
      binary2integer(to_constant_expr(size).value().c_str(), false)
        .to_uint64()));
    source_type = ns.follow(array_type.subtype());
  }
  return shape;
}

std::size_t numpy_shape_element_count(const std::vector<std::size_t> &shape)
{
  return std::accumulate(
    shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>());
}

std::optional<std::vector<long long>>
numpy_raw_reshape_sequence(const nlohmann::json &shape_arg)
{
  if (!shape_arg.contains("elts"))
    return std::nullopt;

  std::vector<long long> raw_shape;
  for (const auto &dim_node : shape_arg["elts"])
  {
    std::optional<long long> dim = literal_int_value(dim_node);
    if (!dim)
      return std::nullopt;
    raw_shape.push_back(*dim);
  }

  return raw_shape;
}

std::optional<std::vector<long long>>
numpy_raw_reshape_method_shape(const nlohmann::json &args)
{
  std::vector<long long> raw_shape;
  for (std::size_t i = 1; i < args.size(); ++i)
  {
    std::optional<long long> dim = literal_int_value(args[i]);
    if (!dim)
      return std::nullopt;
    raw_shape.push_back(*dim);
  }

  return raw_shape;
}

std::optional<std::vector<long long>>
numpy_raw_reshape_shape(const nlohmann::json &node)
{
  if (
    !is_numpy_reshape_call_node(node) || !node.contains("args") ||
    !node["args"].is_array() || node["args"].size() < 2)
    return std::nullopt;

  const nlohmann::json &shape_arg = node["args"][1];
  if (
    shape_arg.is_object() && shape_arg.contains("_type") &&
    (shape_arg["_type"] == "Tuple" || shape_arg["_type"] == "List"))
    return numpy_raw_reshape_sequence(shape_arg);

  if (node.value("_numpy_method_form", false) && node["args"].size() > 2)
    return numpy_raw_reshape_method_shape(node["args"]);

  std::optional<long long> dim = literal_int_value(shape_arg);
  if (!dim)
    return std::nullopt;

  return std::vector<long long>{*dim};
}

std::optional<std::vector<std::size_t>> normalize_numpy_reshape_shape(
  const std::vector<long long> &raw_shape,
  std::size_t total)
{
  std::vector<std::size_t> shape;
  std::size_t inferred_idx = raw_shape.size();
  std::size_t known_product = 1;
  for (std::size_t i = 0; i < raw_shape.size(); ++i)
  {
    if (raw_shape[i] == -1)
    {
      if (inferred_idx != raw_shape.size())
        return std::nullopt;
      inferred_idx = i;
      shape.push_back(0);
      continue;
    }
    if (raw_shape[i] < 0)
      return std::nullopt;
    shape.push_back(static_cast<std::size_t>(raw_shape[i]));
    known_product *= shape.back();
  }

  if (inferred_idx != raw_shape.size())
  {
    if (known_product == 0 || total % known_product != 0)
      return std::nullopt;
    shape[inferred_idx] = total / known_product;
  }

  return numpy_shape_element_count(shape) == total
           ? std::optional<std::vector<std::size_t>>(shape)
           : std::nullopt;
}

std::optional<std::vector<std::size_t>>
parse_numpy_reshape_shape(const nlohmann::json &node, std::size_t total)
{
  std::optional<std::vector<long long>> raw_shape =
    numpy_raw_reshape_shape(node);
  if (!raw_shape)
    return std::nullopt;

  return normalize_numpy_reshape_shape(*raw_shape, total);
}

std::optional<std::vector<std::size_t>> numpy_squeeze_view_shape(
  const nlohmann::json &node,
  const std::vector<std::size_t> &source_shape)
{
  if (
    !is_numpy_squeeze_call_node(node) || !node.contains("args") ||
    !node["args"].is_array() || node["args"].empty())
    return std::nullopt;

  std::vector<std::size_t> view_shape;
  if (node["args"].size() == 1)
  {
    for (std::size_t dim : source_shape)
      if (dim != 1)
        view_shape.push_back(dim);
    return view_shape;
  }

  std::optional<long long> raw_axis = literal_int_value(node["args"][1]);
  if (!raw_axis)
    return std::nullopt;

  std::optional<std::size_t> axis =
    normalize_numpy_axis(*raw_axis, source_shape.size(), false);
  if (!axis || source_shape[*axis] != 1)
    return std::nullopt;

  for (std::size_t i = 0; i < source_shape.size(); ++i)
    if (i != *axis)
      view_shape.push_back(source_shape[i]);
  return view_shape;
}

std::optional<std::vector<std::size_t>> numpy_expand_dims_view_shape(
  const nlohmann::json &node,
  const std::vector<std::size_t> &source_shape)
{
  if (
    !is_numpy_expand_dims_call_node(node) || !node.contains("args") ||
    !node["args"].is_array() || node["args"].size() < 2)
    return std::nullopt;

  std::optional<long long> raw_axis = literal_int_value(node["args"][1]);
  if (!raw_axis)
    return std::nullopt;

  std::optional<std::size_t> axis =
    normalize_numpy_axis(*raw_axis, source_shape.size(), true);
  if (!axis)
    return std::nullopt;

  std::vector<std::size_t> view_shape = source_shape;
  view_shape.insert(view_shape.begin() + *axis, 1);
  return view_shape;
}

bool numpy_shapes_broadcast_to(
  const std::vector<std::size_t> &source_shape,
  const std::vector<std::size_t> &view_shape)
{
  if (source_shape.size() > view_shape.size())
    return false;

  const std::size_t offset = view_shape.size() - source_shape.size();
  for (std::size_t axis = 0; axis < source_shape.size(); ++axis)
  {
    const std::size_t source_dim = source_shape[axis];
    const std::size_t view_dim = view_shape[axis + offset];
    if (source_dim != view_dim && source_dim != 1)
      return false;
  }
  return true;
}

std::optional<std::vector<std::size_t>> numpy_broadcast_to_view_shape(
  const nlohmann::json &node,
  const std::vector<std::size_t> &source_shape)
{
  if (
    !is_numpy_broadcast_to_call_node(node) || !node.contains("args") ||
    !node["args"].is_array() || node["args"].size() < 2)
    return std::nullopt;

  std::optional<std::vector<long long>> raw_shape =
    numpy_raw_reshape_sequence(node["args"][1]);
  if (!raw_shape)
    return std::nullopt;

  std::vector<std::size_t> view_shape;
  for (long long dim : *raw_shape)
  {
    if (dim < 0)
      return std::nullopt;
    view_shape.push_back(static_cast<std::size_t>(dim));
  }

  if (view_shape.empty() || view_shape.size() > 2)
    return std::nullopt;
  return numpy_shapes_broadcast_to(source_shape, view_shape)
           ? std::optional<std::vector<std::size_t>>(view_shape)
           : std::nullopt;
}

std::optional<std::vector<std::size_t>> numpy_shape_only_view_shape(
  const nlohmann::json &node,
  const std::vector<std::size_t> &source_shape)
{
  if (is_numpy_reshape_call_node(node))
    return parse_numpy_reshape_shape(
      node, numpy_shape_element_count(source_shape));
  if (is_numpy_squeeze_call_node(node))
    return numpy_squeeze_view_shape(node, source_shape);
  if (is_numpy_expand_dims_call_node(node))
    return numpy_expand_dims_view_shape(node, source_shape);
  if (is_numpy_broadcast_to_call_node(node))
    return numpy_broadcast_to_view_shape(node, source_shape);
  return std::nullopt;
}

std::optional<std::vector<long long>> numpy_broadcast_source_indices(
  const std::vector<long long> &view_indices,
  const std::vector<std::size_t> &view_shape,
  const std::vector<std::size_t> &source_shape)
{
  if (view_indices.size() != view_shape.size())
    return std::nullopt;

  const std::size_t offset = view_shape.size() - source_shape.size();
  std::vector<long long> source_indices;
  for (std::size_t axis = 0; axis < source_shape.size(); ++axis)
  {
    const long long view_index = view_indices[axis + offset];
    source_indices.push_back(source_shape[axis] == 1 ? 0 : view_index);
  }
  return source_indices;
}

bool numpy_indices_equal(
  const std::vector<long long> &lhs,
  const std::vector<long long> &rhs)
{
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

std::vector<std::vector<long long>> numpy_broadcast_view_indices_for_source(
  const std::vector<long long> &source_indices,
  const std::vector<std::size_t> &view_shape,
  const std::vector<std::size_t> &source_shape)
{
  std::vector<std::vector<long long>> matches;
  if (view_shape.empty() || view_shape.size() > 2)
    return matches;

  const std::size_t outer = view_shape[0];
  const std::size_t inner = view_shape.size() == 1 ? 1 : view_shape[1];
  for (std::size_t i = 0; i < outer; ++i)
  {
    for (std::size_t j = 0; j < inner; ++j)
    {
      std::vector<long long> current{static_cast<long long>(i)};
      if (view_shape.size() == 2)
        current.push_back(static_cast<long long>(j));

      std::optional<std::vector<long long>> mapped =
        numpy_broadcast_source_indices(current, view_shape, source_shape);
      if (mapped && numpy_indices_equal(*mapped, source_indices))
        matches.push_back(std::move(current));
    }
  }
  return matches;
}

std::optional<std::size_t> numpy_flat_index(
  const std::vector<long long> &indices,
  const std::vector<std::size_t> &shape)
{
  if (indices.size() != shape.size())
    return std::nullopt;

  std::size_t flat = 0;
  for (std::size_t i = 0; i < shape.size(); ++i)
  {
    if (indices[i] < 0 || static_cast<std::size_t>(indices[i]) >= shape[i])
      return std::nullopt;
    flat = flat * shape[i] + static_cast<std::size_t>(indices[i]);
  }
  return flat;
}

std::optional<std::vector<long long>>
numpy_unravel_index(std::size_t flat, const std::vector<std::size_t> &shape)
{
  if (shape.empty() || shape.size() > 2)
    return std::nullopt;

  if (shape.size() == 1)
    return std::vector<long long>{static_cast<long long>(flat)};

  if (shape[1] == 0)
    return std::nullopt;

  return std::vector<long long>{
    static_cast<long long>(flat / shape[1]),
    static_cast<long long>(flat % shape[1])};
}

std::optional<std::vector<long long>> numpy_shape_view_source_indices(
  const std::vector<long long> &view_indices,
  const std::vector<std::size_t> &view_shape,
  const std::vector<std::size_t> &source_shape,
  const bool broadcast)
{
  if (broadcast)
    return numpy_broadcast_source_indices(
      view_indices, view_shape, source_shape);

  std::optional<std::size_t> flat = numpy_flat_index(view_indices, view_shape);
  return flat ? numpy_unravel_index(*flat, source_shape) : std::nullopt;
}

nlohmann::json numpy_constant_index_node(const std::size_t value)
{
  return {{"_type", "Constant"}, {"value", static_cast<long long>(value)}};
}

nlohmann::json numpy_name_load_node(const std::string &name)
{
  return {{"_type", "Name"}, {"id", name}, {"ctx", {{"_type", "Load"}}}};
}

nlohmann::json
numpy_subscript_node(nlohmann::json value, const std::size_t index)
{
  return {
    {"_type", "Subscript"},
    {"value", std::move(value)},
    {"slice", numpy_constant_index_node(index)},
    {"ctx", {{"_type", "Load"}}}};
}

nlohmann::json numpy_subscript_node(
  const std::string &root_name,
  const std::vector<std::size_t> &indices)
{
  nlohmann::json node = numpy_name_load_node(root_name);
  for (const std::size_t index : indices)
    node = numpy_subscript_node(std::move(node), index);
  return node;
}

std::size_t numpy_array_rank(const namespacet &ns, typet source_type)
{
  if (source_type.is_pointer())
    source_type = ns.follow(source_type.subtype());

  std::size_t rank = 0;
  for (typet current = source_type; current.is_array();
       current = ns.follow(to_array_type(current).subtype()))
    ++rank;
  return rank;
}

std::optional<std::vector<long long>> numpy_transpose_cell_indices(
  std::size_t rank,
  bool swaps_axes,
  const std::vector<long long> &indices)
{
  if (rank == 1 && indices.size() == 1)
    return std::vector<long long>{indices[0]};

  if (rank != 2 || indices.size() != 2)
    return std::nullopt;

  if (swaps_axes)
    return std::vector<long long>{indices[1], indices[0]};

  return std::vector<long long>{indices[0], indices[1]};
}

// A bare `:` slice axis (no lower/upper/step), matching
// converter_expr.cpp's own is_full_slice_node used to recognise the
// column-select shape `a[:, j]`.
bool is_full_slice_axis_node(const nlohmann::json &node)
{
  if (node.value("_type", "") != "Slice")
    return false;
  auto absent = [&](const char *key) {
    return !node.contains(key) || node[key].is_null();
  };
  return absent("lower") && absent("upper") && absent("step");
}

// `a[:, j]` column-select shape specifically: axis 0 a full slice, axis 1
// a literal int. converter_expr.cpp's SUBSCRIPT/Tuple dispatch only calls
// build_column_select when is_full_slice_node(idx_nodes[0]) holds -- the
// reverse order, `a[j, :]`, is a row-select-then-chained-slice instead
// (list.index(array, idx_nodes[0]) followed by list.index(current,
// idx_nodes[1])), a different shape this function must not also match, or
// should_rebuild_cached_numpy_row_subscript_rhs forces a fresh conversion
// for it too and current_lhs ends up retyped to a pointer mid-chain by
// try_build_row_pointer_view before the outer `[:]` is ever applied.
bool is_column_select_slice_node(const nlohmann::json &slice)
{
  if (
    slice.value("_type", "") != "Tuple" || !slice.contains("elts") ||
    slice["elts"].size() != 2)
    return false;

  return is_full_slice_axis_node(slice["elts"][0]) &&
         is_literal_int_node(slice["elts"][1]);
}

bool has_non_null_value(const nlohmann::json &node)
{
  return node.contains("value") && !node["value"].is_null();
}

void set_dict_literal_element_type(
  const nlohmann::json &ast_node,
  python_dict_handler &dict_handler,
  typet &element_type)
{
  if (
    has_non_null_value(ast_node) &&
    dict_handler.is_dict_literal(ast_node["value"]))
    element_type = dict_handler.get_dict_struct_type();
}

// A `dp[i] = v` / `dp[i] += v` shape. Such an assignment writes an element,
// not the container, so container-level bookkeeping must sit it out.
bool assignment_target_is_subscript(const nlohmann::json &ast_node)
{
  auto is_subscript = [](const nlohmann::json &t) {
    return t.is_object() && t.value("_type", "") == "Subscript";
  };
  return (ast_node.contains("targets") && ast_node["targets"].is_array() &&
          !ast_node["targets"].empty() &&
          is_subscript(ast_node["targets"][0])) ||
         (ast_node.contains("target") && is_subscript(ast_node["target"]));
}

bool is_same_name_assignment(
  const nlohmann::json &target,
  const nlohmann::json &ast_node)
{
  if (target.value("_type", "") != "Name" || !has_non_null_value(ast_node))
    return false;

  const nlohmann::json &value = ast_node["value"];
  return value.value("_type", "") == "Name" &&
         target.value("id", "") == value.value("id", "");
}

bool should_detach_numpy_pointer_views_for_assignment(
  const nlohmann::json &target,
  const nlohmann::json &ast_node,
  const symbolt *lhs_symbol)
{
  return target.value("_type", "") == "Name" && lhs_symbol &&
         !is_same_name_assignment(target, ast_node);
}

bool ast_imports_numpy_module(const nlohmann::json &ast)
{
  if (!ast.is_object() || !ast.contains("body") || !ast["body"].is_array())
    return false;

  for (const auto &stmt : ast["body"])
  {
    if (
      !stmt.is_object() || stmt.value("_type", std::string()) != "Import" ||
      !stmt.contains("names") || !stmt["names"].is_array())
      continue;

    for (const auto &alias : stmt["names"])
      if (
        alias.is_object() && alias.value("_type", std::string()) == "alias" &&
        alias.value("name", std::string()) == "numpy")
        return true;
  }

  return false;
}

// RAII bump of the get_block() nesting depth, and optionally the function-body
// or loop-body depth. Depth 1 is an unconditional top-level (module) statement;
// anything deeper is nested in a function or a conditional body. Straight-line
// retyping (#4770/#4774) is sound on the unconditional spine (module body plus
// enclosing function bodies): exactly block_nesting_ == function_body_depth_
// + 1. loop_body_depth_ counts enclosing while/for bodies: a loop target
// variable leaks past the loop in Python (so its retype must not be reverted at
// the body's join), so dynamic retyping is refused inside a loop body and left
// to the existing fallback, matching the pre-#5716 behaviour.
struct block_nesting_guard
{
  unsigned &depth;
  unsigned *fb_depth;
  unsigned *loop_depth;
  explicit block_nesting_guard(
    unsigned &d,
    unsigned *fb = nullptr,
    unsigned *loop = nullptr)
    : depth(d), fb_depth(fb), loop_depth(loop)
  {
    ++depth;
    if (fb_depth)
      ++*fb_depth;
    if (loop_depth)
      ++*loop_depth;
  }
  ~block_nesting_guard()
  {
    --depth;
    if (fb_depth)
      --*fb_depth;
    if (loop_depth)
      --*loop_depth;
  }
};

// RAII snapshot/restore of the dynamic-retyping alias map across a conditional
// body. A numeric<->string reassignment inside an if/while/for/try body retypes
// the variable so in-body reads observe the new type (sound: there is no join
// before the body ends). On body exit the alias map is reverted so the
// post-join view keeps the variable's pre-conditional type — the branch-taken
// retype must not leak across the control-flow join (see
// retype_str_cond_gated). Inactive on the unconditional spine, where retypes
// persist for the whole body.
struct retype_alias_scope_guard
{
  std::unordered_map<std::string, std::string> &aliases;
  std::unordered_map<std::string, std::string> saved;
  bool active;
  retype_alias_scope_guard(
    std::unordered_map<std::string, std::string> &a,
    bool act)
    : aliases(a), active(act)
  {
    if (active)
      saved = aliases;
  }
  ~retype_alias_scope_guard()
  {
    if (active)
      aliases = std::move(saved);
  }
};

} // namespace

// External linkage: shared with numpy_call_expr.cpp (declared in
// python_converter.h) so both can verify a Name/Attribute receiver actually
// resolves to the imported numpy module.
bool is_imported_numpy_module_alias(
  const nlohmann::json &ast,
  const std::string &name)
{
  if (
    name.empty() || !ast.is_object() || !ast.contains("body") ||
    !ast["body"].is_array())
    return false;

  for (const auto &stmt : ast["body"])
  {
    if (
      !stmt.is_object() || stmt.value("_type", std::string()) != "Import" ||
      !stmt.contains("names") || !stmt["names"].is_array())
      continue;

    for (const auto &alias : stmt["names"])
    {
      if (
        !alias.is_object() || alias.value("_type", std::string()) != "alias" ||
        alias.value("name", std::string()) != "numpy")
        continue;

      const nlohmann::json &asname = alias.value("asname", nlohmann::json());
      const std::string bound_name =
        asname.is_null() ? std::string("numpy") : asname.get<std::string>();
      if (bound_name == name)
        return true;
    }
  }

  return false;
}

void python_converter::adjust_statement_types(exprt &lhs, exprt &rhs) const
{
  typet &lhs_type = lhs.type();
  typet &rhs_type = rhs.type();

  // Case 0: assigning a migrated class pointer (`Class*`) into a by-value
  // class lvalue (`Class`). After the object-model migration a class
  // parameter/return is a pointer, but a struct field declared with the class
  // type is still by-value (e.g. `self.value: Tile`). Dereference the pointer
  // so the field receives the pointee struct rather than the address; without
  // this the GOTO carries a `struct = pointer` assignment that crashes SMT
  // encoding. Guarded by pointee/lvalue struct identity so genuine pointer
  // fields are untouched.
  if (rhs_type.is_pointer())
  {
    const typet lhs_follow = ns.follow(lhs_type);
    if (lhs_follow.is_struct() && ns.follow(rhs_type.subtype()) == lhs_follow)
    {
      // V.3: build the deref in IREP2 (byte-identical round-trip; the helper
      // restores the exact result type and falls back to legacy for dyn-array
      // pointees).
      rhs = python_expr::build_dereference(rhs, lhs_type);
      return;
    }
  }

  // Case 1: Promote RHS integer constant to float if LHS expects a float
  if (
    lhs_type.is_floatbv() && rhs.is_constant() &&
    type_utils::is_integer_type(rhs_type))
  {
    try
    {
      // Convert binary string value to integer
      BigInt value(
        binary2integer(rhs.value().as_string(), rhs_type.is_signedbv()));

      // Create a float literal string (e.g., "42.0")
      std::string rhs_float = std::to_string(value.to_int64()) + ".0";

      // Replace RHS with a float expression
      convert_float_literal(rhs_float, rhs);

      // Update the symbol table entry for RHS if needed
      update_symbol(rhs);
    }
    catch (const std::exception &e)
    {
      log_error(
        "adjust_statement_types: Failed to promote integer to float: {}",
        e.what());
    }
  }
  // Case 2: For Python assignments, if RHS is float but LHS is integer,
  // promote LHS to float to maintain Python's dynamic typing semantics
  else if (rhs_type.is_floatbv() && type_utils::is_integer_type(lhs_type))
  {
    // Update LHS variable type to match RHS float type
    lhs.type() = rhs_type;

    // Update symbol table if LHS is a symbol
    if (lhs.is_symbol())
      update_symbol(lhs);
  }
  // Case 3: Handles Python's / operator by promoting operands to floats
  // to ensure floating-point division, preventing division by zero, and
  // setting the result type to floatbv.
  else if (
    (rhs.id() == "/" || rhs.id() == "ieee_div") && rhs.operands().size() == 2)
  {
    auto &ops = rhs.operands();
    exprt &lhs_op = ops[0];
    exprt &rhs_op = ops[1];

    // Promote both operands to IEEE float (double precision) to match Python
    // semantics
    const typet float_type =
      double_type(); // Python default float is double-precision

    // Handle constant operands
    if (lhs_op.is_constant() && type_utils::is_integer_type(lhs_op.type()))
      math_handler_.promote_int_to_float(lhs_op, float_type);
    // For non-constant operands, create explicit typecast
    else if (!lhs_op.type().is_floatbv())
      lhs_op = typecast_exprt(lhs_op, float_type);

    if (rhs_op.is_constant() && type_utils::is_integer_type(rhs_op.type()))
      math_handler_.promote_int_to_float(rhs_op, float_type);
    else if (!rhs_op.type().is_floatbv())
      rhs_op = typecast_exprt(rhs_op, float_type);

    // For in-place division (like x /= y), ensure LHS variable is promoted to
    // float
    lhs.type() = float_type;
    if (lhs.is_symbol())
      update_symbol(lhs);

    // Update the division expression type and operator ID
    rhs.type() = float_type;
    rhs.id(python_frontend::map_operator("div", float_type));
  }
  // Case 4: Special case for IEEE division results - ensure LHS is float
  else if (rhs.id() == "ieee_div" && !lhs_type.is_floatbv())
  {
    // For any IEEE division result assigned to an integer variable,
    // promote the variable to float to avoid truncation
    const typet float_type = double_type();
    lhs.type() = float_type;

    if (lhs.is_symbol())
      update_symbol(lhs);

    // Ensure RHS type is also float
    if (!rhs_type.is_floatbv())
      rhs.type() = float_type;
  }
  // Case 5 (P19): Promote real RHS to complex when LHS is complex.
  // Must come BEFORE the width-alignment case: a complex struct is 128-bit
  // while a scalar float is 64-bit, so width alignment would otherwise fire
  // first and corrupt the float by assigning struct type to it.
  // Handles: z = 1.0, z = n, z = True where z is declared as complex.
  // Note: is_bool() must be explicit since is_integer_type() excludes bool.
  else if (
    is_complex_type(lhs_type) && !is_complex_type(rhs_type) &&
    (rhs_type.is_floatbv() || type_utils::is_integer_type(rhs_type) ||
     rhs_type.is_bool()))
  {
    rhs = promote_to_complex(rhs);
  }
  // Case 6: Align bit-widths between LHS and RHS if they differ. Never
  // "align" a tuple struct against a non-tuple: demoting the LHS symbol to
  // the scalar's type corrupts already-emitted tuple member reads (see the
  // fresh-alias gate in get_var_assign).
  else if (
    lhs_type.width() != rhs_type.width() &&
    tuple_handler_->is_tuple_type(lhs_type) ==
      tuple_handler_->is_tuple_type(rhs_type))
  {
    try
    {
      const int lhs_width = type_handler_.get_type_width(lhs_type);
      const int rhs_width = type_handler_.get_type_width(rhs_type);

      if (lhs_width > rhs_width)
      {
        // Promote RHS to LHS type
        rhs_type = lhs_type;
        if (rhs.is_symbol())
          update_symbol(rhs);
      }
      else
      {
        // Promote LHS to RHS type
        lhs_type = rhs_type;
        if (lhs.is_symbol())
          update_symbol(lhs);
      }
    }
    catch (const std::exception &e)
    {
      log_error(
        "adjust_statement_types: Failed to parse type widths: {}", e.what());
    }
  }
}
/// True when @p spelling names the built-in @p lower / @p upper and no
/// user-defined class of that name shadows it.
static bool names_builtin(
  const std::string &spelling,
  const char *lower,
  const char *upper,
  const nlohmann::json &ast)
{
  if (spelling != lower && spelling != upper)
    return false;
  return !json_utils::is_class(spelling, ast);
}

std::pair<std::string, typet>
python_converter::extract_type_info(const nlohmann::json &var_node)
{
  typet var_typet;
  std::string var_type_str("");

  if (var_node.contains("annotation") && !var_node["annotation"].is_null())
  {
    // Get type from annotation node
    size_t type_size = get_type_size(var_node);
    const auto &ann = var_node["annotation"];

    if (ann.contains("_type") && ann["_type"] == "Subscript")
    {
      if (ann.contains("value") && ann["value"].contains("id"))
        var_type_str = ann["value"]["id"];
      // Handle annotations written as ``typing.Tuple[...]`` (or any aliased
      // typing module): the Subscript base is an Attribute, not a Name.
      else if (
        ann.contains("value") && ann["value"].contains("_type") &&
        ann["value"]["_type"] == "Attribute" && ann["value"].contains("attr"))
        var_type_str = ann["value"]["attr"];

      // Preserve concrete tuple element types for Tuple[...] annotations
      // instead of resolving to the typing.Tuple class type.
      if (var_type_str == "Tuple" || var_type_str == "tuple")
      {
        var_typet = get_type_from_annotation(ann, var_node);
        return {var_type_str, var_typet};
      }
    }
    else if (
      ann.contains("_type") && ann["_type"] == "Attribute" &&
      ann.contains("attr"))
      var_type_str = ann["attr"];
    else if (ann.contains("id"))
      var_type_str = var_node["annotation"]["id"];
    else if (ann.contains("_type") && ann["_type"] == "BinOp")
    {
      // Handle union types (e.g., re.Match[str] | None)
      // Use get_type_from_annotation which has proper union handling
      var_typet = get_type_from_annotation(ann, var_node);
      return {var_type_str, var_typet};
    }

    if (var_type_str.empty())
      return {var_type_str, var_typet};

    // A spelled `Callable[[A], R]` keeps its signature, so a call through the
    // variable recovers R. A bare one -- what the annotation pass infers for a
    // variable bound to a function value -- resolves to a pointer whose code
    // type returns void, leaving that call nondet: worse than no annotation at
    // all, since an unannotated binding takes the callee's own return type. So
    // defer to the RHS instead (#6640).
    if (var_type_str == "Callable")
      return {
        var_type_str,
        ann.contains("slice") ? get_callable_type(ann, var_node) : typet()};

    // User-defined classes named "list"/"List" or "dict"/"Dict" take priority
    // over the built-in types when used as a plain Name annotation.
    if (names_builtin(var_type_str, "dict", "Dict", *ast_json))
      var_typet = dict_handler_->get_dict_struct_type();
    else if (names_builtin(var_type_str, "list", "List", *ast_json))
      var_typet = type_handler_.get_list_type();
    else
      var_typet = type_handler_.get_typet(var_type_str, type_size);
  }

  return {var_type_str, var_typet};
}

exprt python_converter::create_lhs_expression(
  const nlohmann::json &target,
  symbolt *lhs_symbol,
  const locationt &location)
{
  exprt lhs;
  const auto &target_type = target["_type"];

  if (target_type == "Attribute" || target_type == "Subscript")
  {
    is_converting_lhs = true;
    const nlohmann::json *saved_store_target = lhs_store_target_;
    lhs_store_target_ = &target;
    lhs = get_expr(target);
    lhs_store_target_ = saved_store_target;
    is_converting_lhs = false;
  }
  else
    lhs = symbol_expr(*lhs_symbol);

  lhs.location() = location;
  return lhs;
}

void python_converter::handle_assignment_type_adjustments(
  symbolt *lhs_symbol,
  exprt &lhs,
  exprt &rhs,
  const std::string &lhs_type,
  const nlohmann::json &ast_node,
  bool is_ctor_call)
{
  const bool has_annotation =
    ast_node.contains("annotation") && !ast_node["annotation"].is_null();

  // Don't rewrite lhs_symbol's type for a subscript target.
  if (assignment_target_is_subscript(ast_node))
    return;

  // Assigning to a struct member (self.attr = value): an unannotated parameter
  // is typed as the any-type carrier (void*, i.e. a pointer whose subtype is
  // empty) but holds an integer value round-tripped as (int*)n. Writing that
  // into a non-pointer integer field aborts with2t's type-compat assertion.
  // Cast the any-type RHS to the member's declared integer type, recovering the
  // integer. Restricted to the empty-subtype carrier (matching is_any_ptr in
  // converter_binop): a genuine pointer value (None, a class instance, a list)
  // is left untouched so it is not silently reinterpreted as an integer, and a
  // float member (which would need a bit-reinterpretation) is left unchanged.
  if (
    lhs.id() == "member" && rhs.type().is_pointer() &&
    rhs.type().subtype().id() == "empty" &&
    (lhs.type().is_signedbv() || lhs.type().is_unsignedbv()))
  {
    rhs = typecast_exprt(rhs, lhs.type());
    return;
  }

  // Handle assignment of function to function pointer variable
  if (
    lhs.type().is_pointer() && lhs.type().subtype().is_code() &&
    rhs.type().is_code() && rhs.is_symbol())
  {
    rhs = address_of_exprt(rhs);
    if (lhs_symbol && !is_ctor_call)
      lhs_symbol->set_value(rhs);
    return;
  }

  // When a variable is assigned a function pointer returned from a
  // higher-order lambda call (e.g. `inner = outer(5)` or `inner:int =
  // outer(5)`), override any incorrect annotation (void*, int, …) with the
  // concrete function pointer type so the subsequent indirect call resolves
  // correctly instead of crashing in to_code_type.
  if (
    lhs_symbol && !is_ctor_call && rhs.type().is_pointer() &&
    rhs.type().subtype().is_code() &&
    !(lhs.type().is_pointer() && lhs.type().subtype().is_code()))
  {
    lhs_symbol->set_type(rhs.type());
    lhs.type() = rhs.type();
  }

  // Handle lambda assignments
  if (lambda_handler_->is_lambda_assignment(ast_node) && rhs.is_symbol())
  {
    lambda_handler_->handle_lambda_assignment(lhs_symbol, lhs, rhs);
    return;
  }
  // Handle tuple assignments with generic tuple annotation
  else if (
    lhs_symbol && lhs_symbol->get_type().id() == "empty" &&
    rhs.type().id() == "struct")
  {
    const struct_typet &rhs_struct = to_struct_type(rhs.type());

    // Check if RHS is a tuple (has tuple tag pattern)
    if (rhs_struct.tag().as_string().find("tag-tuple") == 0)
    {
      // Update symbol type from empty to concrete tuple type
      lhs_symbol->set_type(rhs.type());
      lhs.type() = rhs.type();
      lhs_symbol->set_value(rhs);
    }
  }
  else if (lhs_symbol)
  {
    // Handle explicit Any-typed annotation assignments
    // Only applies when the user explicitly wrote `from typing import Any`
    // and annotated `x: Any = value`.
    // Preprocessor-generated AnnAssign nodes
    // with Any annotation are excluded.
    // Constructor calls must fall through to the regular ctor machinery:
    // returning early here would emit no constructor call at all, leaving a
    // nil assignment that crashes the SMT encoder.
    if (
      ast_node.contains("_type") && ast_node["_type"] == "AnnAssign" &&
      !ast_node.value("_inferred_annotation", false) &&
      !ast_node.value("esbmc_synthesized", false) && has_annotation &&
      ast_node["annotation"].contains("id") &&
      ast_node["annotation"]["id"] == "Any" && lhs.type().is_pointer() &&
      !is_ctor_call && !rhs.type().is_code() &&
      json_utils::is_imported_from(*ast_json, "typing", "Any"))
    {
      if (rhs.type().is_array())
      {
        rhs = string_handler_.get_array_base_address(rhs);
        if (rhs.type() != lhs.type())
          rhs = typecast_exprt(rhs, lhs.type());
      }
      else if (rhs.type().is_struct() && lhs_symbol->get_value().is_nil())
      {
        // A struct value (e.g. a tuple read out of a dict) cannot round-trip
        // through the void* cast below — every later component access would
        // misread. Any is not a constraint, so adopt the rhs type. Only on
        // the first binding: a re-annotation (`x: Any = 5; ...; x: Any =
        // (1, 2)`) must not retype uses already emitted at the old type.
        lhs_symbol->set_type(rhs.type());
        lhs.type() = rhs.type();
      }
      else if (!rhs.type().is_pointer() && !rhs.type().is_empty())
      {
        if (rhs.type().is_floatbv())
        {
          unsigned width =
            static_cast<const bv_typet &>(rhs.type()).get_width();
          exprt bitcast("bitcast", unsignedbv_typet(width));
          bitcast.copy_to_operands(rhs);
          rhs = bitcast;
        }
        rhs = typecast_exprt(rhs, lhs.type());
      }
      if (!rhs.type().is_empty() && !is_ctor_call)
        lhs_symbol->set_value(rhs);
      return;
    }
    // Handle string-to-string variable assignments
    if (lhs_type == "str" && rhs.is_symbol())
    {
      symbolt *rhs_symbol = symbol_table_.find_symbol(rhs.identifier());
      if (
        rhs_symbol && rhs_symbol->get_value().is_constant() &&
        rhs_symbol->get_value().type().is_array())
      {
        rhs = rhs_symbol->get_value();
        lhs_symbol->set_type(rhs.type());
        lhs.type() = rhs.type();
      }
    }
    // Array to pointer decay
    else if (lhs.type().id().empty() && rhs.type().is_array())
    {
      // TODO: This case is used to infer an unknown type.
      // Should we model it uniformly using char* ?
      const typet &element_type = to_array_type(rhs.type()).subtype();
      typet pointer_type = gen_pointer_type(element_type);
      lhs_symbol->set_type(pointer_type);
      lhs.type() = pointer_type;
      rhs = string_handler_.get_array_base_address(rhs);
    }
    else if (
      lhs.type().is_pointer() && rhs.type().is_array() &&
      lhs.type() != type_handler_.get_list_type())
    {
      // Array to pointer typecast
      // skip the list type until the list is moved to symex
      // TODO: remove list condition
      rhs = string_handler_.get_array_base_address(rhs);
    }
    // String and list type size adjustments
    else if (
      lhs_type == "str" || lhs_type == "chr" || lhs_type == "ord" ||
      lhs_type == "list" || rhs.type().is_array() ||
      rhs.type() == type_handler_.get_list_type())
    {
      if (!rhs.type().is_empty())
      {
        // An RHS typed any_type() (void*) means "type unknown" — e.g. an
        // instance attribute of a class the scanner could not resolve — not
        // "the object is a void*". Demoting a list-annotated LHS to void*
        // makes the for-loop lowering fall back to the array protocol (raw
        // __ESBMC_get_object_size + pointer indexing), which aborts symex on
        // the PyListObj struct (#4805). Keep the annotated list type and
        // cast the RHS instead.
        if (
          lhs.type() == type_handler_.get_list_type() &&
          rhs.type() == any_type())
        {
          rhs = typecast_exprt(rhs, lhs.type());
        }
        else
        {
          // Prevent type change from scalar (int/float/bool) to string/array
          // when a prior declaration exists with the scalar type, as this
          // creates a type inconsistency in the GOTO program. A void-typed
          // symbol (annotation pass could not infer, e.g. a tuple-unpack
          // target) holds no prior scalar, so adopting the array type is the
          // only consistent choice (#5571).
          bool is_incompatible =
            rhs.type().is_array() && !lhs_symbol->get_type().is_array() &&
            !lhs_symbol->get_type().is_pointer() &&
            !lhs_symbol->get_type().id().empty() &&
            lhs_symbol->get_type().id() != "empty" &&
            !lhs_symbol->get_type().is_nil() &&
            lhs_symbol->get_type() != type_handler_.get_list_type();
          if (!is_incompatible)
          {
            lhs_symbol->set_type(rhs.type());
            lhs.type() = rhs.type();
          }
        }
      }
    }
    else if (rhs.type() == none_type())
    {
      // None/Optional unification (#4796), step C: when the lvalue is already a
      // class reference (`Class*` — a migrated instance), keep that type and
      // store a typed NULL, so a later `x = Class(...)` construction allocates
      // a properly-sized object. Retyping it to none_type() (pointer-to-bool)
      // would shrink the pointee and corrupt the allocation.
      if (is_user_class_pointer(lhs.type()))
        rhs = typecast_exprt(rhs, lhs.type());
      else
      {
        // Adjust pointer_type() to pointer_typet(empty_typet())
        lhs_symbol->set_type(rhs.type());
        lhs.type() = rhs.type();
      }
    }
    // No annotation or an inferred Any: propagate rhs type to lhs. An "Any"
    // annotation counts as inferred when flagged by the annotation pass,
    // when the preprocessor marked it as synthesized (e.g. the items()-loop
    // value variable `v: Any = ESBMC_vals_N[i]`), or when the module never
    // imports typing.Any (then no top-level user annotation can be a live
    // Any). Without this, the declared void* overrides a concrete rhs type —
    // a tuple dict-value read then misfolds every component access (#5444
    // latent item).
    else if (
      (!has_annotation ||
       (ast_node["annotation"].value("id", std::string()) == "Any" &&
        (ast_node.value("_inferred_annotation", false) ||
         ast_node.value("esbmc_synthesized", false) ||
         !json_utils::is_imported_from(*ast_json, "typing", "Any")))) &&
      !rhs.type().is_empty() && lhs.type() != rhs.type() &&
      !rhs.type().is_code() &&
      !(rhs.type().is_pointer() && rhs.type().subtype().id() == "empty") &&
      // Never re-type a tuple-struct symbol to a non-tuple in place — it
      // corrupts already-emitted tuple member reads (see the fresh-alias
      // gate in get_var_assign, which handles such rebinds on the
      // straight-line spine); elsewhere keep the struct type.
      !(tuple_handler_->is_tuple_type(lhs_symbol->get_type()) &&
        !tuple_handler_->is_tuple_type(rhs.type())))
    {
      lhs_symbol->set_type(rhs.type());
      lhs.type() = rhs.type();
    }

    if (!rhs.type().is_empty() && !is_ctor_call)
      lhs_symbol->set_value(rhs);
  }
}

void python_converter::handle_array_unpacking(
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  exprt &rhs,
  codet &target_block)
{
  const auto &targets = target["elts"];

  for (size_t i = 0; i < targets.size(); i++)
  {
    if (targets[i]["_type"] != "Name")
    {
      throw std::runtime_error(
        "Array unpacking only supports simple names, not " +
        targets[i]["_type"].get<std::string>());
    }

    std::string var_name = targets[i]["id"].get<std::string>();
    symbol_id var_sid = create_symbol_id();
    var_sid.set_object(var_name);

    symbolt *var_symbol = find_symbol(var_sid.to_string());

    if (!var_symbol)
    {
      locationt loc = get_location_from_decl(targets[i]);
      typet elem_type = rhs.type().subtype();

      symbolt new_symbol = create_symbol(
        loc.get_file().as_string(),
        var_name,
        var_sid.to_string(),
        loc,
        elem_type);
      new_symbol.lvalue = true;
      new_symbol.file_local = true;
      new_symbol.is_extern = false;
      var_symbol = symbol_table_.move_symbol_to_context(new_symbol);
    }

    // Create subscript: rhs[i]. V.3: IREP2 index access (exact round-trip of
    // index_exprt); this path runs only when rhs is array-typed (see the
    // is_array() guard at the call site), so the index2t source precondition
    // holds.
    exprt index_expr = from_integer(i, size_type());
    expr2tc rhs2, idx2;
    migrate_expr(rhs, rhs2);
    migrate_expr(index_expr, idx2);
    exprt subscript = migrate_expr_back(
      index2tc(migrate_type(rhs.type().subtype()), rhs2, idx2));

    code_assignt assign(symbol_expr(*var_symbol), subscript);
    assign.location() = get_location_from_decl(ast_node);
    target_block.copy_to_operands(assign);
  }
}

void python_converter::handle_list_literal_unpacking(
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  codet &target_block)
{
  const auto &value_node = ast_node["value"];
  const auto &elements = value_node["elts"];
  const auto &targets = target["elts"];

  // Find starred target (if any)
  int star_idx = -1;
  for (size_t i = 0; i < targets.size(); i++)
  {
    if (targets[i]["_type"] == "Starred")
    {
      star_idx = static_cast<int>(i);
      break;
    }
  }

  if (star_idx < 0)
  {
    // No starred target: strict size check
    if (elements.size() != targets.size())
    {
      throw std::runtime_error(
        "Cannot unpack list: expected " + std::to_string(targets.size()) +
        " values, got " + std::to_string(elements.size()));
    }
  }
  else
  {
    size_t non_star_count = targets.size() - 1;
    if (elements.size() < non_star_count)
    {
      throw std::runtime_error(
        "Cannot unpack list: not enough values (expected at least " +
        std::to_string(non_star_count) + ", got " +
        std::to_string(elements.size()) + ")");
    }
  }

  size_t before_star =
    (star_idx >= 0) ? static_cast<size_t>(star_idx) : targets.size();
  size_t after_star =
    (star_idx >= 0) ? targets.size() - static_cast<size_t>(star_idx) - 1 : 0;

  // Assign targets before the star
  for (size_t i = 0; i < before_star; i++)
  {
    if (targets[i]["_type"] != "Name")
    {
      throw std::runtime_error(
        "List unpacking only supports simple names, not " +
        targets[i]["_type"].get<std::string>());
    }

    std::string var_name = targets[i]["id"].get<std::string>();
    symbol_id var_sid = create_symbol_id();
    var_sid.set_object(var_name);

    symbolt *var_symbol = find_symbol(var_sid.to_string());

    is_converting_rhs = true;
    exprt elem_expr = get_expr(elements[i]);
    is_converting_rhs = false;

    if (!var_symbol)
    {
      locationt loc = get_location_from_decl(targets[i]);

      symbolt new_symbol = create_symbol(
        loc.get_file().as_string(),
        var_name,
        var_sid.to_string(),
        loc,
        elem_expr.type());
      new_symbol.lvalue = true;
      new_symbol.file_local = true;
      new_symbol.is_extern = false;
      var_symbol = symbol_table_.move_symbol_to_context(new_symbol);
    }

    code_assignt assign(symbol_expr(*var_symbol), elem_expr);
    assign.location() = get_location_from_decl(ast_node);
    target_block.copy_to_operands(assign);
  }

  // Handle starred target: collect remaining elements into a list
  if (star_idx >= 0)
  {
    const auto &starred_node = targets[static_cast<size_t>(star_idx)];
    const auto &star_value = starred_node["value"];

    if (star_value["_type"] != "Name")
    {
      throw std::runtime_error(
        "Starred unpacking only supports simple names, not " +
        star_value["_type"].get<std::string>());
    }

    // Build a synthetic list JSON node with the starred elements
    nlohmann::json star_list_node = value_node;
    star_list_node["_type"] = "List";
    star_list_node["elts"] = nlohmann::json::array();
    for (size_t j = before_star; j < elements.size() - after_star; j++)
      star_list_node["elts"].push_back(elements[j]);

    python_list star_list(*this, star_list_node);
    exprt list_expr = star_list.get();

    std::string var_name = star_value["id"].get<std::string>();
    symbol_id var_sid = create_symbol_id();
    var_sid.set_object(var_name);

    symbolt *var_symbol = find_symbol(var_sid.to_string());

    if (!var_symbol)
    {
      locationt loc = get_location_from_decl(star_value);

      symbolt new_symbol = create_symbol(
        loc.get_file().as_string(),
        var_name,
        var_sid.to_string(),
        loc,
        list_expr.type());
      new_symbol.lvalue = true;
      new_symbol.file_local = true;
      new_symbol.is_extern = false;
      var_symbol = symbol_table_.move_symbol_to_context(new_symbol);
    }

    code_assignt assign(symbol_expr(*var_symbol), list_expr);
    assign.location() = get_location_from_decl(ast_node);
    target_block.copy_to_operands(assign);
  }

  // Assign targets after the star (from the end)
  for (size_t i = 0; i < after_star; i++)
  {
    size_t target_idx = static_cast<size_t>(star_idx) + 1 + i;
    size_t elem_idx = elements.size() - after_star + i;

    if (targets[target_idx]["_type"] != "Name")
    {
      throw std::runtime_error(
        "List unpacking only supports simple names, not " +
        targets[target_idx]["_type"].get<std::string>());
    }

    std::string var_name = targets[target_idx]["id"].get<std::string>();
    symbol_id var_sid = create_symbol_id();
    var_sid.set_object(var_name);

    symbolt *var_symbol = find_symbol(var_sid.to_string());

    is_converting_rhs = true;
    exprt elem_expr = get_expr(elements[elem_idx]);
    is_converting_rhs = false;

    if (!var_symbol)
    {
      locationt loc = get_location_from_decl(targets[target_idx]);

      symbolt new_symbol = create_symbol(
        loc.get_file().as_string(),
        var_name,
        var_sid.to_string(),
        loc,
        elem_expr.type());
      new_symbol.lvalue = true;
      new_symbol.file_local = true;
      new_symbol.is_extern = false;
      var_symbol = symbol_table_.move_symbol_to_context(new_symbol);
    }

    code_assignt assign(symbol_expr(*var_symbol), elem_expr);
    assign.location() = get_location_from_decl(ast_node);
    target_block.copy_to_operands(assign);
  }
}

exprt python_converter::get_rhs_with_dict_resolution(
  const nlohmann::json &ast_node,
  const typet &target_type)
{
  if (!type_utils::is_dict_subscript(ast_node["value"]))
    return get_expr(ast_node["value"]);

  // Check if we need special dict subscript handling for typed variables
  // Including list type and dict type
  typet list_type = type_handler_.get_list_type();
  if (
    !target_type.is_signedbv() && !target_type.is_unsignedbv() &&
    !target_type.is_bool() && target_type != list_type &&
    !dict_handler_->is_dict_type(target_type))
  {
    return get_expr(ast_node["value"]);
  }

  exprt dict_expr = get_expr(ast_node["value"]["value"]);
  if (
    !dict_expr.type().is_struct() ||
    !dict_handler_->is_dict_type(dict_expr.type()))
    return get_expr(ast_node["value"]);

  return dict_handler_->handle_dict_subscript(
    dict_expr, ast_node["value"]["slice"], target_type);
}

std::string
python_converter::resolve_name_symbol_id(const std::string &name) const
{
  symbol_id sid = create_symbol_id();
  sid.set_object(name);
  if (symbol_table_.find_symbol(sid.to_string()) != nullptr)
    return sid.to_string();

  sid.set_function("");
  if (symbol_table_.find_symbol(sid.to_string()) != nullptr)
    return sid.to_string();

  return "";
}

std::string
python_converter::root_name_from_subscript(const nlohmann::json &node) const
{
  if (!node.is_object() || !node.contains("_type"))
    return "";

  if (node["_type"] == "Name" && node.contains("id"))
    return node["id"].get<std::string>();

  if (node["_type"] == "Subscript" && node.contains("value"))
    return root_name_from_subscript(node["value"]);

  if (node["_type"] == "Attribute" && node.contains("value"))
    return root_name_from_subscript(node["value"]);

  return "";
}

static bool json_contains_slice_node(const nlohmann::json &node)
{
  if (!node.is_object() && !node.is_array())
    return false;

  if (node.is_object())
  {
    if (node.value("_type", "") == "Slice")
      return true;

    for (auto it = node.begin(); it != node.end(); ++it)
      if (json_contains_slice_node(it.value()))
        return true;
  }
  else
  {
    for (const auto &elem : node)
      if (json_contains_slice_node(elem))
        return true;
  }

  return false;
}

static bool json_literal_contains_boolean(const nlohmann::json &node)
{
  if (!node.is_object() && !node.is_array())
    return false;

  if (node.is_object())
  {
    if (
      node.value("_type", "") == "Constant" && node.contains("value") &&
      node["value"].is_boolean())
      return true;

    for (auto it = node.begin(); it != node.end(); ++it)
      if (json_literal_contains_boolean(it.value()))
        return true;
  }
  else
  {
    for (const auto &elem : node)
      if (json_literal_contains_boolean(elem))
        return true;
  }

  return false;
}

bool python_converter::is_basic_numpy_view_subscript(
  const nlohmann::json &node) const
{
  if (
    !node.is_object() || node.value("_type", "") != "Subscript" ||
    !node.contains("value") || !node.contains("slice"))
    return false;

  auto is_boolean_mask_index = [&](const nlohmann::json &idx) {
    nlohmann::json value = idx;
    if (idx.value("_type", "") == "Name" && idx.contains("id"))
    {
      nlohmann::json decl =
        json_utils::find_var_decl(idx["id"], current_func_name_, *ast_json);
      if (decl.contains("value") && decl["value"].is_object())
        value = decl["value"];
    }

    if (
      value.value("_type", "") != "Call" || !value.contains("func") ||
      !value["func"].is_object() ||
      value["func"].value("_type", "") != "Attribute" ||
      value["func"].value("attr", "") != "array" || !value.contains("args") ||
      !value["args"].is_array() || value["args"].empty())
      return false;

    return json_literal_contains_boolean(value["args"][0]);
  };

  auto is_basic_index = [&](const nlohmann::json &idx) {
    if (is_boolean_mask_index(idx))
      return false;
    const std::string type = idx.value("_type", "");
    return type == "Constant" || type == "UnaryOp" || type == "Name" ||
           type == "Slice";
  };

  const nlohmann::json &slice = node["slice"];
  if (slice.value("_type", "") == "Tuple" && slice.contains("elts"))
  {
    for (const auto &idx : slice["elts"])
      if (!is_basic_index(idx))
        return false;
    return true;
  }

  return is_basic_index(slice);
}

bool python_converter::is_numpy_array_constructor_expr(
  const nlohmann::json &node) const
{
  if (
    !node.is_object() || node.value("_type", "") != "Call" ||
    !node.contains("func") || !node["func"].is_object() ||
    node["func"].value("_type", "") != "Attribute" ||
    !node["func"].contains("value") || !node["func"]["value"].is_object() ||
    node["func"]["value"].value("_type", "") != "Name")
    return false;

  const std::string module_name = node["func"]["value"].value("id", "");
  if (!is_imported_numpy_module_alias(*ast_json, module_name))
    return false;

  static const std::set<std::string> constructors = {
    "array",
    "zeros",
    "ones",
    "full",
    "empty",
    "arange",
    "eye",
    "identity",
    "linspace"};
  return constructors.count(node["func"].value("attr", "")) != 0;
}

// a.<method>(...) on a tracked numpy array is only ever resolved as numpy
// when it takes the `np.<method>(a, ...)`-shaped AST a module-form call
// would have produced: name lookup has no other route to the numpy
// operational model for a method call. Centralised here so every call site
// that converts a Call node (assignment RHS, or any other expression
// context) shares one recogniser instead of growing its own copy that can
// drift out of sync with the constructor/method lists above.
static bool is_method_call_node_shape(const nlohmann::json &call_node)
{
  return call_node.is_object() && call_node.value("_type", "") == "Call" &&
         call_node.contains("func") && call_node["func"].is_object() &&
         call_node["func"].value("_type", "") == "Attribute" &&
         call_node["func"].contains("value");
}

bool python_converter::method_base_is_imported_module(
  const std::string &method_base_name) const
{
  return method_base_name == "np" || method_base_name == "numpy" ||
         (!method_base_name.empty() && is_imported_module(method_base_name));
}

// A method name like sum()/max()/min() is not exclusive to numpy (e.g.
// Decimal.max(), a plain module-level function called through an aliased
// import); only rewrite when the receiver is actually a tracked numpy
// array.
bool python_converter::method_base_is_tracked_numpy_array(
  const std::string &method_base_name) const
{
  if (method_base_name.empty())
    return false;
  const std::string method_base_id = resolve_name_symbol_id(method_base_name);
  return !method_base_id.empty() &&
         numpy_array_symbols_.count(method_base_id) != 0;
}

std::tuple<bool, std::string, nlohmann::json, bool, bool>
python_converter::classify_numpy_method_call(
  const nlohmann::json &call_node) const
{
  if (!is_method_call_node_shape(call_node))
    return {false, "", nlohmann::json(), false, false};

  const std::string method_name = call_node["func"].value("attr", "");
  const nlohmann::json &method_base = call_node["func"]["value"];
  const std::string method_base_name =
    method_base.value("_type", "") == "Name" && method_base.contains("id")
      ? method_base["id"].get<std::string>()
      : std::string();
  const bool receiver_is_rewritable =
    !method_base_is_imported_module(method_base_name) &&
    method_base_is_tracked_numpy_array(method_base_name);

  // transpose()/reshape()/ravel() are view-like (see is_numpy_view_copy_expr,
  // which handles them separately); flatten()/sum()/mean()/min()/max()/
  // prod()/std()/var() are not, but the method form still needs the same
  // np.<name>(a, ...)-shaped rewrite to dispatch to the existing
  // np.<name>() handler.
  static const std::set<std::string> dispatch_rewrite_methods = {
    "transpose",
    "reshape",
    "ravel",
    "flatten",
    "sum",
    "mean",
    "min",
    "max",
    "prod",
    "std",
    "var",
    "diagonal",
    "argmin",
    "argmax",
    "argsort",
    "searchsorted"};
  const bool supported_dispatch_rewrite_method =
    receiver_is_rewritable && dispatch_rewrite_methods.count(method_name) != 0;
  const bool supported_copy_method =
    receiver_is_rewritable && method_name == "copy";

  return {
    true,
    method_name,
    method_base,
    supported_copy_method,
    supported_dispatch_rewrite_method};
}

nlohmann::json python_converter::build_numpy_method_rewrite_node(
  const nlohmann::json &call_node,
  const std::string &method_name,
  const nlohmann::json &method_base) const
{
  std::string numpy_alias = "np";
  for (const auto &entry : imported_modules)
  {
    if (entry.second == "numpy")
    {
      numpy_alias = entry.first;
      break;
    }
  }

  nlohmann::json module_name;
  module_name["_type"] = "Name";
  module_name["id"] = numpy_alias;
  module_name["ctx"] = {{"_type", "Load"}};
  copy_location_fields_from_decl(call_node, module_name);

  nlohmann::json rewritten;
  rewritten["_type"] = "Call";
  rewritten["func"] = {
    {"_type", "Attribute"},
    {"value", module_name},
    {"attr", method_name},
    {"ctx", {{"_type", "Load"}}}};
  rewritten["args"] = nlohmann::json::array({method_base});
  if (call_node.contains("args") && call_node["args"].is_array())
    for (const auto &arg : call_node["args"])
      rewritten["args"].push_back(arg);
  rewritten["keywords"] = call_node.value("keywords", nlohmann::json::array());
  // numpy.reshape(a, newshape, order='C') has no split-dimension form
  // (a third positional argument is `order`, not another dimension);
  // only the method form a.reshape(d1, d2, ...) is equivalent to
  // a.reshape((d1, d2, ...)). Mark this rewrite so the reshape handler
  // can tell the two shapes apart and reject a genuine
  // np.reshape(a, 2, 3) call instead of silently accepting it.
  rewritten["_numpy_method_form"] = true;
  copy_location_fields_from_decl(call_node, rewritten);
  copy_location_fields_from_decl(call_node, rewritten["func"]);
  return rewritten;
}

std::optional<nlohmann::json> python_converter::rewrite_numpy_method_call_node(
  const nlohmann::json &call_node) const
{
  const auto
    [is_method_call,
     method_name,
     method_base,
     supported_copy_method,
     supported_dispatch_rewrite_method] = classify_numpy_method_call(call_node);
  if (!is_method_call)
    return std::nullopt;

  if (supported_copy_method)
  {
    nlohmann::json copied = method_base;
    copied["_numpy_copy_method"] = true;
    return copied;
  }

  if (!supported_dispatch_rewrite_method)
    return std::nullopt;

  return build_numpy_method_rewrite_node(call_node, method_name, method_base);
}

bool python_converter::is_numpy_view_copy_call_node(
  const nlohmann::json &node) const
{
  if (
    node.value("_type", "") != "Call" || !node.contains("func") ||
    !node["func"].is_object() || node["func"].value("_type", "") != "Attribute")
    return false;

  static const std::set<std::string> view_functions = {
    "transpose", "reshape", "ravel", "diagonal"};
  return view_functions.count(node["func"].value("attr", "")) != 0;
}

bool python_converter::is_numpy_view_copy_expr(const nlohmann::json &node) const
{
  if (!node.is_object())
    return false;

  if (is_basic_numpy_view_subscript(node))
    return true;

  if (
    node.value("_type", "") == "Attribute" && node.value("attr", "") == "T" &&
    node.contains("value"))
    return !root_name_from_subscript(node["value"]).empty();

  if (!is_numpy_view_copy_call_node(node))
    return false;

  if (node.contains("args") && node["args"].is_array() && !node["args"].empty())
    return !root_name_from_subscript(node["args"][0]).empty();

  return node["func"].contains("value") &&
         !root_name_from_subscript(node["func"]["value"]).empty();
}

std::string python_converter::root_name_from_numpy_view_copy_expr(
  const nlohmann::json &node) const
{
  if (!node.is_object())
    return "";

  if (is_basic_numpy_view_subscript(node))
    return root_name_from_subscript(node["value"]);

  if (
    node.value("_type", "") == "Attribute" && node.value("attr", "") == "T" &&
    node.contains("value"))
    return root_name_from_subscript(node["value"]);

  if (is_numpy_view_copy_call_node(node))
  {
    if (
      node.contains("args") && node["args"].is_array() && !node["args"].empty())
      return root_name_from_subscript(node["args"][0]);

    if (node["func"].contains("value"))
      return root_name_from_subscript(node["func"]["value"]);
  }

  return "";
}

bool python_converter::is_tracked_numpy_view_name_node(
  const nlohmann::json &node)
{
  if (node.value("_type", "") != "Name" || !node.contains("id"))
    return false;

  const std::string id = resolve_name_symbol_id(node["id"].get<std::string>());
  return !id.empty() && is_tracked_numpy_view_id(id);
}

bool python_converter::is_basic_numpy_view_subscript_escape(
  const nlohmann::json &node)
{
  if (!is_basic_numpy_view_subscript(node))
    return false;

  const std::string root_name = root_name_from_subscript(node["value"]);
  if (root_name.empty())
    return false;

  const std::string root_id = resolve_name_symbol_id(root_name);
  if (root_id.empty())
    return false;

  bool root_is_numpy_view_source = numpy_array_symbols_.count(root_id) != 0 ||
                                   is_tracked_numpy_view_id(root_id);
  if (!root_is_numpy_view_source)
  {
    const symbolt *root_symbol = symbol_table_.find_symbol(root_id);
    if (root_symbol != nullptr)
    {
      const namespacet ns(symbol_table_);
      const typet root_type = ns.follow(root_symbol->get_type());
      root_is_numpy_view_source =
        root_type.is_array() ||
        (root_type.is_pointer() && ns.follow(root_type.subtype()).is_array());
    }
  }
  if (!root_is_numpy_view_source)
    return false;

  code_blockt scratch_block;
  code_blockt *saved_block = current_block;
  exprt *saved_lhs = current_lhs;
  current_block = &scratch_block;
  current_lhs = nullptr;
  exprt probe;
  try
  {
    probe = get_expr(node);
  }
  catch (...)
  {
    current_block = saved_block;
    current_lhs = saved_lhs;
    throw;
  }
  current_block = saved_block;
  current_lhs = saved_lhs;
  return !contains_cpp_throw(probe) && probe.type().is_array();
}

bool python_converter::contains_tracked_numpy_view_object(
  const nlohmann::json &node)
{
  const std::string node_type = node.value("_type", "");
  if (
    node_type == "GeneratorExp" || node_type == "ListComp" ||
    node_type == "SetComp" || node_type == "DictComp")
    return false;

  if (is_tracked_numpy_view_name_node(node))
    return true;

  if (is_basic_numpy_view_subscript_escape(node))
    return true;

  if (
    node_type == "Subscript" && node.contains("value") &&
    node.contains("slice") && !json_contains_slice_node(node["slice"]) &&
    contains_tracked_numpy_view_name(node["value"]))
    return contains_tracked_numpy_view_name(node["slice"]);

  for (auto it = node.begin(); it != node.end(); ++it)
    if (contains_tracked_numpy_view_name(it.value()))
      return true;

  return false;
}

bool python_converter::contains_tracked_numpy_view_name(
  const nlohmann::json &node)
{
  if (node.is_object())
    return contains_tracked_numpy_view_object(node);

  if (!node.is_array())
    return false;

  for (const auto &elem : node)
    if (contains_tracked_numpy_view_name(elem))
      return true;

  return false;
}

void python_converter::reject_numpy_view_mutating_method_call(
  const nlohmann::json &node)
{
  if (
    !node.is_object() || node.value("_type", "") != "Call" ||
    !node.contains("func") || !node["func"].is_object() ||
    node["func"].value("_type", "") != "Attribute" ||
    !node["func"].contains("value"))
    return;

  static const std::set<std::string> mutating_methods = {"fill", "sort"};
  if (mutating_methods.count(node["func"].value("attr", "")) == 0)
    return;

  const std::string root_name = root_name_from_subscript(node["func"]["value"]);
  if (root_name.empty())
    return;

  const std::string root_id = resolve_name_symbol_id(root_name);
  if (root_id.empty())
    return;

  if (numpy_view_copy_sources_.count(root_id) != 0)
    throw std::runtime_error(
      "TypeError: writing through a copied numpy view is not supported");
}

bool python_converter::is_tracked_numpy_view_id(
  const std::string &symbol_id) const
{
  return numpy_view_copy_sources_.count(symbol_id) != 0 ||
         numpy_transpose_view_info_.count(symbol_id) != 0 ||
         numpy_reshape_view_info_.count(symbol_id) != 0;
}

void python_converter::reject_nonconstant_numpy_view_write(
  const nlohmann::json &target) const
{
  const std::string root_name = root_name_from_subscript(target);
  const std::string root_id = resolve_name_symbol_id(root_name);
  if (!root_id.empty() && is_tracked_numpy_view_id(root_id))
    throw std::runtime_error(
      "TypeError: writing through a numpy view with a non-constant index is "
      "not supported");
}

std::optional<std::vector<nlohmann::json>>
python_converter::build_numpy_nditer_logical_elements(
  const nlohmann::json &arg) const
{
  if (
    !arg.is_object() || arg.value("_type", "") != "Name" ||
    !arg.contains("id") || !arg["id"].is_string())
    return std::nullopt;

  const std::string root_name = arg["id"].get<std::string>();
  const std::string root_id = resolve_name_symbol_id(root_name);
  if (root_id.empty())
    return std::nullopt;

  std::optional<std::vector<std::size_t>> shape =
    get_numpy_nditer_logical_shape(root_id);
  if (!shape || shape->empty() || shape->size() > 2)
    return std::nullopt;

  std::vector<nlohmann::json> result;
  if (shape->size() == 1)
  {
    for (std::size_t i = 0; i < (*shape)[0]; ++i)
      result.push_back(
        numpy_subscript_node(root_name, std::vector<std::size_t>{i}));
    return result;
  }

  for (std::size_t i = 0; i < (*shape)[0]; ++i)
    for (std::size_t j = 0; j < (*shape)[1]; ++j)
      result.push_back(
        numpy_subscript_node(root_name, std::vector<std::size_t>{i, j}));
  return result;
}

std::optional<exprt> python_converter::build_numpy_descriptor_materialized_list(
  const nlohmann::json &arg,
  const bool nested)
{
  auto materialized = build_numpy_descriptor_materialized_elements(
    arg,
    "TypeError: numpy.ndarray.tolist() currently supports rank 1 or 2 arrays");
  if (!materialized)
    return std::nullopt;

  const std::vector<std::size_t> &shape = materialized->first;
  const std::vector<exprt> &elems = materialized->second;

  nlohmann::json list_node{
    {"_type", "List"}, {"elts", nlohmann::json::array()}};
  python_list list(*this, list_node);
  if (!nested || shape.size() == 1)
    return list.build_list_from_exprs(elems);

  std::vector<exprt> rows;
  const std::size_t cols = shape[1];
  for (std::size_t row = 0; row < shape[0]; ++row)
  {
    const auto first = elems.begin() + static_cast<std::ptrdiff_t>(row * cols);
    const auto last = first + static_cast<std::ptrdiff_t>(cols);
    const std::vector<exprt> row_elems(first, last);
    rows.push_back(list.build_list_from_exprs(row_elems));
  }
  return list.build_list_from_exprs(rows);
}

std::optional<std::pair<std::vector<std::size_t>, std::vector<exprt>>>
python_converter::build_numpy_descriptor_materialized_elements(
  const nlohmann::json &arg,
  const std::string &unsupported_rank_error)
{
  if (
    !arg.is_object() || arg.value("_type", "") != "Name" ||
    !arg.contains("id") || !arg["id"].is_string())
    return std::nullopt;

  const std::string root_id =
    resolve_name_symbol_id(arg["id"].get<std::string>());
  std::optional<std::vector<std::size_t>> shape =
    get_numpy_nditer_logical_shape(root_id);
  if (!shape)
    return std::nullopt;
  if (shape->empty() || shape->size() > 2)
    throw std::runtime_error(unsupported_rank_error);

  if (auto pointer_it = numpy_pointer_view_info_.find(root_id);
      pointer_it != numpy_pointer_view_info_.end())
  {
    const symbolt *symbol = symbol_table_.find_symbol(root_id);
    if (symbol == nullptr)
      return std::nullopt;

    const namespacet ns(symbol_table_);
    const typet pointer_type = ns.follow(symbol->get_type());
    if (!pointer_type.is_pointer())
      return std::nullopt;

    const typet elem_type = ns.follow(pointer_type.subtype());
    std::vector<exprt> elems;
    elems.reserve(pointer_it->second.length);
    for (std::size_t i = 0; i < pointer_it->second.length; ++i)
    {
      const long long offset =
        static_cast<long long>(i) * pointer_it->second.stride;
      exprt element_ptr = python_expr::build_add(
        symbol_expr(*symbol), from_integer(offset, size_type()), pointer_type);
      elems.push_back(python_expr::build_dereference(element_ptr, elem_type));
    }
    return std::make_pair(*shape, elems);
  }

  std::optional<std::vector<nlohmann::json>> element_nodes =
    build_numpy_nditer_logical_elements(arg);
  if (!element_nodes)
    return std::nullopt;

  std::vector<exprt> elems;
  elems.reserve(element_nodes->size());
  for (const nlohmann::json &node : *element_nodes)
    elems.push_back(get_expr(node));

  return std::make_pair(*shape, elems);
}

static exprt build_numpy_descriptor_array_value(
  const std::vector<std::size_t> &shape,
  const std::vector<exprt> &elems,
  const typet &elem_type,
  type_handler &type_handler)
{
  typet result_type = type_handler.build_array(elem_type, shape.back());
  if (shape.size() == 2)
    result_type = type_handler.build_array(result_type, shape[0]);

  exprt value = gen_zero(result_type);
  if (shape.size() == 1)
  {
    for (std::size_t i = 0; i < elems.size(); ++i)
      value.operands().at(i) = elems[i];
    return value;
  }

  const std::size_t cols = shape[1];
  for (std::size_t row = 0; row < shape[0]; ++row)
    for (std::size_t col = 0; col < cols; ++col)
      value.operands().at(row).operands().at(col) = elems[(row * cols) + col];
  return value;
}

std::optional<exprt>
python_converter::build_numpy_descriptor_materialized_array(
  const nlohmann::json &arg)
{
  auto materialized = build_numpy_descriptor_materialized_elements(
    arg,
    "TypeError: numpy descriptor materialization currently supports rank 1 "
    "or 2 arrays");
  if (!materialized)
    return std::nullopt;

  std::optional<typet> empty_elem_type;
  if (materialized->second.empty())
  {
    const std::string root_id =
      resolve_name_symbol_id(arg["id"].get<std::string>());
    empty_elem_type = get_numpy_descriptor_element_type(root_id);
    if (!empty_elem_type)
      return std::nullopt;
  }

  const typet &elem_type = materialized->second.empty()
                             ? *empty_elem_type
                             : materialized->second.front().type();
  exprt value = build_numpy_descriptor_array_value(
    materialized->first, materialized->second, elem_type, type_handler_);

  symbolt &tmp =
    create_tmp_symbol(arg, "$numpy_descriptor_copy$", value.type(), value);
  exprt tmp_expr = symbol_expr(tmp);
  code_declt decl(tmp_expr);
  decl.operands().push_back(value);
  if (current_block != nullptr)
    current_block->copy_to_operands(decl);
  return tmp_expr;
}

std::optional<std::vector<std::size_t>>
python_converter::get_numpy_nditer_logical_shape(
  const std::string &root_id) const
{
  if (auto pointer_it = numpy_pointer_view_info_.find(root_id);
      pointer_it != numpy_pointer_view_info_.end())
    return std::vector<std::size_t>{pointer_it->second.length};

  if (auto reshape_it = numpy_reshape_view_info_.find(root_id);
      reshape_it != numpy_reshape_view_info_.end())
    return reshape_it->second.view_shape;

  if (auto transpose_it = numpy_transpose_view_info_.find(root_id);
      transpose_it != numpy_transpose_view_info_.end())
  {
    const symbolt *source = symbol_table_.find_symbol(
      resolve_numpy_array_storage_alias_id(transpose_it->second.source_id));
    if (source == nullptr)
      return std::nullopt;

    const namespacet ns(symbol_table_);
    std::vector<std::size_t> shape =
      numpy_shape_from_type(ns, ns.follow(source->get_type()));
    if (transpose_it->second.rank == 2 && transpose_it->second.swaps_axes)
      std::reverse(shape.begin(), shape.end());
    return shape;
  }

  // Fallback: no registered pointer/reshape/transpose view entry for this
  // id (this also covers a plain ndarray and a view-copy tracked only via
  // numpy_view_copy_sources_, both of which still carry their own concrete
  // array_typet). Derive shape straight from that type, the same way every
  // view branch above eventually does for its source. This is what lets
  // .tolist()/.any()/.all() reuse the exact same descriptor materialization
  // path for a bare `np.array(...)` instead of needing one of their own.
  // Rank is capped at 2 to match that path's own scope -- without it, a
  // 3-D+ array would get a shape here instead of declining, and reach the
  // descriptor path's "rank 1 or 2" rejection instead of this family's own
  // "constant numeric inputs only" one (regression/numpy/
  // sum_constructor_non_numeric_fail pins the latter).
  if (numpy_array_symbols_.count(root_id) == 0)
    return std::nullopt;

  const symbolt *plain = symbol_table_.find_symbol(root_id);
  if (plain == nullptr)
    return std::nullopt;

  const namespacet ns(symbol_table_);
  std::vector<std::size_t> shape =
    numpy_shape_from_type(ns, ns.follow(plain->get_type()));
  if (shape.empty() || shape.size() > 2)
    return std::nullopt;
  return shape;
}

std::optional<typet> python_converter::get_numpy_descriptor_element_type(
  const std::string &root_id) const
{
  const symbolt *symbol = symbol_table_.find_symbol(root_id);
  if (symbol == nullptr)
    return std::nullopt;

  const namespacet ns(symbol_table_);
  typet current = ns.follow(symbol->get_type());
  if (current.is_pointer())
    return ns.follow(current.subtype());

  while (current.is_array())
    current = ns.follow(to_array_type(current).subtype());

  return current;
}

bool python_converter::is_numpy_readonly_view_arg(
  const nlohmann::json &arg) const
{
  if (
    !arg.is_object() || arg.value("_type", "") != "Name" ||
    !arg.contains("id") || !arg["id"].is_string())
    return false;

  const std::string root_id =
    resolve_name_symbol_id(arg["id"].get<std::string>());
  if (root_id.empty())
    return false;

  if (auto pointer_it = numpy_pointer_view_info_.find(root_id);
      pointer_it != numpy_pointer_view_info_.end())
    return pointer_it->second.readonly;

  if (auto reshape_it = numpy_reshape_view_info_.find(root_id);
      reshape_it != numpy_reshape_view_info_.end())
    return reshape_it->second.readonly;

  return false;
}

bool python_converter::has_numpy_transpose_view_of(
  const std::string &source_id) const
{
  if (numpy_transpose_view_info_.count(source_id) != 0)
    return true;

  const std::string storage_id =
    resolve_numpy_array_storage_alias_id(source_id);
  for (const auto &entry : numpy_transpose_view_info_)
  {
    if (
      resolve_numpy_array_storage_alias_id(entry.second.source_id) ==
      storage_id)
      return true;
  }

  return false;
}

void python_converter::reject_unknown_numpy_view_call(
  const nlohmann::json &node)
{
  if (
    !node.is_object() || node.value("_type", "") != "Call" ||
    !node.contains("func") || !node["func"].is_object() ||
    !node.contains("args") || !node["args"].is_array())
    return;

  if (node["func"].value("_type", "") != "Name")
    return;

  const std::string func_name = node["func"].value("id", "");
  if (
    func_name == "len" || func_name == "bool" || func_name == "int" ||
    func_name == "float")
    return;

  for (const auto &arg : node["args"])
  {
    if (contains_tracked_numpy_view_name(arg))
      throw std::runtime_error(
        "TypeError: passing a copied numpy view to an unknown function is not "
        "supported");
  }
}

void python_converter::reject_numpy_view_identity_query(
  const nlohmann::json &node)
{
  if (!node.is_object())
    return;

  if (node.value("_type", "") == "Attribute")
  {
    const std::string attr = node.value("attr", "");
    if (attr == "base" || attr == "owndata")
    {
      const std::string root_name = node.contains("value")
                                      ? root_name_from_subscript(node["value"])
                                      : std::string();
      const std::string root_id =
        root_name.empty() ? std::string() : resolve_name_symbol_id(root_name);
      if (
        !root_id.empty() && (numpy_array_symbols_.count(root_id) != 0 ||
                             is_tracked_numpy_view_id(root_id)))
      {
        throw std::runtime_error(
          "TypeError: numpy view identity is not supported");
      }
    }

    if (node.contains("value"))
      reject_numpy_view_identity_query(node["value"]);
    return;
  }

  if (
    node.value("_type", "") == "Call" && node.contains("func") &&
    node["func"].is_object() && node["func"].value("_type", "") == "Attribute")
  {
    const std::string attr = node["func"].value("attr", "");
    if (attr != "shares_memory" && attr != "may_share_memory")
      return;

    const nlohmann::json &func = node["func"];
    if (
      !func.contains("value") || !func["value"].is_object() ||
      func["value"].value("_type", "") != "Name" ||
      !is_imported_numpy_module_alias(*ast_json, func["value"].value("id", "")))
      return;

    throw std::runtime_error("TypeError: numpy view identity is not supported");
  }
}

// dict_handler_ intercepts a Dict-literal assignment before the generic
// List/Tuple/Dict escape check further down the caller ever runs, so a
// copied-view escape into a dict literal (named or inline, e.g.
// {"row": x[0]}) has to be caught here too, or the view ends up embedded in
// the dict's runtime representation in a way that crashes SMT encoding
// instead of producing a diagnostic (mismatched sort widths in
// z3_convt::mk_eq).
void python_converter::reject_copied_numpy_view_in_container(
  const nlohmann::json &ast_node,
  const std::set<std::string> &container_types)
{
  if (!ast_node.contains("value") || !ast_node["value"].is_object())
    return;

  const nlohmann::json &value_node = ast_node["value"];
  if (
    container_types.count(value_node.value("_type", "")) == 0 ||
    !contains_tracked_numpy_view_name(value_node))
    return;

  throw std::runtime_error(
    "TypeError: storing a copied numpy view in a container is not "
    "supported");
}

std::optional<nlohmann::json> python_converter::select_return_value_for_call(
  const nlohmann::json &call_node) const
{
  if (
    !call_node.is_object() || call_node.value("_type", "") != "Call" ||
    !call_node.contains("func") ||
    call_node["func"].value("_type", "") != "Name" ||
    !call_node.contains("args") || !call_node["args"].is_array() ||
    (call_node.contains("keywords") && !call_node["keywords"].empty()))
    return std::nullopt;

  const std::string func_name = call_node["func"]["id"].get<std::string>();
  const nlohmann::json func_node =
    json_utils::try_find_function((*ast_json)["body"], func_name);
  if (
    func_node.empty() || !func_node.contains("body") ||
    !func_node["body"].is_array() || !func_node.contains("args") ||
    !func_node["args"].contains("args") ||
    !func_node["args"]["args"].is_array())
    return std::nullopt;

  const nlohmann::json &params = func_node["args"]["args"];
  if (params.size() != call_node["args"].size())
    return std::nullopt;

  auto is_trivial_arg = [](const nlohmann::json &node) {
    return node.value("_type", "") == "Name" ||
           node.value("_type", "") == "Constant";
  };
  for (const auto &arg : call_node["args"])
    if (!is_trivial_arg(arg))
      return std::nullopt;

  auto param_index =
    [&](const std::string &name) -> std::optional<std::size_t> {
    for (std::size_t i = 0; i < params.size(); ++i)
      if (params[i].value("arg", "") == name)
        return i;
    return std::nullopt;
  };

  auto bool_constant = [&](const nlohmann::json &node) -> std::optional<bool> {
    if (
      node.value("_type", "") == "Constant" && node.contains("value") &&
      node["value"].is_boolean())
      return node["value"].get<bool>();

    if (node.value("_type", "") == "Name" && node.contains("id"))
    {
      std::optional<std::size_t> idx =
        param_index(node["id"].get<std::string>());
      if (!idx)
        return std::nullopt;
      const nlohmann::json &arg = call_node["args"][*idx];
      if (
        arg.value("_type", "") == "Constant" && arg.contains("value") &&
        arg["value"].is_boolean())
        return arg["value"].get<bool>();
    }

    return std::nullopt;
  };

  struct return_scan_result
  {
    bool invalid = false;
    bool found = false;
    nlohmann::json value;
  };

  std::function<return_scan_result(const nlohmann::json &)> scan =
    [&](const nlohmann::json &body) -> return_scan_result {
    for (const auto &stmt : body)
    {
      if (stmt.value("_type", "") == "Return")
      {
        if (!stmt.contains("value") || stmt["value"].is_null())
          return {true, false, nlohmann::json()};
        return {false, true, stmt["value"]};
      }

      if (stmt.value("_type", "") == "If")
      {
        std::optional<bool> test_value = bool_constant(stmt["test"]);
        if (!test_value)
          return {true, false, nlohmann::json()};

        const nlohmann::json &chosen =
          *test_value ? stmt["body"] : stmt["orelse"];
        if (chosen.is_array())
        {
          return_scan_result ret = scan(chosen);
          if (ret.invalid || ret.found)
            return ret;
        }
        continue;
      }

      return {true, false, nlohmann::json()};
    }

    return {};
  };

  return_scan_result result = scan(func_node["body"]);
  if (result.invalid || !result.found)
    return std::nullopt;
  return result.value;
}

nlohmann::json python_converter::substitute_call_arguments(
  const nlohmann::json &node,
  const nlohmann::json &call_node) const
{
  if (
    !call_node.is_object() || call_node.value("_type", "") != "Call" ||
    !call_node.contains("func") ||
    call_node["func"].value("_type", "") != "Name" ||
    !call_node.contains("args") || !call_node["args"].is_array())
    return node;

  const std::string func_name = call_node["func"]["id"].get<std::string>();
  const nlohmann::json func_node =
    json_utils::try_find_function((*ast_json)["body"], func_name);
  if (
    func_node.empty() || !func_node.contains("args") ||
    !func_node["args"].contains("args") ||
    !func_node["args"]["args"].is_array())
    return node;

  const nlohmann::json &params = func_node["args"]["args"];
  if (node.is_object())
  {
    if (node.value("_type", "") == "Name" && node.contains("id"))
    {
      const std::string name = node["id"].get<std::string>();
      for (std::size_t i = 0; i < params.size() && i < call_node["args"].size();
           ++i)
        if (params[i].value("arg", "") == name)
          return call_node["args"][i];
    }

    nlohmann::json out = node;
    for (auto it = out.begin(); it != out.end(); ++it)
      it.value() = substitute_call_arguments(it.value(), call_node);
    return out;
  }

  if (node.is_array())
  {
    nlohmann::json out = nlohmann::json::array();
    for (const auto &elem : node)
      out.push_back(substitute_call_arguments(elem, call_node));
    return out;
  }

  return node;
}

// Recursively checks that every Name leaf in `node` satisfies `name_is_safe`
// -- used to decide whether a return expression built around a call (e.g.
// `np.transpose(a)`) is safe to substitute wholesale: substitute_call_arguments
// only ever rewrites a Name matching a parameter, so anything else it would
// leave untouched (a module alias, a literal) must resolve correctly in the
// caller's own scope for the substituted tree to mean the same thing there.
static bool expr_only_references_safe_names(
  const nlohmann::json &node,
  const std::function<bool(const std::string &)> &name_is_safe)
{
  if (node.is_object())
  {
    if (node.value("_type", "") == "Name" && node.contains("id"))
      return name_is_safe(node["id"].get<std::string>());
    for (auto it = node.begin(); it != node.end(); ++it)
      if (!expr_only_references_safe_names(it.value(), name_is_safe))
        return false;
    return true;
  }
  if (node.is_array())
  {
    for (const auto &elem : node)
      if (!expr_only_references_safe_names(elem, name_is_safe))
        return false;
  }
  return true;
}

// Resolves call_node to its callee's FunctionDef node, or an empty json when
// it isn't a plain `name(...)` call to a locally-defined function with a
// concrete parameter list. Split out of return_value_uses_call_argument to
// keep that function's own decision count down.
static nlohmann::json resolve_func_node_with_params(
  const nlohmann::json &call_node,
  const nlohmann::json &ast_body)
{
  if (
    !call_node.is_object() || call_node.value("_type", "") != "Call" ||
    !call_node.contains("func") ||
    call_node["func"].value("_type", "") != "Name")
    return nlohmann::json();

  const std::string func_name = call_node["func"]["id"].get<std::string>();
  const nlohmann::json func_node =
    json_utils::try_find_function(ast_body, func_name);
  if (
    func_node.empty() || !func_node.contains("args") ||
    !func_node["args"].contains("args") ||
    !func_node["args"]["args"].is_array())
    return nlohmann::json();

  return func_node;
}

bool python_converter::return_value_uses_call_argument(
  const nlohmann::json &return_value,
  const nlohmann::json &call_node) const
{
  const nlohmann::json func_node =
    resolve_func_node_with_params(call_node, (*ast_json)["body"]);
  if (func_node.empty())
    return false;

  auto is_param_name = [&](const nlohmann::json &node) {
    if (node.value("_type", "") != "Name" || !node.contains("id"))
      return false;
    const std::string name = node["id"].get<std::string>();
    for (const auto &param : func_node["args"]["args"])
      if (param.value("arg", "") == name)
        return true;
    return false;
  };

  if (is_param_name(return_value))
    return true;

  if (
    return_value.value("_type", "") == "Subscript" &&
    return_value.contains("value") && is_param_name(return_value["value"]))
    return true;

  // return <call>(<param>, ...): e.g. `def transposed(a): return
  // np.transpose(a)`. Split out to keep this function's own decision count
  // down; see that method for why this shape is safe to substitute too.
  if (return_value.value("_type", "") == "Call")
    return return_call_only_references_params_or_modules(
      return_value, func_node["args"]["args"]);

  return false;
}

// True when `alias` is bound by a top-level `import ... as alias` (or a bare
// `import alias`) in the module's own body. imported_modules is a single
// flat, program-wide map -- it also holds aliases bound by an import nested
// inside some OTHER function's body (convert_module_imports hoists those
// into the same map), which are not actually in scope wherever a substituted
// return value ends up spliced into. Restricting to module-level imports
// matches the idiomatic `import numpy as np` at the top of the file, which
// is visible everywhere.
static bool is_module_level_import_alias(
  const nlohmann::json &ast_body,
  const std::string &alias)
{
  for (const auto &stmt : ast_body)
  {
    if (stmt.value("_type", "") != "Import" || !stmt.contains("names"))
      continue;
    for (const auto &name : stmt["names"])
      if (name.value("asname", name.value("name", "")) == alias)
        return true;
  }
  return false;
}

bool python_converter::return_call_only_references_params_or_modules(
  const nlohmann::json &return_value,
  const nlohmann::json &params) const
{
  // Safe to substitute under the same reasoning as the bare-param/subscript
  // cases in return_value_uses_call_argument as long as every Name the call
  // expression references is either a parameter (substituted) or an
  // imported module alias (left as-is, and resolved identically in the
  // caller's own scope).
  auto name_is_safe = [&](const std::string &name) {
    if (
      imported_modules.find(name) != imported_modules.end() &&
      is_module_level_import_alias((*ast_json)["body"], name))
      return true;
    for (const auto &param : params)
      if (param.value("arg", "") == name)
        return true;
    return false;
  };
  return expr_only_references_safe_names(return_value, name_is_safe);
}

/// Item assignment on an immutable container is a TypeError. A string that is
/// not intercepted here updates its char array with a whole string value, which
/// trips with2t::assert_consistency and aborts instead of reporting anything.
bool python_converter::reject_immutable_item_assignment(
  const typet &container_type,
  codet &target_block)
{
  const char *kind = tuple_handler_->is_tuple_type(container_type) ? "tuple"
                     : type_utils::is_string_type(container_type)  ? "str"
                                                                   : nullptr;
  if (!kind)
    return false;

  exprt raise = get_exception_handler().gen_exception_raise(
    "TypeError",
    std::string("'") + kind + "' object does not support item assignment");
  codet throw_code("expression");
  throw_code.operands().push_back(raise);
  target_block.copy_to_operands(throw_code);
  return true;
}

void python_converter::reject_unsafe_numpy_view_target(
  const nlohmann::json &target)
{
  if (!target.is_object() || target.value("_type", "") != "Subscript")
    return;

  const std::string root_name = root_name_from_subscript(target);
  if (root_name.empty())
    return;

  const std::string root_id = resolve_name_symbol_id(root_name);
  if (root_id.empty())
    return;

  reject_unsafe_numpy_view_write_to(root_id);
}

// The actual write-safety check, taking an already-resolved root symbol id
// rather than a Subscript-shaped AST node: shared by
// reject_unsafe_numpy_view_target (a[i] = x) and
// try_handle_flat_index_assignment (a.flat[i] = x, whose receiver is
// resolved through a rewritten ravel Call node with no Subscript for
// root_name_from_subscript to walk).
void python_converter::reject_unsafe_numpy_view_write_to(
  const std::string &root_id)
{
  // A read-only pointer-backed view (the main-diagonal view) rejects a
  // direct write through it with NumPy's own diagnostic, independent of
  // the copy-divergence checks below: unlike those, this has nothing to
  // do with whether the view safely aliases its source (it does) -- a
  // source write is still observed by a live diagonal view exactly like a
  // writable one (diagonal_view_source_write_success), only writing
  // *through* the view itself is refused.
  {
    auto it = numpy_pointer_view_info_.find(root_id);
    if (it != numpy_pointer_view_info_.end() && it->second.readonly)
      throw std::runtime_error(
        "ValueError: assignment destination is read-only");
  }

  {
    auto it = numpy_reshape_view_info_.find(root_id);
    if (it != numpy_reshape_view_info_.end() && it->second.readonly)
      throw std::runtime_error(
        "ValueError: assignment destination is read-only");
  }

  // A view symbol this PR's 1-D slice aliasing retyped to a pointer (see
  // list_access.cpp's handle_range_slice, which populates
  // numpy_pointer_view_info_ exactly for that case) genuinely aliases
  // its source's storage: writing through it, or through the source while
  // it is live, is sound pointer semantics, not the copy-divergence this
  // guard otherwise exists to reject. Checking membership in that map
  // (rather than just "is this symbol's type a pointer") avoids misreading
  // some other, unrelated pointer-typed symbol as one of these views.
  auto is_pointer_backed = [this](const std::string &id) {
    return numpy_pointer_view_info_.count(id) != 0;
  };

  if (numpy_view_copy_sources_.count(root_id) != 0)
  {
    if (is_pointer_backed(root_id))
      return;
    throw std::runtime_error(
      "TypeError: writing through a copied numpy view is not supported");
  }

  for (const auto &entry : numpy_view_copy_sources_)
    if (entry.second == root_id && !is_pointer_backed(entry.first))
      throw std::runtime_error(
        "TypeError: writing to a numpy array with a live copied view is not "
        "supported");
}

void python_converter::reject_numpy_view_slice_assignment(
  const nlohmann::json &target)
{
  const std::string root_name = root_name_from_subscript(target);
  if (root_name.empty())
    return;

  const std::string root_id = resolve_name_symbol_id(root_name);
  if (root_id.empty())
    return;

  if (
    numpy_pointer_view_info_.count(root_id) != 0 ||
    numpy_view_copy_sources_.count(root_id) != 0)
    throw std::runtime_error(
      "TypeError: slice assignment through a numpy view is not supported");
}

void python_converter::record_numpy_view_copy(
  const exprt &lhs,
  const nlohmann::json &rhs_node)
{
  if (!lhs.is_symbol())
    return;

  nlohmann::json view_node = rhs_node;
  if (!is_numpy_view_copy_expr(view_node))
  {
    if (rhs_node.value("_type", "") == "Call")
    {
      std::optional<nlohmann::json> ret_val =
        select_return_value_for_call(rhs_node);
      if (!ret_val || !return_value_uses_call_argument(*ret_val, rhs_node))
      {
        clear_numpy_view_copy(lhs);
        return;
      }
      view_node = substitute_call_arguments(*ret_val, rhs_node);
    }
  }

  if (!is_numpy_view_copy_expr(view_node))
  {
    clear_numpy_view_copy(lhs);
    return;
  }

  const std::string root_name = root_name_from_numpy_view_copy_expr(view_node);
  if (root_name.empty())
  {
    clear_numpy_view_copy(lhs);
    return;
  }

  const std::string source_id = resolve_name_symbol_id(root_name);
  if (source_id.empty())
  {
    clear_numpy_view_copy(lhs);
    return;
  }

  const std::string storage_id =
    resolve_numpy_array_storage_alias_id(source_id);

  if (numpy_array_symbols_.count(storage_id) == 0)
  {
    clear_numpy_view_copy(lhs);
    return;
  }

  const std::string lhs_id = lhs.identifier().as_string();
  numpy_view_copy_sources_[lhs_id] = storage_id;
  numpy_array_symbols_.insert(lhs_id);
}

bool python_converter::record_numpy_transpose_view(
  const exprt &lhs,
  const nlohmann::json &view_node)
{
  if (!lhs.is_symbol())
    return false;

  bool swaps_axes = true;
  const nlohmann::json *source_node =
    numpy_transpose_source_node(view_node, swaps_axes);
  if (!source_node)
    return false;

  const std::string root_name = root_name_from_subscript(*source_node);
  const std::string source_id = resolve_name_symbol_id(root_name);
  if (source_id.empty())
    return false;

  const symbolt *source = symbol_table_.find_symbol(source_id);
  if (!source)
    return false;

  const std::size_t rank = numpy_array_rank(ns, ns.follow(source->get_type()));

  if (rank == 0 || rank > 2)
    return false;

  std::optional<bool> axis_swaps =
    numpy_axis_permutation_swaps_axes(view_node, rank, swaps_axes);
  if (!axis_swaps)
    return false;
  swaps_axes = *axis_swaps;

  const std::string lhs_id = lhs.identifier().as_string();
  const std::string view_source_id =
    resolve_numpy_array_storage_alias_id(source_id);
  numpy_transpose_view_info_[lhs_id] = {
    view_source_id, rank, swaps_axes && rank == 2};
  numpy_array_symbols_.insert(lhs_id);
  return true;
}

bool python_converter::record_numpy_reshape_view(
  const exprt &lhs,
  const nlohmann::json &view_node)
{
  if (!lhs.is_symbol() || !is_numpy_shape_only_view_call_node(view_node))
    return false;

  if (
    !view_node.contains("args") || !view_node["args"].is_array() ||
    view_node["args"].empty())
    return false;

  const std::string root_name = root_name_from_subscript(view_node["args"][0]);
  const std::string source_id = resolve_name_symbol_id(root_name);
  if (source_id.empty() || is_tracked_numpy_view_id(source_id))
    return false;

  const symbolt *source = symbol_table_.find_symbol(source_id);
  if (!source)
    return false;

  const std::vector<std::size_t> source_shape =
    numpy_shape_from_type(ns, ns.follow(source->get_type()));
  if (source_shape.empty() || source_shape.size() > 2)
    return false;

  std::optional<std::vector<std::size_t>> view_shape =
    numpy_shape_only_view_shape(view_node, source_shape);
  if (!view_shape || view_shape->empty() || view_shape->size() > 2)
    return false;

  const std::string lhs_id = lhs.identifier().as_string();
  const bool readonly = is_numpy_broadcast_to_call_node(view_node);
  numpy_reshape_view_info_[lhs_id] = {
    source_id, source_shape, *view_shape, readonly, readonly};
  numpy_array_symbols_.insert(lhs_id);
  return true;
}

bool python_converter::record_numpy_shape_stride_view(
  const exprt &lhs,
  const nlohmann::json &rhs_node)
{
  if (is_numpy_transpose_view_call_node(rhs_node))
  {
    clear_numpy_view_copy(lhs);
    return record_numpy_transpose_view(lhs, rhs_node);
  }

  if (is_numpy_shape_only_view_call_node(rhs_node))
  {
    clear_numpy_view_copy(lhs);
    return record_numpy_reshape_view(lhs, rhs_node);
  }

  return false;
}

symbolt *
python_converter::resolve_numpy_array_storage_alias(symbolt *symbol) const
{
  if (!symbol)
    return symbol;

  symbolt *storage = symbol_table_.find_symbol(
    resolve_numpy_array_storage_alias_id(symbol->id.as_string()));
  return storage ? storage : symbol;
}

std::string python_converter::resolve_numpy_array_storage_alias_id(
  const std::string &symbol_id) const
{
  std::string storage_id = symbol_id;
  std::set<std::string> seen_aliases;
  auto alias_it = numpy_array_storage_aliases_.find(storage_id);
  while (alias_it != numpy_array_storage_aliases_.end() &&
         seen_aliases.insert(alias_it->first).second)
  {
    storage_id = alias_it->second;
    alias_it = numpy_array_storage_aliases_.find(storage_id);
  }
  return storage_id;
}

void python_converter::clear_numpy_view_copy(const exprt &lhs)
{
  if (!lhs.is_symbol())
    return;
  // Every call site here means lhs is being rebound away from view-copy
  // tracking; a stale pointer-backed view entry (ADR-NP-003 etapa 2, set by
  // list_access.cpp's try_build_1d_pointer_view) must not survive that
  // rebind either, or len()/.shape/.ndim/write guards could misread a
  // reused symbol id against the old view's tracked length.
  const std::string lhs_id = lhs.identifier().as_string();
  numpy_view_copy_sources_.erase(lhs_id);
  numpy_pointer_view_info_.erase(lhs_id);
  numpy_transpose_view_info_.erase(lhs_id);
  numpy_reshape_view_info_.erase(lhs_id);
}

void python_converter::detach_numpy_pointer_views_of(
  const std::string &rebound_id,
  const locationt &location,
  codet &target_block)
{
  const namespacet ns(symbol_table_);

  // numpy_view_copy_sources_ is mutated below (erase), so collect the
  // matching keys first rather than erasing mid-iteration.
  std::vector<std::string> view_ids;
  for (const auto &entry : numpy_view_copy_sources_)
    if (entry.second == rebound_id)
      view_ids.push_back(entry.first);

  for (const std::string &view_id : view_ids)
  {
    auto info_it = numpy_pointer_view_info_.find(view_id);
    if (info_it == numpy_pointer_view_info_.end())
      continue; // a plain copied view (etapa 1); already independent

    symbolt *view_symbol = symbol_table_.find_symbol(view_id);
    if (!view_symbol)
      continue;

    // The view's own DECL was already emitted with pointer_typet at its
    // creation point; retyping the symbol table entry now would desync it
    // from that DECL. Keep the declared type and just repoint the pointer
    // *value* at a fresh, independent snapshot instead.
    const exprt old_ptr = symbol_expr(*view_symbol);
    const typet ptr_type = old_ptr.type();
    const typet elem_type = ns.follow(view_symbol->get_type()).subtype();
    const std::size_t length = info_it->second.length;
    const long long stride = info_it->second.stride;

    array_typet snapshot_type(elem_type, from_integer(length, size_type()));
    symbolt &snapshot =
      create_tmp_symbol(location, "$view_snapshot$", snapshot_type, exprt());
    code_declt snap_decl(symbol_expr(snapshot));
    snap_decl.location() = location;
    target_block.copy_to_operands(snap_decl);

    // Copy what the view currently sees (still the pre-rebind source
    // storage at this point in program order) into the snapshot. The
    // source read is scaled by the view's own stride (1 for a unit-stride
    // slice/row view, but e.g. num_cols for a column view); the snapshot
    // itself is always densely packed, so the destination index is not.
    for (std::size_t i = 0; i < length; ++i)
    {
      // stride is signed (row/column/unit-stride views are positive; a
      // reversed slice is negative) but src_idx is unsigned size_type():
      // i*stride's two's-complement bit pattern for a negative product is
      // exactly SIZE_MAX-derived, and the pointer add below (as well as
      // ESBMC's own SMT-level bounds reasoning over it) treats that
      // correctly as a backward offset -- confirmed by a rebind-detach
      // regression over a[::-1] (view_strided_reverse_rebind_source_edge).
      exprt src_idx =
        from_integer(static_cast<long long>(i) * stride, size_type());
      exprt dst_idx = from_integer(i, size_type());
      exprt src = python_expr::build_index(old_ptr, src_idx, elem_type);
      exprt dst =
        python_expr::build_index(symbol_expr(snapshot), dst_idx, elem_type);
      code_assignt elem_assign(dst, src);
      elem_assign.location() = location;
      target_block.copy_to_operands(elem_assign);
    }

    // Address-of the snapshot symbol directly rather than
    // build_index(snapshot, 0, ...): a zero-length view (e.g. a[3:3])
    // makes that index2tc an out-of-bounds subscript on an empty array.
    exprt new_ptr = python_expr::build_typecast(
      python_expr::build_address_of(symbol_expr(snapshot)), ptr_type);
    code_assignt repoint(old_ptr, new_ptr);
    repoint.location() = location;
    target_block.copy_to_operands(repoint);

    // The detached view's own storage is a fresh, densely-packed snapshot,
    // so it is unit-stride from here regardless of what stride it had into
    // rebound_id's buffer; length and read-only-ness (a diagonal view) are
    // unchanged by detaching.
    info_it->second.stride = 1;

    // The view no longer aliases rebound_id's storage; drop the source
    // link so a later write to the (new) rebound_id array is not held
    // responsible for a view it can no longer affect.
    numpy_view_copy_sources_.erase(view_id);
  }
}

void python_converter::clear_numpy_transpose_views_of(
  const std::string &source_id)
{
  for (auto it = numpy_transpose_view_info_.begin();
       it != numpy_transpose_view_info_.end();)
  {
    if (it->second.source_id == source_id)
      it = numpy_transpose_view_info_.erase(it);
    else
      ++it;
  }

  for (auto it = numpy_reshape_view_info_.begin();
       it != numpy_reshape_view_info_.end();)
  {
    if (it->second.source_id == source_id)
      it = numpy_reshape_view_info_.erase(it);
    else
      ++it;
  }
}

void python_converter::emit_numpy_view_cell_assignment(
  const std::string &symbol_id,
  const std::vector<long long> &cell_indices,
  const exprt &rhs,
  const locationt &location,
  codet &target_block)
{
  const symbolt *symbol = symbol_table_.find_symbol(symbol_id);
  if (!symbol)
    return;

  const namespacet ns(symbol_table_);
  exprt cell = build_numpy_array_cell(ns, *symbol, cell_indices);
  if (cell.is_nil())
    return;

  code_assignt mirror(cell, rhs);
  mirror.location() = location;
  target_block.copy_to_operands(mirror);
}

void python_converter::mirror_numpy_source_write_to_views(
  const std::string &source_id,
  const std::vector<long long> &source_indices,
  const exprt &rhs,
  const locationt &location,
  codet &target_block,
  const std::string &skip_view_id)
{
  const std::string storage_source_id =
    resolve_numpy_array_storage_alias_id(source_id);

  for (const auto &entry : numpy_transpose_view_info_)
  {
    if (entry.first == skip_view_id)
      continue;

    const numpy_transpose_view_infot &view = entry.second;
    if (
      resolve_numpy_array_storage_alias_id(view.source_id) != storage_source_id)
      continue;

    std::optional<std::vector<long long>> view_indices =
      numpy_transpose_cell_indices(view.rank, view.swaps_axes, source_indices);
    if (view_indices)
      emit_numpy_view_cell_assignment(
        entry.first, *view_indices, rhs, location, target_block);
  }

  for (const auto &entry : numpy_reshape_view_info_)
  {
    if (entry.first == skip_view_id)
      continue;

    const numpy_reshape_view_infot &view = entry.second;
    if (
      resolve_numpy_array_storage_alias_id(view.source_id) != storage_source_id)
      continue;

    if (view.broadcast)
    {
      for (const auto &current : numpy_broadcast_view_indices_for_source(
             source_indices, view.view_shape, view.source_shape))
        emit_numpy_view_cell_assignment(
          entry.first, current, rhs, location, target_block);
      continue;
    }

    std::optional<std::size_t> flat =
      numpy_flat_index(source_indices, view.source_shape);
    std::optional<std::vector<long long>> view_indices =
      flat ? numpy_unravel_index(*flat, view.view_shape) : std::nullopt;
    if (view_indices)
      emit_numpy_view_cell_assignment(
        entry.first, *view_indices, rhs, location, target_block);
  }
}

void python_converter::emit_numpy_transpose_mirror_assignment(
  const std::string &symbol_id,
  const std::vector<long long> &cell_indices,
  const exprt &rhs,
  const locationt &location,
  codet &target_block)
{
  emit_numpy_view_cell_assignment(
    symbol_id, cell_indices, rhs, location, target_block);

  auto view_it = numpy_transpose_view_info_.find(symbol_id);
  if (view_it == numpy_transpose_view_info_.end())
    return;

  std::optional<std::vector<long long>> source_indices =
    numpy_transpose_cell_indices(
      view_it->second.rank, view_it->second.swaps_axes, cell_indices);
  if (!source_indices)
    return;

  const std::string source_id = view_it->second.source_id;
  emit_numpy_transpose_mirror_assignment(
    source_id, *source_indices, rhs, location, target_block);
  const std::string storage_source_id =
    resolve_numpy_array_storage_alias_id(source_id);
  if (storage_source_id != source_id)
    emit_numpy_view_cell_assignment(
      storage_source_id, *source_indices, rhs, location, target_block);
  mirror_numpy_source_write_to_views(
    storage_source_id, *source_indices, rhs, location, target_block, symbol_id);
}

void python_converter::mirror_numpy_transpose_assignment(
  const nlohmann::json &target,
  const exprt &rhs,
  const locationt &location,
  codet &target_block)
{
  if (!target.is_object() || target.value("_type", "") != "Subscript")
    return;

  std::string root_name;
  const std::vector<long long> indices =
    subscript_indices_from_root(target, root_name);
  if (root_name.empty())
  {
    reject_nonconstant_numpy_view_write(target);
    return;
  }

  const std::string root_id = resolve_name_symbol_id(root_name);
  if (root_id.empty())
    return;
  const std::string storage_root_id =
    resolve_numpy_array_storage_alias_id(root_id);

  auto direct = numpy_transpose_view_info_.find(root_id);
  if (direct != numpy_transpose_view_info_.end())
  {
    std::optional<std::vector<long long>> cell_indices =
      numpy_transpose_cell_indices(
        direct->second.rank, direct->second.swaps_axes, indices);
    if (cell_indices)
    {
      const std::string source_id = direct->second.source_id;
      emit_numpy_transpose_mirror_assignment(
        source_id, *cell_indices, rhs, location, target_block);
      const std::string storage_source_id =
        resolve_numpy_array_storage_alias_id(source_id);
      mirror_numpy_source_write_to_views(
        storage_source_id, *cell_indices, rhs, location, target_block, root_id);
    }
    return;
  }

  for (const auto &entry : numpy_transpose_view_info_)
  {
    const numpy_transpose_view_infot &view = entry.second;
    if (resolve_numpy_array_storage_alias_id(view.source_id) != storage_root_id)
      continue;

    std::optional<std::vector<long long>> cell_indices =
      numpy_transpose_cell_indices(view.rank, view.swaps_axes, indices);
    if (cell_indices)
      emit_numpy_transpose_mirror_assignment(
        entry.first, *cell_indices, rhs, location, target_block);
  }
}

void python_converter::mirror_numpy_reshape_assignment(
  const nlohmann::json &target,
  const exprt &rhs,
  const locationt &location,
  codet &target_block)
{
  if (!target.is_object() || target.value("_type", "") != "Subscript")
    return;

  std::string root_name;
  const std::vector<long long> indices =
    subscript_indices_from_root(target, root_name);
  if (root_name.empty())
  {
    reject_nonconstant_numpy_view_write(target);
    return;
  }

  const std::string root_id = resolve_name_symbol_id(root_name);
  if (root_id.empty())
    return;

  auto direct = numpy_reshape_view_info_.find(root_id);
  if (direct != numpy_reshape_view_info_.end())
  {
    std::optional<std::vector<long long>> source_indices =
      numpy_shape_view_source_indices(
        indices,
        direct->second.view_shape,
        direct->second.source_shape,
        direct->second.broadcast);
    if (source_indices)
    {
      const std::string source_id =
        resolve_numpy_array_storage_alias_id(direct->second.source_id);
      emit_numpy_view_cell_assignment(
        source_id, *source_indices, rhs, location, target_block);
      mirror_numpy_source_write_to_views(
        source_id, *source_indices, rhs, location, target_block, root_id);
    }
    return;
  }

  const std::string storage_root_id =
    resolve_numpy_array_storage_alias_id(root_id);
  for (const auto &entry : numpy_reshape_view_info_)
  {
    const numpy_reshape_view_infot &view = entry.second;
    if (resolve_numpy_array_storage_alias_id(view.source_id) != storage_root_id)
      continue;

    if (view.broadcast)
    {
      for (const auto &current : numpy_broadcast_view_indices_for_source(
             indices, view.view_shape, view.source_shape))
        emit_numpy_transpose_mirror_assignment(
          entry.first, current, rhs, location, target_block);
      continue;
    }

    std::optional<std::size_t> flat =
      numpy_flat_index(indices, view.source_shape);
    std::optional<std::vector<long long>> view_indices =
      flat ? numpy_unravel_index(*flat, view.view_shape) : std::nullopt;
    if (view_indices)
      emit_numpy_transpose_mirror_assignment(
        entry.first, *view_indices, rhs, location, target_block);
  }
}

void python_converter::mirror_numpy_transpose_assignment_from_targets(
  const nlohmann::json &ast_node,
  const exprt &rhs,
  const locationt &location,
  codet &target_block)
{
  if (
    !ast_node.contains("targets") || !ast_node["targets"].is_array() ||
    ast_node["targets"].empty())
    return;

  mirror_numpy_transpose_assignment(
    ast_node["targets"][0], rhs, location, target_block);
  mirror_numpy_reshape_assignment(
    ast_node["targets"][0], rhs, location, target_block);
}

void python_converter::update_numpy_array_binding(
  const exprt &lhs,
  const nlohmann::json &rhs_node)
{
  if (!lhs.is_symbol())
    return;

  const std::string lhs_id = lhs.identifier().as_string();

  if (rhs_node.value("_type", "") == "Name" && rhs_node.contains("id"))
  {
    const std::string rhs_id =
      resolve_name_symbol_id(rhs_node["id"].get<std::string>());
    if (rhs_id == lhs_id)
      return;

    if (rhs_node.value("_numpy_copy_method", false))
    {
      clear_numpy_array_storage_aliases_for(lhs_id);
      clear_numpy_view_copy(lhs);
      numpy_array_symbols_.insert(lhs_id);
      return;
    }

    auto view_it = numpy_view_copy_sources_.find(rhs_id);
    if (view_it != numpy_view_copy_sources_.end())
    {
      clear_numpy_array_storage_aliases_for(lhs_id);
      numpy_view_copy_sources_[lhs_id] = view_it->second;
      numpy_array_symbols_.insert(lhs_id);
      return;
    }
    if (numpy_array_symbols_.count(rhs_id) != 0)
    {
      clear_numpy_view_copy(lhs);
      numpy_array_symbols_.insert(lhs_id);
      bind_numpy_array_storage_alias(lhs_id, rhs_id);
      return;
    }
  }

  clear_numpy_transpose_views_of(lhs_id);
  clear_numpy_array_storage_aliases_for(lhs_id);

  if (record_numpy_view_copy_from_returned_argument(lhs, lhs_id, rhs_node))
    return;

  if (record_numpy_shape_stride_view(lhs, rhs_node))
    return;

  if (is_numpy_view_copy_expr(rhs_node))
  {
    record_numpy_view_copy(lhs, rhs_node);
    return;
  }

  const bool unconditional_assignment =
    block_nesting_ == function_body_depth_ + 1;
  if (unconditional_assignment || numpy_view_copy_sources_.count(lhs_id) == 0)
    clear_numpy_view_copy(lhs);

  if (is_numpy_array_constructor_expr(rhs_node))
    numpy_array_symbols_.insert(lhs_id);
  else
    numpy_array_symbols_.erase(lhs_id);
}

bool python_converter::record_numpy_view_copy_from_returned_argument(
  const exprt &lhs,
  const std::string &lhs_id,
  const nlohmann::json &rhs_node)
{
  if (rhs_node.value("_type", "") != "Call")
    return false;

  std::optional<nlohmann::json> ret_val =
    select_return_value_for_call(rhs_node);
  if (!ret_val || !return_value_uses_call_argument(*ret_val, rhs_node))
    return false;

  nlohmann::json substituted = substitute_call_arguments(*ret_val, rhs_node);
  if (!is_numpy_view_copy_expr(substituted))
    return false;

  record_numpy_view_copy(lhs, substituted);
  return numpy_view_copy_sources_.count(lhs_id) != 0;
}

void python_converter::clear_numpy_array_storage_aliases_for(
  const std::string &symbol_id)
{
  numpy_array_storage_aliases_.erase(symbol_id);
  for (auto it = numpy_array_storage_aliases_.begin();
       it != numpy_array_storage_aliases_.end();)
  {
    if (it->second == symbol_id)
      it = numpy_array_storage_aliases_.erase(it);
    else
      ++it;
  }
}

void python_converter::bind_numpy_array_storage_alias(
  const std::string &lhs_id,
  const std::string &rhs_id)
{
  numpy_array_storage_aliases_[lhs_id] =
    resolve_numpy_array_storage_alias_id(rhs_id);
}

bool python_converter::should_rebuild_cached_numpy_row_subscript_rhs(
  const nlohmann::json &rhs_node) const
{
  if (
    !has_cached_any_subscript_rhs_ ||
    rhs_node.value("_type", "") != "Subscript")
    return false;

  if (
    !rhs_node.contains("value") ||
    rhs_node["value"].value("_type", "") != "Name" ||
    !rhs_node.contains("slice"))
    return false;

  // Row view: a[i]. Column view: a[:, j]. Both hit
  // resolve_any_subscript_array_type's type-inference probe (current_lhs
  // unset there) before the real assignment ever runs, so the cached probe
  // result -- an array-typed copy, not the pointer view this rebuild
  // forces -- must not be reused for either shape.
  const nlohmann::json &slice = rhs_node["slice"];
  if (!is_literal_int_node(slice) && !is_column_select_slice_node(slice))
    return false;

  const std::string source_id =
    resolve_name_symbol_id(rhs_node["value"]["id"].get<std::string>());
  return is_tracked_2d_numpy_array_symbol(source_id);
}

bool python_converter::is_tracked_2d_numpy_array_symbol(
  const std::string &source_id) const
{
  if (source_id.empty() || numpy_array_symbols_.count(source_id) == 0)
    return false;

  const symbolt *source = symbol_table_.find_symbol(source_id);
  if (!source)
    return false;

  const namespacet ns(symbol_table_);
  typet source_type = ns.follow(source->get_type());
  if (!source_type.is_array())
    return false;
  source_type = ns.follow(to_array_type(source_type).subtype());
  if (!source_type.is_array())
    return false;
  source_type = ns.follow(to_array_type(source_type).subtype());
  return !source_type.is_array();
}

std::string python_converter::infer_type_from_any_annotation(
  const nlohmann::json &ast_node,
  const std::string &lhs_type)
{
  if (ast_node["value"].is_null() || ast_node["value"]["_type"] != "Call")
    return lhs_type;

  const auto &func_node = ast_node["value"]["func"];
  std::string func_name;

  if (func_node["_type"] == "Name")
    func_name = func_node["id"].get<std::string>();
  else if (func_node["_type"] == "Attribute")
    func_name = func_node["attr"].get<std::string>();

  if (func_name.empty())
    return lhs_type;

  symbol_id func_sid(current_python_file, "", func_name);
  symbolt *func_symbol = symbol_table_.find_symbol(func_sid.to_string());

  // For method calls (e.g., b.f()), the method symbol is stored under the
  // class scope (py:file@C@ClassName@F@method), not the top-level scope.
  // Look up the object's class and retry the symbol lookup.
  if (
    !func_symbol && func_node["_type"] == "Attribute" &&
    func_node["value"].contains("id"))
  {
    const std::string obj_name = func_node["value"]["id"].get<std::string>();
    symbol_id obj_sid = create_symbol_id();
    obj_sid.set_object(obj_name);
    const symbolt *obj_sym = symbol_table_.find_symbol(obj_sid.to_string());
    if (obj_sym)
    {
      typet obj_type = ns.follow(obj_sym->get_type());
      std::string class_name;
      if (obj_type.is_struct())
        class_name = to_struct_type(obj_type).tag().as_string();
      if (class_name.rfind("tag-", 0) == 0)
        class_name = class_name.substr(4);
      if (!class_name.empty())
      {
        symbol_id method_sid(current_python_file, class_name, func_name);
        func_symbol = symbol_table_.find_symbol(method_sid.to_string());
      }
    }
  }

  if (func_symbol && func_symbol->get_type().is_code())
  {
    const code_typet &func_type = to_code_type(func_symbol->get_type());
    const typet &ret_type = func_type.return_type();

    if (lhs_type == "Any")
    {
      // For Any-annotated variables, always use the function's return type.
      current_element_type = ret_type;
      return ""; // Clear to avoid further "Any" processing
    }

    // If the callee's real return type is tagged, adopt it straight from
    // the symbol table rather than relying on the (skipped, see below)
    // probe build of this same call.
    if (
      ast_node.value("_inferred_annotation", false) &&
      type_handler_.is_tagged_scalar_type(ret_type))
    {
      current_element_type = ret_type;
      return lhs_type;
    }

    // Python type annotations are hints only and do not enforce runtime types.
    // When a function explicitly returns str (char*) but the variable is
    // annotated with a scalar type (e.g. y: int = f() where f() -> str),
    // use the actual return type so comparisons like y == "x" work correctly.
    bool ret_is_charptr =
      ret_type.is_pointer() && ret_type.subtype() == char_type();
    bool lhs_is_scalar =
      !current_element_type.is_pointer() && !current_element_type.is_array() &&
      !current_element_type.is_struct() && !current_element_type.id().empty();
    if (ret_is_charptr && lhs_is_scalar)
    {
      current_element_type = ret_type;
      return "";
    }
  }

  return lhs_type;
}

typet python_converter::resolve_any_subscript_array_type(
  const nlohmann::json &ast_node,
  const typet &current_type)
{
  if (
    current_type != any_type() || ast_node["value"].is_null() ||
    ast_node["value"].value("_type", std::string()) != "Subscript")
    return current_type;

  // Evaluate the actual subscript result before deciding anything: a fully
  // scalar access (`a[0, 0, 1]`) or an out-of-range/rank-mismatched index
  // must fall through unchanged (resp. propagate its own real error) rather
  // than be pre-empted by the 3-D source check below, which only applies
  // once we know the result itself is an array.
  exprt subscript_probe = get_expr(ast_node["value"]);
  if (contains_cpp_throw(subscript_probe))
    return current_type;

  const typet probed_type = ns.follow(subscript_probe.type());
  if (!probed_type.is_array())
    return current_type;

  // A supported N-D mixed slice/index tuple (exactly one full-slice axis `:`
  // and every other axis a literal/resolvable integer, e.g. `a[:, 0, 0]` -
  // see build_mixed_slice_tuple_select) legitimately produces a 1-D result
  // from a 3-D+ source, so it must skip the depth check below rather than be
  // rejected just because the source is deep.
  bool is_supported_mixed_slice_tuple = false;
  if (
    ast_node["value"].contains("slice") &&
    ast_node["value"]["slice"].value("_type", "") == "Tuple" &&
    ast_node["value"]["slice"].contains("elts"))
  {
    auto is_full_slice = [](const nlohmann::json &node) {
      if (node.value("_type", "") != "Slice")
        return false;
      auto absent = [&](const char *k) {
        return !node.contains(k) || node[k].is_null();
      };
      return absent("lower") && absent("upper") && absent("step");
    };
    auto is_literal_int = [](const nlohmann::json &node) {
      if (
        node.value("_type", "") == "Constant" && node.contains("value") &&
        node["value"].is_number_integer())
        return true;
      return node.value("_type", "") == "UnaryOp" && node.contains("op") &&
             node["op"].value("_type", "") == "USub" &&
             node.contains("operand") &&
             node["operand"].value("_type", "") == "Constant" &&
             node["operand"].contains("value") &&
             node["operand"]["value"].is_number_integer();
    };
    auto is_supported_slice = [&](const nlohmann::json &node) {
      if (is_full_slice(node))
        return true;
      for (const char *key : {"lower", "upper", "step"})
        if (
          node.contains(key) && !node[key].is_null() &&
          !is_literal_int(node[key]))
          return false;
      return true;
    };

    std::size_t slice_count = 0;
    bool all_slices_supported = true;
    for (const auto &elt : ast_node["value"]["slice"]["elts"])
    {
      if (elt.value("_type", "") != "Slice")
        continue;
      ++slice_count;
      if (!is_supported_slice(elt))
        all_slices_supported = false;
    }
    is_supported_mixed_slice_tuple = slice_count != 0 && all_slices_supported;
  }

  // Reject a source array of more than 2 dimensions: n-D indexing is out of
  // scope, and the resulting slice's nesting depth alone can't be told apart
  // from a legitimate 2-D row/column/fancy/mask selection (both are a
  // 2-level nested array), so the check has to look at the source instead.
  const nlohmann::json &source_node = ast_node["value"]["value"];
  exprt source_probe = get_expr(source_node);
  if (!contains_cpp_throw(source_probe) && !is_supported_mixed_slice_tuple)
  {
    std::size_t source_depth = 0;
    typet source_type = ns.follow(source_probe.type());
    // A numpy array crossing a function boundary is pointer-to-array (e.g.
    // int (*)[N][M]), not a plain array_typet; peel the pointer so the depth
    // walk below still sees through to the real dimensionality instead of
    // stopping at 0 and silently letting a 3-D+ source slip past.
    if (source_type.is_pointer())
      source_type = ns.follow(source_type.subtype());
    while (source_type.is_array())
    {
      ++source_depth;
      source_type = ns.follow(to_array_type(source_type).subtype());
    }
    if (source_depth > 2)
    {
      std::ostringstream msg;
      msg << "TypeError: assigning a 3-D+ array-typed subscript result to "
             "a variable is not supported";
      const locationt loc = get_location_from_decl(ast_node);
      if (!loc.is_nil())
        msg << " at " << loc.get_file() << ":" << loc.get_line();
      throw std::runtime_error(msg.str());
    }
  }

  // A materialized helper result (fancy/mask/column selection) is already a
  // plain symbol and copies via a normal whole-array code_assignt; a bare
  // `a[i]` chain is a raw index expression instead, which the final store
  // must copy element by element (see the flag's doc comment).
  any_subscript_array_needs_copy_ = !subscript_probe.is_symbol();

  // Reuse this exact conversion as the RHS later instead of converting the
  // same Subscript node again from scratch.
  cached_any_subscript_rhs_ = subscript_probe;
  has_cached_any_subscript_rhs_ = true;

  return probed_type;
}

typet python_converter::resolve_call_argument_array_type(
  const nlohmann::json &ast_node,
  const typet &current_type)
{
  if (
    ast_node["value"].is_null() ||
    ast_node["value"].value("_type", std::string()) != "Call")
    return current_type;

  typet followed_current_type = ns.follow(current_type);
  if (followed_current_type.is_array())
    return current_type;

  const nlohmann::json &call_node = ast_node["value"];
  std::optional<nlohmann::json> ret_val =
    select_return_value_for_call(call_node);
  if (!ret_val || !return_value_uses_call_argument(*ret_val, call_node))
    return current_type;

  nlohmann::json substituted = substitute_call_arguments(*ret_val, call_node);
  exprt call_probe = get_expr(substituted);
  if (contains_cpp_throw(call_probe))
    return current_type;

  typet probed_type = ns.follow(call_probe.type());
  if (probed_type.is_pointer())
    probed_type = ns.follow(probed_type.subtype());

  if (!probed_type.is_array())
    return current_type;

  any_subscript_array_needs_copy_ = !call_probe.is_symbol();
  cached_any_subscript_rhs_ = call_probe;
  has_cached_any_subscript_rhs_ = true;

  return probed_type;
}

typet python_converter::resolve_numpy_reducer_call_array_type(
  const nlohmann::json &ast_node,
  const typet &current_type)
{
  if (ast_node["value"].is_null())
    return current_type;

  const nlohmann::json &call_node = ast_node["value"];
  if (
    call_node.value("_type", "") != "Call" ||
    call_node["func"].value("_type", "") != "Attribute" ||
    call_node["func"]["value"].value("_type", "") != "Name")
    return current_type;

  // sum/mean/min/max/argmin/argmax's numpy.py signature necessarily declares
  // -> Any (its real shape is data-dependent: scalar when flattened, array
  // along an axis), so the static annotator's guess for a np.<reducer>(...)
  // call carrying axis= is unreliable -- sometimes Any (void*), sometimes a
  // plain scalar type inferred from the input literal's element type -- and
  // either one boxes/truncates the genuinely concrete array numpy_call_expr's
  // axis-aware fast paths compute. An axis= keyword is only ever legal on
  // these functions and only ever produces (on success) a 1-D array result,
  // so it is safe to always trust a probe of the real call over the guess.
  static const std::set<std::string> axis_aware_reducers = {
    "sum", "prod", "mean", "min", "max", "argmin", "argmax"};
  if (axis_aware_reducers.count(call_node["func"].value("attr", "")) == 0)
    return current_type;

  bool has_axis_keyword = false;
  for (const auto &kw : call_node.value("keywords", nlohmann::json::array()))
    if (kw.value("arg", "") == "axis")
      has_axis_keyword = true;
  if (!has_axis_keyword)
    return current_type;

  // imported_modules maps an alias ("np") to the resolved operational-model
  // file it was imported from, not the bare module name -- mirrors
  // function_call_builder::is_numpy_call's own filename check.
  const std::string module_alias =
    call_node["func"]["value"]["id"].get<std::string>();
  auto module_it = imported_modules.find(module_alias);
  if (
    module_it == imported_modules.end() ||
    !boost::algorithm::ends_with(module_it->second, "/models/numpy.py"))
    return current_type;

  is_converting_rhs = true;
  in_rhs_type_probe_ = true;
  exprt call_probe = get_expr(call_node);
  in_rhs_type_probe_ = false;
  is_converting_rhs = false;
  if (contains_cpp_throw(call_probe))
    return current_type;

  const typet probed_type = ns.follow(call_probe.type());
  if (probed_type.is_empty() || probed_type == any_type())
    return current_type;

  any_subscript_array_needs_copy_ = !call_probe.is_symbol();
  cached_any_subscript_rhs_ = call_probe;
  has_cached_any_subscript_rhs_ = true;

  return probed_type;
}

bool python_converter::handle_unpacking_assignment(
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  codet &target_block)
{
  const auto &target_type = target["_type"];

  if (target_type != "Tuple" && target_type != "List")
    return false;

  // Targets that write through an lvalue (a[i], obj.attr) need the normal
  // single-assignment store semantics (e.g. invalidating literal const-folding
  // so later reads see the write). When such a target is present and the RHS
  // is a tuple/list literal of matching arity, desugar into temp-mediated
  // single assignments so the swap a[i], a[j] = a[j], a[i] is sound (#4792).
  {
    const auto &targets = target["elts"];
    const auto &value = ast_node["value"];
    bool has_lvalue_target = false;
    bool only_name_or_lvalue = true;
    for (const auto &t : targets)
    {
      const auto &tt = t["_type"];
      if (tt == "Subscript" || tt == "Attribute")
        has_lvalue_target = true;
      else if (tt != "Name")
        only_name_or_lvalue = false; // Starred etc. — leave to existing paths
    }
    const bool rhs_is_literal_seq =
      value.is_object() && value.contains("_type") &&
      (value["_type"] == "Tuple" || value["_type"] == "List") &&
      value.contains("elts") && value["elts"].is_array();
    if (
      has_lvalue_target && only_name_or_lvalue && rhs_is_literal_seq &&
      value["elts"].size() == targets.size())
    {
      desugar_unpacking_with_lvalue_targets(ast_node, target, target_block);
      return true;
    }
  }

  // Get RHS
  is_converting_rhs = true;
  exprt rhs = get_expr(ast_node["value"]);
  is_converting_rhs = false;

  // Prepare RHS if it's a function call
  rhs = tuple_handler_->prepare_rhs_for_unpacking(ast_node, rhs, target_block);

  // Handle different unpacking types
  if (rhs.type().id() == "struct")
  {
    tuple_handler_->handle_tuple_unpacking(ast_node, target, rhs, target_block);
    return true;
  }
  else if (rhs.type().is_array())
  {
    handle_array_unpacking(ast_node, target, rhs, target_block);
    return true;
  }
  else if (rhs.type().is_pointer())
  {
    typet pointed_type = ns.follow(rhs.type().subtype());
    if (
      pointed_type.id() == "struct" &&
      tuple_handler_->is_tuple_type(pointed_type))
    {
      // V.3: build the deref in IREP2 (resolved tuple-struct pointee).
      exprt tuple_value = python_expr::build_dereference(rhs, pointed_type);
      tuple_value.location() = rhs.location();
      tuple_handler_->handle_tuple_unpacking(
        ast_node, target, tuple_value, target_block);
      return true;
    }

    const auto &value_node = ast_node["value"];
    if (value_node["_type"] == "List")
    {
      handle_list_literal_unpacking(ast_node, target, target_block);
      return true;
    }
    if (rhs.type() == get_type_handler().get_list_type())
    {
      python_list list(*this, ast_node["value"]);
      list.handle_list_var_unpacking(ast_node, target, rhs, target_block);
      return true;
    }
  }

  throw std::runtime_error(
    "Cannot unpack " + rhs.type().id_string() +
    " - only tuples and arrays can be unpacked");
}

void python_converter::desugar_unpacking_with_lvalue_targets(
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  codet &target_block)
{
  const auto &targets = target["elts"];
  const auto &values = ast_node["value"]["elts"];
  const size_t n = targets.size();

  // Per-statement-stable temp prefix; reusing the same names across repeated
  // evaluations of the statement (loops, repeated calls) is fine — they are
  // simply reassigned, like other frontend temporaries.
  const std::string prefix =
    "__ESBMC_unpack_" + std::to_string(reinterpret_cast<uintptr_t>(&ast_node)) +
    "_";

  // Build a Name AST node, cloning location fields from a nearby node.
  auto make_name =
    [&](const std::string &id, const nlohmann::json &loc_src, const char *ctx) {
      nlohmann::json node;
      node["_type"] = "Name";
      node["id"] = id;
      node["ctx"] = {{"_type", ctx}};
      copy_location_fields_from_decl(loc_src, node);
      return node;
    };

  // Build an `Assign` AST node for `tgt = val`.
  auto make_assign = [&](const nlohmann::json &tgt, const nlohmann::json &val) {
    nlohmann::json node;
    node["_type"] = "Assign";
    node["targets"] = nlohmann::json::array({tgt});
    node["value"] = val;
    copy_location_fields_from_decl(ast_node, node);
    return node;
  };

  // Phase 1: evaluate every RHS element into its own temporary. Python
  // evaluates the entire RHS before any assignment, so this snapshots the
  // values before the (possibly aliasing) stores below.
  std::vector<nlohmann::json> temps;
  temps.reserve(n);
  for (size_t i = 0; i < n; i++)
  {
    nlohmann::json tmp_tgt =
      make_name(prefix + std::to_string(i), values[i], "Store");
    nlohmann::json assign = make_assign(tmp_tgt, values[i]);
    get_var_assign(assign, target_block);
    temps.push_back(make_name(prefix + std::to_string(i), targets[i], "Load"));
  }

  // Phase 2: store each target from its snapshot temp via the normal
  // single-assignment path.
  for (size_t i = 0; i < n; i++)
  {
    nlohmann::json assign = make_assign(targets[i], temps[i]);
    get_var_assign(assign, target_block);
  }
}

symbolt *python_converter::create_symbol_for_unannotated_assign(
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  const symbol_id &sid,
  bool is_global)
{
  if (is_global)
    return nullptr;

  if (!ast_node.contains("value") || !ast_node["value"].contains("_type"))
    return nullptr;

  const std::string &value_type = ast_node["value"]["_type"];
  locationt location = get_location_from_decl(target);
  std::string module_name = location.get_file().as_string();
  std::string name;

  if (target["_type"] == "Name")
    name = target["id"].get<std::string>();
  else if (target["_type"] == "Attribute")
    name = target["attr"].get<std::string>();

  typet inferred_type;

  if (value_type == "Lambda")
  {
    inferred_type = any_type();
  }
  else if (
    value_type == "Call" &&
    ast_node["value"]["func"].value("_type", "") == "Attribute" &&
    ast_node["value"]["func"]["value"].value("_type", "") == "Name")
  {
    // For dict method calls that emit instructions via
    // converter_.add_instruction() (pop, get, setdefault), calling get_expr()
    // here for type inference would execute the side effects a second time when
    // the actual assignment is processed.  Pop is especially harmful: the first
    // evaluation removes the key, so the second evaluation can't find it and
    // throws KeyError. Instead, infer the return type directly from the dict's
    // value annotation.
    const std::string &method =
      ast_node["value"]["func"]["attr"].get<std::string>();
    const std::string &obj_name =
      ast_node["value"]["func"]["value"]["id"].get<std::string>();

    // Disambiguate by checking the actual symbol type, not just the annotation,
    // so that unannotated dict variables are also handled correctly.
    symbol_id obj_sid = create_symbol_id();
    obj_sid.set_object(obj_name);
    const symbolt *obj_sym = symbol_table_.find_symbol(obj_sid.to_string());

    // Value-returning dict methods emit IR instructions as a side-effect and
    // must not be called via get_expr() during type inference (double-eval).
    bool is_dict_method =
      python_dict_handler::is_value_returning_method(method) &&
      obj_sym != nullptr &&
      dict_handler_->is_dict_type(ns.follow(obj_sym->get_type()));

    if (is_dict_method)
    {
      // obj_sym != nullptr is guaranteed by the is_dict_method check above
      if (method == "popitem")
      {
        // popitem() returns (key, value) tuple — infer the full tuple type
        inferred_type =
          dict_handler_->get_popitem_tuple_type(symbol_expr(*obj_sym));
      }
      else if (method == "copy")
      {
        // copy() returns a new dict, not a single element value.
        inferred_type = dict_handler_->get_dict_struct_type();
      }
      else
      {
        inferred_type = dict_handler_->resolve_expected_type_for_dict_subscript(
          symbol_expr(*obj_sym));
        if (inferred_type.is_nil() || inferred_type.is_empty())
        {
          // Untyped dict (e.g. `a = {}`): infer the return type from the
          // default arg. Any concrete literal (list, dict, int, float, str,
          // bool, None) is more precise than the `long_int` fallback applied
          // just below.
          const std::string shape =
            python_annotation_utils::infer_type_from_default_arg_shape(
              ast_node["value"]["args"]);
          if (shape == "list")
            inferred_type = type_handler_.get_list_type();
          else if (shape == "dict")
            inferred_type = dict_handler_->get_dict_struct_type();
          else if (
            !shape.empty() && shape != "Any" &&
            type_utils::is_builtin_type(shape))
            inferred_type = type_handler_.get_typet(shape, 0);
          if (inferred_type.is_nil() || inferred_type.is_empty())
            inferred_type = long_int_type();
        }
      }
    }
    else
    {
      is_converting_rhs = true;
      in_rhs_type_probe_ = true;
      exprt rhs_expr = get_expr(ast_node["value"]);
      in_rhs_type_probe_ = false;
      is_converting_rhs = false;
      inferred_type = rhs_expr.type();
      if (inferred_type.is_empty())
        inferred_type = any_type();
      else if (inferred_type.is_code())
        inferred_type = gen_pointer_type(inferred_type);
    }
  }
  else
  {
    // Evaluate the RHS for any expression type (Call, BoolOp, Attribute,
    // Name, BinOp, Subscript, …) so that its type can be inferred.
    // If the expression is itself invalid — e.g. accessing a non-existent
    // attribute — get_expr will raise the correct, precise error at the
    // point of access rather than the misleading "Type undefined" later.
    is_converting_rhs = true;
    in_rhs_type_probe_ = true;
    exprt rhs_expr = get_expr(ast_node["value"]);
    in_rhs_type_probe_ = false;
    is_converting_rhs = false;

    inferred_type = rhs_expr.type();
    if (inferred_type.is_empty())
      inferred_type = any_type();
    // Function alias assignment (g = f): store as function pointer,
    // mirroring how lambda assignments are handled.
    else if (inferred_type.is_code())
      inferred_type = gen_pointer_type(inferred_type);
  }

  symbolt symbol =
    create_symbol(module_name, name, sid.to_string(), location, inferred_type);
  symbol.lvalue = true;
  symbol.file_local = true;
  symbol.is_extern = false;
  return symbol_table_.move_symbol_to_context(symbol);
}

/// The bare name an AST node denotes: `id` for a Name, `attr` for an Attribute
/// -- a qualified `module.Class`, which is what a base inherited from an
/// operational model looks like. \p fallback covers any other shape.
static std::string
ast_node_name(const nlohmann::json &node, const std::string &fallback = "")
{
  if (node.contains("id"))
    return node["id"].get<std::string>();
  return node.value("attr", fallback);
}

/// Replace \p dest's recorded element types with \p src's, but only when every
/// one of src's entries has the same type. sorted()/reversed() permute their
/// argument, so a per-position copy would misattribute the elements of a
/// heterogeneous list; a homogeneous one is permutation-invariant.
static void
copy_homogeneous_elem_types(const std::string &src, const std::string &dest)
{
  const size_t n = python_list::get_list_type_map_size(src);
  if (n == 0)
    return;

  const typet first = python_list::get_list_element_type(src, 0);
  if (first.is_nil())
    return;
  for (size_t i = 1; i < n; ++i)
    if (python_list::get_list_element_type(src, i) != first)
      return;

  python_list::copy_type_info(src, dest);
}

/// sorted()/reversed()/list() reorder or copy their argument, they do not
/// retype it, so the result's elements are the argument's. Without this the
/// runtime path leaves the result untyped and a tuple element reads back as an
/// int -- `for u, v in sorted(d, key=d.__getitem__)` then fails to unpack.
/// Only reached when nothing more precise has typed the destination.
const nlohmann::json *
python_converter::reordering_builtin_arg(const nlohmann::json &ast_node)
{
  if (!ast_node.contains("value") || !ast_node["value"].is_object())
    return nullptr;

  const auto &call = ast_node["value"];
  if (
    !call.contains("func") || !call["func"].is_object() ||
    call["func"].value("_type", "") != "Name")
    return nullptr;

  const std::string builtin = call["func"].value("id", "");
  if (builtin != "sorted" && builtin != "reversed" && builtin != "list")
    return nullptr;

  if (!call.contains("args") || call["args"].empty())
    return nullptr;

  return &call["args"][0];
}

void python_converter::copy_elem_types_from_reordering_builtin(
  const nlohmann::json &ast_node,
  const std::string &lhs_id)
{
  const nlohmann::json *arg_p = reordering_builtin_arg(ast_node);
  if (arg_p == nullptr)
    return;
  const auto &arg = *arg_p;

  if (arg.value("_type", "") == "Name")
  {
    symbol_id arg_sid = create_symbol_id();
    arg_sid.set_object(arg["id"].get<std::string>());
    copy_homogeneous_elem_types(arg_sid.to_string(), lhs_id);
    return;
  }

  // The dict-iterating form: the preprocessor rewrites `sorted(d, ...)` to
  // `sorted(d.keys(), ...)`, so the argument is a call, not a name.
  if (
    arg.value("_type", "") != "Call" || !arg.contains("func") ||
    arg["func"].value("_type", "") != "Attribute" ||
    arg["func"]["value"].value("_type", "") != "Name")
    return;

  const std::string component = arg["func"].value("attr", "");
  if (component != "keys" && component != "values")
    return;

  symbol_id dict_sid = create_symbol_id();
  dict_sid.set_object(arg["func"]["value"]["id"].get<std::string>());
  // Named local, not a temporary argument: GCC's -Wdangling-reference flags
  // binding a reference to a call whose arguments are temporaries, even though
  // get_internal_list_id returns into a static map.
  const std::string dict_id = dict_sid.to_string();
  const std::string &src =
    python_dict_handler::get_internal_list_id(dict_id, component == "keys");
  copy_homogeneous_elem_types(src, lhs_id);
}

void python_converter::handle_function_call_rhs(
  const nlohmann::json &ast_node,
  symbolt *lhs_symbol,
  exprt &lhs,
  exprt &rhs,
  const locationt &location,
  bool is_ctor_call,
  codet &target_block)
{
  if (is_ctor_call)
  {
    std::string func_name = ast_node_name(ast_node["value"]["func"]);

    if (base_ctor_called)
    {
      auto class_node = json_utils::find_class((*ast_json)["body"], func_name);
      func_name = ast_node_name(class_node["bases"][0], func_name);
      base_ctor_called = false;
    }

    update_instance_from_self(func_name, func_name, lhs_symbol->id.as_string());
  }
  else
  {
    // The callee may not be in the symbol table yet when the called
    // function is defined later in the module than its call site (a forward
    // reference, e.g. `def f(): w = make()` with `make` defined afterwards).
    // The block below only propagates instance-attribute type hints from the
    // callee's return object to the LHS; it does not affect the GOTO call
    // itself, which is built from the function identifier elsewhere. When the
    // callee symbol is not available yet, skip the best-effort copy instead of
    // aborting.
    symbolt *func_symbol =
      symbol_table_.find_symbol(rhs.op1().identifier().c_str());
    if (
      func_symbol && !static_cast<const code_typet &>(func_symbol->get_type())
                        .return_type()
                        .is_empty())
    {
      if (auto ret = get_return_from_func(func_symbol->id.c_str());
          !ret.is_nil())
      {
        copy_instance_attributes(
          ret.op0().identifier().as_string(), lhs_symbol->id.as_string());
      }
    }
  }

  // Copy attributes from function arguments
  if (!is_ctor_call)
  {
    const code_function_callt &call =
      static_cast<const code_function_callt &>(rhs);
    for (const auto &arg : call.arguments())
    {
      const exprt *arg_ptr = &arg;
      if (arg.is_address_of())
        arg_ptr = &arg.op0();

      if (arg_ptr->is_symbol())
      {
        copy_instance_attributes(
          arg_ptr->identifier().as_string(), lhs_symbol->id.as_string());
      }
    }
  }

  // Stage 1 object-model migration (#3067): the callee now returns a class
  // *reference* (`Cls*`), but the assignment target may still have been typed
  // as the value struct `Cls` — the annotator infers a value-struct type for an
  // RHS it cannot see through (an imported function, or a subscript dispatching
  // to `__getitem__` that returns `self`). Binding `rhs.op0() = lhs` then
  // stores a pointer into a struct slot, and the later `lhs.field` read trips
  // value-set's make_member assertion (#4513/#4514, transitive-imports). Retype
  // the target to the returned `Cls*` so the reference is bound directly and
  // the field read auto-dereferences, matching the other migrated assignment
  // paths. Restricted to a plain symbol target (`x = ...`): for a
  // subscript/attribute target `lhs_symbol` is the *container/base* symbol
  // while `lhs` is the element/member expression, so retyping `lhs_symbol`
  // would corrupt the whole container — the sibling migrations guard on a Name
  // target for the same reason.
  if (
    !is_ctor_call && lhs_symbol && lhs.is_symbol() &&
    is_user_class_pointer(rhs.type()) && is_user_class_struct_type(lhs.type()))
  {
    lhs.type() = rhs.type();
    lhs_symbol->set_type(rhs.type());
  }

  // Set return destination
  if (rhs.type().is_pointer() && !is_ctor_call)
  {
    rhs.op0() = lhs;
  }
  else if (!rhs.type().is_pointer() && !rhs.type().is_empty() && !is_ctor_call)
    rhs.op0() = lhs;

  // Special handling for list return type
  if (rhs.type() == type_handler_.get_list_type())
  {
    if (auto ret = get_return_from_func(rhs.op1().identifier().c_str());
        !ret.is_nil())
    {
      python_list::copy_type_info(
        ret.op0().identifier().as_string(), lhs.identifier().as_string());
    }

    // If list_type_map is still empty for the LHS
    // e.g. the list was passed through as a parameter inside the function,
    // fall back to the called function's return-type annotation
    // to determine the element type.
    const std::string &lhs_id = lhs.identifier().as_string();
    copy_elem_types_from_reordering_builtin(ast_node, lhs_id);
    if (python_list::get_list_type_map_size(lhs_id) == 0)
    {
      std::string func_name;
      if (
        ast_node.contains("value") && ast_node["value"].contains("func") &&
        ast_node["value"]["func"].is_object())
      {
        const auto &func_ref = ast_node["value"]["func"];
        if (func_ref.contains("id") && func_ref["id"].is_string())
          func_name = func_ref["id"].get<std::string>();
        else if (func_ref.contains("attr") && func_ref["attr"].is_string())
          func_name = func_ref["attr"].get<std::string>();
      }

      if (!func_name.empty())
      {
        const auto &func_def =
          json_utils::try_find_function((*ast_json)["body"], func_name);
        if (
          !func_def.empty() && func_def.contains("returns") &&
          !func_def["returns"].is_null())
        {
          const auto &returns = func_def["returns"];
          // Handle list[T] annotation
          // Subscript node with value.id == "list"
          if (
            returns.is_object() && returns.contains("_type") &&
            returns["_type"] == "Subscript" && returns.contains("value") &&
            returns["value"].is_object() && returns["value"].contains("id") &&
            returns["value"]["id"].is_string())
          {
            const std::string val_id =
              returns["value"]["id"].get<std::string>();
            if (val_id == "list" || val_id == "List")
            {
              // Extract element type from the slice, e.g. int in list[int]
              if (
                returns.contains("slice") && returns["slice"].is_object() &&
                returns["slice"].contains("id") &&
                returns["slice"]["id"].is_string())
              {
                typet elem_type = type_handler_.get_typet(
                  returns["slice"]["id"].get<std::string>());
                if (elem_type != typet())
                {
                  python_list::add_type_info_entry(
                    lhs_id, std::string(), elem_type);
                }
              }
            }
          }
        }
      }
    }

    typet l_type = type_handler_.get_list_type();
    symbolt &tmp_var_symbol =
      create_tmp_symbol(ast_node, "tmp_var", l_type, gen_zero(l_type));

    code_declt tmp_var_decl(symbol_expr(tmp_var_symbol));
    tmp_var_decl.location() = get_location_from_decl(ast_node);
    target_block.copy_to_operands(tmp_var_decl);

    rhs.op0() = symbol_expr(tmp_var_symbol);
    target_block.copy_to_operands(rhs);

    code_assignt code_assign(lhs, symbol_expr(tmp_var_symbol));
    code_assign.location() = location;
    rhs = code_assign;
  }

  target_block.copy_to_operands(rhs);
  mirror_numpy_transpose_assignment_from_targets(
    ast_node, lhs, location, target_block);
}

exprt python_converter::handle_string_literal_rhs(
  const nlohmann::json &ast_node,
  const std::string &lhs_type,
  const exprt &rhs)
{
  if (lhs_type != "str" || !type_utils::is_integer_type(rhs.type()))
    return rhs;

  if (
    ast_node["value"]["_type"] != "Constant" ||
    !ast_node["value"]["value"].is_string())
    return rhs;

  std::string str_value = ast_node["value"]["value"].get<std::string>();

  typet string_type =
    type_handler_.build_array(char_type(), str_value.length() + 1);
  exprt str_array = gen_zero(string_type);

  for (size_t i = 0; i < str_value.length(); ++i)
  {
    BigInt char_val(static_cast<unsigned char>(str_value[i]));
    exprt char_expr = constant_exprt(
      integer2binary(char_val, 8), integer2string(char_val), char_type());
    str_array.operands().at(i) = char_expr;
  }

  return str_array;
}

bool python_converter::is_global_variable(const symbol_id &sid) const
{
  for (const std::string &s : global_declarations)
  {
    if (s == sid.global_to_string())
      return true;
  }
  return false;
}

// np.ravel(a): the array is the call's first argument (this is the shape
// the preprocessor's .flat rewrite always produces). a.ravel(): the array
// is the Attribute's own receiver. Empty if unresolvable.
nlohmann::json python_converter::get_ravel_receiver_node(
  const nlohmann::json &ravel_call) const
{
  if (!ravel_call["func"].contains("value"))
    return nlohmann::json();

  const nlohmann::json &func_value = ravel_call["func"]["value"];
  const bool is_module_form =
    func_value.is_object() && func_value.value("_type", "") == "Name" &&
    is_imported_numpy_module_alias(*ast_json, func_value.value("id", ""));

  if (!is_module_form)
    return func_value;

  if (
    ravel_call.contains("args") && ravel_call["args"].is_array() &&
    !ravel_call["args"].empty())
    return ravel_call["args"][0];

  return nlohmann::json();
}

bool python_converter::is_numpy_ravel_receiver(
  const nlohmann::json &ravel_call) const
{
  const nlohmann::json receiver = get_ravel_receiver_node(ravel_call);
  const std::string receiver_name = root_name_from_subscript(receiver);
  if (receiver_name.empty())
    return false;

  const std::string receiver_id = resolve_name_symbol_id(receiver_name);
  return !receiver_id.empty() &&
         (numpy_array_symbols_.count(receiver_id) != 0 ||
          numpy_view_copy_sources_.count(receiver_id) != 0);
}

namespace
{
// True when a ravel Call node carries an order argument (positional or
// keyword), regardless of its literal value. Same scope limit as
// try_build_ravel_pointer_view: order='F' (or anything but the default)
// flattens column-major and is a copy in real NumPy, not the pointer view
// -- a directly-written a.ravel('F')[i] or np.ravel(a, order='F')[i] must
// not match flat_subscript_receiver_node. The `.flat` rewrite that
// function's main target never carries an order argument itself (`.flat`
// always iterates C-order), so this only ever declines a genuine
// explicit-order ravel call.
//
// This checks the raw, unrewritten Call node -- unlike
// numpy_call_expr::handle_ravel_pointer_view_attempt(), which only ever
// sees a call already normalised to module form by
// build_numpy_method_rewrite_node (receiver spliced in as args[0]) --
// so a directly-written method call still has order at args[0], not
// args[1]: is_module_form must be checked before picking the threshold.
bool ravel_call_has_order_arg(
  const nlohmann::json &ast,
  const nlohmann::json &value_node)
{
  const bool is_module_form =
    value_node.contains("func") && value_node["func"].contains("value") &&
    value_node["func"]["value"].is_object() &&
    value_node["func"]["value"].value("_type", "") == "Name" &&
    is_imported_numpy_module_alias(
      ast, value_node["func"]["value"].value("id", ""));
  const std::size_t order_index = is_module_form ? 1 : 0;

  if (
    value_node.contains("args") && value_node["args"].is_array() &&
    value_node["args"].size() > order_index)
    return true;

  return value_node.contains("keywords") && value_node["keywords"].is_array() &&
         std::any_of(
           value_node["keywords"].begin(),
           value_node["keywords"].end(),
           [](const nlohmann::json &kw) {
             return kw.value("_type", "") == "keyword" &&
                    !kw["arg"].is_null() && kw["arg"] == "order";
           });
}
} // namespace

// Detects Subscript(value=Call(ravel(a)), slice=i) -- the shape every
// .flat access (read or assignment target) is rewritten to -- and returns
// a's receiver node, or a null json if the shape doesn't match or a isn't a
// tracked numpy array/view. Shared by try_handle_flat_index_assignment and
// try_build_flat_index_read.
nlohmann::json python_converter::flat_subscript_receiver_node(
  const nlohmann::json &subscript_node) const
{
  if (
    subscript_node.value("_type", "") != "Subscript" ||
    !subscript_node.contains("value") || !subscript_node.contains("slice"))
    return nlohmann::json();

  const nlohmann::json &value_node = subscript_node["value"];
  if (
    value_node.value("_type", "") != "Call" || !value_node.contains("func") ||
    value_node["func"].value("_type", "") != "Attribute" ||
    value_node["func"].value("attr", "") != "ravel" ||
    !is_numpy_ravel_receiver(value_node) ||
    ravel_call_has_order_arg(*ast_json, value_node))
    return nlohmann::json();

  return get_ravel_receiver_node(value_node);
}

// a.flat[i] = x: target here is really Subscript(value=Call(ravel(a)),
// slice=i) -- see extract_target_name's comment for why that shape has no
// symbol id to extract the normal way. Handled as a special case ahead of
// the generic Subscript-target path: builds a's flat pointer view inline
// (try_build_flat_index_assignment_target, same eligibility/math as
// np.ravel(a) itself) and emits the write directly, bypassing the
// symbol-based lhs machinery entirely. Returns false (handles nothing) for
// any shape flat_subscript_receiver_node doesn't recognise (including a
// non-numpy-tracked receiver), leaving those to extract_target_name's
// existing "not supported" diagnostic.
bool python_converter::try_handle_flat_index_assignment(
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  codet &target_block)
{
  const nlohmann::json receiver = flat_subscript_receiver_node(target);
  if (receiver.is_null())
    return false;

  // a.flat[i] = x has no Subscript target for reject_unsafe_numpy_view_target
  // to walk (the target is a rewritten ravel Call), but the write is exactly
  // as unsafe as a[i] = x would be if some other live view still depends on
  // a's current storage -- apply the same check against the resolved root.
  const std::string root_name = root_name_from_subscript(receiver);
  if (!root_name.empty())
  {
    const std::string root_id = resolve_name_symbol_id(root_name);
    if (!root_id.empty())
    {
      reject_unsafe_numpy_view_write_to(root_id);
      if (has_numpy_transpose_view_of(root_id))
        throw std::runtime_error(
          "TypeError: mutation through .flat with a live numpy transpose view "
          "is not supported");
    }
  }

  exprt array_expr = get_expr(receiver);
  python_list list(*this, ast_node);
  std::optional<exprt> lhs =
    list.try_build_flat_index_assignment_target(array_expr, target["slice"]);
  if (!lhs)
    throw std::runtime_error(
      "TypeError: mutation through .flat is not supported");

  is_converting_rhs = true;
  exprt rhs = get_expr(ast_node["value"]);
  is_converting_rhs = false;

  code_assignt assign(*lhs, rhs);
  assign.location() = get_location_from_decl(ast_node);
  target_block.copy_to_operands(assign);
  return true;
}

// a.flat[i] (read): the generic Subscript path nulls current_lhs before
// converting the base, which correctly makes np.ravel(a) decline its own
// pointer-view path and fall back to an independent copy for a genuinely
// nested use -- but that copy is reconstructed from a's *literal*
// declaration, stale against any runtime mutation of a since
// (flat_mutation_source_write_success). Reading through the same pointer
// math the assignment-target path uses
// (try_build_flat_index_assignment_target, used here as an rvalue) keeps
// this observing a's live buffer like every other pointer view. Returns
// nullopt for any shape flat_subscript_receiver_node doesn't recognise, and
// the generic Subscript path handles it unchanged.
std::optional<exprt>
python_converter::try_build_flat_index_read(const nlohmann::json &element)
{
  const nlohmann::json receiver = flat_subscript_receiver_node(element);
  if (receiver.is_null())
    return std::nullopt;

  exprt *saved_lhs = current_lhs;
  current_lhs = nullptr;
  exprt array_expr = get_expr(receiver);
  current_lhs = saved_lhs;

  python_list list(*this, element);
  return list.try_build_flat_index_assignment_target(
    array_expr, element["slice"]);
}

std::string
python_converter::extract_target_name(const nlohmann::json &target) const
{
  const auto &target_type = target["_type"];

  if (target_type == "Name")
    return target["id"].get<std::string>();
  else if (target_type == "Attribute")
    return target["attr"].get<std::string>();
  else if (target_type == "Subscript")
    // Recurse through nested Subscripts (e.g. board[0][0] = x) to reach the
    // root container's Name/Attribute, which carries the symbol id.
    //
    // a.flat[i] = x and np.ravel(a)[i] = x hit this recursion too (the
    // preprocessor rewrites every .flat access to np.ravel(a)): the value
    // here is a Call, with no symbol id to extract the normal way. That
    // shape is fully handled earlier, by try_handle_flat_index_assignment
    // (called ahead of extract_target_name in the Assign dispatch), which
    // either builds the write directly or throws its own diagnostic --
    // this recursion is never reached for it.
    return extract_target_name(target["value"]);

  throw std::runtime_error(
    "Unsupported assignment target type: " + target_type.get<std::string>());
}

std::string python_converter::annotated_optional_class(
  const nlohmann::json &annotation) const
{
  if (!annotation.is_object() || annotation.is_null())
    return "";

  std::string cls;
  // Optional[Class] : Subscript(value=Name("Optional"), slice=Name(Class))
  if (
    annotation.value("_type", "") == "Subscript" &&
    annotation.contains("value") && annotation["value"].contains("id") &&
    annotation["value"]["id"] == "Optional" && annotation.contains("slice") &&
    annotation["slice"].contains("id"))
    cls = annotation["slice"]["id"].get<std::string>();
  // `Class | None` (PEP 604) : BinOp(BitOr, left, right) with one side None.
  else if (
    annotation.value("_type", "") == "BinOp" && annotation.contains("op") &&
    annotation["op"].value("_type", "") == "BitOr" &&
    annotation.contains("left") && annotation.contains("right"))
  {
    auto is_none = [](const nlohmann::json &n) {
      return n.value("_type", "") == "Constant" && n.contains("value") &&
             n["value"].is_null();
    };
    auto name_id = [](const nlohmann::json &n) -> std::string {
      return n.value("_type", "") == "Name" && n.contains("id")
               ? n["id"].get<std::string>()
               : std::string();
    };
    if (is_none(annotation["right"]))
      cls = name_id(annotation["left"]);
    else if (is_none(annotation["left"]))
      cls = name_id(annotation["right"]);
  }

  if (cls.empty() || !json_utils::is_class(cls, *ast_json))
    return "";
  return cls;
}

void python_converter::preregister_global_variables(
  const nlohmann::json &ast_body)
{
  // Pre-register module-level annotated variable symbols so that class methods
  // can reference globals declared later in the source (Python LEGB rule).
  // Only annotated assignments (AnnAssign) carry enough type information for
  // symbol registration; plain Assign without annotation is skipped via the
  // nil-type guard below.
  for (const auto &element : ast_body)
  {
    if (element.value("_type", "") != "AnnAssign")
      continue;

    // Skip implicitly inferred annotations (plain Assign converted by the
    // annotator). Only preregister variables that the user explicitly annotated
    // (e.g., `x: SomeClass = ...`). Inferred globals like `l = [1, 2, 3]`
    // should not be visible inside functions that don't declare `global l`.
    if (element.value("_inferred_annotation", false))
      continue;

    // Skip union-type forward declarations (e.g., `x: str | datetime`).
    // These are bare declarations with no value and the union type cannot be
    // reliably resolved at this stage. The variable will be registered when
    // the actual assignment is processed (after imports are loaded).
    if (
      element.contains("annotation") && !element["annotation"].is_null() &&
      element["annotation"].value("_type", "") == "BinOp" &&
      element.contains("value") && element["value"].is_null())
      continue;

    if (!element.contains("target"))
      continue;

    const auto &target = element["target"];
    if (!target.contains("id"))
      continue;

    const std::string var_name = target["id"].get<std::string>();

    symbol_id sid(current_python_file, "", "");
    sid.set_object(var_name);

    if (symbol_table_.find_symbol(sid.to_string()))
      continue;

    typet var_type;
    // None/Optional unification (#4653/#4796), step B: a global annotated
    // `Optional[Class]` / `Class | None` is a nullable class reference;
    // register it as `Class*` (a zeroable NULL pointer) so it unifies with the
    // pointer instances assigned to it, instead of the legacy pointer-width
    // None handle. The class struct is completed by the main class-build loop;
    // the global's own value (NULL) needs no complete struct, so no build is
    // required here.
    std::string opt_cls;
    if (element.contains("annotation"))
      opt_cls = annotated_optional_class(element["annotation"]);
    if (!opt_cls.empty())
    {
      typet st = type_handler_.get_typet(opt_cls);
      if (st.id() == "symbol" || st.is_struct())
        var_type = gen_pointer_type(st);
    }
    // A default-constructed typet has an empty id (""), which is neither "nil"
    // nor "empty"; the Optional path above only sets var_type for nullable
    // class annotations. For every other annotated global, resolve its real
    // type with extract_type_info so the symbol is pre-registered correctly:
    // a method that references a module global declared later in the file
    // (Python LEGB) must resolve to this symbol (e.g. `counter: int =
    // nondet_int()` — github_3851_4), and a `str` global must be char[N], not
    // an empty placeholder that later decays to char* and forces a runtime
    // strlen on subscript (github_2885).
    if (var_type.is_nil() || var_type.id().empty())
    {
      try
      {
        var_type = extract_type_info(element).second;
      }
      catch (const std::exception &e)
      {
        // Type not yet resolvable (e.g., from an unprocessed import). Skip for
        // now; the variable will be registered when the assignment is processed
        // after imports are loaded.
        log_warning(
          "preregister_global_variables: skipping '{}' ({})",
          element["target"].value("id", "<unknown>"),
          e.what());
        continue;
      }
    }

    // Object-model migration (#3067/#4773): a plain user-class global
    // (`m: C`, whose value is a constructor/call/alias) is registered as a
    // migrated `C*` reference by get_var_assign, NOT here. Pre-registering a
    // speculative `C*` corrupts that construction — it surfaces as
    // "Unexpected type in int/ptr typecast" at SMT encoding
    // (github_4541/github_2997). Skip user-class structs and let get_var_assign
    // register them. An Optional[C] global took the opt_cls pointer path above
    // and is intentionally kept: it is a pointer, not a struct, so it is not
    // caught here.
    if (is_user_class_struct_type(var_type))
      continue;
    // Skip when no usable type resolved (empty/nil placeholder, e.g. a bare
    // annotation whose type is not yet known); get_var_assign registers it
    // later with the real type.
    if (var_type.is_nil() || var_type.is_empty() || var_type.id().empty())
      continue;

    locationt location = get_location_from_decl(element);
    std::string module_name =
      current_python_file.substr(0, current_python_file.find_last_of("."));

    symbolt symbol =
      create_symbol(module_name, var_name, sid.to_string(), location, var_type);
    symbol.lvalue = true;
    // Module-level Python globals are not file-local: they are visible
    // across the entire program. rw_set.cpp uses (mode == "Python" &&
    // !file_local) to recognise them as race-eligible shared state,
    // since the Python frontend leaves static_lifetime=false to avoid
    // the C-side static-init pass picking up its const-prop snapshot.
    symbol.file_local = false;
    symbol.is_extern = false;

    symbol_table_.move_symbol_to_context(symbol);
  }
}

std::string python_converter::flow_lvalue_path(const nlohmann::json &node) const
{
  if (!node.is_object())
    return "";
  const std::string k = node.value("_type", "");
  if (k == "Name" && node.contains("id") && node["id"].is_string())
    return node["id"].get<std::string>();
  if (
    k == "Attribute" && node.contains("attr") && node["attr"].is_string() &&
    node.contains("value") && node["value"].is_object() &&
    node["value"].value("_type", "") == "Name" &&
    node["value"].contains("id") && node["value"]["id"].is_string())
    return node["value"]["id"].get<std::string>() + "." +
           node["attr"].get<std::string>();
  return "";
}

std::string python_converter::flow_rhs_class(const nlohmann::json &rhs) const
{
  if (!rhs.is_object())
    return "";
  const std::string k = rhs.value("_type", "");
  if (
    k == "Call" && rhs.contains("func") && rhs["func"].is_object() &&
    rhs["func"].value("_type", "") == "Name" && rhs["func"].contains("id") &&
    rhs["func"]["id"].is_string())
  {
    const std::string cls = rhs["func"]["id"].get<std::string>();
    return json_utils::is_class(cls, *ast_json) ? cls : std::string();
  }
  if (k == "Name" && rhs.contains("id") && rhs["id"].is_string())
  {
    auto it = flow_class_map_.find(rhs["id"].get<std::string>());
    if (it != flow_class_map_.end())
      return it->second;
  }
  return "";
}

std::string python_converter::call_return_class(const nlohmann::json &rhs) const
{
  if (
    !rhs.is_object() || rhs.value("_type", "") != "Call" ||
    !rhs.contains("func") || !rhs["func"].is_object() ||
    rhs["func"].value("_type", "") != "Name" || !rhs["func"].contains("id") ||
    !rhs["func"]["id"].is_string())
    return "";

  // Constructor calls (`Cls(...)`) are handled by the dedicated path above.
  const std::string fname = rhs["func"]["id"].get<std::string>();
  if (json_utils::is_class(fname, *ast_json))
    return "";

  const auto &fn = json_utils::try_find_function((*ast_json)["body"], fname);
  if (fn.empty() || !fn.contains("returns") || fn["returns"].is_null())
    return "";

  // An @overload stub carries a single-class annotation (`-> Foo`) but the
  // overload set is polymorphic — the real return type is resolved per call
  // site from the argument types. try_find_function returns the first stub, so
  // trusting its annotation would mis-type, e.g., a `Literal["bar"] -> Bar`
  // result as `Foo*` (#3057). Leave such results to the existing call-site
  // overload resolution rather than forcing a single Class* here.
  if (json_utils::has_overload_decorator(fn))
    return "";

  // The return annotation may be a bare Name (`-> Cls`) or a forward-reference
  // string (`-> "Cls"`).
  const auto &ret = fn["returns"];
  std::string cls;
  if (ret.value("_type", "") == "Name" && ret.contains("id"))
    cls = ret["id"].get<std::string>();
  else if (
    ret.value("_type", "") == "Constant" && ret.contains("value") &&
    ret["value"].is_string())
    cls = ret["value"].get<std::string>();

  return json_utils::is_class(cls, *ast_json) ? cls : std::string();
}

symbolt *python_converter::mint_retyped_symbol(
  const symbolt &orig,
  const std::string &alias_key,
  const typet &new_type,
  const locationt &location,
  const symbol_id &sid,
  codet &target_block)
{
  std::string new_id;
  unsigned gen = 1;
  do
  {
    new_id = alias_key + "$ret" + std::to_string(gen++);
  } while (symbol_table_.find_symbol(new_id) != nullptr);

  symbolt new_symbol = create_symbol(
    location.get_file().as_string(),
    orig.name.as_string(),
    new_id,
    location,
    new_type);
  new_symbol.lvalue = true;
  new_symbol.file_local = orig.file_local;
  new_symbol.is_extern = false;

  symbolt *new_symbol_ptr = symbol_table_.move_symbol_to_context(new_symbol);

  // Locals need a declaration; module globals are not declared.
  if (!current_func_name_.empty() && !is_global_variable(sid))
  {
    code_declt decl(symbol_expr(*new_symbol_ptr));
    decl.location() = location;
    target_block.copy_to_operands(decl);
  }

  retype_aliases_[alias_key] = new_id;
  return new_symbol_ptr;
}

/// The Name a single-target Assign/AnnAssign binds, or a null json.
static nlohmann::json assign_name_target(const nlohmann::json &ast_node)
{
  const std::string stmt_type = ast_node.value("_type", "");
  nlohmann::json target;
  if (
    stmt_type == "Assign" && ast_node.contains("targets") &&
    ast_node["targets"].size() == 1)
    target = ast_node["targets"][0];
  else if (stmt_type == "AnnAssign" && ast_node.contains("target"))
    target = ast_node["target"];

  if (target.is_object() && target.value("_type", "") == "Name")
    return target;
  return nlohmann::json();
}

bool python_converter::try_tagged_var_assign(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  const nlohmann::json tag_target = assign_name_target(ast_node);
  if (tag_target.is_null())
    return false;

  const std::string name = tag_target["id"].get<std::string>();
  symbol_id tag_sid = create_symbol_id();
  tag_sid.set_object(name);
  const std::string tag_key = tag_sid.to_string();
  bool is_tagged_already = dynamic_type_handler_.is_tagged(name);

  // A binop between two already-tagged names may produce a result whose
  // type isn't known until conversion, so an untagged target may need to
  // become tagged too. Checked by name to avoid converting the operands
  // twice on the common path where this doesn't apply.
  auto is_tagged_name = [&](const nlohmann::json &operand) {
    return operand.is_object() && operand.value("_type", "") == "Name" &&
           dynamic_type_handler_.is_tagged(operand["id"].get<std::string>());
  };
  bool value_may_tag = false;
  if (!is_tagged_already && ast_node.contains("value"))
  {
    const auto &value = ast_node["value"];
    value_may_tag = value.is_object() && value.value("_type", "") == "BinOp" &&
                    value.contains("op") &&
                    dynamic_type_handler_.tagged_binop_result_may_be_tagged(
                      value["op"].value("_type", "")) &&
                    value.contains("left") && value.contains("right") &&
                    is_tagged_name(value["left"]) &&
                    is_tagged_name(value["right"]);
  }

  // A rebind that already retyped the name away from its tagged slot wins: the
  // live value is in the retype target, so this is an ordinary assignment to
  // that symbol (#7075).
  if (retype_aliases_.count(tag_key) || (!is_tagged_already && !value_may_tag))
    return false;

  if (ast_node.contains("value") && !ast_node["value"].is_null())
  {
    const locationt location = get_location_from_decl(ast_node);
    exprt rhs = get_expr(ast_node["value"]);
    if (type_handler_.is_tagged_scalar_type(rhs.type()))
    {
      if (value_may_tag)
        dynamic_type_handler_.declare_dynamic_type_names({name}, ast_node);
      dynamic_type_handler_.assign_tagged_object(
        rhs, location, name, target_block);
      return true;
    }
    assert(
      !value_may_tag && "tagged 'x + y' always converts to a tagged result");
    if (
      type_handler_.is_numeric_scalar_type(rhs.type()) ||
      type_handler_.is_string_type(rhs.type()))
    {
      dynamic_type_handler_.assign(rhs, location, name, target_block);
      return true;
    }

    // The tagged slot's payload is a fixed-width scalar copy, so it cannot
    // hold a container or an object. Python rebinds the name outright, so give
    // the new value its own slot and redirect later loads to it, exactly as
    // the numeric<->string retype does (#7075). Restricted to the
    // unconditional spine: inside a conditional body retype_aliases_ is
    // reverted at the join, which would leave later reads observing the stale
    // tagged value instead of the container.
    symbolt *tag_symbol =
      symbol_table_.find_symbol(dynamic_type_handler_.tagged_symbol_id(name));
    if (
      tag_symbol && current_class_name_.empty() && loop_body_depth_ == 0 &&
      block_nesting_ == function_body_depth_ + 1 && !rhs.type().is_empty() &&
      !rhs.type().is_code())
    {
      symbolt *fresh = mint_retyped_symbol(
        *tag_symbol, tag_key, rhs.type(), location, tag_sid, target_block);
      code_assignt assign(symbol_expr(*fresh), rhs);
      assign.location() = location;
      target_block.copy_to_operands(assign);
      return true;
    }
  }

  throw std::runtime_error(
    "assigning a value of this type to a dynamically-typed variable "
    "is not yet supported");
}

nlohmann::json
python_converter::rewrite_assign_rhs_node(const nlohmann::json &ast_node) const
{
  nlohmann::json effective_ast_node = ast_node;
  if (
    ast_node.contains("value") && ast_node["value"].is_object() &&
    ast_node["value"].value("_type", "") == "Attribute" &&
    ast_node["value"].value("attr", "") == "T" &&
    ast_node["value"].contains("value"))
  {
    std::string numpy_alias = "np";
    for (const auto &entry : imported_modules)
    {
      if (entry.second == "numpy")
      {
        numpy_alias = entry.first;
        break;
      }
    }

    nlohmann::json module_name;
    module_name["_type"] = "Name";
    module_name["id"] = numpy_alias;
    module_name["ctx"] = {{"_type", "Load"}};
    copy_location_fields_from_decl(ast_node["value"], module_name);

    nlohmann::json call_node;
    call_node["_type"] = "Call";
    call_node["func"] = {
      {"_type", "Attribute"},
      {"value", module_name},
      {"attr", "transpose"},
      {"ctx", {{"_type", "Load"}}}};
    call_node["args"] = nlohmann::json::array({ast_node["value"]["value"]});
    call_node["keywords"] = nlohmann::json::array();
    copy_location_fields_from_decl(ast_node["value"], call_node);
    copy_location_fields_from_decl(ast_node["value"], call_node["func"]);
    effective_ast_node["value"] = call_node;
  }
  else if (ast_node.contains("value") && ast_node["value"].is_object())
  {
    if (
      std::optional<nlohmann::json> rewritten =
        rewrite_numpy_method_call_node(ast_node["value"]))
      effective_ast_node["value"] = std::move(*rewritten);
  }

  return effective_ast_node;
}

void python_converter::propagate_dict_member_list_type_info(
  const exprt &rhs,
  const std::string &lhs_identifier)
{
  const exprt &dict_sym = rhs.op0();
  // get_component_name() returns an irep_idt by value; bind the string
  // by value so it is copied out before that temporary is destroyed
  // (GCC -Wdangling-reference under -Werror).
  const std::string component =
    to_member_expr(rhs).get_component_name().as_string();
  if (!dict_sym.is_symbol() || (component != "keys" && component != "values"))
    return;

  const std::string &dict_id = dict_sym.identifier().as_string();
  const std::string &src =
    python_dict_handler::get_internal_list_id(dict_id, component == "keys");
  if (src.empty())
    return;

  python_list::copy_type_info(src, lhs_identifier);

  // Tuple values are recorded under the $dict_value_types$ key, not the
  // values-list id (github_3719_4), so the copy above is a no-op for them.
  // Propagate the stored tuple struct type so the generic list tuple-element
  // read resolves it.
  if (component != "values")
    return;

  typet tuple_t = dict_handler_->recorded_tuple_value_type(dict_sym);
  if (
    !tuple_t.is_nil() && !tuple_t.is_empty() &&
    python_list::get_list_type_map_size(lhs_identifier) == 0)
    python_list::add_type_info_entry(lhs_identifier, std::string(), tuple_t);
}

void python_converter::propagate_list_type_info(
  const exprt &lhs,
  const exprt &rhs,
  symbolt *lhs_symbol)
{
  const std::string &lhs_identifier = lhs.identifier().as_string();
  const std::string &rhs_identifier = rhs.identifier().as_string();
  python_list::copy_type_info(rhs_identifier, lhs_identifier);

  // When rhs is dict_sym.keys / dict_sym.values (a member expression
  // rather than a list symbol), rhs_identifier is empty and
  // copy_type_info above is a no-op.  Look up the dict's internal
  // keys-list or values-list symbol and propagate from there instead.
  if (rhs_identifier.empty() && rhs.id() == exprt::member)
    propagate_dict_member_list_type_info(rhs, lhs_identifier);

  if (lhs_symbol)
  {
    const symbolt *rhs_symbol = nullptr;
    if (rhs.is_symbol())
      rhs_symbol = find_symbol(rhs.identifier().as_string());
    if (rhs_symbol && rhs_symbol->is_set)
      lhs_symbol->is_set = true;
  }
}

void python_converter::get_var_assign(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  if (try_tagged_var_assign(ast_node, target_block))
    return;

  // Extract type information
  auto [lhs_type, element_type] = extract_type_info(ast_node);

  // Check if the RHS is a dictionary literal - set the element type
  set_dict_literal_element_type(ast_node, *dict_handler_, element_type);

  current_element_type = element_type;
  any_subscript_array_needs_copy_ = false;
  has_cached_any_subscript_rhs_ = false;
  typet annotated_type = element_type;
  std::vector<typet> annotation_types;
  bool can_emit_annotation_check = false;
  locationt annotation_location;
  std::string annotated_name;
  std::vector<typet> annotation_candidates;

  exprt lhs;
  symbolt *lhs_symbol = nullptr;
  locationt location_begin;
  symbol_id sid = create_symbol_id();
  // Set wherever a `code_declt` is already emitted for `lhs_symbol` during
  // symbol creation below, so the any_subscript_array_needs_copy_ copy-loop
  // further down (which needs its own decl when nothing declared the symbol
  // yet, e.g. the plain-Assign path) does not emit a second one for the same
  // symbol.
  bool lhs_already_declared = false;

  const auto &target = (ast_node.contains("targets")) ? ast_node["targets"][0]
                                                      : ast_node["target"];

  if (
    ast_node.contains("value") && ast_node["value"].is_object() &&
    contains_tracked_numpy_view_name(ast_node["value"]))
  {
    if (target.value("_type", "") == "Attribute")
      throw std::runtime_error(
        "TypeError: storing a copied numpy view in an attribute is not "
        "supported");

    if (
      target.value("_type", "") == "Name" && !current_func_name_.empty() &&
      target.contains("id"))
    {
      symbol_id target_sid = create_symbol_id();
      target_sid.set_object(target["id"].get<std::string>());
      if (is_global_variable(target_sid))
        throw std::runtime_error(
          "TypeError: storing a copied numpy view in a global is not "
          "supported");
    }
  }

  if (ast_node.contains("value") && ast_node["value"].is_object())
  {
    reject_numpy_view_identity_query(ast_node["value"]);
    reject_unknown_numpy_view_call(ast_node["value"]);
  }

  // Stage 1 object-model migration (#3067/#4773): a simple Name target bound to
  // a class instance — either a constructor call `o = ClassName(...)` or an
  // alias `b = a` of an existing instance — becomes a *reference* (pointer) to
  // the object, matching CPython's reference semantics. This makes escaping
  // objects survive their defining function and makes `b = a` a pointer copy
  // (shared object) rather than a struct copy. Type the LHS as pointer-to-class
  // up front, before the declaration is emitted below; function_call_expr then
  // allocates the object and passes the pointer as `self`.
  if (
    ast_node.contains("value") && !ast_node["value"].is_null() &&
    target.contains("_type") && target["_type"] == "Name")
  {
    std::string cls;
    if (
      type_handler_.is_constructor_call(ast_node["value"]) &&
      ast_node["value"]["func"].contains("id"))
      cls = ast_node["value"]["func"]["id"].get<std::string>();
    else
      cls = flow_rhs_class(ast_node["value"]); // aliasing: `b = a`
    // `y = f(...)` where f returns a class: type y as `Cls*` so the returned
    // reference is bound (not value-copied into a struct local).
    if (cls.empty())
      cls = call_return_class(ast_node["value"]);
    // A target *explicitly annotated* as a user class (`v: Cls = obj.method()`)
    // is an instance reference too: bind it as `Cls*` so a migrated class
    // return is not value-copied into a struct slot (which mismatches the
    // pointer the callee now returns and trips value-set's make_member
    // assertion). This is the only path covering an annotated *method*-call
    // return — call_return_class above handles only plain `Name` function
    // calls.
    //
    // Gate on the *resolved* annotation type via is_user_class_struct_type —
    // the same predicate the funcdef return migration uses — not on the
    // annotation *name* through json_utils::is_class: the latter also matches
    // the built-in model classes `Tuple`/`List`/`int`, so `coord: Coordinate`
    // (= `Tuple[int, int]`) would be mistyped as a pointer-to-tuple and fault
    // on read. And require a *user-written* annotation: the annotator injects
    // an inferred `annotation` on plain assignments, naming a class for an RHS
    // that yields no instance — `_ = B() + B()` infers `B` though `__add__`
    // returns an int, `b = create("bar")` infers an overload's `Bar` though the
    // callee returns a value (#3057/#3091/#3286/#3921).
    if (
      cls.empty() && ast_node.contains("annotation") &&
      !ast_node["annotation"].is_null() &&
      !ast_node.value("_inferred_annotation", false) &&
      is_user_class_struct_type(current_element_type))
      cls = lhs_type;
    if (!cls.empty())
    {
      typet st = type_handler_.get_typet(cls);
      if (st.id() == "symbol" || st.is_struct())
      {
        current_element_type = gen_pointer_type(st);
        element_type = current_element_type;
      }
    }
  }

  // None/Optional unification (#4653/#4796), step B: a local target annotated
  // `Optional[Class]` / `Class | None` is a nullable class reference — type it
  // `Class*` (NULL for None) so it unifies with the pointer instances assigned
  // to it. Build the class on demand first (process_forward_reference) so its
  // struct symbol is complete, not a null/incomplete stub. Scoped to nullable
  // annotations of user classes only, so non-Optional class variables and the
  // object-lifetime flip are unaffected.
  if (
    target.contains("_type") && target["_type"] == "Name" &&
    !current_element_type.is_pointer() && ast_node.contains("annotation"))
  {
    const std::string opt_cls =
      annotated_optional_class(ast_node["annotation"]);
    if (!opt_cls.empty())
    {
      nlohmann::json cls_ref;
      cls_ref["_type"] = "Name";
      cls_ref["id"] = opt_cls;
      process_forward_reference(cls_ref, target_block);
      typet st = type_handler_.get_typet(opt_cls);
      if (st.id() == "symbol" || st.is_struct())
      {
        current_element_type = gen_pointer_type(st);
        element_type = current_element_type;
      }
    }
  }

  // Flow-sensitive class tracking (#4771/#4772): at an unconditional top-level
  // (depth-1) assignment, record the class most recently assigned to the target
  // lvalue ("v" or "v.attr"), last-write-wins. Read back in converter_expr to
  // resolve nested attribute access on a field the usage-site scanner left as
  // any_type(). Depth-1 gating + clearing on nested-body entry (get_block) keep
  // it from adopting a class across a control-flow join.
  if (
    block_nesting_ == 1 && ast_node.contains("value") &&
    !ast_node["value"].is_null())
  {
    const std::string path = flow_lvalue_path(target);
    if (!path.empty())
    {
      // Rebinding a bare variable `v` makes its previously-tracked attributes
      // (`v.attr`) refer to the old object; drop them so a later `v.attr` read
      // can't reuse a stale class.
      if (path.find('.') == std::string::npos)
      {
        const std::string prefix = path + ".";
        for (auto it = flow_class_map_.begin(); it != flow_class_map_.end();)
        {
          if (it->first.rfind(prefix, 0) == 0)
            it = flow_class_map_.erase(it);
          else
            ++it;
        }
      }
      std::string cls = flow_rhs_class(ast_node["value"]);
      if (cls.empty())
        cls =
          call_return_class(ast_node["value"]); // `v = f()` returning a class
      if (!cls.empty())
        flow_class_map_[path] = cls;
      else
        flow_class_map_.erase(path);
    }
  }

  // Handle forward references
  if (
    ast_node.contains("value") && !ast_node["value"].is_null() &&
    ast_node["value"]["_type"] == "Call" &&
    type_handler_.is_constructor_call(ast_node["value"]))
  {
    process_forward_reference(ast_node["value"]["func"], target_block);
  }

  // Handle dict subscript assignment: dict[key] = value
  if (dict_handler_->handle_subscript_assignment_check(
        *this, ast_node, target, target_block))
    return;

  if (target.contains("_type") && target["_type"] == "Subscript")
  {
    reject_unsafe_numpy_view_target(target);

    exprt container_expr = get_expr(target["value"]);
    typet container_type = container_expr.type();

    if (reject_immutable_item_assignment(container_type, target_block))
      return;

    // Handle object subscript assignment via __setitem__:
    //   obj[key] = value  ->  obj.__setitem__(key, value)
    if (
      target.contains("value") && target.contains("slice") &&
      ast_node.contains("value") && !ast_node["value"].is_null() &&
      has_dunder_method(target["value"], "__setitem__"))
    {
      nlohmann::json args = nlohmann::json::array();
      args.push_back(target["slice"]);
      args.push_back(ast_node["value"]);
      nlohmann::json call_node =
        build_dunder_call(target["value"], "__setitem__", args, ast_node);
      exprt setitem_call = get_function_call(call_node);
      target_block.copy_to_operands(convert_expression_to_code(setitem_call));
      return;
    }

    // List slice assignment (a[i:j:k] = ...) is lowered to the
    // __ESBMC_list_slice_assign model, which mutates the target list in
    // place with CPython semantics. Falling through to the generic store
    // instead would evaluate get_expr(a[i:j]) — a *copy* of the slice — and
    // assign into that temporary, leaving the original list unchanged; a
    // later read then sees stale values, so ESBMC would report a buggy
    // program as SUCCESSFUL (silent unsoundness). Reject non-list containers
    // (e.g. strings) explicitly instead. Object slice __setitem__ is handled
    // above; tuples raise TypeError above; dict subscripts are handled by
    // handle_subscript_assignment_check earlier.
    if (
      target.contains("slice") && target["slice"].is_object() &&
      target["slice"].value("_type", "") == "Slice")
    {
      const namespacet ns(symbol_table_);
      const typet resolved_container = ns.follow(container_type);
      const typet resolved_list = ns.follow(type_handler_.get_list_type());
      const bool container_is_list =
        resolved_container == resolved_list ||
        (resolved_container.is_pointer() &&
         ns.follow(resolved_container.subtype()) == resolved_list);

      if (
        container_is_list && ast_node.contains("value") &&
        !ast_node["value"].is_null())
      {
        python_list list_handler(*this, target);
        list_handler.handle_slice_assignment(
          container_expr, target["slice"], ast_node["value"]);
        return;
      }

      reject_numpy_view_slice_assignment(target);
      throw std::runtime_error(
        "Slice assignment is only supported on list targets");
    }
  }

  if (ast_node["_type"] == "AnnAssign")
  {
    // Extract name and set in symbol ID
    std::string name = extract_target_name(target);
    sid.set_object(name);
    annotated_name = name;

    // Check if this is a forward declaration with union type and no value
    // e.g., dt: str | datetime (without assignment)
    // These should be skipped; wait for the actual assignment
    bool is_union_type = false;
    if (
      ast_node.contains("annotation") && !ast_node["annotation"].is_null() &&
      ast_node["annotation"].contains("_type") &&
      ast_node["annotation"]["_type"] == "BinOp")
    {
      is_union_type = true;
    }

    if (is_union_type && ast_node["value"].is_null())
    {
      // Skip this forward declaration; wait for the actual assignment
      // that will give us the type information
      return;
    }

    // Infer type from function return if annotation is "Any"
    lhs_type = infer_type_from_any_annotation(ast_node, lhs_type);

    // Process RHS before LHS if in function scope, or for a global that
    // infer_type_from_any_annotation above already resolved to a tagged
    // return type (needs checking against the actual RHS type).
    exprt rhs;
    if (
      (sid.to_string().find("@F") != std::string::npos &&
       sid.to_string().find("@C") == std::string::npos) ||
      (type_handler_.is_tagged_scalar_type(current_element_type) &&
       current_func_name_.empty()))
    {
      is_right = true;
      if (!ast_node["value"].is_null())
      {
        // This RHS build only exists to probe rhs.type() for the Any/char*
        // string adjustment below; the value is discarded and the real RHS is
        // built again later. Skip it for kinds whose type is statically known
        // and never a char* string, otherwise get_expr emits the whole
        // construction a second time as dead code. Dict literals were already
        // skipped (handled specially later); list literals and comprehensions
        // matter most — eliding their dead duplicate roughly halves
        // list-construction cost on construction-heavy programs (#5121). Calls
        // are skipped too, to avoid re-running their side effects (e.g. a
        // second list.pop()).
        const std::string rhs_kind =
          ast_node["value"].value("_type", std::string());
        const bool rhs_kind_skips_type_probe =
          dict_handler_->is_dict_literal(ast_node["value"]) ||
          rhs_kind == "List" || rhs_kind == "ListComp";
        if (!rhs_kind_skips_type_probe)
        {
          if (rhs_kind != "Call")
          {
            // Discarded probe: suppress the ZeroDivisionError guard so a
            // division here is not emitted (and its divisor not evaluated) an
            // extra time; the real RHS build below emits it once.
            in_rhs_type_probe_ = true;
            rhs = get_rhs_with_dict_resolution(ast_node, current_element_type);
            in_rhs_type_probe_ = false;
          }
        }
      }
      is_right = false;
    }

    // When the annotation resolves to `Any` (void*) but the RHS is a concrete
    // string (char*), adopt the string type for the symbol. Comprehension and
    // `for`-loop targets over an unannotated string parameter are lowered as
    // `char: Any = s[i]`; leaving the target void* makes list.append() miss the
    // string-pointer storage branch in build_list_push_call, so the element
    // pointer is byte-copied and corrupted (esbmc/esbmc#5158). This mirrors the
    // Call-RHS handling in infer_type_from_any_annotation for a non-Call RHS.
    // `rhs` is default-constructed (empty type) outside function scope, where
    // is_pointer() is false, so no extra nil guard is needed.
    if (
      current_element_type == any_type() && rhs.type().is_pointer() &&
      rhs.type().subtype() == char_type())
    {
      current_element_type = rhs.type();
    }

    // An inferred annotation is just a guess; adopt the tagged type if the
    // RHS turns out tagged, same as an Any-annotated target below.
    if (
      (current_element_type == any_type() ||
       ast_node.value("_inferred_annotation", false)) &&
      type_handler_.is_tagged_scalar_type(rhs.type()))
    {
      current_element_type = rhs.type();
    }

    current_element_type =
      resolve_any_subscript_array_type(ast_node, current_element_type);
    current_element_type =
      resolve_call_argument_array_type(ast_node, current_element_type);
    current_element_type =
      resolve_numpy_reducer_call_array_type(ast_node, current_element_type);

    // Location and symbol lookup
    location_begin = get_location_from_decl(target);
    annotation_location = location_begin;
    can_emit_annotation_check = true;
    lhs_symbol = symbol_table_.find_symbol(sid.to_string().c_str());

    bool is_global = is_global_variable(sid);
    if (is_global)
      lhs_symbol = symbol_table_.find_symbol(sid.global_to_string().c_str());

    // Symbol creation
    if (!lhs_symbol || !is_global)
    {
      std::string module_name = location_begin.get_file().as_string();

      symbolt symbol = create_symbol(
        module_name,
        name,
        sid.to_string(),
        location_begin,
        current_element_type);
      symbol.lvalue = true;
      // Module-level Python globals are not file-local (see
      // preregister_global_variables). Function-local annotated assigns
      // stay file-local.
      symbol.file_local = !current_func_name_.empty();
      symbol.is_extern = false;

      bool symbol_created = (lhs_symbol == nullptr);
      lhs_symbol = symbol_table_.move_symbol_to_context(symbol);

      if (!symbol_created)
        retype_placeholder_to_class(*lhs_symbol, current_element_type);

      // Add declaration statement ONLY for newly created local variables
      if (symbol_created && !current_func_name_.empty() && !is_global)
      {
        code_declt decl(symbol_expr(*lhs_symbol));
        decl.location() = location_begin;
        target_block.copy_to_operands(decl);
        lhs_already_declared = true;
      }
    }
    else
    {
      // A pre-registered module global reached from a `global` declaration
      // skips the branch above, so it never got the placeholder widening and
      // the constructor overran its scalar storage (#6243).
      retype_placeholder_to_class(*lhs_symbol, current_element_type);
    }

    if (lhs_symbol && ast_node.contains("annotation"))
      get_typechecker().cache_annotation_types(
        *lhs_symbol, ast_node["annotation"]);

    if (
      type_assertions_enabled() && lhs_symbol &&
      ast_node.contains("annotation"))
    {
      auto &tc = get_typechecker();
      annotation_types = tc.get_annotation_types(lhs_symbol->id.as_string());
      if (
        !annotation_types.empty() &&
        !tc.should_skip_type_assertion(annotated_type))
      {
        annotated_type = annotation_types.front();
        can_emit_annotation_check = true;
        annotation_location = location_begin;
        annotated_name = name;
        annotation_candidates = annotation_types;
      }
    }

    // Check for uninitialized usage
    if (lhs_symbol)
    {
      for (std::string &s : local_loads)
      {
        if (lhs_symbol->id.as_string() == s)
        {
          throw std::runtime_error(
            "Variable " + sid.get_object() + " in function " +
            current_func_name_ + " is uninitialized.");
        }
      }
    }

    // Create LHS expression
    lhs = create_lhs_expression(target, lhs_symbol, location_begin);

    reject_copied_numpy_view_in_container(ast_node, {"Dict"});

    // Handle dict literal assignment specially - after LHS is created
    if (dict_handler_->handle_literal_assignment_check(*this, ast_node, lhs))
    {
      if (type_assertions_enabled() && can_emit_annotation_check)
        get_typechecker().emit_type_annotation_assertion(
          lhs,
          annotated_type,
          annotation_types,
          annotated_name,
          annotation_location,
          target_block);
      return;
    }
  }
  else if (ast_node["_type"] == "Assign")
  {
    const auto &target = ast_node["targets"][0];
    location_begin = get_location_from_decl(target);

    // Handle tuple/list unpacking
    if (handle_unpacking_assignment(ast_node, target, target_block))
      return;

    // a.flat[i] = x
    if (try_handle_flat_index_assignment(ast_node, target, target_block))
      return;

    // Normal assignment handling
    std::string name = extract_target_name(target);
    sid.set_object(name);
    lhs_symbol = resolve_subscript_base_symbol(
      target, name, symbol_table_.find_symbol(sid.to_string()));

    bool is_global = is_global_variable(sid);

    reject_copied_numpy_view_in_container(ast_node, {"Dict"});

    // Handle unannotated dict literal assignment
    if (
      !lhs_symbol && dict_handler_->handle_unannotated_literal_check(
                       *this, ast_node, target, sid))
      return;

    // Create symbol for unannotated assignments with inferrable types.
    // If the annotator injected an "annotation" field and we already have a
    // valid type in current_element_type, use it directly so that user-defined
    // classes named "List"/"Dict" are not mis-resolved to built-in types.
    if (!lhs_symbol && !is_global)
    {
      if (
        ast_node.contains("annotation") && !ast_node["annotation"].is_null() &&
        !current_element_type.is_empty())
      {
        current_element_type =
          resolve_any_subscript_array_type(ast_node, current_element_type);
        current_element_type =
          resolve_call_argument_array_type(ast_node, current_element_type);

        std::string module_name = location_begin.get_file().as_string();
        symbolt symbol = create_symbol(
          module_name,
          name,
          sid.to_string(),
          location_begin,
          current_element_type);
        symbol.lvalue = true;
        // Inferred-annotation Assign: module-scope symbols are not
        // file-local (so rw_set recognises them); function-locals are.
        symbol.file_local = !current_func_name_.empty();
        symbol.is_extern = false;
        lhs_symbol = symbol_table_.move_symbol_to_context(symbol);
      }
      else
      {
        lhs_symbol = create_symbol_for_unannotated_assign(
          ast_node, target, sid, is_global);
      }
    }

    // The is_global branch above only locates pre-registered module-level
    // symbols; it does not synthesise one. If we still have no symbol here,
    // create_lhs_expression would dereference a null symbolt* in symbol_expr
    // and crash. For a verifier, refusing with a diagnostic is the only safe
    // behaviour — silently fabricating a symbol could change the verdict.
    if (!lhs_symbol)
      throw std::runtime_error("Type undefined for \"" + name + "\"");

    lhs = create_lhs_expression(target, lhs_symbol, location_begin);

    if (lhs_symbol && ast_node.contains("annotation"))
      get_typechecker().cache_annotation_types(
        *lhs_symbol, ast_node["annotation"]);

    if (type_assertions_enabled() && lhs_symbol)
    {
      auto &tc = get_typechecker();
      annotation_types = tc.get_annotation_types(lhs_symbol->id.as_string());
      if (
        !annotation_types.empty() &&
        !tc.should_skip_type_assertion(lhs_symbol->get_type()))
      {
        annotated_type = annotation_types.front();
        can_emit_annotation_check = true;
        annotation_location = location_begin;
        annotated_name = name;
        annotation_candidates = annotation_types;
      }
    }
  }

  if (
    type_assertions_enabled() && can_emit_annotation_check &&
    annotation_candidates.empty() &&
    !get_typechecker().should_skip_type_assertion(annotated_type))
    annotation_candidates.push_back(annotated_type);

  const bool has_value = has_non_null_value(ast_node);
  bool is_ctor_call =
    has_value && type_handler_.is_constructor_call(ast_node["value"]);
  current_lhs = &lhs;
  is_converting_lhs = false;

  // A bare `a = ...` rebind is invisible to reject_unsafe_numpy_view_target
  // (Subscript targets only); detach any live pointer-backed view of the
  // old `a` before its storage is overwritten below, so the view keeps
  // observing what it saw pre-rebind instead of silently switching to
  // whatever `a` is rebound to. Common to both the annotated and
  // unannotated assignment branches above -- the annotator may inject an
  // annotation onto what the user wrote as a plain `a = ...`, routing it
  // through either one.
  if (should_detach_numpy_pointer_views_for_assignment(
        target, ast_node, lhs_symbol))
    detach_numpy_pointer_views_of(
      lhs_symbol->id.as_string(), location_begin, target_block);

  reject_copied_numpy_view_in_container(ast_node, {"List", "Tuple", "Dict"});

  // Get RHS
  nlohmann::json effective_ast_node = rewrite_assign_rhs_node(ast_node);

  exprt rhs;
  bool converted_value = false;
  if (!effective_ast_node["value"].is_null())
  {
    if (
      has_cached_any_subscript_rhs_ &&
      !should_rebuild_cached_numpy_row_subscript_rhs(
        effective_ast_node["value"]))
    {
      // Already converted once by resolve_any_subscript_array_type's type
      // probe; reuse it rather than converting the same Subscript node (and
      // re-emitting any temporaries it built) a second time.
      rhs = cached_any_subscript_rhs_;
      has_cached_any_subscript_rhs_ = false;
    }
    else
    {
      has_cached_any_subscript_rhs_ = false;
      is_converting_rhs = true;

      if (lhs_symbol)
        rhs = get_rhs_with_dict_resolution(
          effective_ast_node, lhs_symbol->get_type());
      else
        rhs = get_expr(effective_ast_node["value"]);

      is_converting_rhs = false;
    }

    converted_value = true;

    // Handle string literal conversion
    rhs = handle_string_literal_rhs(effective_ast_node, lhs_type, rhs);
  }

  if (converted_value && rhs != exprt("_init_undefined"))
  {
    auto try_follow_symbol_type = [this](const typet &type) -> typet {
      if (type.id() != "symbol")
        return type;

      const irep_idt &symbol_id = type.identifier();
      if (symbol_id.empty())
        return type;

      if (symbol_table_.find_symbol(symbol_id) == nullptr)
        return type;

      return ns.follow(type);
    };

    auto is_list_model_type = [this,
                               &try_follow_symbol_type](const typet &in_type) {
      typet t = in_type;
      t = try_follow_symbol_type(t);
      if (t.is_pointer())
        t = t.subtype();
      t = try_follow_symbol_type(t);
      if (!t.is_struct())
        return false;
      return to_struct_type(t).tag().as_string().find("__ESBMC_PyListObj") !=
             std::string::npos;
    };

    auto resolve_runtime_type = [this,
                                 &try_follow_symbol_type](const exprt &expr) {
      typet t = expr.type();
      if (expr.is_symbol())
      {
        if (const symbolt *sym = symbol_table_.find_symbol(expr.identifier()))
          t = sym->get_type();
      }
      return try_follow_symbol_type(t);
    };

    const typet lhs_runtime_type =
      lhs_symbol ? lhs_symbol->get_type() : lhs.type();
    const typet rhs_runtime_type = resolve_runtime_type(rhs);
    if (
      dict_handler_->is_dict_type(lhs_runtime_type) &&
      is_list_model_type(rhs_runtime_type))
    {
      throw std::runtime_error(
        "Unsupported reassignment from dict to list for variable '" +
        sid.get_object() + "'");
    }

    // Handle throw expression
    if (rhs.statement() == "cpp-throw")
    {
      rhs.location() = location_begin;
      codet code_expr("expression");
      code_expr.operands().push_back(rhs);
      code_declt decl(symbol_expr(*lhs_symbol));
      decl.location() = location_begin;

      target_block.copy_to_operands(code_expr);
      target_block.copy_to_operands(decl);
      current_lhs = nullptr;
      return;
    }

    // Dynamic retyping (#4770, #4774). A variable whose current static type is
    // a numeric scalar is being reassigned a string value (or vice versa). The
    // GOTO IR binds one type per symbol, so the new value cannot be stored in
    // the old slot. We model the rebinding by minting a fresh symbol of the new
    // type, declaring it, and redirecting later loads of the name to it via
    // retype_aliases_ (keyed by the function-qualified symbol id, so aliases
    // never leak between functions). We then fall through to the normal
    // assignment path with the new, correctly typed symbol.
    //
    // On the unconditional spine (block_nesting_ == function_body_depth_ + 1)
    // the alias persists for the rest of the body — there is no control-flow
    // join that could leave the runtime type ambiguous. Inside an if/else/try
    // body the same rebinding is applied for in-body reads, but get_block
    // snapshots retype_aliases_ on body entry and restores it on exit, so the
    // retype does not leak across the join (the post-join view keeps the
    // pre-conditional type — see retype_str_cond_gated).
    //
    // Refused inside a while/for body (loop_body_depth_ > 0): a loop target
    // variable leaks past the loop in Python, so the retyped value must remain
    // visible after the body rather than be reverted at the join. Leaving the
    // rebinding to the existing fallback there preserves the pre-#5716 verdict
    // (see github_3647_15_fail). Class bodies are excluded via
    // current_class_name_: their attribute symbols are managed separately by
    // python_class_builder.
    if (
      current_class_name_.empty() && lhs_symbol && lhs.is_symbol() &&
      loop_body_depth_ == 0)
    {
      // The LHS lookup above returns the ORIGINAL symbol. If this variable was
      // already retyped, its live value lives in the alias target; resolve to
      // it so a further retype is detected against the current type and a
      // same-type write lands in the live slot (mirrors the load redirect in
      // converter_expr). The alias is always keyed by the original id, which is
      // what loads resolve before redirecting.
      const std::string orig_id = lhs_symbol->id.as_string();
      auto existing = retype_aliases_.find(orig_id);
      if (existing != retype_aliases_.end())
      {
        if (symbolt *live = symbol_table_.find_symbol(existing->second))
        {
          lhs_symbol = live;
          lhs = symbol_expr(*live);
          current_lhs = &lhs;
        }
      }

      // A tuple-struct variable (adopted from an explicit-Any binding) being
      // rebound to a non-tuple value needs the same fresh slot: retyping the
      // shared symbol would turn earlier member reads into member-of-scalar,
      // which trips member2t's source-type assert at GOTO conversion.
      // Code-typed and unknown (void*) rhs are excluded exactly as in the
      // propagate branch of handle_assignment_type_adjustments: neither
      // yields a usable alias slot type.
      const bool tuple_to_nontuple_rebind =
        tuple_handler_->is_tuple_type(lhs.type()) && !rhs.type().is_empty() &&
        !tuple_handler_->is_tuple_type(rhs.type()) && !rhs.type().is_code() &&
        !(rhs.type().is_pointer() && rhs.type().subtype().id() == "empty");

      if (
        is_incompatible_scalar_string_retype(
          type_handler_, lhs.type(), rhs.type()) ||
        tuple_to_nontuple_rebind)
      {
        symbolt *new_symbol_ptr = mint_retyped_symbol(
          *lhs_symbol, orig_id, rhs.type(), location_begin, sid, target_block);

        lhs_symbol = new_symbol_ptr;
        lhs = symbol_expr(*new_symbol_ptr);
        current_lhs = &lhs;
        // Fall through: the normal assignment handling below now stores rhs
        // into the new, type-matched symbol.
      }
    }

    // Python dynamic typing: if a variable already has a numeric type (e.g.
    // double from float()) and is being reassigned to a pointer/string type
    // (e.g. char* from chr()), the GOTO IR cannot represent this type change
    // safely — the old SSA constant and the new pointer type mismatch in both
    // the symex renamer and the SMT encoder. Skip the assignment so the prior
    // type and value are preserved. This is sound for verification as long as
    // the new value is not used in a subsequent assertion.
    if (
      lhs_symbol && !lhs.type().is_pointer() && rhs.type().is_pointer() &&
      rhs.type().subtype() ==
        char_type() && // only skip string (char*) reassignment, not None
                       // (void*/bool*)
      (lhs.type().is_floatbv() || lhs.type().is_signedbv() ||
       lhs.type().is_unsignedbv() || lhs.type().is_bool()))
    {
      // Still emit the RHS as a void call so exceptions/side-effects are
      // preserved (e.g. chr() out-of-range ValueError, or a method that mutates
      // `self` while returning a string that does not fit the annotated scalar
      // slot, as in `n: int = obj.method_returning_str()`). The call reaches
      // here either as a side_effect_function_call expression or as an
      // already-lowered code_function_call statement (nil result); handle both,
      // otherwise the call — and its side effects — would be dropped entirely.
      if (
        rhs.id() == "sideeffect" &&
        rhs.statement() == irep_idt("function_call"))
      {
        const side_effect_expr_function_callt &se =
          to_side_effect_expr_function_call(rhs);
        code_function_callt void_call;
        void_call.function() = se.function();
        void_call.arguments() = se.arguments();
        void_call.location() = rhs.location();
        add_instruction(void_call);
      }
      else if (rhs.is_code() && rhs.statement() == irep_idt("function_call"))
      {
        const code_function_callt &cc = to_code_function_call(to_code(rhs));
        code_function_callt void_call;
        void_call.function() = cc.function();
        void_call.arguments() = cc.arguments();
        void_call.location() = rhs.location();
        add_instruction(void_call);
      }
      current_lhs = nullptr;
      return;
    }

    // Handle type adjustments
    handle_assignment_type_adjustments(
      lhs_symbol, lhs, rhs, lhs_type, ast_node, is_ctor_call);

    // Propagate $input_str$ → $input_len$ companion mapping so that len()
    // on any alias of an input() string can use the symbolic length directly.
    // Must run before type-branching which may take early returns.
    if (lhs.is_symbol())
    {
      if (rhs.is_symbol())
      {
        const std::string rhs_id = rhs.identifier().as_string();
        auto it = input_str_to_len_sym_.find(rhs_id);
        if (it != input_str_to_len_sym_.end())
          input_str_to_len_sym_[lhs.identifier().as_string()] = it->second;
        else
          input_str_to_len_sym_.erase(lhs.identifier().as_string());
      }
      else
      {
        // RHS is not a symbol with a mapped input length: clear any stale
        // mapping
        input_str_to_len_sym_.erase(lhs.identifier().as_string());
      }
    }

    // Function call handling
    if (rhs.is_function_call())
    {
      // Static constructor compatibility check for annotated variables:
      // if var is annotated with a class type (e.g., Animal) and the RHS
      // constructor is a different, non-derived class (e.g., Car), inject
      // an assertion failure.
      if (
        type_assertions_enabled() && can_emit_annotation_check &&
        is_ctor_call && ast_node.contains("annotation") &&
        ast_node["annotation"].contains("id"))
      {
        std::string expected_base =
          ast_node["annotation"]["id"].get<std::string>();
        std::string ctor_name =
          get_typechecker().get_constructor_name(ast_node["value"]["func"]);

        if (
          !expected_base.empty() && !ctor_name.empty() &&
          !get_typechecker().class_derives_from(ctor_name, expected_base))
        {
          // V.3: build the always-fail assert condition in IREP2.
          code_assertt ctor_assert(migrate_expr_back(gen_false_expr()));
          ctor_assert.location() = location_begin;
          ctor_assert.location().comment(
            "Constructor '" + ctor_name +
            "' is incompatible with annotated type '" + expected_base + "'");
          target_block.copy_to_operands(ctor_assert);
        }
      }

      handle_function_call_rhs(
        effective_ast_node,
        lhs_symbol,
        lhs,
        rhs,
        location_begin,
        is_ctor_call,
        target_block);
      if (type_assertions_enabled() && can_emit_annotation_check)
        get_typechecker().emit_type_annotation_assertion(
          lhs,
          annotated_type,
          annotation_types,
          annotated_name,
          annotation_location,
          target_block);
      if (
        effective_ast_node.contains("value") &&
        effective_ast_node["value"].is_object())
        update_numpy_array_binding(lhs, effective_ast_node["value"]);
      else
        clear_numpy_view_copy(lhs);
      current_lhs = nullptr;
      return;
    }

    // Type-incompatible reassignment: scalar variable assigned a
    // string/array value. Must check BEFORE adjust_statement_types
    // which would coerce rhs type to match lhs, hiding the mismatch.
    // Only enforce when type assertions are enabled (--is-instance-check).
    if (
      type_assertions_enabled() && lhs.type() != rhs.type() &&
      !rhs.type().is_code() && !rhs.type().is_empty() &&
      rhs.type().is_array() && !lhs.type().is_array() &&
      !lhs.type().is_pointer())
    {
      // V.3: build the always-fail assert condition in IREP2.
      code_assertt type_assert(migrate_expr_back(gen_false_expr()));
      type_assert.location() = location_begin;
      type_assert.location().comment(
        "Type violation: incompatible types in assignment");
      target_block.copy_to_operands(type_assert);
      if (type_assertions_enabled() && can_emit_annotation_check)
        get_typechecker().emit_type_annotation_assertion(
          lhs,
          annotated_type,
          annotation_types,
          annotated_name,
          annotation_location,
          target_block);
      current_lhs = nullptr;
      return;
    }

    adjust_statement_types(lhs, rhs);

    // Handle list type info propagation. A subscript store rebinds an element,
    // not the container, and its lhs is an unnamed dereference -- propagating
    // would write the element's entries into the empty-key bucket that
    // attribute-rooted lists (self.xs) use as their map key (#7360).
    if (
      lhs.type() == rhs.type() && lhs.type() == type_handler_.get_list_type() &&
      !assignment_target_is_subscript(ast_node))
      propagate_list_type_info(lhs, rhs, lhs_symbol);
    else if (
      rhs.type() != lhs.type() && lhs.type().is_array() &&
      !rhs.type().is_code())
    {
      // Note: lhs.type() may be a fixed-size array (e.g. a string literal's
      // char array) whose size is a constant, so no nil-size invariant holds
      // here. The previous debug assert only passed because binding
      // `const array_typet& = lhs.type()` constructed a throwaway array (with
      // a nil size) rather than reinterpreting the real type; it asserted
      // nothing meaningful and is removed.
      lhs_symbol->set_type(rhs.type());

      code_declt decl(symbol_expr(*lhs_symbol), rhs);
      decl.location() = location_begin;
      target_block.copy_to_operands(decl);
      if (type_assertions_enabled() && can_emit_annotation_check)
        get_typechecker().emit_type_annotation_assertion(
          lhs,
          annotated_type,
          annotation_types,
          annotated_name,
          annotation_location,
          target_block);
      current_lhs = nullptr;
      return;
    }

    if (any_subscript_array_needs_copy_ && lhs.type().is_array())
    {
      any_subscript_array_needs_copy_ = false;

      const array_typet &dst_type = to_array_type(lhs.type());
      if (!dst_type.size().is_constant())
        throw std::runtime_error(
          "TypeError: assigning a symbolic-shape array-typed subscript "
          "result to a variable is not supported");

      if (!lhs_already_declared)
      {
        code_declt decl(symbol_expr(*lhs_symbol));
        decl.location() = location_begin;
        target_block.copy_to_operands(decl);
      }

      const typet elem_type = ns.follow(dst_type.subtype());
      const BigInt len = binary2integer(dst_type.size().value().c_str(), false);
      for (BigInt i = 0; i < len; ++i)
      {
        exprt idx = from_integer(i, size_type());
        exprt src_elem = python_expr::build_index(rhs, idx, elem_type);
        exprt dst_elem = python_expr::build_index(lhs, idx, elem_type);
        code_assignt elem_assign(dst_elem, src_elem);
        elem_assign.location() = location_begin;
        target_block.copy_to_operands(elem_assign);
      }

      if (type_assertions_enabled() && can_emit_annotation_check)
        get_typechecker().emit_type_annotation_assertion(
          lhs,
          annotated_type,
          annotation_types,
          annotated_name,
          annotation_location,
          target_block);
      update_numpy_array_binding(lhs, effective_ast_node["value"]);
      current_lhs = nullptr;
      return;
    }

    code_assignt code_assign(lhs, rhs);
    code_assign.location() = location_begin;
    target_block.copy_to_operands(code_assign);
    mirror_numpy_transpose_assignment(
      target, rhs, location_begin, target_block);
    mirror_numpy_reshape_assignment(target, rhs, location_begin, target_block);
    if (
      effective_ast_node.contains("value") &&
      effective_ast_node["value"].is_object())
      update_numpy_array_binding(lhs, effective_ast_node["value"]);
    else
      clear_numpy_view_copy(lhs);
    if (type_assertions_enabled() && can_emit_annotation_check)
      get_typechecker().emit_type_annotation_assertion(
        lhs,
        annotated_type,
        annotation_types,
        annotated_name,
        annotation_location,
        target_block);
  }
  else
  {
    {
      exprt v = gen_zero(current_element_type, true);
      v.zero_initializer(true);
      lhs_symbol->set_value(std::move(v));
    }

    code_declt decl(symbol_expr(*lhs_symbol));
    decl.location() = location_begin;
    target_block.copy_to_operands(decl);
  }

  current_lhs = nullptr;
}

typet python_converter::resolve_variable_type(
  const std::string &var_name,
  const locationt &loc)
{
  std::string function = loc.get_function().as_string();
  nlohmann::json decl_node = find_var_decl(var_name, function, *ast_json);

  if (!decl_node.empty())
  {
    if (decl_node.contains("annotation") && !decl_node["annotation"].is_null())
    {
      const auto &annotation = decl_node["annotation"];

      try
      {
        // Handle rich annotations such as Union, Optional, module attributes,
        // etc. via the unified helper.
        return get_type_from_annotation(annotation, decl_node);
      }
      catch (const std::exception &e)
      {
        log_warning(
          "Failed to resolve complex annotation for '{}': {}. Falling back to "
          "simple identifier lookup.",
          var_name,
          e.what());
      }

      if (annotation.contains("id"))
      {
        std::string type_annotation = annotation["id"].get<std::string>();
        return type_handler_.get_typet(type_annotation);
      }
    }
  }

  std::string filename = loc.get_file().as_string();
  std::string symbol_id = "py:" + filename + "@F@" + function + "@" + var_name;

  const symbolt *sym = symbol_table_.find_symbol(symbol_id);
  if (sym != nullptr)
    return sym->get_type();
  else
  {
    log_error(
      "Variable '{}' not found in symbol table; cannot determine type.",
      symbol_id);
    abort();
  }
}

// `xs += [...]` rebinds the list without growing its element records, so the
// declaring literal no longer describes it and its recorded length is stale.
void python_converter::mark_augassign_list_escaped(
  const exprt &lhs,
  const exprt &rhs)
{
  if (
    lhs.is_symbol() && rhs.type() == lhs.type() &&
    lhs.type() == type_handler_.get_list_type())
    mark_list_call_escaped(lhs.identifier().as_string());
}

// `c[0] = v` mutates c, it does not bind it, so a nested function's subscript
// target resolves against the enclosing scope. Minting a local here typed it
// from the RHS and refused the base as not subscriptable.
symbolt *python_converter::resolve_subscript_base_symbol(
  const nlohmann::json &target,
  const std::string &name,
  symbolt *found)
{
  if (found || target.value("_type", "") != "Subscript")
    return found;
  return find_symbol_in_enclosing_scopes(name);
}

symbolt *
python_converter::find_symbol_in_enclosing_scopes(const std::string &name)
{
  std::string scope = current_func_name_;
  for (size_t sep = scope.rfind("@F@"); sep != std::string::npos;
       sep = scope.rfind("@F@"))
  {
    scope.erase(sep);
    symbol_id sid = create_symbol_id();
    sid.set_function(scope);
    sid.set_object(name);
    if (symbolt *found = symbol_table_.find_symbol(sid.to_string()))
      return found;
  }
  return nullptr;
}

void python_converter::get_compound_assign(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  locationt loc = get_location_from_decl(ast_node);

  // Set flags for LHS processing
  is_converting_lhs = true;
  const nlohmann::json *saved_store_target = lhs_store_target_;
  lhs_store_target_ = &ast_node["target"];

  // Get the target expression first
  exprt lhs = get_expr(ast_node["target"]);

  // Reset LHS flag and set RHS flag
  lhs_store_target_ = saved_store_target;
  is_converting_lhs = false;
  is_converting_rhs = true;

  std::string var_name;

  // Extract variable name based on target type
  if (ast_node["target"].contains("id"))
  {
    // Simple variable assignment: x += 1
    var_name = ast_node["target"]["id"].get<std::string>();
  }
  else if (ast_node["target"]["_type"] == "Attribute")
  {
    // Don't extract just the attribute name for type resolution
    // The type should come from the LHS expression we just created
    if (ast_node["target"].contains("attr"))
      var_name = ast_node["target"]["attr"].get<std::string>();
  }
  else if (ast_node["target"]["_type"] == "Subscript")
  {
    // Subscript assignment: arr[i] += 1
    throw std::runtime_error(
      "Subscript assignment not supported in compound assignment");
  }
  else
  {
    throw std::runtime_error(
      "Unsupported target type in compound assignment: " +
      ast_node["target"]["_type"].get<std::string>());
  }

  // For attribute assignments, use the type from the LHS expression
  // For other assignments, resolve the variable type
  if (!lhs.type().is_nil() && !lhs.type().id().empty())
    current_element_type = lhs.type();
  else
  {
    // Fallback to resolving the variable type from AST or symbol table
    current_element_type = resolve_variable_type(var_name, loc);
  }

  std::string op = ast_node["op"]["_type"].get<std::string>();

  // Check if this is a string concatenation based on variable annotation
  bool is_string_concat = false;
  if (op == "Add")
  {
    // Standard array-based string concatenation
    if (
      (lhs.type().is_array() && lhs.type().subtype() == char_type()) ||
      (current_element_type.is_array() &&
       current_element_type.subtype() == char_type()))
    {
      is_string_concat = true;
    }
    // Pointer-based string
    else if (
      (lhs.type().is_pointer() && lhs.type().subtype() == char_type()) ||
      (current_element_type.is_pointer() &&
       current_element_type.subtype() == char_type()))
    {
      is_string_concat = true;
    }
    // Check if variable is annotated as str but implemented as single char
    else if (
      type_utils::is_integer_type(lhs.type()) &&
      type_utils::is_integer_type(current_element_type))
    {
      // Check if the variable was declared with str annotation
      nlohmann::json decl_node = get_var_node(var_name, *ast_json);
      if (
        !decl_node.empty() && decl_node.contains("annotation") &&
        decl_node["annotation"].contains("id") &&
        decl_node["annotation"]["id"] == "str")
      {
        is_string_concat = true;
      }
    }
  }

  if (is_string_concat)
  {
    exprt rhs_expr = get_expr(ast_node["value"]);
    nlohmann::json left = ast_node["target"];
    nlohmann::json right = ast_node["value"];
    exprt concatenated =
      string_handler_.handle_string_concatenation(lhs, rhs_expr, left, right);

    // Update the variable's type to match the concatenated result
    // Handle both array and pointer results
    if (
      !var_name.empty() && (concatenated.type().is_array() ||
                            (concatenated.type().is_pointer() &&
                             concatenated.type().subtype() == char_type())))
    {
      symbol_id sid = create_symbol_id();
      sid.set_object(var_name);
      symbolt *symbol = symbol_table_.find_symbol(sid.to_string());
      if (symbol)
      {
        // Update the symbol's type to pointer if concatenated returns pointer
        symbol->set_type(concatenated.type());

        // Update LHS to be a symbol with the new type
        lhs = symbol_exprt(symbol->id, symbol->get_type());

        // For pointer results, don't update the value
        // (it will be assigned via the assignment statement)
        if (concatenated.type().is_array())
        {
          symbol->set_value(concatenated);
        }
      }
    }

    code_assignt code_assign(lhs, concatenated);
    code_assign.location() = loc;
    target_block.copy_to_operands(code_assign);

    // Reset RHS flag
    is_converting_rhs = false;
    return;
  }

  exprt rhs = get_binary_operator_expr(ast_node);

  // Reset RHS flag
  is_converting_rhs = false;

  // P27: Promote real RHS to complex when LHS is complex (AugAssign path).
  // adjust_statement_types() is NOT called on this path, so without this
  // check, `z += 1.0` / `z *= 2` produce a struct/scalar type mismatch in IR.
  if (
    is_complex_type(lhs.type()) && !is_complex_type(rhs.type()) &&
    (rhs.type().is_floatbv() || type_utils::is_integer_type(rhs.type()) ||
     rhs.type().is_bool()))
  {
    rhs = promote_to_complex(rhs);
  }

  mark_augassign_list_escaped(lhs, rhs);

  code_assignt code_assign(lhs, rhs);
  code_assign.location() = loc;
  target_block.copy_to_operands(code_assign);
}

typet resolve_ternary_type(
  const typet &then_type,
  const typet &else_type,
  const typet &default_type)
{
  if (then_type == else_type)
    return then_type;

  // Enhanced numeric promotion: int < float
  if (type_utils::is_integer_type(then_type) && else_type.is_floatbv())
    return else_type;
  if (type_utils::is_integer_type(else_type) && then_type.is_floatbv())
    return then_type;

  // String handling: use pointer type for consistency
  // Handles: array+array, array+pointer, pointer+array
  bool then_is_string =
    (then_type.is_array() && then_type.subtype() == char_type()) ||
    (then_type.is_pointer() && then_type.subtype() == char_type());
  bool else_is_string =
    (else_type.is_array() && else_type.subtype() == char_type()) ||
    (else_type.is_pointer() && else_type.subtype() == char_type());

  if (then_is_string && else_is_string)
    return gen_pointer_type(char_type());

  // Both arrays (non-strings)
  if (then_type.is_array() && else_type.is_array())
    return then_type;

  // Mixed signed/unsigned integers - prefer signed for safety
  if (then_type.is_signedbv() && else_type.is_unsignedbv())
    return then_type;
  if (then_type.is_unsignedbv() && else_type.is_signedbv())
    return else_type;

  // Incompatible types
  log_debug(
    "python-frontend",
    "[resolve_ternary_type] Ternary branches have incompatible types: {} vs "
    "{}, using default {}",
    then_type.id_string(),
    else_type.id_string(),
    default_type.id_string());

  return default_type;
}

bool python_converter::contains_named_expr(const nlohmann::json &node)
{
  if (node.is_object())
  {
    if (node.value("_type", "") == "NamedExpr")
      return true;
    for (auto it = node.begin(); it != node.end(); ++it)
      if (contains_named_expr(it.value()))
        return true;
  }
  else if (node.is_array())
  {
    for (const auto &e : node)
      if (contains_named_expr(e))
        return true;
  }
  return false;
}

/// Truth-testing an object calls its __bool__ when the class defines one. The
/// caller has already resolved @p value_type to the struct behind @p cond
/// (following a pointer if need be); a non-struct or a class without the dunder
/// comes back unchanged.
/// `not obj` is a truth test, so it dispatches __bool__ exactly as `if obj:`
/// does. Kept as its own entry point so the unary-operator conversion stays a
/// single statement.
exprt python_converter::apply_bool_dunder_for_not(
  const std::string &op,
  exprt operand,
  const locationt &location)
{
  return op == "Not" ? apply_bool_dunder(operand, location) : operand;
}

exprt python_converter::apply_bool_dunder(exprt cond, const locationt &location)
{
  typet value_type = ns.follow(cond.type());
  if (value_type.is_pointer())
    value_type = ns.follow(value_type.subtype());
  if (!value_type.is_struct())
    return cond;

  const std::string class_name =
    extract_class_name_from_tag(to_struct_type(value_type).tag().as_string());
  if (class_name.empty())
    return cond;

  symbolt *bool_method = find_dunder_method(class_name, "__bool__");
  if (!bool_method)
    return cond;

  // __bool__ expects self by reference. A migrated instance is already a
  // `Class*` pointer (pass it through); a by-value struct must be a named
  // object whose address we take.
  const bool object_is_ptr = cond.type().is_pointer();
  if (!object_is_ptr && !cond.is_symbol())
    cond = store_call_result(cond, location, "cond_obj");

  side_effect_expr_function_callt bool_call;
  bool_call.function() = symbol_expr(*bool_method);
  bool_call.type() = to_code_type(bool_method->get_type()).return_type();
  bool_call.location() = location;
  bool_call.arguments().push_back(object_is_ptr ? cond : gen_address_of(cond));

  exprt result = store_call_result(bool_call, location, "cond_bool");
  result.location() = location;
  return result;
}

exprt python_converter::get_conditional_stm(const nlohmann::json &ast_node)
{
  // A walrus in a `while` test re-evaluates every iteration, but get_named_expr
  // emits the binding once into the enclosing block (it would go stale). Refuse
  // with a clean diagnostic rather than return an unsound verdict. A plain `if`
  // condition and a comprehension filter evaluate the walrus exactly once, so
  // they remain supported. (Ternary-branch and short-circuit-operand walrus are
  // refused at their own lowering sites: get_expr and
  // get_logical_operator_expr.)
  if (
    ast_node.value("_type", "") == "While" && ast_node.contains("test") &&
    contains_named_expr(ast_node["test"]))
    throw std::runtime_error(
      "Walrus operator ':=' in a while-loop condition is not supported");

  // Copy current type
  typet t = current_element_type;
  // Change to boolean before extracting condition
  current_element_type = bool_type();

  // Check if we need to materialize function calls in the condition
  // This handles cases like: if not math.isnan(x): or if isinstance(x, type):
  auto test_type = ast_node["test"]["_type"].get<std::string>();

  bool has_nested_call = false;
  nlohmann::json call_node;
  bool is_wrapped_in_unary = false;

  // Check for function call wrapped in UnaryOp (e.g., "not func()")
  if (test_type == "UnaryOp" && ast_node["test"].contains("operand"))
  {
    auto operand_type = ast_node["test"]["operand"]["_type"].get<std::string>();
    if (operand_type == "Call")
    {
      has_nested_call = true;
      is_wrapped_in_unary = true;
      call_node = ast_node["test"]["operand"];
    }
  }
  // Check for direct function call
  else if (test_type == "Call")
  {
    has_nested_call = true;
    call_node = ast_node["test"];
  }

  auto type = ast_node["_type"];
  if (type == "While" && has_nested_call)
  {
    locationt location = get_location_from_decl(ast_node);
    locationt call_location = get_location_from_decl(call_node);

    code_blockt transformed;

    // Reuse a single condition temporary to avoid redeclaring symbols
    // at each iteration of the lowered loop.
    symbolt cond_symbol =
      create_return_temp_variable(bool_type(), call_location, "while_cond");
    symbol_table_.add(cond_symbol);
    exprt cond_tmp = symbol_expr(cond_symbol);

    code_declt cond_decl(cond_tmp);
    cond_decl.location() = call_location;
    transformed.copy_to_operands(cond_decl);

    code_blockt loop_body;

    code_blockt *saved_block = current_block;
    current_block = &loop_body;
    exprt *saved_lhs = current_lhs;
    current_lhs = nullptr;
    exprt func_call = get_expr(call_node);
    current_lhs = saved_lhs;
    current_block = saved_block;

    if (func_call.is_function_call())
    {
      if (!func_call.type().is_empty())
        func_call.op0() = cond_tmp;
      loop_body.copy_to_operands(func_call);
    }
    else
    {
      code_assignt cond_assign(cond_tmp, func_call);
      cond_assign.location() = call_location;
      loop_body.copy_to_operands(cond_assign);
    }

    exprt overall_cond = cond_tmp;
    if (is_wrapped_in_unary)
    {
      // V.3: build the unary-not condition in IREP2.
      expr2tc ct2;
      migrate_expr(cond_tmp, ct2);
      overall_cond = migrate_expr_back(not2tc(ct2));
    }

    // V.3: build the break guard !cond in IREP2.
    expr2tc oc2;
    migrate_expr(overall_cond, oc2);
    exprt break_cond = migrate_expr_back(not2tc(oc2));

    code_breakt break_stmt;
    break_stmt.location() = location;
    code_ifthenelset break_if;
    break_if.cond() = break_cond;
    break_if.then_case() = break_stmt;
    break_if.location() = location;
    loop_body.copy_to_operands(break_if);

    exprt body_expr;
    if (ast_node["body"].is_array())
      body_expr = get_block(
        ast_node["body"],
        /*is_function_body=*/false,
        /*is_loop_body=*/true);
    else
      body_expr = get_expr(ast_node["body"]);
    body_expr.location() = location;
    loop_body.copy_to_operands(body_expr);

    codet while_code;
    while_code.set_statement("while");
    while_code.location() = location;
    // V.3: build the `while True` condition in IREP2.
    while_code.copy_to_operands(migrate_expr_back(gen_true_expr()), loop_body);

    transformed.copy_to_operands(while_code);
    current_element_type = t;
    return transformed;
  }

  // Extract condition from AST
  exprt cond;

  // Keep `and` and `or` in conditions short-circuited.
  const bool coverage_mode = is_coverage_mode();
  const bool pytest_generation_mode = is_pytest_generation_mode();
  const bool model_mode = is_model_file(ast_node["test"]);
  auto to_bool_condition =
    [&](const exprt &value_expr_in, const nlohmann::json &value_node) -> exprt {
    exprt value_expr = value_expr_in;

    // A non-folded call (e.g. to a multi-return function) arrives here as a
    // code_function_callt — a *statement*, not a value. Used directly as a
    // boolean-operator operand it is wrapped in an assignment / condition and
    // survives into the SSA as a code_function_call2t whose operands the SMT
    // encoder then dereferences, segfaulting on a null operand (GitHub #4998).
    // Normalise it to a value-producing side-effect call, which goto-conversion
    // correctly hoists into a function-call instruction.
    if (value_expr.is_function_call())
    {
      const code_function_callt &code =
        to_code_function_call(to_code(value_expr));
      typet return_type = code.type();
      // A void/None-returning call has an empty type; fall back to int so the
      // downstream bool typecast is well-defined (mirrors
      // get_logical_operator_expr).
      if (return_type.is_empty() || return_type.id() == typet::t_empty)
        return_type = type_handler_.get_typet("int", 0);
      side_effect_expr_function_callt side_effect;
      side_effect.function() = code.function();
      side_effect.arguments() = code.arguments();
      side_effect.type() = return_type;
      side_effect.location() = code.location();
      value_expr = side_effect;
    }

    if (value_expr.type().is_bool())
      return value_expr;

    typet list_type = type_handler_.get_list_type();
    if (
      value_expr.type() == list_type ||
      (value_expr.type().is_pointer() &&
       value_expr.type().subtype() == list_type))
    {
      const symbolt *size_func =
        symbol_table_.find_symbol("c:@F@__ESBMC_list_size");
      if (!size_func)
        throw std::runtime_error(
          "__ESBMC_list_size not found for list condition check");

      // V.3: build `__ESBMC_list_size(xs) != 0` in IREP2, back-migrating once
      // (mirrors the list-condition path at converter_stmt.cpp:3216). The call
      // is built directly in IREP2; it carries no IREP2 location, so the
      // location is re-attached to the back-migrated operand below.
      exprt size_arg = value_expr.type().is_pointer()
                         ? value_expr
                         : address_of_exprt(value_expr);
      expr2tc size_arg2;
      migrate_expr(size_arg, size_arg2);
      expr2tc size_call2 = side_effect_function_call2tc(
        migrate_type(size_type()), symbol_expr2tc(*size_func), {size_arg2});
      exprt cond = migrate_expr_back(
        notequal2tc(size_call2, gen_zero(migrate_type(size_type()))));
      const locationt cond_loc = get_location_from_decl(value_node);
      cond.location() = cond_loc;
      cond.op0().location() = cond_loc;
      return cond;
    }

    exprt bool_expr = typecast_exprt(value_expr, bool_type());
    bool_expr.location() = get_location_from_decl(value_node);
    return bool_expr;
  };

  if (
    test_type == "BoolOp" && current_block && type != "While" &&
    !coverage_mode && !pytest_generation_mode && !model_mode)
  {
    const auto &test_node = ast_node["test"];
    const auto &operands = test_node["values"];
    if (!operands.empty())
    {
      exprt *saved_lhs = current_lhs;
      current_lhs = nullptr;
      // Start from the leftmost operand and carry the running result forward.
      cond = to_bool_condition(get_expr(operands[0]), operands[0]);
      current_lhs = saved_lhs;

      symbolt result_symbol = create_return_temp_variable(
        bool_type(), get_location_from_decl(test_node), "boolop_cond");
      symbol_table_.add(result_symbol);
      exprt result_expr = symbol_expr(result_symbol);

      code_declt result_decl(result_expr);
      result_decl.location() = get_location_from_decl(test_node);
      current_block->copy_to_operands(result_decl);

      code_assignt result_init(result_expr, cond);
      result_init.location() = get_location_from_decl(test_node);
      current_block->copy_to_operands(result_init);

      const bool is_and = test_node["op"]["_type"] == "And";
      for (size_t i = 1; i < operands.size(); ++i)
      {
        code_blockt next_operand_block;
        code_blockt *saved_block = current_block;
        current_block = &next_operand_block;
        saved_lhs = current_lhs;
        current_lhs = nullptr;
        // Build the next operand only in the branch where it is still needed.
        exprt next_operand =
          to_bool_condition(get_expr(operands[i]), operands[i]);
        current_lhs = saved_lhs;
        current_block = saved_block;

        code_assignt result_update(result_expr, next_operand);
        result_update.location() = get_location_from_decl(operands[i]);
        next_operand_block.copy_to_operands(result_update);

        code_ifthenelset short_circuit_if;
        short_circuit_if.location() = get_location_from_decl(operands[i]);
        short_circuit_if.location().property("skipped");
        // `and` keeps going while the running result is true; `or` keeps
        // going while it is false.
        if (is_and)
          short_circuit_if.cond() = result_expr;
        else
        {
          // V.1k keystone: `not result` over the bool short-circuit accumulator
          // (a symbol) built in IREP2 (exact not2tc round-trip, matching the
          // not2tc uses above).
          expr2tc result2;
          migrate_expr(result_expr, result2);
          short_circuit_if.cond() = migrate_expr_back(not2tc(result2));
        }
        short_circuit_if.then_case() = next_operand_block;
        current_block->copy_to_operands(short_circuit_if);
      }

      cond = result_expr;
    }
  }
  else if (test_type == "BoolOp" && !model_mode)
  {
    exprt boolop_expr(
      python_frontend::map_operator(
        ast_node["test"]["op"]["_type"], bool_type()),
      bool_type());
    for (const auto &operand : ast_node["test"]["values"])
      boolop_expr.copy_to_operands(
        to_bool_condition(get_expr(operand), operand));
    cond = boolop_expr;
  }
  else if (has_nested_call)
  {
    locationt location = get_location_from_decl(call_node);

    auto apply_wrapped_unary = [&](const exprt &base_expr) -> exprt {
      if (!is_wrapped_in_unary)
        return base_expr;

      auto op = ast_node["test"]["op"]["_type"].get<std::string>();
      if (op == "Not")
      {
        exprt unary_expr("not", bool_type());
        unary_expr.copy_to_operands(base_expr);
        return unary_expr;
      }
      return base_expr;
    };

    // Get the function call expression with special handling
    // Temporarily disable the conditional processing to avoid recursion
    exprt *saved_lhs = current_lhs;
    current_lhs = nullptr;
    exprt func_call = get_expr(call_node);
    current_lhs = saved_lhs;

    if (func_call.is_function_call())
    {
      // Create temporary variable for function call result
      symbolt temp_symbol =
        create_return_temp_variable(func_call.type(), location, "cond");
      symbol_table_.add(temp_symbol);
      exprt temp_var_expr = symbol_expr(temp_symbol);

      // Create declaration for temporary
      code_declt temp_decl(temp_var_expr);
      temp_decl.location() = location;

      // Set the LHS of the function call
      if (!func_call.type().is_empty())
        func_call.op0() = temp_var_expr;

      // Add both declaration and function call to current_block
      if (current_block)
      {
        current_block->copy_to_operands(temp_decl);
        current_block->copy_to_operands(func_call);
      }

      cond = apply_wrapped_unary(temp_var_expr);
    }
    else
    {
      cond = apply_wrapped_unary(func_call);
    }
  }
  else
  {
    // Normal path: no function call to materialize
    cond = get_expr(ast_node["test"]);
  }

  if (!(test_type == "BoolOp" && current_block && type != "While" &&
        !coverage_mode && !pytest_generation_mode && !model_mode))
  {
    cond.location() = get_location_from_decl(ast_node["test"]);

    if (!cond.type().is_bool())
    {
      const locationt location = get_location_from_decl(ast_node["test"]);
      typet value_type = ns.follow(cond.type());
      if (value_type.is_pointer())
        value_type = ns.follow(value_type.subtype());

      // Objects in conditions are converted with __bool__() when available.
      cond = apply_bool_dunder(cond, location);

      typet list_type = type_handler_.get_list_type();
      // Python treats lists in conditions by their size, for example:
      // `1 if xs else 0`.
      if (
        current_block &&
        (cond.type() == list_type ||
         (cond.type().is_pointer() && cond.type().subtype() == list_type)))
      {
        const symbolt *size_func =
          symbol_table_.find_symbol("c:@F@__ESBMC_list_size");
        if (!size_func)
          throw std::runtime_error(
            "__ESBMC_list_size not found for list condition check");

        // Keep the size query inside the condition expression so constructs
        // like `while heap:` re-evaluate the current list size every iteration.
        // V.3: build `__ESBMC_list_size(xs) != 0` in IREP2, back-migrating
        // once (mirrors the list-condition path at converter_binop.cpp:208).
        // The call is built directly in IREP2; it carries no IREP2 location,
        // so the location is re-attached to the back-migrated operand below.
        exprt size_arg =
          cond.type().is_pointer() ? cond : address_of_exprt(cond);
        expr2tc size_arg2;
        migrate_expr(size_arg, size_arg2);
        expr2tc size_call2 = side_effect_function_call2tc(
          migrate_type(size_type()), symbol_expr2tc(*size_func), {size_arg2});
        cond = migrate_expr_back(
          notequal2tc(size_call2, gen_zero(migrate_type(size_type()))));
        cond.location() = location;
        cond.op0().location() = location;
      }

      // Python treats strings in conditions by their length: "" is falsy.
      if (type_utils::is_string_type(cond.type()))
      {
        const symbolt *strlen_sym = symbol_table_.find_symbol("c:@F@strlen");
        if (!strlen_sym)
          throw std::runtime_error(
            "strlen not found for string truthiness check");

        // V.3: build `strlen(s) != 0` in IREP2, back-migrating once (mirrors
        // the string-truthiness path at converter_unop.cpp). The call is built
        // directly in IREP2; it carries no IREP2 location, so the location is
        // re-attached to the back-migrated operand below.
        expr2tc strlen_arg2;
        migrate_expr(string_handler_.get_array_base_address(cond), strlen_arg2);
        expr2tc strlen_call2 = side_effect_function_call2tc(
          migrate_type(size_type()),
          symbol_expr2tc(*strlen_sym),
          {strlen_arg2});
        cond = migrate_expr_back(
          notequal2tc(strlen_call2, gen_zero(migrate_type(size_type()))));
        cond.location() = location;
        cond.op0().location() = location;
      }
    }
  }

  // P12: Python truthiness for complex in conditional contexts:
  // bool(z) == (z.real != 0.0 or z.imag != 0.0).
  // Delegates to the single canonical implementation in type_handler.h.
  if (is_complex_type(cond.type()))
  {
    locationt loc = get_location_from_decl(ast_node["test"]);
    cond = complex_to_bool_expr(cond);
    cond.location() = loc;
  }

  // Recover type
  current_element_type = t;

  // Declares the flagged variable's tagged-object symbol before either
  // branch converts, so goto-symex's struct-merge resolves the join for
  // free; get_var_assign fills in the fields per branch.
  std::unordered_set<std::string> dynamic_type_names;
  if (type == "If")
    dynamic_type_names =
      dynamic_type_handler_.detect_dynamic_type_names(ast_node);

  dynamic_type_handler_.declare_dynamic_type_names(
    dynamic_type_names, ast_node);
  dynamic_type_handler::scope_guard tag_scope_guard(
    dynamic_type_handler_, dynamic_type_names);

  // Extract 'then' block from AST
  exprt then;

  // A Python conditional expression `body if test else orelse` short-circuits:
  // only the selected branch is evaluated. When a ternary branch emits
  // side-effecting instructions (notably a subscript's IndexError raise), they
  // must run only when that branch is taken. Capture each branch's side effects
  // into its own block so they can be guarded by the condition below, instead
  // of leaking unconditionally into the enclosing block.
  code_blockt then_side_effects, else_side_effects;

  // Skip the 'then' block when the condition evaluates to false.
  if (cond.is_constant() && cond.value() == "false" && type != "IfExp")
  {
    then = code_blockt();
  }
  else
  {
    if (ast_node["body"].is_array())
      then = get_block(
        ast_node["body"],
        /*is_function_body=*/false,
        /*is_loop_body=*/type == "While");
    else if (type == "IfExp")
    {
      code_blockt *saved_block = current_block;
      current_block = &then_side_effects;
      then = get_expr(ast_node["body"]);
      current_block = saved_block;
    }
    else
      then = get_expr(ast_node["body"]);
  }

  locationt location = get_location_from_decl(ast_node);
  then.location() = location;

  // Extract 'else' block from AST
  exprt else_expr;
  if (ast_node.contains("orelse") && !ast_node["orelse"].empty())
  {
    // Append 'else' block to the statement
    if (ast_node["orelse"].is_array())
      else_expr = get_block(ast_node["orelse"]);
    else if (type == "IfExp")
    {
      code_blockt *saved_block = current_block;
      current_block = &else_side_effects;
      else_expr = get_expr(ast_node["orelse"]);
      current_block = saved_block;
    }
    else
      else_expr = get_expr(ast_node["orelse"]);
  }

  // ternary operator
  if (type == "IfExp")
  {
    // A condition that raised while being evaluated (e.g. ord() of a bad
    // argument) arrives as a cpp-throw side effect, not a boolean value.
    // Propagate the exception instead of building an if2t with a non-boolean
    // condition, which migrates to a null cond and crashes goto_check. Mirrors
    // the cpp-throw guards in get_binary_operator_expr.
    if (cond.statement() == "cpp-throw")
      return cond;

    // Normalize branches: code_function_callt must become side_effect_expr so
    // that migration to irep2 preserves the correct return type in if2t.
    then = to_value_expr(then, ns);
    else_expr = to_value_expr(else_expr, ns);

    bool then_is_none = (then.type() == none_type());
    bool else_is_none = (else_expr.type() == none_type());

    typet result_type;
    if (then_is_none != else_is_none)
    {
      // One branch is None, the other is T → Optional[T] models Python's T |
      // None
      typet concrete_type = then_is_none ? else_expr.type() : then.type();
      result_type = type_handler_.build_optional_type(concrete_type);
      then = wrap_in_optional(then, result_type);
      else_expr = wrap_in_optional(else_expr, result_type);
    }
    else
    {
      // Resolve result type based on branch types
      result_type = resolve_ternary_type(
        then.type(), else_expr.type(), current_element_type);

      // Handle array-to-pointer conversion for ternary expressions
      // When assigning to a pointer (e.g., str field), convert array branches
      // to pointers
      if (
        then.type().is_array() && else_expr.type().is_array() && current_lhs &&
        current_lhs->type().is_pointer())
      {
        then = string_handler_.get_array_base_address(then);
        else_expr = string_handler_.get_array_base_address(else_expr);
        result_type = then.type(); // Use pointer type as result
      }
    }

    // When a branch emits side-effecting instructions (notably a subscript's
    // IndexError raise, or a materialised nested access whose value expression
    // references temps the branch declared), lower the ternary as a
    // short-circuiting if/else into a result temp: each branch runs its own
    // side effects and assigns its value to the temp, so ONLY the selected
    // branch is evaluated — matching Python's short-circuit semantics. Emitting
    // the value assignment inside the guarded branch keeps the branch's temps
    // in scope (a flat value-select would reference them from outside their
    // guard). Keep the pure value-select `if_expr` when neither branch has side
    // effects (the common case), avoiding churn. A pure-expression context has
    // no current_block to emit into, and there its branches carry no side
    // effects.
    if (
      current_block && (!then_side_effects.operands().empty() ||
                        !else_side_effects.operands().empty()))
    {
      symbolt result_symbol =
        create_return_temp_variable(result_type, location, "ternary_result");
      symbol_table_.add(result_symbol);
      exprt result_expr = symbol_expr(result_symbol);

      code_declt result_decl(result_expr);
      result_decl.location() = location;
      current_block->copy_to_operands(result_decl);

      auto coerce = [&](const exprt &branch) -> exprt {
        if (branch.type() == result_type)
          return branch;
        return typecast_exprt(branch, result_type);
      };

      code_assignt then_assign(result_expr, coerce(then));
      then_assign.location() = location;
      then_side_effects.copy_to_operands(then_assign);

      code_assignt else_assign(result_expr, coerce(else_expr));
      else_assign.location() = location;
      else_side_effects.copy_to_operands(else_assign);

      code_ifthenelset guard;
      guard.location() = location;
      guard.cond() = cond;
      guard.then_case() = then_side_effects;
      guard.else_case() = else_side_effects;
      current_block->copy_to_operands(guard);

      return result_expr;
    }

    // Create fully symbolic if expression (pure branches, no side effects)
    exprt if_expr("if", result_type);
    if_expr.copy_to_operands(cond, then, else_expr);
    return if_expr;
  }

  // Create if or while code
  codet code;
  if (type == "If")
    code.set_statement("ifthenelse");
  else if (type == "While")
    code.set_statement("while");

  // Set location for the conditional statement
  code.location() = get_location_from_decl(ast_node);

  // Append "then" block
  code.copy_to_operands(cond, then);
  if (!else_expr.id_string().empty())
    code.copy_to_operands(else_expr);

  return code;
}
exprt python_converter::box_value_on_heap(
  const exprt &value,
  const locationt &location,
  codet &target_block)
{
  return box_value_on_heap(
    value, location, target_block, current_func_return_type_);
}

exprt python_converter::box_value_on_heap(
  const exprt &value,
  const locationt &location,
  codet &target_block,
  const typet &ptr_type)
{
  const symbolt *new_obj_sym =
    symbol_table_.find_symbol("c:@F@__ESBMC_new_object");
  assert(new_obj_sym && "__ESBMC_new_object model required");

  symbolt heap_symbol =
    create_return_temp_variable(ptr_type, location, "ctor_box");
  symbol_table_.add(heap_symbol);
  exprt heap_ptr = symbol_expr(heap_symbol);

  code_declt heap_decl(heap_ptr);
  heap_decl.location() = location;
  target_block.copy_to_operands(heap_decl);

  code_function_callt alloc_call;
  alloc_call.lhs() = heap_ptr;
  alloc_call.function() = symbol_expr(*new_obj_sym);
  alloc_call.location() = location;
  target_block.copy_to_operands(alloc_call);

  exprt deref("dereference", ptr_type.subtype());
  deref.copy_to_operands(heap_ptr);

  // Whole-array assignment through a dereference is rejected by the
  // dereference layer ("Can't construct rvalue reference to array type"),
  // so a boxed array of statically-known size is stored element-wise.
  if (ptr_type.subtype().is_array())
  {
    const array_typet &arr_t = to_array_type(ptr_type.subtype());
    assert(arr_t.size().is_constant());
    const size_t n =
      binary2integer(to_constant_expr(arr_t.size()).value().c_str(), false)
        .to_uint64();
    for (size_t i = 0; i < n; i++)
    {
      exprt idx = from_integer(i, index_type());
      code_assignt store(
        python_expr::build_index(deref, idx, arr_t.subtype()),
        python_expr::build_index(value, idx, arr_t.subtype()));
      store.location() = location;
      target_block.copy_to_operands(store);
    }
    return heap_ptr;
  }

  code_assignt store(deref, value);
  store.location() = location;
  target_block.copy_to_operands(store);

  return heap_ptr;
}

void python_converter::get_return_statements(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  if (ast_node["value"].is_null())
  {
    // Handle bare return statement (return with no value)
    locationt location = get_location_from_decl(ast_node);
    code_returnt return_code;
    return_code.location() = location;

    if (type_handler_.is_tagged_scalar_type(current_func_return_type_))
      throw std::runtime_error(
        "returning a value of this type from a dynamically-typed function "
        "is not yet supported");

    // If the function returns Optional, wrap None in Optional struct
    if (current_func_return_type_.is_struct())
    {
      const struct_typet &st = to_struct_type(current_func_return_type_);
      if (st.tag().as_string().starts_with("tag-Optional_"))
      {
        constant_exprt none_expr(none_type());
        return_code.return_value() =
          wrap_in_optional(none_expr, current_func_return_type_);
      }
    }
    // A bare `return` yields None. When the function returns None — either
    // already typed none_type(), or still an empty placeholder that the funcdef
    // will promote to none_type() (issue #5914) — give the RETURN an explicit
    // None value so it matches the declared none_type() slot at the call site.
    else if (
      current_func_return_type_ == none_type() ||
      current_func_return_type_.is_empty())
      return_code.return_value() = gen_zero(none_type());

    target_block.copy_to_operands(return_code);
    return;
  }

  // Same check Assign already applies to its RHS: a numpy view (copied,
  // transpose, or reshape) stashed inside a list/tuple/dict escapes into a
  // container get_expr cannot build a valid GOTO reference for, crashing
  // deep in irep migration instead of raising a clean diagnostic. `return
  // [a[0]]` is exactly Assign's own `y = [a[0]]` case, just via a Return
  // instead of an Assign target.
  reject_copied_numpy_view_in_container(ast_node, {"List", "Tuple", "Dict"});

  bool is_user_defined_function = false;
  if (
    !current_func_name_.empty() && current_func_name_ != "python_user_main" &&
    ast_json && ast_json->contains("filename") &&
    is_program_file((*ast_json)["filename"].get<std::string>()))
  {
    const std::vector<std::string> function_path =
      json_utils::split_function_path(current_func_name_);
    const nlohmann::json func_node =
      json_utils::find_function_by_path(*ast_json, function_path);
    is_user_defined_function = !func_node.empty() && !is_model_file(func_node);
  }
  const bool returns_name = ast_node["value"].value("_type", "") == "Name" &&
                            ast_node["value"].contains("id");
  if (
    is_user_defined_function && returns_name &&
    contains_tracked_numpy_view_name(ast_node["value"]))
    throw std::runtime_error(
      "TypeError: returning a copied numpy view is not supported");
  const locationt return_location = get_location_from_decl(ast_node);
  const std::string return_file = return_location.get_file().as_string();
  if (
    returns_name && ast_json && is_user_defined_function &&
    is_program_file(return_file))
  {
    const std::string name = ast_node["value"]["id"].get<std::string>();
    const nlohmann::json decl =
      json_utils::find_var_decl(name, current_func_name_, *ast_json);
    if (
      decl.is_object() && decl.value("_type", "") != "arg" &&
      decl.contains("value") && is_numpy_view_copy_expr(decl["value"]))
    {
      const std::string root_name =
        root_name_from_numpy_view_copy_expr(decl["value"]);
      const std::string root_id =
        root_name.empty() ? std::string() : resolve_name_symbol_id(root_name);
      const bool root_is_tracked_numpy =
        !root_id.empty() && (numpy_array_symbols_.count(root_id) != 0 ||
                             numpy_view_copy_sources_.count(root_id) != 0);
      bool root_is_numpy_param = false;
      if (!root_name.empty() && ast_json && ast_imports_numpy_module(*ast_json))
      {
        const nlohmann::json root_decl =
          json_utils::find_var_decl(root_name, current_func_name_, *ast_json);
        root_is_numpy_param =
          root_decl.is_object() && root_decl.value("_type", "") == "arg";
      }
      if (root_is_tracked_numpy || root_is_numpy_param)
      {
        throw std::runtime_error(
          "TypeError: returning a copied numpy view is not supported");
      }
    }
  }

  exprt return_value = get_expr(ast_node["value"]);
  locationt location = get_location_from_decl(ast_node);

  // Coerces `val` to a tagged-object value when the function's return type
  // is tagged. No-op when `val` is already tagged.
  auto coerce_to_tagged_return = [&](exprt &val) {
    if (
      !type_handler_.is_tagged_scalar_type(current_func_return_type_) ||
      type_handler_.is_tagged_scalar_type(val.type()))
      return;
    if (
      type_handler_.is_numeric_scalar_type(val.type()) ||
      type_handler_.is_string_type(val.type()))
      val = dynamic_type_handler_.build_tagged_return_value(
        val, location, target_block);
    else
      throw std::runtime_error(
        "returning a value of this type from a dynamically-typed function "
        "is not yet supported");
  };

  // Check if return value is a function call
  // get_function_call() returns code_function_callt (code statement), not
  // side_effect_expr_function_callt
  bool is_func_call =
    return_value.is_code() && return_value.get("statement") == "function_call";

  if (is_func_call)
  {
    // Extract function name for temporary variable naming.
    // get_expr() also returns a function-call expression when a Subscript
    // dispatches to a user-defined __getitem__ (see GitHub #4541); in that
    // case the AST node type is "Subscript" rather than "Call", but the
    // returned code is still a function_call that needs the same temp-LHS
    // materialisation, otherwise the call expression ends up embedded
    // directly in the GOTO RETURN and trips value-set's make_member
    // assertion at value_set.cpp:1543.
    const std::string ast_type = ast_node["value"]["_type"].get<std::string>();
    std::string func_name = "func";
    if (ast_type == "Call")
    {
      if (ast_node["value"]["func"]["_type"] == "Name")
        func_name = ast_node["value"]["func"]["id"].get<std::string>();
      else if (ast_node["value"]["func"]["_type"] == "Attribute")
        func_name = ast_node["value"]["func"]["attr"].get<std::string>();
    }
    else if (ast_type == "Subscript")
      func_name = "__getitem__";

    // Determine return type: check if it's empty (forward reference)
    typet return_type = return_value.type();

    if (return_type.is_empty() || return_type.id() == typet::t_empty)
    {
      // Forward reference: function not yet processed
      // Look up return type from AST
      const auto &func_node =
        json_utils::try_find_function((*ast_json)["body"], func_name);

      if (
        !func_node.empty() && func_node.contains("returns") &&
        !func_node["returns"].is_null())
        return_type = get_type_from_annotation(func_node["returns"], func_node);
      else
      {
        // Default to void* if we can't determine the type
        return_type = any_type();
      }
    }

    // Create temporary variable to store function call result
    symbolt temp_symbol =
      create_return_temp_variable(return_type, location, func_name);
    symbol_table_.add(temp_symbol);
    exprt temp_var_expr = symbol_expr(temp_symbol);

    // Create declaration for temporary variable
    code_declt temp_decl(temp_var_expr);
    temp_decl.location() = location;
    target_block.copy_to_operands(temp_decl);

    // If a constructor is being invoked, the temporary variable is passed as
    // 'self'. For constructors, we don't set LHS because they modify the object
    // through the first parameter (self), not through LHS.
    bool is_constructor = type_handler_.is_constructor_call(ast_node["value"]);

    // Set the LHS of the function call to our temporary variable (only for
    // non-constructors)
    if (!return_type.is_empty() && !is_constructor)
      return_value.op0() = temp_var_expr;

    if (is_constructor)
    {
      code_function_callt &call =
        static_cast<code_function_callt &>(return_value);

      // Strip any temporary $ctor_self$ parameters and add correct self
      exprt::operandst filtered_args =
        function_call_expr::strip_ctor_self_parameters(call.arguments());
      exprt::operandst new_args;
      new_args.push_back(gen_address_of(temp_var_expr));
      for (const auto &arg : filtered_args)
        new_args.push_back(arg);
      call.arguments() = new_args;
      update_instance_from_self(
        func_name, func_name, temp_var_expr.identifier().as_string());
    }

    // Add the function call statement to the block
    target_block.copy_to_operands(return_value);

    exprt ret_expr = temp_var_expr;
    // `return ClassName(...)` constructs into the stack-local value temp above;
    // when the function returns a migrated class reference (Cls*, #3067), box
    // that value onto a non-expiring heap object and return the pointer so the
    // instance survives the frame (returning &temp would dangle). Mirrors the
    // member/parameter return path's boxing in the else branch below.
    if (
      is_constructor && is_user_class_pointer(current_func_return_type_) &&
      is_user_class_struct_type(temp_var_expr.type()))
      ret_expr = box_value_on_heap(temp_var_expr, location, target_block);
    // Wrap in Optional if the function returns Optional
    else if (current_func_return_type_.is_struct())
    {
      const struct_typet &st = to_struct_type(current_func_return_type_);
      if (st.tag().as_string().starts_with("tag-Optional_"))
        ret_expr = wrap_in_optional(ret_expr, current_func_return_type_);
    }

    coerce_to_tagged_return(ret_expr);

    // Return the temporary variable
    code_returnt return_code;
    return_code.return_value() = ret_expr;
    return_code.location() = location;
    target_block.copy_to_operands(return_code);
  }
  else
  {
    // If we're returning an array but the function expects a pointer,
    // convert the array to a pointer (for string literals).
    const typet &expected_return_type = current_func_return_type_;

    if (expected_return_type.is_pointer() && return_value.type().is_array())
    {
      // For constant array literals (string literals), convert to
      // string_constantt
      if (return_value.is_constant())
      {
        // Extract the string content from the constant array
        std::string str_content;
        for (const auto &operand : return_value.operands())
        {
          if (operand.is_constant())
          {
            BigInt char_val = binary2integer(
              operand.value().as_string(), operand.type().is_signedbv());
            if (char_val == 0)
              break; // Stop at null terminator
            str_content += static_cast<char>(char_val.to_int64());
          }
        }

        // Create a string_constantt with proper type
        typet string_type = return_value.type();
        return_value = string_constantt(
          str_content, string_type, string_constantt::k_default);

        // Get its address (converts array to pointer)
        // V.3: build the address-of in IREP2 (operand is a string constant).
        return_value = python_expr::build_address_of(return_value);
      }
      else if (to_array_type(return_value.type()).size().is_constant())
      {
        // For non-constant arrays (variables), convert to pointer. The
        // array is function-local (e.g. a string copied out of a tuple,
        // #5571), so its storage expires with this frame: box the bytes
        // onto a fresh non-expiring heap object first — the same model as
        // returning a constructed class value (#3067) — and hand back a
        // pointer into that.
        exprt boxed = box_value_on_heap(
          return_value,
          location,
          target_block,
          gen_pointer_type(return_value.type()));
        exprt deref("dereference", return_value.type());
        deref.copy_to_operands(boxed);
        return_value = string_handler_.get_array_base_address(deref);
      }
      else
      {
        // Symbolic-size array: element-wise boxing needs a static count, so
        // keep the plain decay (the pre-#5571 behaviour for this rare case).
        return_value = string_handler_.get_array_base_address(return_value);
      }
    }

    // `return ClassName(...)` lowers to a stack-local `$ctor_self$` *value*
    // struct (function_call_expr's no-LHS constructor path), but the function
    // now returns a class *reference* (Cls*). Box the constructed value onto a
    // fresh non-expiring heap object and return the pointer, so the result
    // survives the callee frame with reference identity (#3067) — the same
    // model as `o = ClassName(...)`. Returning `&$ctor_self$` would instead
    // hand back a dangling stack address.
    if (
      is_user_class_pointer(current_func_return_type_) &&
      is_user_class_struct_type(return_value.type()))
      return_value = box_value_on_heap(return_value, location, target_block);

    // Wrap return value in Optional if the function returns Optional
    if (current_func_return_type_.is_struct())
    {
      const struct_typet &st = to_struct_type(current_func_return_type_);
      if (st.tag().as_string().starts_with("tag-Optional_"))
        return_value =
          wrap_in_optional(return_value, current_func_return_type_);
    }

    coerce_to_tagged_return(return_value);

    code_returnt return_code;
    return_code.return_value() = return_value;
    return_code.location() = location;
    target_block.copy_to_operands(return_code);
  }
}

exprt python_converter::get_block(
  const nlohmann::json &ast_block,
  bool is_function_body,
  bool is_loop_body)
{
  // Track block nesting (and the function-body / loop-body depths) so dynamic
  // retyping (#4770/#4774) fires on the unconditional spine (the module body
  // plus enclosing function bodies, block_nesting_ == function_body_depth_ + 1)
  // and inside if/else/try bodies, but not inside a while/for body, where a
  // retyped loop variable must keep leaking past the join (see get_var_assign).
  block_nesting_guard nesting_guard(
    block_nesting_,
    is_function_body ? &function_body_depth_ : nullptr,
    is_loop_body ? &loop_body_depth_ : nullptr);

  // A conditional body is any block off the unconditional spine (the module
  // body plus enclosing function bodies, block_nesting_ == function_body_depth_
  // + 1). Inside an if/else/try body, dynamic retyping is applied locally for
  // in-body reads but reverted on exit so it does not leak across the join. A
  // loop body retypes nothing (refused in get_var_assign via loop_body_depth_),
  // so the snapshot/restore is a no-op there and merely preserves any alias
  // established before the loop.
  const bool is_conditional_body = (block_nesting_ != function_body_depth_ + 1);
  retype_alias_scope_guard retype_guard(retype_aliases_, is_conditional_body);

  // Entering any nested/conditional body (function, if/while/for, try/except):
  // straight-line flow-sensitive class tracking is no longer valid here, so
  // drop the map rather than risk adopting a class across a control-flow join.
  if (block_nesting_ >= 2)
    flow_class_map_.clear();

  code_blockt block, *old_block = current_block;
  current_block = &block;

  // Iterate over block statements
  for (auto &element : ast_block)
  {
    StatementType type = python_frontend::get_statement_type(element);

    switch (type)
    {
    case StatementType::VARIABLE_ASSIGN:
    {
      // Add an assignment to the block
      get_var_assign(element, block);
      break;
    }
    case StatementType::IF_STATEMENT:
    case StatementType::WHILE_STATEMENT:
    {
      exprt cond = get_conditional_stm(element);
      block.copy_to_operands(cond);
      break;
    }
    case StatementType::FOR_STATEMENT:
    {
      // For loops are transformed to while loops by the preprocessor
      // This case should not be reached in normal operation
      throw std::runtime_error(
        "For loops should be preprocessed before reaching converter");
    }
    case StatementType::COMPOUND_ASSIGN:
    {
      get_compound_assign(element, block);
      break;
    }
    case StatementType::FUNC_DEFINITION:
    {
      // A nested def is converted in the middle of its enclosing function, so
      // save and restore rather than clear: the inner body's own `global`
      // declarations must not outlive it, and the enclosing scope's must
      // survive it. Clearing dropped the enclosing `global x`, after which
      // every later `x = ...` in the outer body bound a fresh local and the
      // module global kept its initial value (#6669). At module scope the
      // saved state is empty, so this matches the previous behaviour.
      std::vector<std::string> saved_globals = global_declarations;
      std::vector<std::string> saved_loads = local_loads;
      get_function_definition(element);
      global_declarations = std::move(saved_globals);
      local_loads = std::move(saved_loads);

      // Bind the closure's capture cells where the `def` executes (#6256).
      exprt::operandst &bindings = pending_captures_.operands();
      block.operands().insert(
        block.operands().end(), bindings.begin(), bindings.end());
      bindings.clear();
      break;
    }
    case StatementType::RETURN:
    {
      get_return_statements(element, block);
      break;
    }
    case StatementType::ASSERT:
    {
      // Fold whole-assertion tests that provably evaluate to True at
      // conversion time (e.g. `f(GLOBAL) == [literal]` for a pure f). This
      // bypasses operational-model loops (strlen/str-slice) whose unwinding
      // would otherwise scale with the data size. Gated on the test containing
      // a function call so plain symbolic asserts stay on the solver path;
      // only a constant True short-circuits — False/unknown fall through so
      // the solver still detects genuine violations.
      if (
        !is_assert_fold_disabled() && element.contains("test") &&
        ast_contains_call(element["test"]))
      {
        python_consteval evaluator(*ast_json);
        auto folded = evaluator.try_eval_global_expr(element["test"]);
        if (folded && folded->kind == PyConstValue::BOOL && folded->bool_val)
        {
          code_assertt proven;
          proven.assertion() = gen_boolean(true);
          proven.location() = get_location_from_decl(element);
          proven.location().comment("assertion proven by constant evaluation");
          block.move_to_operands(proven);
          break;
        }
      }

      current_element_type = bool_type();
      exprt test = get_expr(element["test"]);
      // An object asserted directly is truth-tested like any condition, so a
      // class with __bool__ decides the answer. Without this `assert obj` cast
      // the object to bool and passed whatever the dunder said.
      test = apply_bool_dunder(test, get_location_from_decl(element));
      if (test.statement() == "cpp-throw")
      {
        test.location() = get_location_from_decl(element);
        codet code_expr("expression");
        code_expr.operands().push_back(test);
        block.move_to_operands(code_expr);
        break;
      }

      // Convert dictionary to boolean (truthiness check)
      if (dict_handler_->is_dict_type(test.type()))
      {
        locationt location = get_location_from_decl(element);
        typet list_type = type_handler_.get_list_type();

        // Get dict.keys member. V.3: IREP2 member access (exact round-trip of
        // member_exprt); `test` is dict-typed (is_dict_type ⇒ struct), so the
        // member2t source precondition holds.
        expr2tc dict2;
        migrate_expr(test, dict2);
        exprt keys_member =
          migrate_expr_back(member2tc(migrate_type(list_type), dict2, "keys"));

        // Find __ESBMC_list_size function
        const symbolt *size_func =
          symbol_table_.find_symbol("c:@F@__ESBMC_list_size");
        if (!size_func)
          throw std::runtime_error(
            "__ESBMC_list_size not found for dict truthiness check");

        // Create temporary variable to store the size result
        symbolt &size_result = create_tmp_symbol(
          element, "$dict_size$", size_type(), gen_zero(size_type()));
        code_declt size_decl(symbol_expr(size_result));
        size_decl.location() = location;
        block.copy_to_operands(size_decl);

        // Call __ESBMC_list_size(dict.keys)
        code_function_callt size_call;
        size_call.function() = symbol_expr(*size_func);
        size_call.lhs() = symbol_expr(size_result);
        size_call.arguments().push_back(keys_member);
        size_call.type() = size_type();
        size_call.location() = location;
        block.copy_to_operands(size_call);

        // Replace test with: size != 0 (non-empty dict is truthy)
        // V.3: build `$dict_size$ != 0` in IREP2, back-migrating once
        // (mirrors the list/string truthiness paths above).
        expr2tc size2;
        migrate_expr(symbol_expr(size_result), size2);
        exprt is_not_empty = migrate_expr_back(
          notequal2tc(size2, gen_zero(migrate_type(size_type()))));
        is_not_empty.location() = location;
        test = is_not_empty;
      }

      // Attach assertion message if present
      auto attach_assert_message = [&element](code_assertt &assert_code) {
        if (element.contains("msg") && !element["msg"].is_null())
        {
          std::string msg;
          if (
            element["msg"]["_type"] == "Constant" &&
            element["msg"]["value"].is_string())
          {
            msg = element["msg"]["value"].get<std::string>();
          }
          else if (element["msg"]["_type"] == "JoinedStr")
          {
            // For f-strings, this is just a placeholder
            // TODO: Full f-string evaluation would require more complex
            // handling
            msg = "<formatted string message>";
          }

          if (!msg.empty())
            assert_code.location().comment(msg);
        }
      };

      // Handle list assertions
      if (
        test.type() == type_handler_.get_list_type() ||
        (test.type().is_pointer() &&
         test.type().subtype() == type_handler_.get_list_type()))
      {
        exception_handler_->handle_list_assertion(
          element, test, block, attach_assert_message);
        break;
      }

      // Check for function call assertions
      const exprt *func_call_expr = nullptr;
      bool is_negated = false;

      // Case 1: Direct function call - assert func()
      if (test.id() == "code" && test.get("statement") == "function_call")
      {
        func_call_expr = &test;
        is_negated = false;
      }
      // Case 2: Negated function call - assert not func()
      else if (
        test.id() == "not" && test.operands().size() == 1 &&
        test.operands()[0].id() == "code" &&
        test.operands()[0].get("statement") == "function_call")
      {
        func_call_expr = &test.operands()[0];
        is_negated = true;
      }

      if (func_call_expr != nullptr)
      {
        exception_handler_->handle_function_call_assertion(
          element, *func_call_expr, is_negated, block, attach_assert_message);
      }
      else
      {
        // Direct assertion
        if (!test.type().is_bool())
          test.make_typecast(current_element_type);

        code_assertt assert_code;
        assert_code.assertion() = test;
        assert_code.location() = get_location_from_decl(element);
        attach_assert_message(assert_code);
        block.move_to_operands(assert_code);
      }
      break;
    }
    case StatementType::EXPR:
    {
      // Skip yield expressions: the preprocessor inlines them into assignments.
      // Reject yield from: the preprocessor does not expand it, so reaching
      // here means the generator was not fully lowered and verification would
      // silently produce wrong results.
      if (element.contains("value") && element["value"].contains("_type"))
      {
        const auto &inner_type = element["value"]["_type"];
        if (inner_type == "Yield")
          break;
        if (inner_type == "YieldFrom")
          throw std::runtime_error(
            "'yield from' is not supported in ESBMC's Python frontend");
      }

      // Function calls are handled here
      reject_numpy_view_identity_query(element["value"]);
      reject_numpy_view_mutating_method_call(element["value"]);
      reject_unknown_numpy_view_call(element["value"]);

      exprt empty;
      exprt expr = get_expr(element["value"]);
      if (expr != empty)
      {
        codet code_stmt = convert_expression_to_code(expr);
        // Every sibling statement handler stamps this; EXPR did not, so a bare
        // expression statement -- most commonly a docstring, which lowers to a
        // decayed string literal -- reached goto-convert unlocated. The native
        // body dispatcher declines an unlocated expression statement, and that
        // was ~87 % of the Python corpus's genuine declines
        // (docs/roadmap/frontends-to-irep2.md §13).
        //
        // Fill in only when the statement has no usable location of its own.
        // Assigning unconditionally clobbers one that is already set, and a
        // locationt carries more than a position: __ESBMC_assert's message
        // rides in its comment field, so overwriting it turns a modelled
        // rejection ("Counter.most_common is not modelled") into a bare
        // "assertion 0".
        const locationt &here = code_stmt.location();
        if (here.is_nil() || here.get_file().empty())
          code_stmt.location() = get_location_from_decl(element);
        block.move_to_operands(code_stmt);
      }

      break;
    }
    case StatementType::CLASS_DEFINITION:
    {
      get_class_definition(element, block);
      break;
    }
    case StatementType::BREAK:
    {
      code_breakt break_expr;
      block.move_to_operands(break_expr);
      break;
    }
    case StatementType::CONTINUE:
    {
      code_continuet continue_expr;
      block.move_to_operands(continue_expr);
      break;
    }
    case StatementType::GLOBAL:
    {
      symbol_id sid = create_symbol_id();
      for (const auto &item : element["names"])
      {
        sid.set_object(item);
        global_declarations.push_back(sid.global_to_string());
      }
      break;
    }
    case StatementType::TRY:
    {
      exception_handler_->get_try_statement(element, block);
      break;
    }
    case StatementType::EXCEPTHANDLER:
    {
      exception_handler_->get_except_handler_statement(element, block);
      break;
    }
    case StatementType::RAISE:
    {
      exception_handler_->get_raise_statement(element, block);
      break;
    }
    case StatementType::DELETE_STATEMENT:
    {
      get_delete_statement(element, block);
      break;
    }
    /* "https://docs.python.org/3/tutorial/controlflow.html:
     * "The pass statement does nothing. It can be used when a statement
     *  is required syntactically but the program requires no action." */
    case StatementType::PASS:
    // Imports are handled by parser.py so we can just ignore here.
    case StatementType::IMPORT:
      // PASS and IMPORT need no action here; break to avoid the default throw.
      break;
    case StatementType::UNKNOWN:
    default:
      throw std::runtime_error(
        element["_type"].get<std::string>() + " statements are not supported");
    }
  }

  current_block = old_block;

  return block;
}

exprt python_converter::get_static_array(
  const nlohmann::json &arr,
  const typet &shape)
{
  exprt zero = gen_zero(size_type());
  exprt list = gen_zero(shape);

  unsigned int i = 0;
  for (auto &e : arr["elts"])
  {
    exprt element_expr = get_expr(e);
    list.operands().at(i++) = element_expr;
  }

  symbolt &cl = create_tmp_symbol(arr, "$compound-literal$", shape, list);

  exprt expr = symbol_expr(cl);
  code_declt decl(expr);
  decl.operands().push_back(list);
  assert(current_block);
  current_block->copy_to_operands(decl);

  return expr;
}
void python_converter::get_delete_statement(
  const nlohmann::json &ast_node,
  codet &target_block)
{
  if (!ast_node.contains("targets") || !ast_node["targets"].is_array())
  {
    throw std::runtime_error("Delete statement missing targets");
  }

  for (const auto &target : ast_node["targets"])
  {
    if (target["_type"] == "Subscript")
    {
      exprt dict_expr = get_expr(target["value"]);
      const nlohmann::json &slice = target["slice"];

      typet dict_type = dict_expr.type();
      if (dict_expr.is_symbol())
      {
        const symbolt *sym = symbol_table_.find_symbol(dict_expr.identifier());
        if (sym)
          dict_type = sym->get_type();
      }

      if (dict_type.id() == "symbol")
        dict_type = ns.follow(dict_type);

      // del a[i] on a list removes (and shifts out) the element at index i.
      // This is exactly list.pop(i) with the result discarded, and pop is
      // already modelled (bounds-checked, shifting), so desugar to it instead
      // of requiring a dict.
      if (dict_type == type_handler_.get_list_type())
      {
        // del a[lower:upper] removes the slice — equivalent to a[lower:upper]
        // = []. Desugaring to pop() (below) would pass the Slice node as a pop
        // index, which is invalid; route slice deletes through the existing
        // slice-assignment lowering with an empty replacement instead.
        if (slice.contains("_type") && slice["_type"] == "Slice")
        {
          // Only a contiguous (absent / step-1) slice maps to a[i:j] = []. An
          // extended-step delete is always legal in CPython, but `a[::k] = []`
          // is not (the slice-assign model asserts a size match), so reject the
          // strided form with a clean diagnostic rather than a misleading
          // assignment-flavoured ValueError.
          const nlohmann::json &step =
            slice.contains("step") ? slice["step"] : nlohmann::json();
          bool contiguous = step.is_null();
          if (
            !contiguous && step.is_object() &&
            step.value("_type", "") == "Constant" && step.contains("value") &&
            step["value"].is_number_integer() &&
            step["value"].get<long long>() == 1)
            contiguous = true;
          if (!contiguous)
            throw std::runtime_error(
              "del on a strided list slice (step != 1) is not supported");

          nlohmann::json empty_list;
          empty_list["_type"] = "List";
          empty_list["elts"] = nlohmann::json::array();
          copy_location_fields_from_decl(ast_node, empty_list);
          python_list list_handler(*this, target);
          list_handler.handle_slice_assignment(dict_expr, slice, empty_list);
          continue;
        }

        // del a[i] on a list removes (and shifts out) the element at index i.
        // This is exactly list.pop(i) with the result discarded.
        nlohmann::json pop_call;
        pop_call["_type"] = "Call";
        pop_call["func"] = {
          {"_type", "Attribute"}, {"value", target["value"]}, {"attr", "pop"}};
        pop_call["args"] = nlohmann::json::array({slice});
        pop_call["keywords"] = nlohmann::json::array();
        copy_location_fields_from_decl(ast_node, pop_call);
        exprt pop_expr = get_function_call(pop_call);
        target_block.copy_to_operands(convert_expression_to_code(pop_expr));
        continue;
      }

      if (!dict_type.is_struct())
      {
        throw std::runtime_error(
          "del on subscript requires a dictionary (struct) type");
      }

      // Delegate to dict_handler which handles both constant and variable keys
      dict_handler_->handle_dict_delete(dict_expr, slice, target_block);
    }
    else if (target["_type"] == "Attribute")
    {
      // del obj.attr — Python semantics: remove the instance attribute so that
      // subsequent reads fall back to the class-level attribute.
      // We model this by resetting the struct member to the class default and
      // removing the instance-attribute registration.
      if (target["value"]["_type"] != "Name")
      {
        throw std::runtime_error(
          "del on nested attribute chains is not supported");
      }

      const std::string var_name = target["value"]["id"].get<std::string>();
      const std::string attr_name = target["attr"].get<std::string>();

      // Find the instance symbol (with fallback to global scope).
      symbol_id inst_sid = create_symbol_id();
      inst_sid.set_object(var_name);
      symbolt *inst_sym = find_symbol(inst_sid.to_string());
      if (!inst_sym)
      {
        inst_sid.set_function("");
        inst_sym = find_symbol(inst_sid.to_string());
      }
      if (!inst_sym)
      {
        throw std::runtime_error(
          "del attribute: instance variable '" + var_name + "' not found");
      }

      // Determine the class struct type from the instance symbol type.
      const typet &sym_type = inst_sym->get_type().is_pointer()
                                ? inst_sym->get_type().subtype()
                                : inst_sym->get_type();
      typet resolved = sym_type;
      if (resolved.id() == "symbol")
        resolved = ns.follow(resolved);
      if (resolved.id() != "struct")
      {
        throw std::runtime_error(
          "del attribute: '" + var_name + "' is not a struct instance");
      }

      const struct_typet &struct_type = to_struct_type(resolved);
      const std::string class_tag = struct_type.tag().as_string();
      const std::string class_name = extract_class_name_from_tag(class_tag);

      // Look up the authoritative class-type symbol so we see any dynamically
      // added components (e.g. added during a.x = 2 processing).
      const std::string class_tag_id = "tag-" + class_tag;
      const symbolt *class_type_sym = symbol_table_.find_symbol(class_tag_id);
      const struct_typet &class_struct =
        class_type_sym ? to_struct_type(class_type_sym->get_type())
                       : struct_type;

      // Find the class-level attribute symbol (the default value to restore).
      symbol_id class_sid = create_symbol_id();
      class_sid.set_function("");
      class_sid.set_class(class_name);
      class_sid.set_object(attr_name);
      symbolt *class_attr_sym =
        symbol_table_.find_symbol(class_sid.to_string());
      if (!class_attr_sym)
      {
        throw std::runtime_error(
          "del attribute: class '" + class_name +
          "' has no class-level attribute '" + attr_name + "'");
      }

      // Emit: obj.attr = ClassName::attr  (restore class default)
      if (class_struct.has_component(attr_name))
      {
        const typet &attr_type = class_struct.get_component(attr_name).type();
        exprt lhs = create_member_expression(*inst_sym, attr_name, attr_type);
        exprt rhs = symbol_expr(*class_attr_sym);
        if (rhs.type() != lhs.type())
          rhs = typecast_exprt(rhs, lhs.type());
        code_assignt assign(lhs, rhs);
        target_block.copy_to_operands(assign);
      }

      // Unregister the instance attribute so future reads fall back to the
      // class-level symbol instead of the (now-reset) struct member.
      auto map_it = instance_attr_map.find(inst_sym->id.as_string());
      if (map_it != instance_attr_map.end())
        map_it->second.erase(attr_name);
    }
    else if (target["_type"] == "Name")
    {
      log_warning("del on simple variables is not fully supported");
    }
    else
    {
      throw std::runtime_error(
        "Delete statement target type not supported: " +
        target["_type"].get<std::string>());
    }
  }
}
