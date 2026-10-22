#include <python-frontend/numpy/numpy_nditer.h>
#include <python-frontend/numpy/numpy_call_expr.h>
#include <python-frontend/python-list/python_list.h>
#include <python-frontend/python_converter.h>
#include <util/irep/std_expr.h>
#include <util/lang/c_types.h>

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace
{
using json = nlohmann::json;

// NumPy buffers at most this many elements, and the external loop of a
// buffered iterator yields one buffer at a time.
constexpr long long buffer_size = 8192;

json constant(long long value)
{
  return {{"_type", "Constant"}, {"value", value}};
}

json boolean(bool value)
{
  return {{"_type", "Constant"}, {"value", value}};
}

json name(const std::string &id)
{
  return {{"_type", "Name"}, {"id", id}, {"ctx", {{"_type", "Load"}}}};
}

json binop(const json &lhs, const std::string &op, const json &rhs)
{
  return {
    {"_type", "BinOp"}, {"left", lhs}, {"op", {{"_type", op}}}, {"right", rhs}};
}

json compare(const json &lhs, const std::string &op, const json &rhs)
{
  return {
    {"_type", "Compare"},
    {"left", lhs},
    {"ops", json::array({{{"_type", op}}})},
    {"comparators", json::array({rhs})}};
}

json if_exp(const json &test, const json &body, const json &orelse)
{
  return {
    {"_type", "IfExp"}, {"test", test}, {"body", body}, {"orelse", orelse}};
}

json tuple(const std::vector<json> &elts)
{
  return {
    {"_type", "Tuple"}, {"elts", json(elts)}, {"ctx", {{"_type", "Load"}}}};
}

json index_node(const json &value, const json &index)
{
  return {
    {"_type", "Subscript"},
    {"value", value},
    {"slice", index},
    {"ctx", {{"_type", "Load"}}}};
}

/// Copies the source position of `from` onto `node` and every node in it,
/// so diagnostics and the converter's location-keyed lookups see the call.
void locate(json &node, const json &from)
{
  if (node.is_array())
  {
    for (json &child : node)
      locate(child, from);
    return;
  }
  if (!node.is_object())
    return;
  for (const char *key :
       {"lineno", "col_offset", "end_lineno", "end_col_offset"})
    if (from.contains(key) && !node.contains(key))
      node[key] = from[key];
  for (auto &item : node.items())
    if (item.value().is_structured())
      locate(item.value(), from);
}

const std::set<std::string> &modelled_flags()
{
  static const std::set<std::string> flags = {
    "buffered",
    "external_loop",
    "grow_inner",
    "multi_index",
    "refs_ok",
    "zerosize_ok"};
  return flags;
}

const std::set<std::string> &numpy_flags()
{
  static const std::set<std::string> flags = {
    "buffered",
    "c_index",
    "common_dtype",
    "copy_if_overlap",
    "delay_bufalloc",
    "external_loop",
    "f_index",
    "grow_inner",
    "multi_index",
    "ranged",
    "refs_ok",
    "reduce_ok",
    "zerosize_ok"};
  return flags;
}

bool has(const std::vector<std::string> &flags, const std::string &flag)
{
  return std::find(flags.begin(), flags.end(), flag) != flags.end();
}

long long product(const std::vector<long long> &values)
{
  long long result = 1;
  for (long long value : values)
    result *= value;
  return result;
}

std::vector<long long> c_strides(const std::vector<long long> &shape)
{
  std::vector<long long> strides(shape.size(), 1);
  for (std::size_t axis = shape.size(); axis-- > 1;)
    strides[axis - 1] = strides[axis] * shape[axis];
  return strides;
}

const json *keyword(const json &call, const std::string &arg)
{
  if (!call.contains("keywords"))
    return nullptr;
  for (const json &kw : call["keywords"])
    if (kw["arg"].is_string() && kw["arg"] == arg)
      return &kw["value"];
  return nullptr;
}

/// A shape as NumPy spells it in a message: `(2,)`, `(2,3)`.
std::string shape_text(const std::vector<long long> &shape)
{
  std::string text = "(";
  for (std::size_t i = 0; i < shape.size(); ++i)
    text += std::to_string(shape[i]) +
            (shape.size() == 1 || i + 1 < shape.size() ? "," : "");
  return text + ")";
}

/// The flags of each operand: `op_flags=[...]` applies to every operand,
/// `op_flags=[[...], [...]]` lists them per operand.
std::vector<std::vector<std::string>>
operand_flags(const json &call, std::size_t operands)
{
  std::vector<std::vector<std::string>> result(operands);
  const json *value = keyword(call, "op_flags");
  if (!value)
    return result;
  auto strings = [](const json &list) {
    std::vector<std::string> flags;
    const std::string type = list.value("_type", "");
    if (type != "List" && type != "Tuple")
      throw std::runtime_error(
        "TypeError: numpy.nditer() op_flags must be literal lists of strings");
    for (const json &flag : list["elts"])
    {
      if (flag.value("_type", "") != "Constant" || !flag["value"].is_string())
        throw std::runtime_error(
          "TypeError: numpy.nditer() op_flags must be literal lists of "
          "strings");
      flags.push_back(flag["value"].get<std::string>());
    }
    return flags;
  };
  const json &elts = (*value).value("elts", json::array());
  const bool per_operand =
    !elts.empty() && elts[0].value("_type", "") != "Constant";
  if (per_operand && elts.size() != operands)
    throw std::runtime_error(
      "ValueError: numpy.nditer() op_flags must list one entry per operand");
  for (std::size_t op = 0; op < operands; ++op)
    result[op] = strings(per_operand ? elts[op] : *value);
  return result;
}

bool writable(const std::vector<std::string> &flags)
{
  return has(flags, "readwrite") || has(flags, "writeonly");
}

const std::set<std::string> &numpy_op_flags()
{
  static const std::set<std::string> flags = {
    "aligned",
    "allocate",
    "arraymask",
    "contig",
    "copy",
    "nbo",
    "no_broadcast",
    "no_subtype",
    "overlap_assume_elementwise",
    "readonly",
    "readwrite",
    "updateifcopy",
    "virtual",
    "writemasked",
    "writeonly"};
  return flags;
}

// The message NumPy raises for an operand's flags, or empty.
std::string invalid_op_flags(const std::vector<std::string> &flags)
{
  int modes = 0;
  for (const std::string &flag : flags)
  {
    if (!numpy_op_flags().count(flag))
      return "Unexpected per-op iterator flag \"" + flag + "\"";
    if (flag != "readonly" && flag != "readwrite" && flag != "writeonly")
      throw std::runtime_error(
        "TypeError: numpy.nditer() op_flag '" + flag + "' is not supported");
    ++modes;
  }
  if (modes > 1)
    return "Only one of the iterator flags READWRITE, READONLY, and WRITEONLY "
           "may be specified for an operand";
  return "";
}

// `x[...]` or `x[()]`: the whole 0-d item.
bool whole_item(const json &index)
{
  return (index.value("_type", "") == "Constant" && index["value"] == "...") ||
         (index.value("_type", "") == "Tuple" && index["elts"].empty());
}

/// The stride of operand `op` along iteration axis `axis`, as visited (a
/// flipped axis walks the other way); 0 when the operand is broadcast.
long long visited_stride(
  const numpy_nditer_state &it,
  std::size_t op,
  std::size_t axis,
  bool as_visited)
{
  const numpy_nditer_operand &o = it.operands[op];
  const std::size_t lead = it.shape.size() - o.shape.size();
  if (axis < lead || (o.shape[axis - lead] == 1 && it.shape[axis] != 1))
    return 0;
  const long long s = o.strides[axis - lead];
  return as_visited && it.flipped[axis] ? -s : s;
}
} // namespace

numpy_nditer::numpy_nditer(python_converter &converter) : converter_(converter)
{
}

bool numpy_nditer::is_nditer_call(const json &node, const json &ast)
{
  return node.is_object() && node.value("_type", "") == "Call" &&
         node["func"].value("_type", "") == "Attribute" &&
         node["func"].value("attr", "") == "nditer" &&
         node["func"]["value"].value("_type", "") == "Name" &&
         is_imported_numpy_module_alias(
           ast, node["func"]["value"].value("id", ""));
}

std::string
numpy_nditer::key(const std::string &function, const std::string &name)
{
  return function + "::" + name;
}

const numpy_nditer_state *numpy_nditer::find(const json &node) const
{
  if (!node.is_object() || node.value("_type", "") != "Name")
    return nullptr;
  auto it = converter_.numpy_nditers_.find(
    key(converter_.current_func_name_, node["id"].get<std::string>()));
  return it == converter_.numpy_nditers_.end() ? nullptr : &it->second;
}

const numpy_nditer_item *numpy_nditer::find_item(const json &node) const
{
  if (!node.is_object() || node.value("_type", "") != "Name")
    return nullptr;
  auto it = converter_.numpy_nditer_items_.find(
    key(converter_.current_func_name_, node["id"].get<std::string>()));
  return it == converter_.numpy_nditer_items_.end() ? nullptr : &it->second;
}

json numpy_nditer::prebuilt(const exprt &expr)
{
  converter_.prebuilt_exprs_.push_back(expr);
  return {
    {"_type", "EsbmcExpr"}, {"slot", converter_.prebuilt_exprs_.size() - 1}};
}

void numpy_nditer::assign(const std::string &target, const json &value)
{
  json statement = {
    {"_type", "Assign"},
    {"targets",
     json::array(
       {{{"_type", "Name"}, {"id", target}, {"ctx", {{"_type", "Store"}}}}})},
    {"value", value}};
  locate(statement, converter_.current_nditer_call_);
  // The type probe of an assignment's value is discarded and converted again.
  if (
    !converter_.emit_statement_into_current_block(statement) &&
    !converter_.in_rhs_type_probe_)
    throw std::runtime_error(
      "TypeError: numpy.nditer() is only supported in a statement");
}

void numpy_nditer::raise_if(
  const json &cond,
  const std::string &exception,
  const std::string &message)
{
  json test = cond;
  locate(test, converter_.current_nditer_call_);
  converter_.emit_guarded_raise(
    converter_.get_expr(test),
    exception,
    message,
    converter_.get_location_from_decl(converter_.current_nditer_call_));
}

void numpy_nditer::raise(
  const std::string &exception,
  const std::string &message)
{
  converter_.emit_guarded_raise(
    true_exprt(),
    exception,
    message,
    converter_.get_location_from_decl(converter_.current_nditer_call_));
}

std::vector<std::string> numpy_nditer::flags(const json &call)
{
  std::vector<std::string> result;
  const json *value = keyword(call, "flags");
  if (!value)
    return result;
  const std::string type = value->value("_type", "");
  if (type != "List" && type != "Tuple")
    throw std::runtime_error(
      "TypeError: numpy.nditer() flags must be a literal list of strings");
  for (const json &flag : (*value)["elts"])
  {
    if (flag.value("_type", "") != "Constant" || !flag["value"].is_string())
      throw std::runtime_error(
        "TypeError: numpy.nditer() flags must be a literal list of strings");
    result.push_back(flag["value"].get<std::string>());
  }
  return result;
}

std::string numpy_nditer::order(const json &call)
{
  const json *value = keyword(call, "order");
  if (!value)
    return "K";
  if (value->value("_type", "") != "Constant" || !(*value)["value"].is_string())
    throw std::runtime_error(
      "TypeError: numpy.nditer() order must be a string literal");
  return (*value)["value"].get<std::string>();
}

// The operand as a name: an expression is first bound to a hidden variable,
// evaluated once where the iterator is created.
json numpy_nditer::named_operand(const json &node, std::size_t index)
{
  if (node.value("_type", "") == "Name")
    return node;
  const std::string temp =
    converter_.current_nditer_state_ + "$op" + std::to_string(index);
  assign(temp, node);
  json array = name(temp);
  locate(array, converter_.current_nditer_call_);
  return array;
}

template <class Info>
void numpy_nditer::runtime_layout(
  const Info &info,
  numpy_nditer_operand &result)
{
  if (info.shape.size() > 1)
    throw std::runtime_error(
      "TypeError: numpy.nditer() over an N-D view whose layout is only known "
      "at run time is not supported");
  auto runtime_value =
    [&](const std::vector<std::string> &ids, long long fallback) {
      if (ids.empty() || ids.front().empty())
        return constant(fallback);
      const symbolt *symbol = converter_.symbol_table_.find_symbol(ids.front());
      if (!symbol)
        throw std::runtime_error(
          "TypeError: numpy.nditer() lost a run-time view extent");
      return prebuilt(symbol_expr(*symbol));
    };
  result.shape = {0};
  result.strides = {info.stride};
  result.extent = runtime_value(
    info.shape_symbols,
    info.shape.empty() ? static_cast<long long>(info.length)
                       : static_cast<long long>(info.shape.front()));
  result.stride = runtime_value(info.stride_symbols, info.stride);
}

// A view's shape and strides. A 1-D view keeps a run-time extent and stride
// as expressions; any other view whose layout is only known at run time, or
// that is bound to different arrays on different paths, has no order to
// iterate it in.
template <class Info>
void numpy_nditer::view_layout(const Info &info, numpy_nditer_operand &result)
{
  if (!info.other_sources.empty())
    throw std::runtime_error(
      "TypeError: numpy.nditer() over a view bound to different arrays on "
      "different paths is not supported");
  if (info.is_symbolic())
  {
    runtime_layout(info, result);
    return;
  }
  if (info.shape.empty())
    result.shape = {static_cast<long long>(info.length)};
  else
    result.shape.assign(info.shape.begin(), info.shape.end());
  if (!info.strides.empty())
    result.strides = info.strides;
  else if (result.shape.size() == 1)
    result.strides = {info.stride};
  else
    result.strides = c_strides(result.shape);
}

// The layout of a named array or view, in elements; a number is a 0-d
// operand, broadcast to every position.
numpy_nditer_operand numpy_nditer::operand(const json &node, std::size_t index)
{
  numpy_nditer_operand result;
  result.array = named_operand(node, index);
  const std::string id =
    converter_.resolve_name_symbol_id(result.array["id"].get<std::string>());
  const bool array =
    !id.empty() && converter_.numpy_array_symbols_.count(id) != 0;
  const symbolt *symbol =
    id.empty() ? nullptr : converter_.symbol_table_.find_symbol(id);
  if (
    !array && symbol && !symbol->get_type().is_array() &&
    !symbol->get_type().is_pointer())
    return result;
  if (!array)
    throw std::runtime_error(
      "TypeError: numpy.nditer() operand must be a numpy array or a number");

  auto view = converter_.numpy_pointer_view_info_.find(id);
  if (view != converter_.numpy_pointer_view_info_.end())
  {
    view_layout(view->second, result);
    return result;
  }
  std::optional<std::vector<std::size_t>> shape =
    converter_.get_numpy_nditer_logical_shape(id);
  if (!shape)
    throw std::runtime_error(
      "TypeError: numpy.nditer() operand has no static shape");
  result.shape.assign(shape->begin(), shape->end());
  result.strides = c_strides(result.shape);
  return result;
}

namespace
{
// Whether every operand steps through memory in C (or Fortran) order.
bool contiguous(const numpy_nditer_state &it, bool fortran)
{
  const std::size_t ndim = it.shape.size();
  for (std::size_t op = 0; op < it.operands.size(); ++op)
  {
    long long expected = 1;
    for (std::size_t i = 0; i < ndim; ++i)
    {
      const std::size_t axis = fortran ? i : ndim - 1 - i;
      if (
        it.shape[axis] != 1 && visited_stride(it, op, axis, false) != expected)
        return false;
      expected *= it.shape[axis];
    }
  }
  return true;
}

// Whether axis `outer` goes inside axis `inner` for NumPy's K order: the
// first operand that strides along both decides a swap, and any operand with
// the smaller stride on `inner` keeps the order. Empty when no operand
// strides along both.
std::optional<bool>
swaps(const numpy_nditer_state &it, std::size_t outer, std::size_t inner)
{
  std::optional<bool> swap;
  for (std::size_t op = 0; op < it.operands.size(); ++op)
  {
    const long long s0 = visited_stride(it, op, outer, false);
    const long long s1 = visited_stride(it, op, inner, false);
    if (s0 == 0 || s1 == 0)
      continue;
    if (std::llabs(s1) <= std::llabs(s0))
      swap = false;
    else if (!swap)
      swap = true;
  }
  return swap;
}

// npyiter_find_best_axis_ordering: an insertion sort of the axes, innermost
// first, by stride magnitude.
std::vector<std::size_t> k_order(const numpy_nditer_state &it)
{
  std::vector<std::size_t> perm;
  for (std::size_t axis = it.shape.size(); axis-- > 0;)
    perm.push_back(axis);
  for (std::size_t i0 = 1; i0 < perm.size(); ++i0)
  {
    std::size_t ipos = i0;
    const std::size_t j0 = perm[i0];
    for (std::size_t i1 = i0; i1-- > 0;)
    {
      const std::optional<bool> swap = swaps(it, j0, perm[i1]);
      if (swap && !*swap)
        break;
      if (swap)
        ipos = i1;
    }
    perm.erase(perm.begin() + i0);
    perm.insert(perm.begin() + ipos, j0);
  }
  return {perm.rbegin(), perm.rend()};
}

// npyiter_flip_negative_strides: an axis every operand has a non-positive,
// and some a negative, stride on is walked backwards.
std::vector<bool> flip_negative(const numpy_nditer_state &it)
{
  std::vector<bool> flipped(it.shape.size(), false);
  for (std::size_t axis = 0; axis < it.shape.size(); ++axis)
  {
    bool negative = false;
    bool positive = false;
    for (std::size_t op = 0; op < it.operands.size(); ++op)
    {
      negative = negative || visited_stride(it, op, axis, false) < 0;
      positive = positive || visited_stride(it, op, axis, false) > 0;
    }
    flipped[axis] = negative && !positive;
  }
  return flipped;
}
} // namespace

// NumPy's iteration order. C and F are forced; A is F when every operand is
// Fortran- but not C-contiguous; K sorts the axes by stride magnitude and
// undoes negative strides.
void numpy_nditer::order_axes(numpy_nditer_state &it, const std::string &order)
  const
{
  it.flipped.assign(it.shape.size(), false);
  it.axes.clear();
  for (std::size_t axis = 0; axis < it.shape.size(); ++axis)
    it.axes.push_back(axis);
  if (
    order == "F" ||
    (order == "A" && contiguous(it, true) && !contiguous(it, false)))
    std::reverse(it.axes.begin(), it.axes.end());
  if (order != "K")
    return;
  it.flip_negative = true;
  it.axes = k_order(it);
  it.flipped = flip_negative(it);
}

// The elements an external loop yields at once: the whole iteration when
// buffered, otherwise the innermost axes whose elements are evenly spaced in
// every operand (npyiter_coalesce_axes).
void numpy_nditer::set_chunk(numpy_nditer_state &it) const
{
  it.chunk = 1;
  if (!it.external_loop || it.size == 0)
    return;
  if (it.buffered)
  {
    if (it.size > buffer_size)
      throw std::runtime_error(
        "TypeError: numpy.nditer() with a buffered external loop over more "
        "than 8192 elements is not supported");
    it.chunk = it.size;
    return;
  }
  it.chunk = reported_shape(it).back();
}

namespace
{
void check_keywords(const json &call)
{
  if (call["args"].size() != 1)
    throw std::runtime_error(
      "TypeError: numpy.nditer() takes its operands as the only positional "
      "argument");
  for (const json &kw : call["keywords"])
  {
    const std::string arg =
      kw["arg"].is_string() ? kw["arg"].get<std::string>() : "";
    if (
      arg != "flags" && arg != "order" && arg != "op_flags" &&
      arg != "op_dtypes" && arg != "op_axes" && arg != "casting")
      throw std::runtime_error(
        "TypeError: numpy.nditer() keyword '" + arg + "' is not supported");
  }
}

std::vector<json> operand_nodes(const json &call)
{
  const json &arg = call["args"][0];
  const std::string type = arg.value("_type", "");
  if (type == "List" || type == "Tuple")
    return arg["elts"].get<std::vector<json>>();
  return {arg};
}
} // namespace

// The message NumPy raises for the global flags or the order, or empty.
std::string numpy_nditer::invalid_setup(
  const std::vector<std::string> &flag_list,
  const std::string &order_name) const
{
  for (const std::string &flag : flag_list)
  {
    if (!numpy_flags().count(flag))
      return "Unexpected iterator global flag \"" + flag + "\"";
    if (!modelled_flags().count(flag))
      throw std::runtime_error(
        "TypeError: numpy.nditer() flag '" + flag + "' is not supported");
  }
  if (
    order_name != "C" && order_name != "F" && order_name != "A" &&
    order_name != "K")
    return "order must be one of 'C', 'F', 'A', or 'K' (got '" + order_name +
           "')";
  if (has(flag_list, "external_loop") && has(flag_list, "multi_index"))
    return "Iterator flag EXTERNAL_LOOP cannot be used if an index or "
           "multi-index is being tracked";
  return "";
}

// A writable operand must cover the whole iteration: NumPy does not
// broadcast into an output.
bool numpy_nditer::writable_operands_fit(
  const numpy_nditer_state &it,
  const std::vector<std::vector<std::string>> &op_flags)
{
  for (std::size_t op = 0; op < op_flags.size(); ++op)
    if (writable(op_flags[op]) && it.operands[op].shape != it.shape)
    {
      raise(
        "ValueError",
        "non-broadcastable output operand with shape " +
          shape_text(it.operands[op].shape) +
          " doesn't match the broadcast shape " + shape_text(it.shape));
      return false;
    }
  for (std::size_t op = 0; op < op_flags.size(); ++op)
  {
    if (!writable(op_flags[op]))
      continue;
    if (it.operands[op].shape.empty())
      throw std::runtime_error(
        "TypeError: numpy.nditer() writes only to array operands");
    if (it.buffered)
      throw std::runtime_error(
        "TypeError: numpy.nditer() writes through a buffered iterator are "
        "not supported");
  }
  return true;
}

// Zero-sized iteration is an error unless `zerosize_ok`; a run-time extent is
// checked when the iterator is created.
bool numpy_nditer::check_size(numpy_nditer_state &it, bool zerosize_ok)
{
  if (it.runtime)
  {
    if (!zerosize_ok)
      raise_if(
        compare(it.operands.front().extent, "Eq", constant(0)),
        "ValueError",
        "Iteration of zero-sized operands is not enabled");
    return true;
  }
  it.size = product(it.shape);
  if (it.size == 0 && !zerosize_ok)
  {
    raise("ValueError", "Iteration of zero-sized operands is not enabled");
    return false;
  }
  set_chunk(it);
  return true;
}

std::optional<numpy_nditer_state>
numpy_nditer::describe(const json &call, const std::string &state)
{
  converter_.current_nditer_call_ = call;
  converter_.current_nditer_state_ = state;
  check_keywords(call);
  const std::vector<json> ops = operand_nodes(call);
  const std::vector<std::vector<std::string>> op_flags =
    operand_flags(call, ops.size());
  for (std::size_t op = 0; op < ops.size(); ++op)
    if (
      writable(op_flags[op]) && converter_.is_numpy_readonly_view_arg(ops[op]))
      throw std::runtime_error(
        "ValueError: assignment destination is read-only");

  const std::vector<std::string> flag_list = flags(call);
  const std::string order_name = order(call);
  std::string error = invalid_setup(flag_list, order_name);
  for (std::size_t op = 0; op < ops.size() && error.empty(); ++op)
    error = invalid_op_flags(op_flags[op]);
  if (!error.empty())
  {
    raise("ValueError", error);
    return std::nullopt;
  }

  numpy_nditer_state it;
  it.state = state;
  it.alias = call["func"]["value"]["id"].get<std::string>();
  it.multi_index = has(flag_list, "multi_index");
  it.external_loop = has(flag_list, "external_loop");
  it.buffered = has(flag_list, "buffered");
  for (std::size_t op = 0; op < ops.size(); ++op)
  {
    it.operands.push_back(operand(ops[op], op));
    it.operands.back().writable = writable(op_flags[op]);
  }
  if (
    !apply_op_axes(it, call) || !broadcast(it) ||
    !writable_operands_fit(it, op_flags) || !apply_op_dtypes(it, call))
    return std::nullopt;
  if (it.runtime && it.external_loop)
    throw std::runtime_error(
      "TypeError: numpy.nditer() external_loop over a view whose layout is "
      "only known at run time is not supported");
  order_axes(it, order_name);
  if (!check_size(it, has(flag_list, "zerosize_ok")))
    return std::nullopt;
  return it;
}

namespace
{
// An integer literal, possibly negated; nullopt for anything else.
std::optional<long long> literal_int(const json &node)
{
  if (
    node.value("_type", "") == "Constant" && node["value"].is_number_integer())
    return node["value"].get<long long>();
  if (
    node.value("_type", "") == "UnaryOp" &&
    node["op"].value("_type", "") == "USub")
    if (std::optional<long long> value = literal_int(node["operand"]))
      return -*value;
  return std::nullopt;
}

// The `op_axes` entry of each operand, null when the operand has none.
std::vector<json>
per_operand(const json &call, const std::string &arg, std::size_t operands)
{
  const json *value = keyword(call, arg);
  if (!value)
    return std::vector<json>(operands);
  const std::string type = value->value("_type", "");
  if (
    (type != "List" && type != "Tuple") || (*value)["elts"].size() != operands)
    throw std::runtime_error(
      "TypeError: numpy.nditer() " + arg +
      " must be a literal list with one entry per operand");
  std::vector<json> entries;
  for (const json &entry : (*value)["elts"])
    entries.push_back(
      entry.value("_type", "") == "Constant" && entry["value"].is_null()
        ? json()
        : entry);
  return entries;
}
} // namespace

// `op_axes`: iteration axis i reads axis op_axes[op][i] of the operand, or a
// new length-1 axis for -1. Every operand axis must be iterated (a reduction
// is not modelled).
bool numpy_nditer::apply_op_axes(numpy_nditer_state &it, const json &call)
{
  const std::vector<json> entries =
    per_operand(call, "op_axes", it.operands.size());
  for (std::size_t op = 0; op < entries.size(); ++op)
  {
    if (entries[op].is_null())
      continue;
    numpy_nditer_operand &o = it.operands[op];
    if (!o.extent.is_null())
      throw std::runtime_error(
        "TypeError: numpy.nditer() op_axes over a view whose layout is only "
        "known at run time is not supported");
    std::vector<long long> axes;
    for (const json &axis : entries[op].value("elts", json::array()))
    {
      const std::optional<long long> value = literal_int(axis);
      if (!value)
        throw std::runtime_error(
          "TypeError: numpy.nditer() op_axes must be integer literals");
      axes.push_back(*value);
    }
    const std::string error = invalid_op_axes(o, axes, op);
    if (!error.empty())
    {
      raise("ValueError", error);
      return false;
    }
    std::vector<long long> shape, strides;
    for (long long axis : axes)
    {
      shape.push_back(axis < 0 ? 1 : o.shape[axis]);
      strides.push_back(axis < 0 ? 0 : o.strides[axis]);
    }
    o.shape = shape;
    o.strides = strides;
    o.axes = axes;
  }
  return true;
}

// The message NumPy raises for an operand's op_axes, or empty.
std::string numpy_nditer::invalid_op_axes(
  const numpy_nditer_operand &o,
  const std::vector<long long> &axes,
  std::size_t op) const
{
  const long long rank = static_cast<long long>(o.shape.size());
  std::vector<bool> seen(o.shape.size(), false);
  for (std::size_t i = 0; i < axes.size(); ++i)
  {
    if (axes[i] < -1 || axes[i] >= rank)
      return "Iterator input op_axes[" + std::to_string(op) + "][" +
             std::to_string(i) + "] (==" + std::to_string(axes[i]) +
             ") is not a valid axis of op[" + std::to_string(op) +
             "], which has " + std::to_string(rank) + " dimensions ";
    if (axes[i] < 0)
      continue;
    if (seen[axes[i]])
      return "The 'op_axes' provided to the iterator constructor for operand " +
             std::to_string(op) + " contained duplicate value " +
             std::to_string(axes[i]);
    seen[axes[i]] = true;
  }
  if (std::find(seen.begin(), seen.end(), false) != seen.end())
    throw std::runtime_error(
      "TypeError: numpy.nditer() op_axes that leave an operand axis out (a "
      "reduction) are not supported");
  return "";
}

namespace
{
/// A dtype by NumPy name, alias, scalar type or builtin; an empty type when
/// unknown.
typet dtype_type(const json &node)
{
  std::string name;
  if (node.value("_type", "") == "Constant" && node["value"].is_string())
    name = node["value"].get<std::string>();
  else if (node.value("_type", "") == "Attribute")
    name = node["attr"].get<std::string>();
  else if (node.value("_type", "") == "Name")
    name = node["id"].get<std::string>();
  static const std::map<std::string, std::pair<char, unsigned>> table = {
    {"bool", {'b', 1}},    {"?", {'b', 1}},        {"bool_", {'b', 1}},
    {"int8", {'i', 8}},    {"i1", {'i', 8}},       {"int16", {'i', 16}},
    {"i2", {'i', 16}},     {"int32", {'i', 32}},   {"i4", {'i', 32}},
    {"int64", {'i', 64}},  {"i8", {'i', 64}},      {"int", {'i', 64}},
    {"uint8", {'u', 8}},   {"u1", {'u', 8}},       {"uint16", {'u', 16}},
    {"u2", {'u', 16}},     {"uint32", {'u', 32}},  {"u4", {'u', 32}},
    {"uint64", {'u', 64}}, {"u8", {'u', 64}},      {"float32", {'f', 32}},
    {"f4", {'f', 32}},     {"float64", {'f', 64}}, {"f8", {'f', 64}},
    {"float", {'f', 64}},  {"double", {'f', 64}}};
  auto it = table.find(name);
  if (it == table.end())
    return typet();
  const auto [kind, width] = it->second;
  if (kind == 'b')
    return bool_type();
  if (kind == 'f')
    return width == 32 ? float_type() : double_type();
  return kind == 'i' ? typet(signedbv_typet(width))
                     : typet(unsignedbv_typet(width));
}

char dtype_kind(const typet &type)
{
  if (type.is_bool())
    return 'b';
  if (type.is_floatbv())
    return 'f';
  return type.is_unsignedbv() ? 'u' : 'i';
}

unsigned dtype_width(const typet &type)
{
  if (type.is_bool())
    return 1;
  return std::stoul(type.width().as_string());
}

/// NumPy's "safe" casting: no value of `from` changes in `to`.
bool safe_cast(char fk, unsigned fw, char tk, unsigned tw)
{
  if (fk == 'b' || (fk == tk && fw <= tw))
    return true;
  if (fk == 'u' && tk == 'i')
    return fw < tw;
  return tk == 'f' && fk != 'f' && (fw < tw || tw == 64);
}

/// NumPy's can_cast for the bool, integer and float dtypes.
bool can_cast(const typet &from, const typet &to, const std::string &rule)
{
  const char fk = dtype_kind(from), tk = dtype_kind(to);
  const unsigned fw = dtype_width(from), tw = dtype_width(to);
  if (rule == "unsafe")
    return true;
  if (rule == "no" || rule == "equiv")
    return fk == tk && fw == tw;
  const bool safe = safe_cast(fk, fw, tk, tw);
  return safe ||
         (rule == "same_kind" && (fk == tk || (fk == 'u' && tk == 'i')));
}

std::string dtype_name(const typet &type)
{
  const char kind = dtype_kind(type);
  if (kind == 'b')
    return "bool";
  const std::string width = std::to_string(dtype_width(type));
  return (kind == 'f' ? "float" : kind == 'u' ? "uint" : "int") + width;
}
} // namespace

// `op_dtypes`: each operand is read through a cast to its dtype, which NumPy
// allows under the `casting` rule and only through a buffer.
bool numpy_nditer::apply_op_dtypes(numpy_nditer_state &it, const json &call)
{
  const std::vector<json> entries =
    per_operand(call, "op_dtypes", it.operands.size());
  const json *casting = keyword(call, "casting");
  const std::string rule =
    casting ? casting->value("value", json("")).get<std::string>() : "safe";
  if (
    rule != "no" && rule != "equiv" && rule != "safe" && rule != "same_kind" &&
    rule != "unsafe")
  {
    raise(
      "ValueError",
      "casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'");
    return false;
  }
  for (std::size_t op = 0; op < entries.size(); ++op)
  {
    if (entries[op].is_null())
      continue;
    numpy_nditer_operand &o = it.operands[op];
    const typet to = dtype_type(entries[op]);
    const typet from = element_type(o);
    if (to.id().empty())
      throw std::runtime_error(
        "TypeError: numpy.nditer() op_dtypes entry is not a supported dtype");
    if (from == to)
      continue;
    std::string error;
    if (!it.buffered)
      error =
        "Iterator operand required copying or buffering, but neither "
        "copying nor buffering was enabled";
    else if (!can_cast(from, to, rule))
      error = "Iterator operand " + std::to_string(op) +
              " dtype could not be cast from dtype('" + dtype_name(from) +
              "') to dtype('" + dtype_name(to) + "') according to the rule '" +
              rule + "'";
    if (!error.empty())
    {
      raise("TypeError", error);
      return false;
    }
    o.dtype = to;
  }
  return true;
}

// The element type an operand holds.
typet numpy_nditer::element_type(const numpy_nditer_operand &o) const
{
  json first = o.array;
  if (!o.shape.empty())
  {
    std::size_t rank = o.axes.empty() ? o.shape.size() : 0;
    for (long long axis : o.axes)
      rank += axis >= 0;
    first = index_node(
      o.array,
      rank == 1 ? constant(0) : tuple(std::vector<json>(rank, constant(0))));
  }
  locate(first, converter_.current_nditer_call_);
  return converter_.get_expr(first).type();
}

// The iteration shape: the operands' shapes broadcast together, aligned on
// their last axes. A 1-D view with a run-time layout is iterated alone.
bool numpy_nditer::broadcast(numpy_nditer_state &it)
{
  it.runtime = !it.operands.front().extent.is_null();
  if (it.runtime && it.operands.size() > 1)
    throw std::runtime_error(
      "TypeError: numpy.nditer() over several operands with a view whose "
      "layout is only known at run time is not supported");
  for (const numpy_nditer_operand &o : it.operands)
    if (o.shape.size() > it.shape.size())
      it.shape.insert(it.shape.begin(), o.shape.size() - it.shape.size(), 1);
  for (const numpy_nditer_operand &o : it.operands)
  {
    const std::size_t lead = it.shape.size() - o.shape.size();
    for (std::size_t j = 0; j < o.shape.size(); ++j)
    {
      long long &extent = it.shape[j + lead];
      if (extent == 1)
        extent = o.shape[j];
      else if (o.shape[j] != 1 && o.shape[j] != extent)
      {
        std::string shapes;
        for (const numpy_nditer_operand &each : it.operands)
          shapes += shape_text(each.shape) + " ";
        raise(
          "ValueError",
          "operands could not be broadcast together with shapes " + shapes);
        return false;
      }
    }
  }
  return true;
}

void numpy_nditer::create(const std::string &target, const json &call)
{
  const std::string state = target + "$nditer";
  const std::string id = key(converter_.current_func_name_, target);
  converter_.numpy_nditers_.erase(id);
  std::optional<numpy_nditer_state> it = describe(call, state);
  if (!it)
    return;
  assign(state + "$k", constant(0));
  assign(state + "$started", boolean(false));
  converter_.numpy_nditers_[id] = *it;
}

json numpy_nditer::position(const numpy_nditer_state &it) const
{
  return name(it.state + "$k");
}

json numpy_nditer::items(const numpy_nditer_state &it) const
{
  if (it.runtime)
    return it.operands.front().extent;
  return constant(it.size == 0 ? 0 : it.size / it.chunk);
}

json numpy_nditer::finished(const numpy_nditer_state &it) const
{
  return compare(position(it), "GtE", items(it));
}

// The axis extents `shape` reports: the logical ones while a multi-index is
// tracked, otherwise the visited axes merged where every operand steps evenly
// across them (npyiter_coalesce_axes).
std::vector<long long>
numpy_nditer::reported_shape(const numpy_nditer_state &it)
{
  if (it.multi_index || it.axes.empty())
    return it.shape;
  std::vector<long long> merged;
  std::size_t inner = it.axes.back();
  long long extent = it.shape[inner];
  for (std::size_t i = it.axes.size() - 1; i-- > 0;)
  {
    const std::size_t outer = it.axes[i];
    bool mergeable = true;
    for (std::size_t op = 0; op < it.operands.size(); ++op)
      mergeable = mergeable && (it.shape[outer] == 1 ||
                                visited_stride(it, op, outer, true) ==
                                  extent * visited_stride(it, op, inner, true));
    if (mergeable)
    {
      extent *= it.shape[outer];
      continue;
    }
    merged.insert(merged.begin(), extent);
    inner = outer;
    extent = it.shape[outer];
  }
  merged.insert(merged.begin(), extent);
  return merged;
}

// The iteration index along each axis of the element visited at
// `element_position`, in the axes' own order.
std::vector<json> numpy_nditer::indices(
  const numpy_nditer_state &it,
  const json &element_position) const
{
  if (it.runtime)
  {
    const numpy_nditer_operand &o = it.operands.front();
    if (!it.flip_negative)
      return {element_position};
    json last = binop(o.extent, "Sub", constant(1));
    return {if_exp(
      compare(o.stride, "Lt", constant(0)),
      binop(last, "Sub", element_position),
      element_position)};
  }
  std::vector<json> result(it.shape.size());
  long long inner = 1;
  for (std::size_t i = it.axes.size(); i-- > 0;)
  {
    const std::size_t axis = it.axes[i];
    json digit = inner == 1
                   ? element_position
                   : binop(element_position, "FloorDiv", constant(inner));
    if (i != 0)
      digit = binop(digit, "Mod", constant(it.shape[axis]));
    if (it.flipped[axis])
      digit = binop(constant(it.shape[axis] - 1), "Sub", digit);
    result[axis] = digit;
    inner *= it.shape[axis];
  }
  return result;
}

json numpy_nditer::element(
  const numpy_nditer_state &it,
  std::size_t op,
  const json &element_position) const
{
  const numpy_nditer_operand &o = it.operands[op];
  const std::vector<json> axis_index = indices(it, element_position);
  const std::size_t lead = it.shape.size() - o.shape.size();
  std::vector<json> index;
  for (std::size_t j = 0; j < o.shape.size(); ++j)
  {
    const std::size_t axis = j + lead;
    index.push_back(
      !it.runtime && o.shape[j] == 1 && it.shape[axis] != 1 ? constant(0)
                                                            : axis_index[axis]);
  }
  if (!o.axes.empty())
  {
    // Operand axis o.axes[i] is iterated as axis i; -1 is a new axis.
    std::vector<json> mapped(index.size());
    std::size_t rank = 0;
    for (std::size_t i = 0; i < o.axes.size(); ++i)
      if (o.axes[i] >= 0)
      {
        mapped[o.axes[i]] = index[i];
        ++rank;
      }
    mapped.resize(rank);
    index = mapped;
  }
  if (index.empty())
    return o.array;
  return index_node(o.array, index.size() == 1 ? index.front() : tuple(index));
}

// The element as a read sees it: cast to the operand's requested dtype.
json numpy_nditer::value(
  const numpy_nditer_state &it,
  std::size_t op,
  const json &element_position)
{
  json node = element(it, op, element_position);
  const typet &type = it.operands[op].dtype;
  if (type.id().empty())
    return node;
  locate(node, converter_.current_nditer_call_);
  return prebuilt(typecast_exprt(converter_.get_expr(node), type));
}

// The chunk an external loop yields, read from the operand now.
exprt numpy_nditer::chunk_value(
  const numpy_nditer_state &it,
  const numpy_nditer_item &item)
{
  const json start = binop(name(item.position), "Mult", constant(it.chunk));
  std::vector<exprt> elements;
  for (long long j = 0; j < it.chunk; ++j)
  {
    json node = value(it, item.operand, binop(start, "Add", constant(j)));
    locate(node, converter_.current_nditer_call_);
    elements.push_back(converter_.get_expr(node));
  }
  const typet element_type = elements.front().type();
  exprt array(
    "array", array_typet(element_type, from_integer(it.chunk, size_type())));
  for (const exprt &e : elements)
    array.copy_to_operands(
      e.type() == element_type ? e : typecast_exprt(e, element_type));
  return array;
}

// A numpy array holding the chunk as it reads now, for a consumer that needs
// a named array (`chunk.sum()`, `len(chunk)`).
json numpy_nditer::materialize(const numpy_nditer_item &item)
{
  const numpy_nditer_state &it = converter_.numpy_nditers_.at(item.iterator);
  const std::string temp =
    it.state + "$chunk" + std::to_string(++converter_.nditer_counter_);
  assign(temp, prebuilt(chunk_value(it, item)));
  converter_.numpy_array_symbols_.insert(
    converter_.resolve_name_symbol_id(temp));
  json node = name(temp);
  locate(node, converter_.current_nditer_call_);
  return node;
}

bool numpy_nditer::substitute_chunks(json &node)
{
  if (node.is_array())
  {
    bool changed = false;
    for (json &child : node)
      changed = substitute_chunks(child) || changed;
    return changed;
  }
  if (!node.is_object())
    return false;
  if (node.value("_type", "") == "Name")
  {
    const numpy_nditer_item *item = find_item(node);
    if (!item || !converter_.numpy_nditers_.at(item->iterator).external_loop)
      return false;
    node = materialize(*item);
    return true;
  }
  bool changed = false;
  for (auto &child : node.items())
    if (child.value().is_structured())
      changed = substitute_chunks(child.value()) || changed;
  return changed;
}

void numpy_nditer::advance(const numpy_nditer_state &it)
{
  const json k = position(it);
  assign(
    it.state + "$k",
    if_exp(name(it.state + "$started"), binop(k, "Add", constant(1)), k));
  assign(it.state + "$started", boolean(true));
}

void numpy_nditer::bind_name(
  const json &target,
  const numpy_nditer_state &it,
  const std::string &iterator,
  std::size_t op)
{
  if (target.value("_type", "") != "Name")
    throw std::runtime_error(
      "TypeError: numpy.nditer() items can only be bound to names");
  const std::string target_name = target["id"].get<std::string>();
  numpy_nditer_item item;
  item.iterator = iterator;
  item.operand = op;
  item.position = it.state + "$" + target_name + "$p";
  assign(item.position, position(it));
  converter_
    .numpy_nditer_items_[key(converter_.current_func_name_, target_name)] =
    item;
}

namespace
{
// The iterator a binding or call reads from, and how: a method call,
// `next(it)`, `it.value` or `it[i]`; a null receiver for anything else.
std::pair<const json *, std::string> source(const json &value)
{
  const std::string type = value.value("_type", "");
  const json &func = value.value("func", json::object());
  if (type == "Call" && func.value("_type", "") == "Attribute")
    return {&value["func"]["value"], func["attr"].get<std::string>()};
  if (
    type == "Call" && func.value("_type", "") == "Name" &&
    func["id"] == "next" && value["args"].size() == 1)
    return {&value["args"][0], "next"};
  if (type == "Attribute" && value["attr"] == "value")
    return {&value["value"], "value"};
  if (type == "Subscript")
    return {&value["value"], "[]"};
  return {nullptr, ""};
}

bool binds_item(const std::string &form)
{
  return form == "_esbmc_nditer_item" || form == "next" || form == "value" ||
         form == "[]";
}
} // namespace

// The current item of every operand, a tuple for several.
json numpy_nditer::current_values(const numpy_nditer_state &it)
{
  std::vector<json> values;
  for (std::size_t op = 0; op < it.operands.size(); ++op)
    values.push_back(value(it, op, position(it)));
  return values.size() == 1 ? values.front() : tuple(values);
}

bool numpy_nditer::bind(const json &target, const json &value)
{
  const auto [receiver, form] = source(value);
  const numpy_nditer_state *found = receiver ? find(*receiver) : nullptr;
  if (!found || (form != "__enter__" && !binds_item(form)))
    return false;
  const numpy_nditer_state it = *found;
  const std::string iterator =
    key(converter_.current_func_name_, (*receiver)["id"].get<std::string>());
  converter_.current_nditer_call_ = value;
  if (form == "__enter__")
  {
    if (target.value("_type", "") != "Name")
      throw std::runtime_error(
        "TypeError: numpy.nditer() context can only be bound to a name");
    converter_.numpy_nditers_[key(
      converter_.current_func_name_, target["id"].get<std::string>())] = it;
    return true;
  }
  if (form == "next")
  {
    advance(it);
    raise_if(finished(it), "StopIteration", "");
  }
  else if (form != "_esbmc_nditer_item")
    raise_if(finished(it), "ValueError", "Iterator is past the end");
  if (form == "[]")
    bind_operand(target, it, iterator, value["slice"]);
  else
    bind_items(target, it, iterator);
  return true;
}

// `x = it[i]`: operand i, or an IndexError past the last operand.
void numpy_nditer::bind_operand(
  const json &target,
  const numpy_nditer_state &it,
  const std::string &iterator,
  const json &index)
{
  if (
    index.value("_type", "") != "Constant" ||
    !index["value"].is_number_integer() || index["value"].get<long long>() < 0)
    throw std::runtime_error(
      "TypeError: numpy.nditer() operands are indexed by a non-negative "
      "integer literal");
  if (index["value"].get<std::size_t>() >= it.operands.size())
    raise("IndexError", "Iterator operand index is out of bounds");
  else
    bind_name(target, it, iterator, index["value"].get<std::size_t>());
}

// `x = next(it)` and the for-loop item: one name per operand.
void numpy_nditer::bind_items(
  const json &target,
  const numpy_nditer_state &it,
  const std::string &iterator)
{
  if (target.value("_type", "") == "Tuple" && it.operands.size() > 1)
  {
    if (target["elts"].size() != it.operands.size())
      throw std::runtime_error(
        "ValueError: numpy.nditer() yields one item per operand");
    for (std::size_t op = 0; op < it.operands.size(); ++op)
      bind_name(target["elts"][op], it, iterator, op);
    return;
  }
  if (it.operands.size() != 1)
    throw std::runtime_error(
      "TypeError: binding the items of a multi-operand numpy.nditer() to one "
      "name is not supported");
  bind_name(target, it, iterator, 0);
}

// What the iterator reports about itself, independent of its position; null
// for an attribute of the current position.
json numpy_nditer::layout_attribute(
  const numpy_nditer_state &it,
  const std::string &attr) const
{
  if (attr == "finished")
    return finished(it);
  if (attr == "itersize")
    return it.runtime ? items(it) : constant(it.size);
  if (attr == "operands")
  {
    std::vector<json> arrays;
    for (const numpy_nditer_operand &o : it.operands)
      arrays.push_back(o.array);
    return tuple(arrays);
  }
  if (attr != "shape" && attr != "ndim")
    return json();
  std::vector<json> extents;
  if (it.runtime)
    extents.push_back(items(it));
  else
    for (long long extent : reported_shape(it))
      extents.push_back(constant(extent));
  return attr == "shape" ? tuple(extents)
                         : constant(static_cast<long long>(extents.size()));
}

// `multi_index`, `iterindex` and `value`: they raise past the end, and
// `multi_index` without the flag.
json numpy_nditer::position_attribute(
  const numpy_nditer_state &it,
  const std::string &attr)
{
  if (attr != "multi_index" && attr != "iterindex" && attr != "value")
    throw std::runtime_error(
      "TypeError: numpy.nditer attribute '" + attr + "' is not supported");
  if (attr == "multi_index" && !it.multi_index)
  {
    raise("ValueError", "Iterator is not tracking a multi-index");
    return tuple(std::vector<json>(it.shape.size(), constant(0)));
  }
  raise_if(finished(it), "ValueError", "Iterator is past the end");
  if (attr == "multi_index")
    return tuple(indices(it, position(it)));
  if (attr == "iterindex")
    return binop(position(it), "Mult", constant(it.chunk));
  return current_values(it);
}

std::optional<exprt> numpy_nditer::attribute(const json &node)
{
  const numpy_nditer_state *found = find(node["value"]);
  if (!found)
    return std::nullopt;
  converter_.current_nditer_call_ = node;
  const std::string attr = node["attr"].get<std::string>();
  // `shape` describes the remaining iteration, so it raises past the end too.
  if (attr == "shape")
    raise_if(finished(*found), "ValueError", "Iterator is past the end");
  json result = layout_attribute(*found, attr);
  if (result.is_null())
    result = position_attribute(*found, attr);
  locate(result, node);
  return converter_.get_expr(result);
}

// `for x, y in it` with a different number of names than operands.
void numpy_nditer::check_unpack(
  const numpy_nditer_state &it,
  long long expected)
{
  const long long got = static_cast<long long>(it.operands.size());
  if (expected > got)
    raise(
      "ValueError",
      "not enough values to unpack (expected " + std::to_string(expected) +
        ", got " + std::to_string(got) + ")");
  else if (expected < got)
    raise(
      "ValueError",
      "too many values to unpack (expected " + std::to_string(expected) + ")");
}

std::optional<exprt> numpy_nditer::call(const json &node)
{
  json substituted = node;
  converter_.current_nditer_call_ = node;
  if (substitute_chunks(substituted))
    return converter_.get_expr(substituted);
  const auto [receiver, method] = source(node);
  const numpy_nditer_state *found =
    receiver && node.value("_type", "") == "Call" ? find(*receiver) : nullptr;
  if (!found)
    return std::nullopt;
  json result = call_method(*found, method, node);
  locate(result, node);
  return converter_.get_expr(result);
}

json numpy_nditer::call_method(
  const numpy_nditer_state &it,
  const std::string &method,
  const json &node)
{
  const json none = {{"_type", "Constant"}, {"value", nullptr}};
  const json k = position(it);
  if (method == "_esbmc_nditer_unpack")
    check_unpack(it, node["args"][0]["value"].get<long long>());
  else if (method == "_esbmc_nditer_advance" || method == "next")
    advance(it);
  else if (method == "iternext")
  {
    assign(
      it.state + "$k",
      if_exp(compare(k, "Lt", items(it)), binop(k, "Add", constant(1)), k));
    assign(it.state + "$started", boolean(true));
    return compare(k, "Lt", items(it));
  }
  else if (method == "reset")
  {
    assign(it.state + "$k", constant(0));
    assign(it.state + "$started", boolean(false));
  }
  else if (method != "__enter__" && method != "__exit__" && method != "close")
    throw std::runtime_error(
      "TypeError: numpy.nditer method '" + method + "' is not supported");
  if (method != "next")
    return none;
  raise_if(finished(it), "StopIteration", "");
  return current_values(it);
}

std::optional<exprt> numpy_nditer::read_item(const json &node)
{
  const numpy_nditer_item *item = find_item(node);
  if (!item)
    return std::nullopt;
  const numpy_nditer_state &it = converter_.numpy_nditers_.at(item->iterator);
  converter_.current_nditer_call_ = node;
  if (it.external_loop)
    return chunk_value(it, *item);
  json read = value(it, item->operand, name(item->position));
  locate(read, node);
  return converter_.get_expr(read);
}

std::optional<exprt> numpy_nditer::subscript(const json &node)
{
  const json &base = node["value"];
  const json &index = node["slice"];
  converter_.current_nditer_call_ = node;
  if (const numpy_nditer_state *it = find(base))
  {
    if (
      index.value("_type", "") != "Constant" ||
      !index["value"].is_number_integer() ||
      index["value"].get<long long>() < 0 ||
      index["value"].get<std::size_t>() >= it->operands.size())
      throw std::runtime_error(
        "TypeError: numpy.nditer() operands are indexed by an integer "
        "literal in range");
    raise_if(finished(*it), "ValueError", "Iterator is past the end");
    json value =
      this->value(*it, index["value"].get<std::size_t>(), position(*it));
    locate(value, node);
    return converter_.get_expr(value);
  }

  // `chunk[j]` on what an external loop yields.
  const numpy_nditer_item *item = find_item(base);
  if (!item)
    return std::nullopt;
  const numpy_nditer_state &it = converter_.numpy_nditers_.at(item->iterator);
  if (!it.external_loop && whole_item(index))
    return read_item(base);
  if (!it.external_loop || index.value("_type", "") == "Slice")
    return std::nullopt;
  json offset = index;
  raise_if(
    {{"_type", "BoolOp"},
     {"op", {{"_type", "Or"}}},
     {"values",
      json::array(
        {compare(index, "Lt", constant(-it.chunk)),
         compare(index, "GtE", constant(it.chunk))})}},
    "IndexError",
    "index is out of bounds for the numpy.nditer() chunk");
  offset = if_exp(
    compare(index, "Lt", constant(0)),
    binop(index, "Add", constant(it.chunk)),
    index);
  json value = element(
    it,
    item->operand,
    binop(
      binop(name(item->position), "Mult", constant(it.chunk)), "Add", offset));
  locate(value, node);
  return converter_.get_expr(value);
}

// The operand and position a subscript target writes: `x[...]` on a bound
// item, or `it[i][...]` on the iterator's current position.
std::optional<std::tuple<const numpy_nditer_state *, std::size_t, json>>
numpy_nditer::written(const json &target)
{
  const json &base = target["value"];
  if (const numpy_nditer_item *item = find_item(base))
  {
    const numpy_nditer_state &it = converter_.numpy_nditers_.at(item->iterator);
    if (it.external_loop)
      throw std::runtime_error(
        "TypeError: writing a numpy.nditer() external-loop chunk is not "
        "supported");
    return std::make_tuple(&it, item->operand, name(item->position));
  }
  if (base.value("_type", "") != "Subscript")
    return std::nullopt;
  const numpy_nditer_state *it = find(base["value"]);
  const json &index = base["slice"];
  if (
    !it || index.value("_type", "") != "Constant" ||
    !index["value"].is_number_integer() ||
    index["value"].get<long long>() < 0 ||
    index["value"].get<std::size_t>() >= it->operands.size())
    return std::nullopt;
  raise_if(finished(*it), "ValueError", "Iterator is past the end");
  return std::make_tuple(it, index["value"].get<std::size_t>(), position(*it));
}

bool numpy_nditer::write(const json &target, const json &value)
{
  if (target.value("_type", "") != "Subscript")
    return false;
  converter_.current_nditer_call_ = target;
  const auto destination = written(target);
  if (!destination)
    return false;
  const auto &[it, op, at] = *destination;
  if (!whole_item(target["slice"]))
    raise("IndexError", "too many indices for array: array is 0-dimensional");
  else if (!it->operands[op].writable)
    raise("ValueError", "assignment destination is read-only");
  else
  {
    json store = element(*it, op, at);
    store["ctx"] = {{"_type", "Store"}};
    json statement = {
      {"_type", "Assign"}, {"targets", json::array({store})}, {"value", value}};
    locate(statement, target);
    converter_.emit_statement_into_current_block(statement);
  }
  return true;
}

exprt numpy_nditer::as_list(const json &call)
{
  std::optional<numpy_nditer_state> it = describe(
    call, "__esbmc_nditer$" + std::to_string(++converter_.nditer_counter_));
  std::vector<exprt> elements;
  if (it && it->external_loop)
    throw std::runtime_error(
      "TypeError: numpy.nditer() with external_loop is only supported as a "
      "for loop or a named iterator");
  if (it && it->runtime)
    throw std::runtime_error(
      "TypeError: numpy.nditer() over a view whose layout is only known at "
      "run time is only supported as a for loop or a named iterator");
  if (it)
    for (long long position = 0; position < it->size; ++position)
    {
      std::vector<json> values;
      for (std::size_t op = 0; op < it->operands.size(); ++op)
        values.push_back(value(*it, op, constant(position)));
      json node = values.size() == 1 ? values.front() : tuple(values);
      locate(node, call);
      elements.push_back(converter_.get_expr(node));
    }
  json list_node = {{"_type", "List"}, {"elts", json::array()}};
  locate(list_node, call);
  python_list list(converter_, list_node);
  return list.build_list_from_exprs(elements);
}
