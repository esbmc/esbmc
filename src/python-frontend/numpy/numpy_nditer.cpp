#include <python-frontend/numpy/numpy_nditer.h>
#include <python-frontend/numpy/numpy_call_expr.h>
#include <python-frontend/python-list/python_list.h>
#include <python-frontend/python_converter.h>
#include <util/irep/std_expr.h>

#include <algorithm>
#include <set>
#include <stdexcept>

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
  if (!converter_.emit_statement_into_current_block(statement))
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

// The layout of a named array or view, in elements. A 1-D view keeps a
// run-time extent and stride as expressions; any other view whose layout is
// only known at run time, or that is bound to different arrays on different
// paths, has no order to iterate it in.
numpy_nditer_operand numpy_nditer::operand(const json &node, std::size_t index)
{
  json array = node;
  if (array.value("_type", "") != "Name")
  {
    const std::string temp =
      converter_.current_nditer_state_ + "$op" + std::to_string(index);
    assign(temp, node);
    array = name(temp);
    locate(array, converter_.current_nditer_call_);
  }

  const std::string id =
    converter_.resolve_name_symbol_id(array["id"].get<std::string>());
  if (id.empty() || converter_.numpy_array_symbols_.count(id) == 0)
    throw std::runtime_error(
      "TypeError: numpy.nditer() operand must be a numpy array");

  numpy_nditer_operand result;
  result.array = array;
  auto view = converter_.numpy_pointer_view_info_.find(id);
  if (view == converter_.numpy_pointer_view_info_.end())
  {
    std::optional<std::vector<std::size_t>> shape =
      converter_.get_numpy_nditer_logical_shape(id);
    if (!shape)
      throw std::runtime_error(
        "TypeError: numpy.nditer() operand has no static shape");
    result.shape.assign(shape->begin(), shape->end());
    result.strides = c_strides(result.shape);
    return result;
  }

  const auto &info = view->second;
  if (!info.other_sources.empty())
    throw std::runtime_error(
      "TypeError: numpy.nditer() over a view bound to different arrays on "
      "different paths is not supported");
  if (info.is_symbolic())
  {
    if (info.shape.size() > 1)
      throw std::runtime_error(
        "TypeError: numpy.nditer() over an N-D view whose layout is only "
        "known at run time is not supported");
    auto runtime_value =
      [&](const std::vector<std::string> &ids, long long fallback) {
        if (ids.empty() || ids.front().empty())
          return constant(fallback);
        const symbolt *symbol =
          converter_.symbol_table_.find_symbol(ids.front());
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
    return result;
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
  return result;
}

// NumPy's iteration order. C and F are forced; A is F when every operand is
// Fortran- but not C-contiguous; K sorts the axes by stride magnitude with
// npyiter_find_best_axis_ordering's insertion sort, then walks backwards
// along an axis every operand has a non-positive stride on
// (npyiter_flip_negative_strides).
void numpy_nditer::order_axes(numpy_nditer_state &it, const std::string &order)
  const
{
  const std::size_t ndim = it.shape.size();
  it.flipped.assign(ndim, false);
  it.axes.clear();
  for (std::size_t axis = 0; axis < ndim; ++axis)
    it.axes.push_back(axis);

  auto stride = [&](std::size_t op, std::size_t axis) {
    return visited_stride(it, op, axis, false);
  };
  auto contiguous = [&](bool fortran) {
    for (std::size_t op = 0; op < it.operands.size(); ++op)
    {
      long long expected = 1;
      for (std::size_t i = 0; i < ndim; ++i)
      {
        const std::size_t axis = fortran ? i : ndim - 1 - i;
        if (it.shape[axis] != 1 && stride(op, axis) != expected)
          return false;
        expected *= it.shape[axis];
      }
    }
    return true;
  };

  if (order == "F" || (order == "A" && contiguous(true) && !contiguous(false)))
  {
    std::reverse(it.axes.begin(), it.axes.end());
    return;
  }
  if (order != "K")
    return;
  it.flip_negative = true;

  // perm[0] is the innermost axis, as in NumPy's axis data.
  std::vector<std::size_t> perm(it.axes.rbegin(), it.axes.rend());
  for (std::size_t i0 = 1; i0 < ndim; ++i0)
  {
    std::size_t ipos = i0;
    const std::size_t j0 = perm[i0];
    for (std::size_t i1 = i0; i1-- > 0;)
    {
      bool ambiguous = true;
      bool swap = false;
      for (std::size_t op = 0; op < it.operands.size(); ++op)
      {
        const long long s0 = stride(op, j0);
        const long long s1 = stride(op, perm[i1]);
        if (s0 == 0 || s1 == 0)
          continue;
        if (std::llabs(s1) <= std::llabs(s0))
          swap = false;
        else if (ambiguous)
          swap = true;
        ambiguous = false;
      }
      if (ambiguous)
        continue;
      if (!swap)
        break;
      ipos = i1;
    }
    if (ipos != i0)
    {
      perm.erase(perm.begin() + i0);
      perm.insert(perm.begin() + ipos, j0);
    }
  }
  it.axes.assign(perm.rbegin(), perm.rend());

  for (std::size_t axis = 0; axis < ndim; ++axis)
  {
    bool negative = false;
    bool positive = false;
    for (std::size_t op = 0; op < it.operands.size(); ++op)
    {
      negative = negative || stride(op, axis) < 0;
      positive = positive || stride(op, axis) > 0;
    }
    it.flipped[axis] = negative && !positive;
  }
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

std::optional<numpy_nditer_state>
numpy_nditer::describe(const json &call, const std::string &state)
{
  converter_.current_nditer_call_ = call;
  converter_.current_nditer_state_ = state;

  if (call["args"].size() != 1)
    throw std::runtime_error(
      "TypeError: numpy.nditer() takes its operands as the only positional "
      "argument");
  for (const json &kw : call["keywords"])
  {
    const std::string arg =
      kw["arg"].is_string() ? kw["arg"].get<std::string>() : "";
    if (arg == "op_flags")
    {
      if (converter_.is_numpy_readonly_view_arg(call["args"][0]))
        throw std::runtime_error(
          "ValueError: assignment destination is read-only");
      throw std::runtime_error(
        "TypeError: numpy.nditer() op_flags are not supported");
    }
    if (arg != "flags" && arg != "order")
      throw std::runtime_error(
        "TypeError: numpy.nditer() keyword '" + arg + "' is not supported");
  }

  const json &ops = call["args"][0];
  if (ops.value("_type", "") == "List" || ops.value("_type", "") == "Tuple")
    throw std::runtime_error(
      "TypeError: numpy.nditer() currently supports a single operand");

  const std::vector<std::string> flag_list = flags(call);
  for (const std::string &flag : flag_list)
  {
    if (!numpy_flags().count(flag))
    {
      raise("ValueError", "Unexpected iterator global flag \"" + flag + "\"");
      return std::nullopt;
    }
    if (!modelled_flags().count(flag))
      throw std::runtime_error(
        "TypeError: numpy.nditer() flag '" + flag + "' is not supported");
  }
  const std::string order_name = order(call);
  if (
    order_name != "C" && order_name != "F" && order_name != "A" &&
    order_name != "K")
  {
    raise(
      "ValueError",
      "order must be one of 'C', 'F', 'A', or 'K' (got '" + order_name + "')");
    return std::nullopt;
  }

  numpy_nditer_state it;
  it.state = state;
  it.alias = call["func"]["value"]["id"].get<std::string>();
  it.multi_index = has(flag_list, "multi_index");
  it.external_loop = has(flag_list, "external_loop");
  it.buffered = has(flag_list, "buffered");
  if (it.external_loop && it.multi_index)
  {
    raise(
      "ValueError",
      "Iterator flag EXTERNAL_LOOP cannot be used if an index or multi-index "
      "is being tracked");
    return std::nullopt;
  }

  it.operands.push_back(operand(ops, 0));
  it.shape = it.operands.front().shape;
  it.runtime = !it.operands.front().extent.is_null();
  if (it.runtime && it.external_loop)
    throw std::runtime_error(
      "TypeError: numpy.nditer() external_loop over a view whose layout is "
      "only known at run time is not supported");
  order_axes(it, order_name);
  if (it.runtime)
  {
    if (!has(flag_list, "zerosize_ok"))
      raise_if(
        compare(it.operands.front().extent, "Eq", constant(0)),
        "ValueError",
        "Iteration of zero-sized operands is not enabled");
    return it;
  }
  it.size = product(it.shape);
  if (it.size == 0 && !has(flag_list, "zerosize_ok"))
  {
    raise("ValueError", "Iteration of zero-sized operands is not enabled");
    return std::nullopt;
  }
  set_chunk(it);
  return it;
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
  if (index.empty())
    return o.array;
  return index_node(o.array, index.size() == 1 ? index.front() : tuple(index));
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
    json node = element(it, item.operand, binop(start, "Add", constant(j)));
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

bool numpy_nditer::bind(const json &target, const json &value)
{
  const json *receiver = nullptr;
  std::string form;
  const std::string type = value.value("_type", "");
  if (type == "Call" && value["func"].value("_type", "") == "Attribute")
  {
    receiver = &value["func"]["value"];
    form = value["func"]["attr"].get<std::string>();
  }
  else if (
    type == "Call" && value["func"].value("_type", "") == "Name" &&
    value["func"]["id"] == "next" && value["args"].size() == 1)
  {
    receiver = &value["args"][0];
    form = "next";
  }
  else if (type == "Attribute" && value["attr"] == "value")
  {
    receiver = &value["value"];
    form = "value";
  }
  else if (type == "Subscript")
  {
    receiver = &value["value"];
    form = "[]";
  }
  const numpy_nditer_state *found = receiver ? find(*receiver) : nullptr;
  if (!found)
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
  if (
    form != "_esbmc_nditer_item" && form != "next" && form != "value" &&
    form != "[]")
    return false;

  if (form == "next")
  {
    advance(it);
    raise_if(finished(it), "StopIteration", "");
  }
  else if (form != "_esbmc_nditer_item")
    raise_if(finished(it), "ValueError", "Iterator is past the end");

  if (form == "[]")
  {
    const json &index = value["slice"];
    if (
      index.value("_type", "") != "Constant" ||
      !index["value"].is_number_integer() ||
      index["value"].get<long long>() < 0 ||
      index["value"].get<std::size_t>() >= it.operands.size())
      throw std::runtime_error(
        "TypeError: numpy.nditer() operands are indexed by an integer "
        "literal in range");
    bind_name(target, it, iterator, index["value"].get<std::size_t>());
    return true;
  }
  if (target.value("_type", "") == "Tuple" && it.operands.size() > 1)
  {
    if (target["elts"].size() != it.operands.size())
      throw std::runtime_error(
        "ValueError: numpy.nditer() yields one item per operand");
    for (std::size_t op = 0; op < it.operands.size(); ++op)
      bind_name(target["elts"][op], it, iterator, op);
    return true;
  }
  if (it.operands.size() != 1)
    throw std::runtime_error(
      "TypeError: binding the items of a multi-operand numpy.nditer() to one "
      "name is not supported");
  bind_name(target, it, iterator, 0);
  return true;
}

std::optional<exprt> numpy_nditer::attribute(const json &node)
{
  const numpy_nditer_state *found = find(node["value"]);
  if (!found)
    return std::nullopt;
  const numpy_nditer_state &it = *found;
  converter_.current_nditer_call_ = node;
  const std::string attr = node["attr"].get<std::string>();
  json result;
  if (attr == "finished")
    result = finished(it);
  else if (attr == "itersize")
    result = it.runtime ? items(it) : constant(it.size);
  else if (attr == "shape" || attr == "ndim")
  {
    std::vector<json> extents;
    if (it.runtime)
      extents.push_back(items(it));
    else
      for (long long extent : reported_shape(it))
        extents.push_back(constant(extent));
    result = attr == "shape" ? tuple(extents)
                             : constant(static_cast<long long>(extents.size()));
  }
  else if (attr == "operands")
  {
    std::vector<json> arrays;
    for (const numpy_nditer_operand &o : it.operands)
      arrays.push_back(o.array);
    result = tuple(arrays);
  }
  else if (attr == "multi_index")
  {
    if (!it.multi_index)
      raise("ValueError", "Iterator is not tracking a multi-index");
    else
      raise_if(finished(it), "ValueError", "Iterator is past the end");
    result = tuple(
      it.multi_index ? indices(it, position(it))
                     : std::vector<json>(it.shape.size(), constant(0)));
  }
  else if (attr == "iterindex" || attr == "value")
  {
    raise_if(finished(it), "ValueError", "Iterator is past the end");
    if (attr == "iterindex")
      result = binop(position(it), "Mult", constant(it.chunk));
    else if (it.operands.size() == 1)
      result = element(it, 0, position(it));
    else
    {
      std::vector<json> values;
      for (std::size_t op = 0; op < it.operands.size(); ++op)
        values.push_back(element(it, op, position(it)));
      result = tuple(values);
    }
  }
  else
    throw std::runtime_error(
      "TypeError: numpy.nditer attribute '" + attr + "' is not supported");
  locate(result, node);
  return converter_.get_expr(result);
}

std::optional<exprt> numpy_nditer::call(const json &node)
{
  json substituted = node;
  converter_.current_nditer_call_ = node;
  if (substitute_chunks(substituted))
    return converter_.get_expr(substituted);

  const json &func = node["func"];
  const bool next_call = func.value("_type", "") == "Name" &&
                         func["id"] == "next" && node["args"].size() == 1;
  const json *receiver =
    next_call
      ? &node["args"][0]
      : (func.value("_type", "") == "Attribute" ? &func["value"] : nullptr);
  const numpy_nditer_state *found = receiver ? find(*receiver) : nullptr;
  if (!found)
    return std::nullopt;
  const numpy_nditer_state it = *found;
  const std::string method =
    next_call ? "next" : func["attr"].get<std::string>();
  json none = {{"_type", "Constant"}, {"value", nullptr}};
  locate(none, node);

  if (method == "_esbmc_nditer_advance")
  {
    advance(it);
    return converter_.get_expr(none);
  }
  if (method == "next")
  {
    advance(it);
    raise_if(finished(it), "StopIteration", "");
    std::vector<json> values;
    for (std::size_t op = 0; op < it.operands.size(); ++op)
      values.push_back(element(it, op, position(it)));
    json value = values.size() == 1 ? values.front() : tuple(values);
    locate(value, node);
    return converter_.get_expr(value);
  }
  if (method == "iternext")
  {
    const json k = position(it);
    assign(
      it.state + "$k",
      if_exp(compare(k, "Lt", items(it)), binop(k, "Add", constant(1)), k));
    assign(it.state + "$started", boolean(true));
    json more = compare(k, "Lt", items(it));
    locate(more, node);
    return converter_.get_expr(more);
  }
  if (method == "reset")
  {
    assign(it.state + "$k", constant(0));
    assign(it.state + "$started", boolean(false));
    return converter_.get_expr(none);
  }
  if (method == "__enter__" || method == "__exit__" || method == "close")
    return converter_.get_expr(none);
  throw std::runtime_error(
    "TypeError: numpy.nditer method '" + method + "' is not supported");
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
  json value = element(it, item->operand, name(item->position));
  locate(value, node);
  return converter_.get_expr(value);
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
    json value = element(*it, index["value"].get<std::size_t>(), position(*it));
    locate(value, node);
    return converter_.get_expr(value);
  }

  // `chunk[j]` on what an external loop yields.
  const numpy_nditer_item *item = find_item(base);
  if (!item)
    return std::nullopt;
  const numpy_nditer_state &it = converter_.numpy_nditers_.at(item->iterator);
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
      json node = element(*it, 0, constant(position));
      locate(node, call);
      elements.push_back(converter_.get_expr(node));
    }
  json list_node = {{"_type", "List"}, {"elts", json::array()}};
  locate(list_node, call);
  python_list list(converter_, list_node);
  return list.build_list_from_exprs(elements);
}
