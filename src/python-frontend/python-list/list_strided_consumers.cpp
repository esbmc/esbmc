#include "python_list_internal.h"
#include <algorithm>
#include <functional>
#include <set>
using namespace python_expr;

// Consumers of numpy views whose extents or strides are only known at run
// time (ADR-NP-003 etapa 3): copies, reducers and lists are built with loops
// over the view's own descriptor instead of a compile-time element list.

namespace
{
typet ll_type()
{
  return signedbv_typet(64);
}
} // namespace

exprt python_list::axis_extent(const strided_axis &axis)
{
  return axis.extent_expr.is_not_nil() ? axis.extent_expr
                                       : from_integer(axis.extent, ll_type());
}

exprt python_list::axis_stride(const strided_axis &axis)
{
  return axis.stride_expr.is_not_nil() ? axis.stride_expr
                                       : from_integer(axis.stride, ll_type());
}

std::optional<python_list::strided_view_desc>
python_list::symbolic_view_operand(const nlohmann::json &arg)
{
  if (!arg.is_object() || arg.value("_type", "") != "Name")
    return std::nullopt;

  const auto view = converter_.numpy_pointer_view_info_.find(
    converter_.resolve_name_symbol_id(arg.value("id", "")));
  if (
    view == converter_.numpy_pointer_view_info_.end() ||
    !view->second.is_symbolic())
    return std::nullopt;

  exprt *lhs = converter_.current_lhs;
  converter_.current_lhs = nullptr;
  const exprt array = converter_.get_expr(arg);
  converter_.current_lhs = lhs;
  return describe_strided_view(array);
}

exprt python_list::view_element(
  const strided_view_desc &view,
  const std::vector<exprt> &index) const
{
  exprt position = from_integer(0, ll_type());
  for (std::size_t axis = 0; axis < index.size(); ++axis)
    position = build_add(
      position,
      build_mul(index[axis], axis_stride(view.axes[axis]), ll_type()),
      ll_type());
  return build_dereference(
    build_add(
      view.base,
      build_typecast(position, size_type()),
      pointer_typet(view.elem_type)),
    view.elem_type);
}

void python_list::emit_counted_loop(
  const nlohmann::json &node,
  const exprt &extent,
  const std::function<void(const exprt &)> &body)
{
  const locationt loc = converter_.get_location_from_decl(node);
  symbolt &idx_sym =
    converter_.create_tmp_symbol(node, "$view_loop_i$", ll_type(), exprt());
  const exprt idx = build_symbol(idx_sym);
  code_declt idx_decl(idx);
  idx_decl.location() = loc;
  converter_.add_instruction(idx_decl);
  code_assignt idx_init(idx, from_integer(0, ll_type()));
  idx_init.location() = loc;
  converter_.add_instruction(idx_init);

  code_blockt loop_body;
  code_blockt *saved_block = converter_.current_block;
  converter_.current_block = &loop_body;
  body(idx);
  converter_.current_block = saved_block;
  loop_body.copy_to_operands(
    code_assignt(idx, build_add(idx, from_integer(1, ll_type()), ll_type())));

  codet loop;
  loop.set_statement("while");
  loop.copy_to_operands(build_less_than(idx, extent), loop_body);
  loop.location() = loc;
  converter_.add_instruction(loop);
}

void python_list::emit_view_loops(
  const nlohmann::json &node,
  const strided_view_desc &view,
  const std::function<void(const exprt &)> &leaf)
{
  std::vector<exprt> index;
  std::function<void(std::size_t)> level = [&](std::size_t axis) {
    emit_counted_loop(node, axis_extent(view.axes[axis]), [&](const exprt &i) {
      index.push_back(i);
      if (axis + 1 == view.axes.size())
        leaf(view_element(view, index));
      else
        level(axis + 1);
      index.pop_back();
    });
  };
  level(0);
}

exprt python_list::view_element_count(
  const nlohmann::json &node,
  const strided_view_desc &view)
{
  exprt count = from_integer(1, ll_type());
  for (const strided_axis &axis : view.axes)
    count = build_mul(count, axis_extent(axis), ll_type());
  return emit_ll_temp(node, "$view_count$", count);
}

exprt python_list::emit_view_accumulator(
  const nlohmann::json &arg,
  const typet &type,
  const exprt &init)
{
  const locationt loc = converter_.get_location_from_decl(arg);
  symbolt &sym =
    converter_.create_tmp_symbol(arg, "$view_reduce$", type, exprt());
  const exprt acc = build_symbol(sym);
  code_declt decl(acc);
  decl.location() = loc;
  converter_.add_instruction(decl);
  code_assignt assign(acc, init);
  assign.location() = loc;
  converter_.add_instruction(assign);
  return acc;
}

exprt python_list::reduce_view_truth(
  bool is_any,
  const nlohmann::json &arg,
  const strided_view_desc &view)
{
  const locationt loc = converter_.get_location_from_decl(arg);
  const exprt acc =
    emit_view_accumulator(arg, bool_type(), gen_boolean(!is_any));
  emit_view_loops(arg, view, [&](const exprt &element) {
    const exprt truthy = build_notequal(element, gen_zero(view.elem_type));
    code_ifthenelset update;
    update.cond() = is_any ? truthy : build_not(truthy);
    update.then_case() = code_assignt(acc, gen_boolean(is_any));
    update.location() = loc;
    converter_.add_instruction(update);
  });
  return acc;
}

exprt python_list::reduce_view_extreme(
  const std::string &function,
  const nlohmann::json &arg,
  const strided_view_desc &view,
  const exprt &count)
{
  const locationt loc = converter_.get_location_from_decl(arg);
  exprt raise = converter_.get_exception_handler().gen_exception_raise(
    "ValueError", "numpy." + function + "() arg is an empty sequence");
  codet throw_code("expression");
  throw_code.operands().push_back(raise);
  throw_code.location() = loc;
  code_ifthenelset empty_guard;
  empty_guard.cond() = build_equal(count, from_integer(0, ll_type()));
  empty_guard.then_case() = throw_code;
  empty_guard.location() = loc;
  empty_guard.location().property("skipped");
  converter_.add_instruction(empty_guard);

  // Seed with the first element, then fold the rest in.
  std::vector<exprt> origin(view.axes.size(), from_integer(0, ll_type()));
  const exprt acc =
    emit_view_accumulator(arg, view.elem_type, view_element(view, origin));
  emit_view_loops(arg, view, [&](const exprt &element) {
    code_ifthenelset update;
    update.cond() = function == "min" ? build_less_than(element, acc)
                                      : build_greater_than(element, acc);
    update.then_case() = code_assignt(acc, element);
    update.location() = loc;
    converter_.add_instruction(update);
  });
  return acc;
}

std::optional<exprt> python_list::try_reduce_symbolic_view(
  const std::string &function,
  const nlohmann::json &arg)
{
  static const std::set<std::string> reducers = {
    "sum", "mean", "min", "max", "any", "all"};
  if (reducers.count(function) == 0)
    return std::nullopt;
  std::optional<strided_view_desc> view = symbolic_view_operand(arg);
  if (!view)
    return std::nullopt;

  if (function == "any" || function == "all")
    return reduce_view_truth(function == "any", arg, *view);
  const exprt count = view_element_count(arg, *view);
  if (function == "min" || function == "max")
    return reduce_view_extreme(function, arg, *view, count);

  const typet sum_type =
    view->elem_type.is_floatbv() ? view->elem_type : ll_type();
  const exprt acc = emit_view_accumulator(arg, sum_type, gen_zero(sum_type));
  emit_view_loops(arg, *view, [&](const exprt &element) {
    converter_.add_instruction(code_assignt(
      acc, build_add(acc, build_typecast(element, sum_type), sum_type)));
  });
  if (function == "sum")
    return acc;

  expr2tc total2, count2;
  migrate_expr(build_typecast(acc, double_type()), total2);
  migrate_expr(build_typecast(count, double_type()), count2);
  return migrate_expr_back(div2tc(migrate_type(double_type()), total2, count2));
}

std::optional<exprt> python_list::try_build_symbolic_view_list(
  const nlohmann::json &arg,
  bool nested)
{
  std::optional<strided_view_desc> view = symbolic_view_operand(arg);
  if (!view)
    return std::nullopt;

  if (!nested || view->axes.size() == 1)
  {
    symbolt &list_symbol = create_list();
    emit_view_loops(arg, *view, [&](const exprt &element) {
      converter_.add_instruction(
        build_push_list_call(list_symbol, list_value_, element));
    });
    elem_types().record(list_symbol.id.as_string(), "", view->elem_type);
    return build_symbol(list_symbol);
  }

  std::vector<exprt> index;
  std::function<exprt(std::size_t)> level = [&](std::size_t axis) {
    symbolt &list_symbol = create_list();
    exprt child = build_symbol(list_symbol);
    emit_counted_loop(arg, axis_extent(view->axes[axis]), [&](const exprt &i) {
      index.push_back(i);
      const bool leaf = axis + 1 == view->axes.size();
      const exprt value = leaf ? view_element(*view, index) : level(axis + 1);
      converter_.add_instruction(
        build_push_list_call(list_symbol, list_value_, value));
      if (leaf)
        elem_types().record(list_symbol.id.as_string(), "", view->elem_type);
      else
        elem_types().record(
          list_symbol.id.as_string(),
          value.is_symbol() ? value.identifier().as_string() : "",
          value.type());
      index.pop_back();
    });
    return child;
  };
  return level(0);
}

// The value to return without building a result: a type-probe placeholder.
// An N-D result with run-time extents needs a name to carry its shape.
std::optional<exprt> python_list::unnamed_nd_view_placeholder(
  const strided_view_desc &view,
  bool as_view,
  bool flat_result,
  const char *unnamed_error) const
{
  if (!as_view && !flat_result)
  {
    if (converter_.in_rhs_type_probe_ || converter_.in_scratch_probe_)
      return view.base;
    throw std::runtime_error(unnamed_error);
  }
  if (as_view && converter_.in_rhs_type_probe_)
    return view.base;
  return std::nullopt;
}

// ravel() of a contiguous view is the same storage; of any other view a
// copy -- and which one applies cannot be told for a run-time stride.
std::optional<exprt> python_list::alias_contiguous_ravel(
  const nlohmann::json &arg,
  const strided_view_desc &view)
{
  const view_contiguity layout = classify_view_contiguity(view);
  if (layout == view_contiguity::unknown)
    throw std::runtime_error(
      "TypeError: numpy.ravel() of a view with a run-time stride is not "
      "supported (whether it copies depends on the stride)");
  if (layout != view_contiguity::contiguous)
    return std::nullopt;
  strided_axis flat;
  flat.extent_expr = view_element_count(arg, view);
  flat.stride = 1;
  return register_strided_view(view, from_integer(0, size_type()), {flat});
}

// Own storage: a flat variable-length buffer filled in row-major order.
exprt python_list::emit_view_copy_buffer(
  const nlohmann::json &arg,
  const strided_view_desc &view,
  const exprt &count)
{
  const locationt loc = converter_.get_location_from_decl(arg);
  array_typet buffer_type(view.elem_type, build_typecast(count, size_type()));
  symbolt &buffer = converter_.create_tmp_symbol(
    arg, "$numpy_view_copy$", buffer_type, exprt());
  code_declt buffer_decl(build_symbol(buffer));
  buffer_decl.location() = loc;
  converter_.add_instruction(buffer_decl);

  symbolt &next_sym =
    converter_.create_tmp_symbol(arg, "$view_copy_k$", ll_type(), exprt());
  const exprt next = build_symbol(next_sym);
  code_declt next_decl(next);
  next_decl.location() = loc;
  converter_.add_instruction(next_decl);
  converter_.add_instruction(code_assignt(next, from_integer(0, ll_type())));
  emit_view_loops(arg, view, [&](const exprt &element) {
    converter_.add_instruction(code_assignt(
      build_index(
        build_symbol(buffer),
        build_typecast(next, size_type()),
        view.elem_type),
      element));
    converter_.add_instruction(code_assignt(
      next, build_add(next, from_integer(1, ll_type()), ll_type())));
  });
  return build_symbol(buffer);
}

// Axes of a view over a fresh copy: same extents, or one flat axis,
// contiguous.
std::vector<python_list::strided_axis> python_list::owned_copy_axes(
  const nlohmann::json &arg,
  const strided_view_desc &view,
  bool flatten,
  const exprt &count)
{
  std::vector<strided_axis> axes;
  if (flatten)
  {
    strided_axis flat;
    flat.extent_expr = count;
    flat.stride = 1;
    axes.push_back(flat);
    return axes;
  }
  axes.resize(view.axes.size());
  exprt running = from_integer(1, ll_type());
  bool running_is_constant = true;
  long long running_constant = 1;
  for (std::size_t axis = view.axes.size(); axis-- > 0;)
  {
    axes[axis].extent = view.axes[axis].extent;
    axes[axis].extent_expr = view.axes[axis].extent_expr;
    if (running_is_constant)
      axes[axis].stride = running_constant;
    else
      axes[axis].stride_expr = emit_ll_temp(arg, "$view_copy_stride$", running);
    if (view.axes[axis].extent_expr.is_not_nil())
      running_is_constant = false;
    else
      running_constant *= view.axes[axis].extent;
    running = build_mul(running, axis_extent(view.axes[axis]), ll_type());
  }
  return axes;
}

std::optional<exprt> python_list::try_copy_symbolic_view(
  const nlohmann::json &arg,
  bool flatten,
  bool alias_if_contiguous)
{
  std::optional<strided_view_desc> view = symbolic_view_operand(arg);
  if (!view)
    return std::nullopt;

  exprt *lhs = converter_.current_lhs;
  const bool as_view = lhs && lhs->is_symbol();
  if (
    std::optional<exprt> placeholder = unnamed_nd_view_placeholder(
      *view,
      as_view,
      flatten || view->axes.size() == 1,
      "TypeError: copying an N-D numpy view with run-time extents requires "
      "assigning it to a name"))
    return placeholder;

  if (as_view && flatten && alias_if_contiguous)
    if (std::optional<exprt> aliased = alias_contiguous_ravel(arg, *view))
      return aliased;

  const exprt count = view_element_count(arg, *view);
  const exprt buffer = emit_view_copy_buffer(arg, *view, count);
  if (!as_view)
    return buffer;

  strided_view_desc owned;
  owned.elem_type = view->elem_type;
  owned.base =
    build_typecast(build_address_of(buffer), pointer_typet(view->elem_type));
  return register_strided_view(
    owned,
    from_integer(0, size_type()),
    owned_copy_axes(arg, *view, flatten, count));
}

python_list::view_contiguity
python_list::classify_view_contiguity(const strided_view_desc &view)
{
  long long expected = 1;
  bool expected_known = true;
  for (std::size_t axis = view.axes.size(); axis-- > 0;)
  {
    const strided_axis &described = view.axes[axis];
    if (described.stride_expr.is_not_nil())
      return view_contiguity::unknown;
    if (described.extent_expr.is_nil() && described.extent == 1)
      continue; // a size-1 axis never constrains the layout
    if (!expected_known)
      return view_contiguity::unknown;
    if (described.stride != expected)
      return view_contiguity::not_contiguous;
    if (described.extent_expr.is_not_nil())
      expected_known = false;
    else
      expected *= described.extent;
  }
  return view_contiguity::contiguous;
}

void python_list::emit_reshape_size_guard(
  const nlohmann::json &arg,
  const exprt &count,
  const std::vector<std::size_t> &new_shape)
{
  const locationt loc = converter_.get_location_from_decl(arg);
  long long requested = 1;
  for (std::size_t dim : new_shape)
    requested *= static_cast<long long>(dim);
  exprt raise = converter_.get_exception_handler().gen_exception_raise(
    "ValueError", "cannot reshape array into the requested shape");
  codet throw_code("expression");
  throw_code.operands().push_back(raise);
  throw_code.location() = loc;
  code_ifthenelset size_guard;
  size_guard.cond() = build_notequal(count, from_integer(requested, ll_type()));
  size_guard.then_case() = throw_code;
  size_guard.location() = loc;
  size_guard.location().property("skipped");
  converter_.add_instruction(size_guard);
}

// One source axis can always be split into a view: each new axis steps over
// the product of the later ones, in units of the source stride.
void python_list::scale_split_axis_strides(
  const nlohmann::json &arg,
  const strided_axis &source,
  std::vector<strided_axis> &axes)
{
  for (strided_axis &axis : axes)
  {
    if (source.stride_expr.is_nil())
    {
      axis.stride *= source.stride;
      continue;
    }
    axis.stride_expr = emit_ll_temp(
      arg,
      "$view_reshape_stride$",
      build_mul(
        source.stride_expr, from_integer(axis.stride, ll_type()), ll_type()));
  }
}

// try_copy_symbolic_view registered the target as a flat view over the new
// buffer; give it the requested shape.
void python_list::reshape_registered_copy(
  const std::string &lhs_id,
  const std::vector<std::size_t> &new_shape)
{
  auto info = converter_.numpy_pointer_view_info_.find(lhs_id);
  if (info == converter_.numpy_pointer_view_info_.end())
    return;
  info->second.shape.assign(new_shape.begin(), new_shape.end());
  info->second.shape_symbols.assign(new_shape.size(), std::string());
  info->second.stride_symbols.assign(new_shape.size(), std::string());
  const std::vector<long long> extents(new_shape.begin(), new_shape.end());
  info->second.strides = contiguous_strides(extents);
  info->second.length = new_shape.front();
  info->second.stride = info->second.strides.front();
  if (symbolt *symbol = converter_.find_symbol(lhs_id))
    converter_.numpy_pointer_view_info_[symbol->id.as_string()] = info->second;
}

std::optional<exprt> python_list::try_reshape_symbolic_view(
  const nlohmann::json &arg,
  const std::vector<std::size_t> &new_shape)
{
  std::optional<strided_view_desc> view = symbolic_view_operand(arg);
  if (!view)
    return std::nullopt;

  exprt *lhs = converter_.current_lhs;
  const bool as_view = lhs && lhs->is_symbol();
  if (
    std::optional<exprt> placeholder = unnamed_nd_view_placeholder(
      *view,
      as_view,
      new_shape.size() == 1,
      "TypeError: reshaping a numpy view with run-time extents to N-D "
      "requires assigning it to a name"))
    return placeholder;

  const view_contiguity layout = classify_view_contiguity(*view);
  const bool single_axis = view->axes.size() == 1;
  if (as_view && layout == view_contiguity::unknown && !single_axis)
    throw std::runtime_error(
      "TypeError: numpy.reshape() of an N-D view with a run-time stride is "
      "not supported (whether it copies depends on the strides)");

  // The element count is only known at run time.
  emit_reshape_size_guard(arg, view_element_count(arg, *view), new_shape);

  const std::vector<long long> extents(new_shape.begin(), new_shape.end());
  const std::vector<long long> strides = contiguous_strides(extents);
  std::vector<strided_axis> axes(new_shape.size());
  for (std::size_t axis = 0; axis < axes.size(); ++axis)
  {
    axes[axis].extent = extents[axis];
    axes[axis].stride = strides[axis];
  }

  if (as_view && (layout == view_contiguity::contiguous || single_axis))
  {
    if (layout != view_contiguity::contiguous)
      scale_split_axis_strides(arg, view->axes.front(), axes);
    return register_strided_view(*view, from_integer(0, size_type()), axes);
  }

  // A non-contiguous source is copied (as NumPy does) into a flat buffer.
  std::optional<exprt> copied = try_copy_symbolic_view(arg, true);
  if (copied && as_view)
    reshape_registered_copy(lhs->identifier().as_string(), new_shape);
  return copied;
}
