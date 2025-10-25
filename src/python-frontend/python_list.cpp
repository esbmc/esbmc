#include <python-frontend/python_list.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/json_utils.h>
#include <util/expr.h>
#include <util/type.h>
#include <util/symbol.h>
#include <util/expr_util.h>
#include <util/arith_tools.h>
#include <util/std_code.h>
#include <util/symbolic_types.h>
#include <string>

// Extract element type from annotation
static typet get_elem_type_from_annotation(
  const nlohmann::json &node,
  const type_handler &type_handler_)
{
  // Check if annotation exists and has the expected structure
  if (
    node.contains("annotation") && node["annotation"].is_object() &&
    node["annotation"].contains("slice") &&
    node["annotation"]["slice"].is_object() &&
    node["annotation"]["slice"].contains("id") &&
    node["annotation"]["slice"]["id"].is_string())
  {
    return type_handler_.get_typet(
      node["annotation"]["slice"]["id"].get<std::string>());
  }

  // Check for direct type annotation
  if (
    node.contains("annotation") && node["annotation"].is_object() &&
    node["annotation"].contains("id") && node["annotation"]["id"].is_string())
  {
    return type_handler_.get_typet(node["annotation"]["id"].get<std::string>());
  }

  // Return empty type if annotation structure is not as expected
  return typet();
}

std::unordered_map<std::string, std::vector<std::pair<std::string, typet>>>
  python_list::list_type_map{};

list_elem_info
python_list::get_list_element_info(const nlohmann::json &op, const exprt &elem)
{
  const type_handler type_handler_ = converter_.get_type_handler();
  locationt location = converter_.get_location_from_decl(op);
  const std::string elem_type_name = type_handler_.type_to_string(elem.type());

  // Create type name as null-terminated char array
  const typet type_name_type =
    type_handler_.build_array(char_type(), elem_type_name.size() + 1);
  std::vector<unsigned char> type_name_str(
    elem_type_name.begin(), elem_type_name.end());
  type_name_str.push_back('\0');
  exprt type_name_expr =
    converter_.make_char_array_expr(type_name_str, type_name_type);

  // Create and declare temporary symbol for element type
  symbolt &elem_type_sym = converter_.create_tmp_symbol(
    op, "$list_elem_type$", size_type(), type_name_expr);
  code_declt elem_type_decl(symbol_expr(elem_type_sym));
  elem_type_decl.location() = location;
  converter_.add_instruction(elem_type_decl);

  // Call hash function to get type hash
  const symbolt *hash_func_symbol =
    converter_.symbol_table().find_symbol("c:list.c@F@list_hash_string");
  if (!hash_func_symbol)
  {
    throw std::runtime_error("Hash function symbol not found");
  }

  code_function_callt list_type_hash_func_call;
  list_type_hash_func_call.function() = symbol_expr(*hash_func_symbol);
  list_type_hash_func_call.arguments().push_back(
    converter_.get_string_handler().get_array_base_address(type_name_expr));
  list_type_hash_func_call.lhs() = symbol_expr(elem_type_sym);
  list_type_hash_func_call.type() = size_type();
  list_type_hash_func_call.location() = location;
  converter_.add_instruction(list_type_hash_func_call);

  // Create and declare temporary symbol for list element
  symbolt &elem_symbol =
    converter_.create_tmp_symbol(op, "$list_elem$", elem.type(), elem);
  code_declt elem_decl(symbol_expr(elem_symbol));
  elem_decl.copy_to_operands(elem);
  elem_decl.location() = location;
  converter_.add_instruction(elem_decl);

  // Calculate element size in bytes
  exprt elem_size;

  // For string pointers (char*), calculate length at runtime using strlen
  if (
    elem_symbol.type.is_pointer() && elem_symbol.type.subtype() == char_type())
  {
    // Call strlen to get actual string length
    const symbolt *strlen_symbol =
      converter_.symbol_table().find_symbol("c:@F@strlen");
    if (!strlen_symbol)
    {
      throw std::runtime_error("strlen function not found in symbol table");
    }

    // Create temp variable to store strlen result
    symbolt &strlen_result = converter_.create_tmp_symbol(
      op, "$strlen_result$", size_type(), gen_zero(size_type()));
    code_declt strlen_decl(symbol_expr(strlen_result));
    strlen_decl.location() = location;
    converter_.add_instruction(strlen_decl);

    // Call strlen(elem_symbol)
    code_function_callt strlen_call;
    strlen_call.function() = symbol_expr(*strlen_symbol);
    strlen_call.lhs() = symbol_expr(strlen_result);
    strlen_call.arguments().push_back(symbol_expr(elem_symbol));
    strlen_call.type() = size_type();
    strlen_call.location() = location;
    converter_.add_instruction(strlen_call);

    // Add 1 for null terminator: size = strlen(s) + 1
    // Use strlen_result.type to ensure exact type match
    exprt one_const = from_integer(1, strlen_result.type);
    elem_size = exprt("+", strlen_result.type);
    elem_size.copy_to_operands(symbol_expr(strlen_result), one_const);
  }
  else
  {
    // Handle arrays and other types
    constexpr size_t BITS_PER_BYTE = 8;
    constexpr size_t DEFAULT_SIZE = 1;

    size_t elem_size_bytes = DEFAULT_SIZE;
    try
    {
      if (elem_symbol.type.is_array())
      {
        const size_t subtype_size_bits =
          std::stoull(elem.type().subtype().width().as_string(), nullptr, 10);

        const array_typet &array_type =
          static_cast<const array_typet &>(elem_symbol.type);

        const size_t array_length =
          std::stoull(array_type.size().value().as_string(), nullptr, 2);

        elem_size_bytes = (array_length * subtype_size_bits) / BITS_PER_BYTE;
      }
      else
      {
        const size_t type_width_bits =
          std::stoull(elem_symbol.type.width().as_string(), nullptr, 10);

        elem_size_bytes = type_width_bits / BITS_PER_BYTE;
      }
    }
    catch (std::invalid_argument &)
    {
      elem_size_bytes = DEFAULT_SIZE;
    }

    if (elem_size_bytes == 0)
    {
      throw std::runtime_error("Element size cannot be zero");
    }

    elem_size = from_integer(BigInt(elem_size_bytes), size_type());
  }

  // Build and return the push function call
  list_elem_info elem_info;
  elem_info.elem_type_sym = &elem_type_sym;
  elem_info.elem_symbol = &elem_symbol;
  elem_info.elem_size = elem_size;
  elem_info.location = location;
  return elem_info;
}

exprt python_list::build_push_list_call(
  const symbolt &list,
  const nlohmann::json &op,
  const exprt &elem)
{
  list_elem_info elem_info = get_list_element_info(op, elem);

  const symbolt *push_func_sym =
    converter_.symbol_table().find_symbol("c:list.c@F@list_push");

  if (!push_func_sym)
  {
    throw std::runtime_error("Push function symbol not found");
  }

  code_function_callt push_func_call;
  push_func_call.function() = symbol_expr(*push_func_sym);
  push_func_call.arguments().push_back(symbol_expr(list)); // list
  push_func_call.arguments().push_back(                    // &element
    address_of_exprt(symbol_expr(*elem_info.elem_symbol)));
  push_func_call.arguments().push_back(
    symbol_expr(*elem_info.elem_type_sym));                  // type hash
  push_func_call.arguments().push_back(elem_info.elem_size); // element size

  push_func_call.type() = bool_type();
  push_func_call.location() = elem_info.location;

  return push_func_call;
}

exprt python_list::build_insert_list_call(
  const symbolt &list,
  const exprt &index,
  const nlohmann::json &op,
  const exprt &elem)
{
  list_elem_info elem_info = get_list_element_info(op, elem);

  const symbolt *insert_func_sym =
    converter_.symbol_table().find_symbol("c:list.c@F@list_insert");
  if (!insert_func_sym)
    throw std::runtime_error("Insert function symbol not found");

  code_function_callt insert_func_call;
  insert_func_call.function() = symbol_expr(*insert_func_sym);
  insert_func_call.arguments().push_back(symbol_expr(list));
  insert_func_call.arguments().push_back(index);
  insert_func_call.arguments().push_back(
    address_of_exprt(symbol_expr(*elem_info.elem_symbol)));
  insert_func_call.arguments().push_back(symbol_expr(*elem_info.elem_type_sym));
  insert_func_call.arguments().push_back(elem_info.elem_size);
  insert_func_call.type() = bool_type();
  insert_func_call.location() = elem_info.location;

  return converter_.convert_expression_to_code(insert_func_call);
}

exprt python_list::build_concat_list_call(
  const exprt &lhs,
  const exprt &rhs,
  const nlohmann::json &element)
{
  // Create destination list
  symbolt &dst_list = create_list();
  const locationt loc = converter_.get_location_from_decl(element);

  // Helpers we’ll call from the C model
  const symbolt *size_sym =
    converter_.symbol_table().find_symbol("c:list.c@F@list_size");
  const symbolt *at_sym =
    converter_.symbol_table().find_symbol("c:list.c@F@list_at");
  const symbolt *push_obj_sym =
    converter_.symbol_table().find_symbol("c:list.c@F@list_push_object");
  assert(size_sym && at_sym && push_obj_sym);

  auto copy_list = [&](const exprt &src_list) {
    // size_t n = list_size(src_list);
    symbolt &n_sym = converter_.create_tmp_symbol(
      element, "$n$", size_type(), gen_zero(size_type()));
    code_declt n_decl(symbol_expr(n_sym));
    converter_.add_instruction(n_decl);

    code_function_callt get_size;
    get_size.function() = symbol_expr(*size_sym);
    // list_size takes const List*, pass address if we have a value
    if (src_list.type().is_pointer())
      get_size.arguments().push_back(src_list);
    else
      get_size.arguments().push_back(address_of_exprt(src_list));
    get_size.lhs() = symbol_expr(n_sym);
    get_size.type() = size_type();
    get_size.location() = loc;
    converter_.add_instruction(get_size);

    // for (size_t i = 0; i < n; ++i) { push_object(dst, list_at(src, i)); }
    symbolt &i_sym = converter_.create_tmp_symbol(
      element, "$i$", size_type(), gen_zero(size_type()));
    code_declt i_decl(symbol_expr(i_sym));
    converter_.add_instruction(i_decl);

    // i = 0
    code_assignt i_init(symbol_expr(i_sym), gen_zero(size_type()));
    converter_.add_instruction(i_init);

    // condition: i < n
    exprt cond("<", bool_type());
    cond.copy_to_operands(symbol_expr(i_sym), symbol_expr(n_sym));

    // body
    code_blockt body;

    // tmp_obj = list_at(src_list, i)
    side_effect_expr_function_callt at_call;
    at_call.function() = symbol_expr(*at_sym);
    if (src_list.type().is_pointer())
      at_call.arguments().push_back(src_list);
    else
      at_call.arguments().push_back(address_of_exprt(src_list));
    at_call.arguments().push_back(symbol_expr(i_sym));
    at_call.type() =
      pointer_typet(converter_.get_type_handler().get_list_element_type());
    at_call.location() = loc;

    symbolt &tmp_obj = converter_.create_tmp_symbol(
      element,
      "tmp_list_at",
      pointer_typet(converter_.get_type_handler().get_list_element_type()),
      exprt());
    code_declt tmp_obj_decl(symbol_expr(tmp_obj));
    tmp_obj_decl.copy_to_operands(at_call);
    body.copy_to_operands(tmp_obj_decl);

    // list_push_object(dst_list, tmp_obj)
    side_effect_expr_function_callt push_call;
    push_call.function() = symbol_expr(*push_obj_sym);
    push_call.arguments().push_back(symbol_expr(dst_list));
    push_call.arguments().push_back(symbol_expr(tmp_obj));
    push_call.type() = bool_type();
    push_call.location() = loc;
    body.copy_to_operands(converter_.convert_expression_to_code(push_call));

    // i = i + 1
    plus_exprt i_inc(symbol_expr(i_sym), gen_one(size_type()));
    code_assignt i_step(symbol_expr(i_sym), i_inc);
    body.copy_to_operands(i_step);

    // while (i < n) { ... }
    codet loop;
    loop.set_statement("while");
    loop.copy_to_operands(cond, body);
    converter_.add_instruction(loop);
  };

  // Copy lhs then rhs
  copy_list(lhs);
  copy_list(rhs);

  // Update list type mapping
  const std::string dst_id = dst_list.id.as_string();
  auto copy_type_info = [&](const exprt &src_list) {
    if (!src_list.is_symbol())
      return;
    const std::string key = src_list.identifier().as_string();
    auto it = list_type_map.find(key);
    if (it != list_type_map.end())
    {
      for (const auto &p : it->second)
        list_type_map[dst_id].push_back(p);
    }
  };
  copy_type_info(lhs);
  copy_type_info(rhs);

  return symbol_expr(dst_list);
}

symbolt &python_list::create_list()
{
  locationt location = converter_.get_location_from_decl(list_value_);
  const type_handler &type_handler = converter_.get_type_handler();

  // Create infinite array type for list storage
  const array_typet inf_array_type(
    type_handler.get_list_element_type(), exprt("infinity", size_type()));

  exprt inf_array_value =
    gen_zero(get_complete_type(inf_array_type, converter_.ns), true);

  // Create and configure infinite array symbol
  symbolt &inf_array_symbol = converter_.create_tmp_symbol(
    list_value_, "$storage$", inf_array_type, inf_array_value);
  inf_array_symbol.value.zero_initializer(true);
  inf_array_symbol.static_lifetime = true;

  // Declare infinite array
  code_declt inf_array_decl(symbol_expr(inf_array_symbol));
  inf_array_decl.location() = location;
  converter_.add_instruction(inf_array_decl);

  // Create list symbol
  const typet list_type = type_handler.get_list_type();
  symbolt &list_symbol =
    converter_.create_tmp_symbol(list_value_, "$list$", list_type, exprt());

  // Declare list
  code_declt list_decl(symbol_expr(list_symbol));
  list_decl.location() = location;
  converter_.add_instruction(list_decl);

  // Initialize list with storage array
  const symbolt *create_func_sym =
    converter_.symbol_table().find_symbol("c:list.c@F@list_create");
  if (!create_func_sym)
  {
    throw std::runtime_error("List creation function symbol not found");
  }

  // Add list_create call to the block
  code_function_callt list_create_func_call;
  list_create_func_call.function() = symbol_expr(*create_func_sym);
  list_create_func_call.lhs() = symbol_expr(list_symbol);
  list_create_func_call.arguments().push_back(
    converter_.get_string_handler().get_array_base_address(
      symbol_expr(inf_array_symbol)));
  list_create_func_call.type() = list_type;
  list_create_func_call.location() = location;
  converter_.add_instruction(list_create_func_call);

  return list_symbol;
}

exprt python_list::get()
{
  symbolt &list_symbol = create_list();
  const std::string &list_id = list_symbol.id.as_string();

  for (auto &e : list_value_["elts"])
  {
    exprt elem = converter_.get_expr(e);
    exprt list_push_func_call =
      build_push_list_call(list_symbol, list_value_, elem);

    converter_.add_instruction(list_push_func_call);

    list_type_map[list_id].push_back(
      std::make_pair(elem.identifier().as_string(), elem.type()));
  }

  return symbol_expr(list_symbol);
}

exprt python_list::build_list_at_call(
  const exprt &list,
  const exprt &index,
  const nlohmann::json &element)
{
  pointer_typet obj_type(converter_.get_type_handler().get_list_element_type());

  const symbolt *list_at_func_sym =
    converter_.symbol_table().find_symbol("c:list.c@F@list_at");
  assert(list_at_func_sym);

  side_effect_expr_function_callt list_at_call;
  list_at_call.function() = symbol_expr(*list_at_func_sym);
  if (list.type().is_pointer())
    list_at_call.arguments().push_back(list); // &l
  else
    list_at_call.arguments().push_back(address_of_exprt(list)); // &l

  list_at_call.arguments().push_back(index);
  list_at_call.type() = obj_type;
  list_at_call.location() = converter_.get_location_from_decl(element);

  return list_at_call;
}

exprt python_list::index(const exprt &array, const nlohmann::json &slice_node)
{
  if (slice_node["_type"] == "Slice") // arr[lower:upper]
  {
    return handle_range_slice(array, slice_node);
  }
  else
  {
    return handle_index_access(array, slice_node);
  }
  return exprt();
}

exprt python_list::handle_range_slice(
  const exprt &array,
  const nlohmann::json &slice_node)
{
  const typet list_type = converter_.get_type_handler().get_list_type();

  // Handle regular array/string slicing (not list slicing)
  if (array.type() != list_type && array.type().is_array())
  {
    const array_typet &src_type = to_array_type(array.type());
    locationt location = converter_.get_location_from_decl(slice_node);

    // Get array length
    exprt array_len = src_type.size();

    // For char arrays (strings), exclude the null terminator from length
    // when calculating negative indices, to match Python string behavior
    exprt logical_len = array_len;
    if (src_type.subtype() == char_type())
      logical_len = minus_exprt(array_len, gen_one(size_type()));

    // Process slice bounds (handles null, negative indices)
    auto process_bound =
      [&](const std::string &bound_name, const exprt &default_value) -> exprt {
      if (!slice_node.contains(bound_name) || slice_node[bound_name].is_null())
        return default_value;

      const auto &bound = slice_node[bound_name];

      // Check if negative index
      if (bound["_type"] == "UnaryOp" && bound["op"]["_type"] == "USub")
      {
        exprt abs_value = converter_.get_expr(bound["operand"]);
        return minus_exprt(logical_len, abs_value);
      }

      return converter_.get_expr(bound);
    };

    // Process bounds
    exprt lower_expr = process_bound("lower", gen_zero(size_type()));
    exprt upper_expr = process_bound("upper", logical_len);

    // Calculate slice length
    minus_exprt slice_len(upper_expr, lower_expr);

    // Create result array type with extra space for null terminator
    plus_exprt result_size(slice_len, gen_one(size_type()));
    array_typet result_type(src_type.subtype(), result_size);

    // Create temporary for sliced array
    symbolt &result = converter_.create_tmp_symbol(
      slice_node, "$array_slice$", result_type, exprt());

    code_declt result_decl(symbol_expr(result));
    result_decl.location() = location;
    converter_.add_instruction(result_decl);

    // Create loop: for i = 0; i < (upper-lower); i++
    symbolt &idx = converter_.create_tmp_symbol(
      slice_node, "$i$", size_type(), gen_zero(size_type()));
    code_assignt idx_init(symbol_expr(idx), gen_zero(size_type()));
    converter_.add_instruction(idx_init);

    exprt cond("<", bool_type());
    cond.copy_to_operands(symbol_expr(idx), slice_len);

    code_blockt body;
    // result[i] = array[lower + i]
    plus_exprt src_idx(lower_expr, symbol_expr(idx));
    index_exprt src(array, src_idx, src_type.subtype());
    index_exprt dst(symbol_expr(result), symbol_expr(idx), src_type.subtype());
    code_assignt assign(dst, src);
    body.copy_to_operands(assign);

    // i++
    plus_exprt incr(symbol_expr(idx), gen_one(size_type()));
    code_assignt update(symbol_expr(idx), incr);
    body.copy_to_operands(update);

    codet loop;
    loop.set_statement("while");
    loop.copy_to_operands(cond, body);
    converter_.add_instruction(loop);

    // Add null terminator at result[slice_len]
    index_exprt null_pos(symbol_expr(result), slice_len, src_type.subtype());
    code_assignt add_null(null_pos, gen_zero(src_type.subtype()));
    add_null.location() = location;
    converter_.add_instruction(add_null);

    return symbol_expr(result);
  }

  // Handle list slicing
  symbolt &sliced_list = create_list();
  const locationt location = converter_.get_location_from_decl(list_value_);

  // Get bound expressions (handles null/missing)
  auto get_list_bound = [&](const std::string &bound_name) -> exprt {
    if (slice_node.contains(bound_name) && !slice_node[bound_name].is_null())
      return converter_.get_expr(slice_node[bound_name]);

    // For lists, we'd need the list size here, but that's not easily accessible
    // For now, keep existing behavior - assumes bounds are present
    throw std::runtime_error(
      "List slicing with missing bounds not yet supported");
  };

  const exprt lower_expr = get_list_bound("lower");
  const exprt upper_expr = get_list_bound("upper");

  // Initialize counter: int counter = lower
  symbolt &counter = converter_.create_tmp_symbol(
    list_value_, "counter", size_type(), lower_expr);
  code_assignt counter_init(symbol_expr(counter), lower_expr);
  converter_.add_instruction(counter_init);

  // Build while loop: while (counter < upper)
  exprt loop_condition("<", bool_type());
  loop_condition.operands().push_back(symbol_expr(counter));
  loop_condition.operands().push_back(upper_expr);

  // Build loop body
  code_blockt loop_body;

  // Get element at current index
  const exprt list_at_call =
    build_list_at_call(array, symbol_expr(counter), list_value_);
  const symbolt &at_result = converter_.create_tmp_symbol(
    list_value_,
    "tmp_list_at",
    pointer_typet(converter_.get_type_handler().get_list_element_type()),
    exprt());

  code_declt at_decl(symbol_expr(at_result));
  at_decl.copy_to_operands(list_at_call);
  loop_body.copy_to_operands(at_decl);

  // Push element to sliced list
  const symbolt *push_func =
    converter_.symbol_table().find_symbol("c:list.c@F@list_push_object");
  if (!push_func)
  {
    throw std::runtime_error("Push function symbol not found");
  }

  side_effect_expr_function_callt push_call;
  push_call.function() = symbol_expr(*push_func);
  push_call.arguments().push_back(symbol_expr(sliced_list));
  push_call.arguments().push_back(symbol_expr(at_result));
  push_call.type() = bool_type();
  push_call.location() = location;
  loop_body.copy_to_operands(converter_.convert_expression_to_code(push_call));

  // Increment counter
  exprt increment("+");
  increment.copy_to_operands(symbol_expr(counter));
  increment.copy_to_operands(gen_one(int_type()));
  code_assignt counter_update(symbol_expr(counter), increment);
  loop_body.copy_to_operands(counter_update);

  // Create and execute while loop
  codet while_loop;
  while_loop.set_statement("while");
  while_loop.copy_to_operands(loop_condition, loop_body);
  converter_.add_instruction(while_loop);

  // Update type map for sliced elements (only if bounds are available)
  if (
    slice_node.contains("lower") && !slice_node["lower"].is_null() &&
    slice_node.contains("upper") && !slice_node["upper"].is_null())
  {
    const auto &list_node = json_utils::get_var_value(
      list_value_["value"]["id"],
      converter_.current_function_name(),
      converter_.ast());

    const size_t lower_bound = slice_node["lower"]["value"].get<size_t>();
    const size_t upper_bound = slice_node["upper"]["value"].get<size_t>();

    // Only update type map for actual lists (not strings or other types)
    if (
      !list_node.is_null() && list_node.contains("value") &&
      list_node["value"].contains("elts") &&
      list_node["value"]["elts"].is_array())
    {
      for (size_t i = lower_bound; i < upper_bound; ++i)
      {
        const exprt element =
          converter_.get_expr(list_node["value"]["elts"][i]);
        list_type_map[sliced_list.id.as_string()].push_back(
          std::make_pair(element.identifier().as_string(), element.type()));
      }
    }
  }

  return symbol_expr(sliced_list);
}

exprt python_list::handle_index_access(
  const exprt &array,
  const nlohmann::json &slice_node)
{
  // Find list node for type information
  nlohmann::json list_node;
  if (list_value_["value"].contains("id"))
  {
    list_node = json_utils::find_var_decl(
      list_value_["value"]["id"],
      converter_.current_function_name(),
      converter_.ast());
  }

  exprt pos_expr = converter_.get_expr(slice_node);
  size_t index = 0;

  // Validate index type
  if (pos_expr.type().is_array())
  {
    locationt l = converter_.get_location_from_decl(list_value_);
    throw std::runtime_error(
      "TypeError at " + l.get_file().as_string() + " " +
      l.get_line().as_string() +
      ": list indices must be integers or slices, not str");
  }

  // Handle negative indices
  if (slice_node.contains("op") && slice_node["op"]["_type"] == "USub")
  {
    if (list_node.is_null() || list_node["value"]["_type"] != "List")
    {
      BigInt v = binary2integer(pos_expr.op0().value().c_str(), true);
      v *= -1;

      const array_typet &t = static_cast<const array_typet &>(array.type());
      BigInt s = binary2integer(t.size().value().c_str(), true);

      // For char arrays (strings), exclude null terminator from logical length
      if (t.subtype() == char_type())
        s -= 1;

      v += s;
      pos_expr = from_integer(v, pos_expr.type());
    }
    else
    {
      index = slice_node["operand"]["value"].get<size_t>();
      index = list_node["value"]["elts"].size() - index;
      pos_expr = from_integer(index, size_type());
    }
  }
  else if (slice_node["_type"] == "Constant")
  {
    index = slice_node["value"].get<size_t>();
  }

  // Handle different array types
  if (array.type().is_symbol() || array.type().subtype().is_symbol())
  {
    // Handle list types (symbol-based)
    typet elem_type;

    // Check for nested list access
    if (array.type() == converter_.get_type_handler().get_list_type())
    {
      const auto &key = array.identifier().as_string();
      auto type_map_it = list_type_map.find(key);
      if (type_map_it != list_type_map.end())
      {
        if (index < type_map_it->second.size())
        {
          const std::string &elem_id = type_map_it->second.at(index).first;
          elem_type = type_map_it->second.at(index).second;

          if (elem_type == converter_.get_type_handler().get_list_type())
          {
            symbolt *nested_list = converter_.find_symbol(elem_id);
            assert(nested_list);
            return symbol_expr(*nested_list);
          }
        }
      }
    }

    // Determine element type
    if (list_node["_type"] == "arg")
    {
      elem_type =
        get_elem_type_from_annotation(list_node, converter_.get_type_handler());
    }
    else if (
      slice_node["_type"] == "Constant" || slice_node["_type"] == "BinOp" ||
      (slice_node["_type"] == "UnaryOp" &&
       slice_node["operand"]["_type"] == "Constant"))
    {
      const std::string &list_name = array.identifier().as_string();

      if (list_type_map[list_name].empty())
      {
        /* Fall back to annotation for function parameters */
        const nlohmann::json list_value_node = json_utils::get_var_value(
          list_value_["value"]["id"],
          converter_.current_function_name(),
          converter_.ast());

        elem_type = get_elem_type_from_annotation(
          list_value_node, converter_.get_type_handler());
      }
      else
      {
        size_t type_index =
          (!list_node.is_null() && list_node["value"]["_type"] == "BinOp")
            ? 0
            : index;

        try
        {
          elem_type = list_type_map[list_name].at(type_index).second;
        }
        catch (const std::out_of_range &)
        {
          const locationt l = converter_.get_location_from_decl(list_value_);
          throw std::runtime_error(
            "List out of bounds at " + l.get_file().as_string() +
            " line: " + l.get_line().as_string());
        }
      }
    }
    else if (slice_node["_type"] == "Name")
    {
      // Handle variable-based indexing
      if (!list_node.is_null() && list_node["_type"] == "arg")
      {
        elem_type = get_elem_type_from_annotation(
          list_node, converter_.get_type_handler());
      }
      else
      {
        // Handle case where we need to find the variable declaration
        while (!list_node.is_null() && (!list_node.contains("value") ||
                                        !list_node["value"].contains("elts") ||
                                        !list_node["value"]["elts"].is_array()))
        {
          if (list_node.contains("value") && list_node["value"].contains("id"))
            list_node = json_utils::find_var_decl(
              list_node["value"]["id"],
              converter_.current_function_name(),
              converter_.ast());
          else
          {
            break;
          }
        }

        if (!list_node.is_null() && list_node["_type"] == "arg")
        {
          elem_type = get_elem_type_from_annotation(
            list_node, converter_.get_type_handler());
        }
        else if (elem_type == typet() && list_node.contains("value"))
        {
          elem_type =
            converter_
              .get_expr(json_utils::get_list_element(list_node["value"], 0))
              .type();
        }
      }
    }

    if (pos_expr == exprt() || elem_type == typet())
    {
      throw std::runtime_error(
        "Invalid list access: could not resolve position or element type");
    }

    // Build list access and cast result
    exprt list_at_call = build_list_at_call(array, pos_expr, list_value_);

    // Get obj->value and cast to correct type
    member_exprt obj_value(list_at_call, "value", pointer_typet(empty_typet()));
    {
      exprt &base = obj_value.struct_op();
      exprt deref("dereference");
      deref.type() = base.type().subtype();
      deref.move_to_operands(base);
      base.swap(deref);
    }

    // Cast from void* to target type pointer and dereference
    typecast_exprt tc(obj_value, pointer_typet(elem_type));

    // Dereference to get the actual value
    dereference_exprt deref(elem_type);
    deref.op0() = tc;
    return deref;
  }

  // Handle static arrays
  return index_exprt(array, pos_expr, array.type().subtype());
}

exprt python_list::compare(
  const exprt &l1,
  const exprt &l2,
  const std::string &op)
{
  const symbolt *list_eq_func_sym =
    converter_.symbol_table().find_symbol("c:list.c@F@list_eq");
  assert(list_eq_func_sym);

  const symbolt *lhs_symbol =
    converter_.find_symbol(l1.identifier().as_string());
  const symbolt *rhs_symbol =
    converter_.find_symbol(l2.identifier().as_string());
  assert(lhs_symbol);
  assert(rhs_symbol);

  symbolt &eq_ret = converter_.create_tmp_symbol(
    list_value_, "eq_tmp", bool_type(), gen_boolean(false));
  code_declt eq_ret_decl(symbol_expr(eq_ret));
  converter_.add_instruction(eq_ret_decl);

  code_function_callt list_eq_func_call;
  list_eq_func_call.function() = symbol_expr(*list_eq_func_sym);
  list_eq_func_call.lhs() = symbol_expr(eq_ret);
  // passing arguments
  list_eq_func_call.arguments().push_back(symbol_expr(*lhs_symbol)); // l1
  list_eq_func_call.arguments().push_back(symbol_expr(*rhs_symbol)); // l2
  list_eq_func_call.type() = bool_type();
  list_eq_func_call.location() = converter_.get_location_from_decl(list_value_);
  converter_.add_instruction(list_eq_func_call);

  //return list_eq_func_call;
  exprt cond("=", bool_type());
  cond.copy_to_operands(symbol_expr(eq_ret));
  if (op == "Eq")
    cond.copy_to_operands(gen_boolean(true));
  else
    cond.copy_to_operands(gen_boolean(false));

  return cond;
}

exprt python_list::create_vla(
  const nlohmann::json &element,
  const symbolt *list,
  symbolt *size_var,
  const exprt &list_elem)
{
  // Add counter for while loop
  symbolt &counter = converter_.create_tmp_symbol(
    element, "counter", int_type(), gen_zero(int_type()));

  code_assignt counter_code(symbol_expr(counter), gen_zero(int_type()));
  converter_.add_instruction(counter_code);

  // Build conditional for while loop (counter < len(arr))
  exprt cond("<", bool_type());
  cond.operands().push_back(symbol_expr(counter));
  cond.operands().push_back(symbol_expr(*size_var));

  // Build block with lish_push() calls and counter increment
  code_blockt then;
  exprt list_push_call = build_push_list_call(*list, element, list_elem);
  then.copy_to_operands(list_push_call);

  // increment counter
  exprt incr("+");
  incr.copy_to_operands(symbol_expr(counter));
  incr.copy_to_operands(gen_one(int_type()));
  code_assignt update(symbol_expr(counter), incr);
  then.copy_to_operands(update);

  // add while block for list_push() calls
  codet while_cod;
  while_cod.set_statement("while");
  while_cod.copy_to_operands(cond, then);
  converter_.add_instruction(while_cod);

  return symbol_expr(*list);
}

exprt python_list::list_repetition(
  const nlohmann::json &left_node,
  const nlohmann::json &right_node,
  const exprt &lhs,
  const exprt &rhs)
{
  typet list_type = converter_.get_type_handler().get_list_type();

  BigInt list_size;
  exprt list_elem;

  // Get list size from lhs (e.g.: 3 * [1])
  if (lhs.type() != list_type)
  {
    if (lhs.is_symbol())
    {
      symbolt *size_var = converter_.find_symbol(
        to_symbol_expr(lhs).get_identifier().as_string());
      assert(size_var);
      list_size = std::stoi(size_var->value.value().as_string(), nullptr, 2);
    }
    else if (lhs.is_constant())
      list_size = std::stoi(lhs.value().as_string(), nullptr, 2);

    // List element is the rhs
    list_elem = converter_.get_expr(right_node["elts"][0]);
  }

  // Get list size from rhs (e.g.: [1] * 3)
  if (rhs.type() != list_type)
  {
    // List element is the rhs
    list_elem = converter_.get_expr(left_node["elts"][0]);

    if (rhs.is_symbol()) // (e.g.: [1] * n)
    {
      symbolt *size_var = converter_.find_symbol(
        to_symbol_expr(rhs).get_identifier().as_string());

      assert(size_var);

      symbolt *list_symbol =
        converter_.find_symbol(lhs.identifier().as_string());
      assert(list_symbol);

      if (size_var->value.is_code())
      {
        return create_vla(list_value_, list_symbol, size_var, list_elem);
      }

      list_size = std::stoi(size_var->value.value().as_string(), nullptr, 2);
    }
    else if (rhs.is_constant())
      list_size = std::stoi(rhs.value().as_string(), nullptr, 2);
  }

  symbolt *list_symbol = converter_.find_symbol(lhs.identifier().as_string());
  assert(list_symbol);

  const std::string &list_id = converter_.current_lhs->identifier().as_string();

  for (int64_t i = 0; i < list_size.to_int64() - 1; ++i)
  {
    converter_.add_instruction(
      build_push_list_call(*list_symbol, list_value_, list_elem));

    list_type_map[list_id].push_back(
      std::make_pair(list_elem.identifier().as_string(), list_elem.type()));
  }

  return symbol_expr(*list_symbol);
}

exprt python_list::contains(const exprt &item, const exprt &list)
{
  // Get type and size information for the item
  list_elem_info item_info = get_list_element_info(list_value_, item);

  // Find the list_contains function
  const symbolt *list_contains_func =
    converter_.symbol_table().find_symbol("c:list.c@F@list_contains");
  if (!list_contains_func)
    throw std::runtime_error(
      "list_contains function not found in symbol table");

  // Create a temporary variable to store the result
  symbolt &contains_ret = converter_.create_tmp_symbol(
    list_value_, "contains_tmp", bool_type(), gen_boolean(false));
  code_declt contains_ret_decl(symbol_expr(contains_ret));
  converter_.add_instruction(contains_ret_decl);

  // Build the function call as a statement
  code_function_callt contains_call;
  contains_call.function() = symbol_expr(*list_contains_func);
  contains_call.lhs() = symbol_expr(contains_ret);

  // Pass the list directly
  contains_call.arguments().push_back(list);

  // For pointer types (e.g., string parameters), use the pointer directly
  // For value types, take the address
  exprt item_arg;
  if (item_info.elem_symbol->type.is_pointer())
  {
    // String parameters are pointers - use the pointer value directly
    item_arg = symbol_expr(*item_info.elem_symbol);
  }
  else
  {
    // For arrays or other value types, take the address
    item_arg = address_of_exprt(symbol_expr(*item_info.elem_symbol));
  }

  contains_call.arguments().push_back(item_arg);

  contains_call.arguments().push_back(
    symbol_expr(*item_info.elem_type_sym));                 // item type hash
  contains_call.arguments().push_back(item_info.elem_size); // item size

  contains_call.type() = bool_type();
  contains_call.location() = converter_.get_location_from_decl(list_value_);
  converter_.add_instruction(contains_call);

  exprt result("=", bool_type());
  result.copy_to_operands(symbol_expr(contains_ret));
  result.copy_to_operands(gen_boolean(true));

  return result;
}

exprt python_list::build_extend_list_call(
  const symbolt &list,
  const nlohmann::json &op,
  const exprt &other_list)
{
  const symbolt *extend_func_sym =
    converter_.symbol_table().find_symbol("c:list.c@F@list_extend");
  if (!extend_func_sym)
    throw std::runtime_error("Extend function symbol not found");

  locationt location = converter_.get_location_from_decl(op);

  // Update list_type_map: copy type info from other_list to list
  const std::string &list_name = list.id.as_string();
  const std::string &other_list_name = other_list.identifier().as_string();

  // Copy all type entries from other_list to the end of list
  if (list_type_map.find(other_list_name) != list_type_map.end())
  {
    for (const auto &type_entry : list_type_map[other_list_name])
    {
      list_type_map[list_name].push_back(type_entry);
    }
  }

  code_function_callt extend_func_call;
  extend_func_call.function() = symbol_expr(*extend_func_sym);
  extend_func_call.arguments().push_back(symbol_expr(list));
  extend_func_call.arguments().push_back(other_list);
  extend_func_call.type() = empty_typet();
  extend_func_call.location() = location;

  return extend_func_call;
}
