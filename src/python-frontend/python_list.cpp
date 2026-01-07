#include <python-frontend/python_list.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/symbol_id.h>
#include <util/expr.h>
#include <util/type.h>
#include <util/symbol.h>
#include <util/expr_util.h>
#include <util/arith_tools.h>
#include <util/std_code.h>
#include <util/std_expr.h>
#include <util/mp_arith.h>
#include <util/python_types.h>
#include <util/symbolic_types.h>
#include <string>
#include <functional>

// Extract element type from annotation
static typet get_elem_type_from_annotation(
  const nlohmann::json &node,
  const type_handler &type_handler_)
{
  // Extract element type from a Subscript node such as list[T]
  auto extract_subscript_elem = [&](const nlohmann::json &ann) -> typet {
    if (
      ann.contains("slice") && ann["slice"].is_object() &&
      ann["slice"].contains("id") && ann["slice"]["id"].is_string())
    {
      return type_handler_.get_typet(ann["slice"]["id"].get<std::string>());
    }
    return typet();
  };

  if (!node.contains("annotation") || !node["annotation"].is_object())
    return typet();

  const auto &annotation = node["annotation"];

  // Case 1: Direct subscript annotation like list[str]
  if (annotation.is_object() && annotation.contains("slice"))
  {
    typet elem_type = extract_subscript_elem(annotation);
    if (elem_type != typet())
      return elem_type;
  }

  // Case 2: Union type annotation such as list[str] | None
  if (
    annotation.is_object() && annotation.contains("_type") &&
    annotation["_type"] == "BinOp")
  {
    // Try left side first (e.g., handles list[str] | None)
    if (
      annotation.contains("left") && annotation["left"].is_object() &&
      annotation["left"].contains("_type") &&
      annotation["left"]["_type"] == "Subscript")
    {
      typet elem_type = extract_subscript_elem(annotation["left"]);
      if (elem_type != typet())
        return elem_type;
    }

    // Try right side (e.g., handles None | list[str])
    if (
      annotation.contains("right") && annotation["right"].is_object() &&
      annotation["right"].contains("_type") &&
      annotation["right"]["_type"] == "Subscript")
    {
      typet elem_type = extract_subscript_elem(annotation["right"]);
      if (elem_type != typet())
        return elem_type;
    }
  }

  // Case 3: Direct type annotation such as str, int
  if (annotation.contains("id") && annotation["id"].is_string())
    return type_handler_.get_typet(annotation["id"].get<std::string>());

  // Return empty type if annotation structure is not recognized
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

  // TODO: Eventually we should build a reverse index of hash => type into the context
  // this will allow better verification counter-examples.
  constant_exprt hash_value(size_type());
  hash_value.set_value(integer2binary(
    std::hash<std::string>{}(elem_type_name), config.ansi_c.address_width));
  code_assignt hash_assignment(symbol_expr(elem_type_sym), hash_value);
  hash_assignment.location() = location;
  converter_.add_instruction(hash_assignment);

  // Create and declare temporary symbol for list element
  symbolt &elem_symbol =
    converter_.create_tmp_symbol(op, "$list_elem$", elem.type(), elem);
  code_declt elem_decl(symbol_expr(elem_symbol));
  elem_decl.copy_to_operands(elem);
  elem_decl.location() = location;
  converter_.add_instruction(elem_decl);

  // Calculate element size in bytes
  exprt elem_size;

  // For list pointers (PyListObj*), use pointer size
  typet list_type = converter_.get_type_handler().get_list_type();
  // None type: store pointer directly without copying
  // Set size to 0 so memcpy is skipped and NULL is preserved
  if (
    elem_symbol.type.is_pointer() && elem_symbol.type.subtype() == bool_type())
  {
    elem_size = from_integer(BigInt(0), size_type());
  }
  // For list pointers (PyListObj*), use pointer size
  else if (elem_symbol.type == list_type)
  {
    // This is a pointer to PyListObj: use pointer size
    const size_t pointer_size_bytes = config.ansi_c.pointer_width() / 8;
    elem_size = from_integer(BigInt(pointer_size_bytes), size_type());
  }
  // Handle struct types (such as dictionaries): store by reference
  else if (elem_symbol.type.is_struct())
  {
    // Calculate actual struct size by counting pointer-sized components
    const struct_union_typet &struct_type =
      to_struct_union_type(elem_symbol.type);

    // For dictionary structs, each component is a pointer (keys, values)
    size_t num_components = struct_type.components().size();
    size_t total_size = num_components * (config.ansi_c.pointer_width() / 8);

    elem_size = from_integer(BigInt(total_size), size_type());
  }
  // For string pointers (char*), calculate length at runtime using strlen
  else if (
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
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_push");

  if (!push_func_sym)
  {
    throw std::runtime_error("Push function symbol not found");
  }

  code_function_callt push_func_call;
  push_func_call.function() = symbol_expr(*push_func_sym);
  push_func_call.arguments().push_back(symbol_expr(list)); // list

  // For string types (pointer to char), we must pass the pointer value directly
  // For other types (including other pointers such None/bool*), we must pass the address
  exprt element_arg;
  if (
    elem_info.elem_symbol->type.is_pointer() &&
    elem_info.elem_symbol->type.subtype() == char_type())
  {
    // For string type (char*), we must pass the pointer value itself
    element_arg = symbol_expr(*elem_info.elem_symbol);
  }
  else if (
    elem_info.elem_symbol->type.is_pointer() &&
    elem_info.elem_symbol->type.subtype() == bool_type())
  {
    // For None type (_Bool*), pass the pointer value itself
    // This allows direct NULL checks without dereferencing
    element_arg = symbol_expr(*elem_info.elem_symbol);
  }
  else if (elem_info.elem_symbol->type.is_struct())
  {
    // For structs (dictionaries), pass address of the struct directly
    element_arg = address_of_exprt(symbol_expr(*elem_info.elem_symbol));
  }
  else
  {
    // For bool types, cast to signed long int before taking address
    // This ensures proper storage and retrieval
    if (elem_info.elem_symbol->type == bool_type())
    {
      symbolt &bool_as_long = converter_.create_tmp_symbol(
        op,
        "$bool_as_long$",
        signedbv_typet(config.ansi_c.long_int_width),
        exprt());

      typecast_exprt bool_cast(
        symbol_expr(*elem_info.elem_symbol),
        signedbv_typet(config.ansi_c.long_int_width));

      code_declt bool_long_decl(symbol_expr(bool_as_long));
      bool_long_decl.copy_to_operands(bool_cast);
      bool_long_decl.location() = elem_info.location;
      converter_.add_instruction(bool_long_decl);

      element_arg = address_of_exprt(symbol_expr(bool_as_long));

      // Update elem_size to match
      elem_info.elem_size =
        from_integer(BigInt(config.ansi_c.long_int_width / 8), size_type());
    }
    else
    {
      // For all other types, we must pass address of the value
      element_arg = address_of_exprt(symbol_expr(*elem_info.elem_symbol));
    }
  }

  push_func_call.arguments().push_back(element_arg); // element or &element
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
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_insert");
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
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
  const symbolt *at_sym =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_at");
  const symbolt *push_obj_sym =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_push_object");
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

  // Create list symbol
  const typet list_type = type_handler.get_list_type();
  symbolt &list_symbol =
    converter_.create_tmp_symbol(list_value_, "$py_list$", list_type, exprt());

  // Declare list
  code_declt list_decl(symbol_expr(list_symbol));
  list_decl.location() = location;
  converter_.add_instruction(list_decl);

  // Initialize list with storage array
  const symbolt *create_func_sym =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_create");
  assert(create_func_sym);

  // Add list_create call to the block
  code_function_callt list_create_func_call;
  list_create_func_call.function() = symbol_expr(*create_func_sym);
  list_create_func_call.lhs() = symbol_expr(list_symbol);
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
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_at");
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

exprt python_list::build_split_list(
  python_converter &converter,
  const nlohmann::json &call_node,
  const std::string &input,
  const std::string &separator,
  long long count)
{
  if (separator.empty())
    throw std::runtime_error("split() separator cannot be empty");

  if (count == 0)
  {
    nlohmann::json list_node;
    list_node["_type"] = "List";
    list_node["elts"] = nlohmann::json::array();
    converter.copy_location_fields_from_decl(call_node, list_node);

    nlohmann::json elem;
    elem["_type"] = "Constant";
    elem["value"] = input;
    converter.copy_location_fields_from_decl(call_node, elem);
    list_node["elts"].push_back(elem);

    python_list list(converter, list_node);
    return list.get();
  }

  std::vector<std::string> parts;
  size_t start = 0;
  long long splits = 0;
  while (true)
  {
    if (count >= 0 && splits >= count)
    {
      parts.push_back(input.substr(start));
      break;
    }

    size_t pos = input.find(separator, start);
    if (pos == std::string::npos)
    {
      parts.push_back(input.substr(start));
      break;
    }
    parts.push_back(input.substr(start, pos - start));
    start = pos + separator.size();
    ++splits;
  }

  nlohmann::json list_node;
  list_node["_type"] = "List";
  list_node["elts"] = nlohmann::json::array();
  converter.copy_location_fields_from_decl(call_node, list_node);

  for (const auto &part : parts)
  {
    nlohmann::json elem;
    elem["_type"] = "Constant";
    elem["value"] = part;
    converter.copy_location_fields_from_decl(call_node, elem);
    list_node["elts"].push_back(elem);
  }

  python_list list(converter, list_node);
  return list.get();
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

exprt python_list::remove_function_calls_recursive(
  exprt &e,
  const nlohmann::json &node)
{
  // Bounds might generate intermediate calls, we need to add lhs to all of them.
  const auto add_lhs_var_bound = [&](exprt &foo) -> exprt {
    if (!foo.is_function_call())
      return foo;
    code_function_callt &call = static_cast<code_function_callt &>(foo);
    symbolt &lhs = converter_.create_tmp_symbol(
      node, "__python_function_call_lhs$", size_type(), exprt());
    call.lhs() = symbol_expr(lhs);
    converter_.add_instruction(call);
    return symbol_expr(lhs);
  };

  auto res = add_lhs_var_bound(e);
  for (auto &ee : res.operands())
  {
    ee = add_lhs_var_bound(ee);
    remove_function_calls_recursive(ee, node);
  }

  return res;
}

exprt python_list::handle_range_slice(
  const exprt &array,
  const nlohmann::json &slice_node)
{
  const typet list_type = converter_.get_type_handler().get_list_type();

  // Handle regular array/string slicing (not list slicing)
  // String parameters come as pointer-to-char, so handle both arrays and char pointers
  bool is_string_slice =
    (array.type() != list_type && array.type().is_array()) ||
    (array.type().is_pointer() && array.type().subtype() == char_type());

  if (is_string_slice)
  {
    locationt location = converter_.get_location_from_decl(slice_node);

    // Determine element type and logical length
    typet elem_type;
    exprt array_len;
    exprt logical_len;

    if (array.type().is_array())
    {
      const array_typet &src_type = to_array_type(array.type());
      elem_type = src_type.subtype();
      array_len = src_type.size();
      // For char arrays (strings), exclude null terminator from logical length
      logical_len = (elem_type == char_type())
                      ? minus_exprt(array_len, gen_one(size_type()))
                      : array_len;
    }
    else // pointer case
    {
      elem_type = array.type().subtype();
      array_len = exprt();   // Not used for pointers
      logical_len = exprt(); // Will use explicit bounds only
    }

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
        return logical_len.is_nil() ? abs_value
                                    : minus_exprt(logical_len, abs_value);
      }

      exprt e = converter_.get_expr(bound);
      return remove_function_calls_recursive(e, slice_node);
    };

    // Process bounds
    exprt lower_expr = process_bound("lower", gen_zero(size_type()));

    // For pointer types with no upper bound, compute length using strlen
    exprt upper_expr;
    if (array.type().is_pointer() && elem_type == char_type())
    {
      if (!slice_node.contains("upper") || slice_node["upper"].is_null())
      {
        // Call strlen to get the length
        const symbolt *strlen_symbol =
          converter_.symbol_table().find_symbol("c:@F@strlen");
        if (!strlen_symbol)
          throw std::runtime_error("strlen function not found in symbol table");

        symbolt &strlen_result = converter_.create_tmp_symbol(
          slice_node, "$strlen_result$", size_type(), gen_zero(size_type()));
        code_declt strlen_decl(symbol_expr(strlen_result));
        strlen_decl.location() = location;
        converter_.add_instruction(strlen_decl);

        code_function_callt strlen_call;
        strlen_call.function() = symbol_expr(*strlen_symbol);
        strlen_call.lhs() = symbol_expr(strlen_result);
        strlen_call.arguments().push_back(array);
        strlen_call.type() = size_type();
        strlen_call.location() = location;
        converter_.add_instruction(strlen_call);

        upper_expr = symbol_expr(strlen_result);
      }
      else
        upper_expr = process_bound("upper", logical_len);
    }
    else
      upper_expr = process_bound("upper", logical_len);

    // Calculate slice length
    minus_exprt slice_len(upper_expr, lower_expr);

    // Create result array type with extra space for null terminator
    plus_exprt result_size(slice_len, gen_one(size_type()));
    array_typet result_type(elem_type, result_size);

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
    index_exprt src(array, src_idx, elem_type);
    index_exprt dst(symbol_expr(result), symbol_expr(idx), elem_type);
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
    index_exprt null_pos(symbol_expr(result), slice_len, elem_type);
    code_assignt add_null(null_pos, gen_zero(elem_type));
    add_null.location() = location;
    converter_.add_instruction(add_null);

    return symbol_expr(result);
  }

  // Handle list slicing
  symbolt &sliced_list = create_list();
  const locationt location = converter_.get_location_from_decl(list_value_);

  auto get_list_bound =
    [&](const std::string &bound_name, bool is_upper) -> exprt {
    if (slice_node.contains(bound_name) && !slice_node[bound_name].is_null())
      return converter_.get_expr(slice_node[bound_name]);

    if (is_upper)
    {
      const symbolt *size_func =
        converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
      assert(size_func);

      side_effect_expr_function_callt size_call;
      size_call.function() = symbol_expr(*size_func);

      // Check if array is already a pointer, don't take address again
      if (array.type().is_pointer())
        size_call.arguments().push_back(array); // Already a pointer
      else
        size_call.arguments().push_back(
          address_of_exprt(array)); // Take address

      size_call.type() = size_type();
      size_call.location() = converter_.get_location_from_decl(list_value_);

      symbolt &size_sym = converter_.create_tmp_symbol(
        list_value_, "$list_size$", size_type(), exprt());
      code_declt size_decl(symbol_expr(size_sym));
      size_decl.copy_to_operands(size_call);
      converter_.add_instruction(size_decl);

      return symbol_expr(size_sym);
    }
    else
    {
      return gen_zero(size_type());
    }
  };

  const exprt lower_expr = get_list_bound("lower", false);
  const exprt upper_expr = get_list_bound("upper", true);

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
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_push_object");
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
          // Only throw compile-time error if this is a static list with known elements
          // For constant indices on static lists, this is a definite out-of-bounds error
          if (
            (slice_node["_type"] == "Constant" ||
             (slice_node["_type"] == "UnaryOp" &&
              slice_node["operand"]["_type"] == "Constant")) &&
            !list_node.is_null() && list_node.contains("value") &&
            list_node["value"].contains("elts") &&
            list_node["value"]["elts"].is_array())
          {
            const locationt l = converter_.get_location_from_decl(list_value_);
            throw std::runtime_error(
              "List out of bounds at " + l.get_file().as_string() +
              " line: " + l.get_line().as_string());
          }

          // Try annotation fallback for dynamic lists or function parameters
          const nlohmann::json list_value_node = json_utils::get_var_value(
            list_value_["value"]["id"],
            converter_.current_function_name(),
            converter_.ast());

          elem_type = get_elem_type_from_annotation(
            list_value_node, converter_.get_type_handler());

          // Only throw if annotation also fails
          if (elem_type == typet())
          {
            const locationt l = converter_.get_location_from_decl(list_value_);
            throw std::runtime_error(
              "List out of bounds at " + l.get_file().as_string() +
              " line: " + l.get_line().as_string());
          }
        }
      }
    }
    else if (slice_node["_type"] == "Name")
    {
      // First try to get element type from list_node if it's an AnnAssign
      if (
        !list_node.is_null() && list_node["_type"] == "AnnAssign" &&
        list_node.contains("annotation") && elem_type == typet())
      {
        elem_type = get_elem_type_from_annotation(
          list_node, converter_.get_type_handler());
      }

      // If still no elem_type, try to get it from the array variable's type annotation
      if (array.is_symbol() && elem_type == typet())
      {
        // Extract variable name from the symbol identifier
        std::string list_var_name = json_utils::extract_var_name_from_symbol_id(
          array.identifier().as_string());

        // Find the variable's declaration to check for type annotation
        nlohmann::json list_var_decl = json_utils::find_var_decl(
          list_var_name, converter_.current_function_name(), converter_.ast());

        // If the variable has a type annotation such as list[str], extract element type
        if (!list_var_decl.is_null() && list_var_decl.contains("annotation"))
        {
          elem_type = get_elem_type_from_annotation(
            list_var_decl, converter_.get_type_handler());
        }
      }

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
        else if (!list_node.is_null() && list_node.contains("value"))
        {
          // Check if the value is a Subscript (such as d['a'])
          if (list_node["value"]["_type"] == "Subscript")
          {
            // For ESBMC_iter_0 = d['a'], get element type from dict's actual value
            if (list_node["value"]["value"]["_type"] == "Name")
            {
              std::string dict_var_name =
                list_node["value"]["value"]["id"].get<std::string>();

              // Find the dict's declaration
              nlohmann::json dict_node = json_utils::find_var_decl(
                dict_var_name,
                converter_.current_function_name(),
                converter_.ast());

              if (!dict_node.is_null() && dict_node.contains("value"))
              {
                const auto &dict_value = dict_node["value"];

                // Get the key being accessed (e.g., 'a' in d['a'])
                if (list_node["value"].contains("slice"))
                {
                  const auto &key_node = list_node["value"]["slice"];

                  // Handle constant string key
                  if (
                    key_node["_type"] == "Constant" &&
                    key_node.contains("value"))
                  {
                    std::string key = key_node["value"].get<std::string>();

                    // For dict literals, get the corresponding value
                    if (
                      dict_value["_type"] == "Dict" &&
                      dict_value.contains("keys") &&
                      dict_value.contains("values"))
                    {
                      const auto &keys = dict_value["keys"];
                      const auto &values = dict_value["values"];

                      // Find the matching key
                      for (size_t i = 0; i < keys.size(); i++)
                      {
                        if (
                          keys[i]["_type"] == "Constant" &&
                          keys[i]["value"].get<std::string>() == key)
                        {
                          // Found the value: now get its element type
                          const auto &list_value = values[i];

                          // Get the first element from the list using json_utils
                          nlohmann::json first_elem =
                            json_utils::get_list_element(list_value, 0);

                          if (!first_elem.is_null() && !first_elem.empty())
                          {
                            // Use type_handler to infer the element type
                            elem_type = converter_.get_type_handler().get_typet(
                              first_elem);
                          }
                          break;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          else if (
            list_node["value"].contains("elts") &&
            list_node["value"]["elts"].is_array() &&
            !list_node["value"]["elts"].empty())
          {
            // Get element type from first list element using json_utils
            nlohmann::json first_elem =
              json_utils::get_list_element(list_node["value"], 0);

            if (!first_elem.is_null() && !first_elem.empty())
              elem_type = converter_.get_type_handler().get_typet(first_elem);
          }
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

    // For array types, return pointer to element type instead of pointer to array
    // The dereference system doesn't support array types as target types
    // Callers will handle the conversion when needed (similar to single-char string handling)
    if (elem_type.is_array())
    {
      const array_typet &arr_type = to_array_type(elem_type);
      // Cast to pointer to element type (e.g., char* instead of char[2]*)
      typecast_exprt tc(obj_value, pointer_typet(arr_type.subtype()));
      return tc;
    }

    // For char* strings and None (_Bool*), the void* already contains the pointer value
    // For all other types, the void* contains a pointer to the value
    if (
      elem_type.is_pointer() && (elem_type.subtype() == char_type() ||
                                 elem_type.subtype() == bool_type()))
    {
      // String and None case: cast void* directly to the pointer type (no dereference needed)
      typecast_exprt tc(obj_value, elem_type);
      return tc;
    }
    else
    {
      // All other types: cast void* to pointer-to-type, then dereference
      typecast_exprt tc(obj_value, pointer_typet(elem_type));
      dereference_exprt deref(elem_type);
      deref.op0() = tc;
      return deref;
    }
  }

  // Handle static string indexing with safe null fallback
  if (array.type().is_array() && array.type().subtype() == char_type())
  {
    exprt idx = pos_expr;
    if (idx.type() != size_type())
      idx = typecast_exprt(idx, size_type());

    exprt bound = to_array_type(array.type()).size();
    if (bound.type() != size_type())
      bound = typecast_exprt(bound, size_type());

    exprt cond("<", bool_type());
    cond.copy_to_operands(idx, bound);

    index_exprt in_bounds(array, idx, char_type());
    if_exprt result(cond, in_bounds, gen_zero(char_type()));
    result.type() = char_type();
    return result;
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
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_eq");
  assert(list_eq_func_sym);

  // Convert member expressions into temporary symbols
  auto materialize_if_needed = [&](const exprt &e) -> exprt {
    if (e.id() == "member")
    {
      // Extract member expression to a temporary variable
      const member_exprt &member = to_member_expr(e);

      symbolt &temp_sym = converter_.create_tmp_symbol(
        list_value_, "$list_temp$", e.type(), exprt());

      code_declt temp_decl(symbol_expr(temp_sym));
      temp_decl.location() = converter_.get_location_from_decl(list_value_);
      converter_.add_instruction(temp_decl);

      code_assignt temp_assign(symbol_expr(temp_sym), member);
      temp_assign.location() = converter_.get_location_from_decl(list_value_);
      converter_.add_instruction(temp_assign);

      return symbol_expr(temp_sym);
    }
    return e;
  };

  const exprt converted_l1 = materialize_if_needed(l1);
  const exprt converted_l2 = materialize_if_needed(l2);

  const symbolt *lhs_symbol =
    converter_.find_symbol(converted_l1.identifier().as_string());
  const symbolt *rhs_symbol =
    converter_.find_symbol(converted_l2.identifier().as_string());
  assert(lhs_symbol);
  assert(rhs_symbol);

  // Compute list type_id for nested list detection
  const typet &list_type = l1.type();
  const std::string list_type_name =
    converter_.get_type_handler().type_to_string(list_type);
  constant_exprt list_type_id(size_type());
  list_type_id.set_value(integer2binary(
    std::hash<std::string>{}(list_type_name), config.ansi_c.address_width));

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
  list_eq_func_call.arguments().push_back(list_type_id); // list_type_id
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

  auto parse_size_from_symbol = [&](symbolt *size_var, BigInt &out) -> bool {
    if (
      size_var->value.is_code() || size_var->value.is_nil() ||
      !size_var->value.is_constant())
    {
      return false;
    }

    out = binary2integer(size_var->value.value().c_str(), true);
    return true;
  };

  // Get list size from lhs (e.g.: 3 * [1])
  if (lhs.type() != list_type)
  {
    if (lhs.is_symbol())
    {
      symbolt *size_var = converter_.find_symbol(
        to_symbol_expr(lhs).get_identifier().as_string());
      assert(size_var);
      symbolt *list_symbol =
        converter_.find_symbol(rhs.identifier().as_string());
      assert(list_symbol);
      if (!parse_size_from_symbol(size_var, list_size))
        return create_vla(list_value_, list_symbol, size_var, list_elem);
    }
    else if (lhs.is_constant())
      list_size = binary2integer(lhs.value().c_str(), true);

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

      if (!parse_size_from_symbol(size_var, list_size))
        return create_vla(list_value_, list_symbol, size_var, list_elem);
    }
    else if (rhs.is_constant())
      list_size = binary2integer(rhs.value().c_str(), true);
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
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_contains");
  assert(list_contains_func);

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

  // For void/char pointers from iteration, use stored type info from list
  exprt type_hash = symbol_expr(*item_info.elem_type_sym);
  exprt elem_size = item_info.elem_size;

  // Check if item is a void pointer (from loop iteration over strings)
  if (item_info.elem_symbol->type == pointer_typet(empty_typet()))
  {
    const std::string &list_name = list.identifier().as_string();
    auto type_map_it = list_type_map.find(list_name);

    if (type_map_it != list_type_map.end() && !type_map_it->second.empty())
    {
      // Look for a string array type (char array) in the list
      for (const auto &stored_entry : type_map_it->second)
      {
        const typet &stored_type = stored_entry.second;

        // Check if stored type is a char array (string)
        if (stored_type.is_array() && stored_type.subtype() == char_type())
        {
          // Use the stored string array type instead of void pointer type
          const type_handler type_handler_ = converter_.get_type_handler();
          const std::string stored_type_name =
            type_handler_.type_to_string(stored_type);

          constant_exprt stored_hash(size_type());
          stored_hash.set_value(integer2binary(
            std::hash<std::string>{}(stored_type_name),
            config.ansi_c.address_width));
          type_hash = stored_hash;

          // Use strlen for void* strings from iteration
          const symbolt *strlen_symbol =
            converter_.symbol_table().find_symbol("c:@F@strlen");
          if (strlen_symbol)
          {
            // Call strlen to get actual string length
            symbolt &strlen_result = converter_.create_tmp_symbol(
              list_value_,
              "$strlen_result$",
              size_type(),
              gen_zero(size_type()));
            code_declt strlen_decl(symbol_expr(strlen_result));
            strlen_decl.location() = item_info.location;
            converter_.add_instruction(strlen_decl);

            code_function_callt strlen_call;
            strlen_call.function() = symbol_expr(*strlen_symbol);
            strlen_call.lhs() = symbol_expr(strlen_result);
            strlen_call.arguments().push_back(
              symbol_expr(*item_info.elem_symbol));
            strlen_call.type() = size_type();
            strlen_call.location() = item_info.location;
            converter_.add_instruction(strlen_call);

            // Add 1 for null terminator: size = strlen(s) + 1
            exprt one_const = from_integer(1, strlen_result.type);
            elem_size = exprt("+", strlen_result.type);
            elem_size.copy_to_operands(symbol_expr(strlen_result), one_const);
          }

          break; // Found string array type, use it
        }
      }
    }
  }

  contains_call.arguments().push_back(type_hash);
  contains_call.arguments().push_back(elem_size);

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
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_extend");
  assert(extend_func_sym);

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

typet python_list::get_list_element_type(
  const std::string &list_id,
  size_t index)
{
  auto type_map_it = list_type_map.find(list_id);

  if (type_map_it == list_type_map.end() || type_map_it->second.empty())
    return typet();

  // If index is out of bounds, return the first element's type
  if (index >= type_map_it->second.size())
    index = 0;

  return type_map_it->second[index].second;
}

exprt python_list::handle_comprehension(const nlohmann::json &element)
{
  if (!element.contains("generators") || element["generators"].empty())
  {
    throw std::runtime_error(
      "Comprehension expression missing generators clause");
  }

  const auto &generator = element["generators"][0];
  const auto &elt = element["elt"];
  const auto &target = generator["target"];
  const auto &iter = generator["iter"];

  locationt location = converter_.get_location_from_decl(element);
  typet list_type = converter_.get_type_handler().get_list_type();

  // 1. Create result list
  symbolt &result_list = create_list();
  const std::string &result_list_id = result_list.id.as_string();

  // 2. Get iterable expression
  exprt iterable_expr = converter_.get_expr(iter);

  // 3. Create loop variable
  std::string loop_var_name = target["id"].get<std::string>();
  symbol_id loop_var_sid = converter_.create_symbol_id();
  loop_var_sid.set_object(loop_var_name);

  // Infer loop variable type from iterable
  typet loop_var_type;
  if (iterable_expr.type().is_array())
    loop_var_type = iterable_expr.type().subtype();
  else if (iterable_expr.type().is_pointer())
    loop_var_type = iterable_expr.type().subtype();
  else if (iterable_expr.type() == list_type)
    loop_var_type = any_type();
  else
    loop_var_type = any_type();

  symbolt loop_var_symbol = converter_.create_symbol(
    location.get_file().as_string(),
    loop_var_name,
    loop_var_sid.to_string(),
    location,
    loop_var_type);
  loop_var_symbol.lvalue = true;
  loop_var_symbol.file_local = true;
  loop_var_symbol.is_extern = false;
  symbolt *loop_var =
    converter_.symbol_table().move_symbol_to_context(loop_var_symbol);

  // 4. Create index variable
  symbolt &index_var = converter_.create_tmp_symbol(
    element, "comp_i", size_type(), gen_zero(size_type()));

  code_declt index_decl(symbol_expr(index_var));
  index_decl.location() = location;
  converter_.add_instruction(index_decl);

  // 5. Get length of iterable
  exprt length_expr;
  if (iterable_expr.type() == list_type)
  {
    const symbolt *size_func =
      converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
    if (!size_func)
      throw std::runtime_error("__ESBMC_list_size not found in symbol table");

    symbolt &length_var = converter_.create_tmp_symbol(
      element, "comp_len", size_type(), gen_zero(size_type()));

    code_declt length_decl(symbol_expr(length_var));
    length_decl.location() = location;
    converter_.add_instruction(length_decl);

    code_function_callt size_call;
    size_call.function() = symbol_expr(*size_func);
    size_call.arguments().push_back(
      iterable_expr.type().is_pointer() ? iterable_expr
                                        : address_of_exprt(iterable_expr));
    size_call.lhs() = symbol_expr(length_var);
    size_call.type() = size_type();
    size_call.location() = location;
    converter_.add_instruction(size_call);

    length_expr = symbol_expr(length_var);
  }
  else if (iterable_expr.type().is_array())
  {
    const array_typet &arr_type = to_array_type(iterable_expr.type());
    length_expr = arr_type.size();
  }
  else if (iterable_expr.type().is_pointer())
  {
    const symbolt *strlen_func =
      converter_.symbol_table().find_symbol("c:@F@strlen");
    if (strlen_func)
    {
      symbolt &length_var = converter_.create_tmp_symbol(
        element, "comp_len", size_type(), gen_zero(size_type()));

      code_declt length_decl(symbol_expr(length_var));
      length_decl.location() = location;
      converter_.add_instruction(length_decl);

      code_function_callt strlen_call;
      strlen_call.function() = symbol_expr(*strlen_func);
      strlen_call.arguments().push_back(iterable_expr);
      strlen_call.lhs() = symbol_expr(length_var);
      strlen_call.type() = size_type();
      strlen_call.location() = location;
      converter_.add_instruction(strlen_call);

      length_expr = symbol_expr(length_var);
    }
    else
    {
      throw std::runtime_error("strlen not found for string iteration");
    }
  }
  else
  {
    throw std::runtime_error(
      "Unsupported iterable type in comprehension: " +
      iterable_expr.type().id_string());
  }

  // 6. Build while loop body
  code_blockt loop_body;

  // Get current element: loop_var = iterable[i]
  exprt current_element;
  if (iterable_expr.type() == list_type)
  {
    // For lists, use list_at
    current_element =
      build_list_at_call(iterable_expr, symbol_expr(index_var), element);

    // Dereference the returned pointer to get the value
    member_exprt obj_value(
      current_element, "value", pointer_typet(empty_typet()));
    {
      exprt &base = obj_value.struct_op();
      exprt deref("dereference");
      deref.type() = base.type().subtype();
      deref.move_to_operands(base);
      base.swap(deref);
    }

    typecast_exprt tc(obj_value, pointer_typet(loop_var_type));
    dereference_exprt final_deref(loop_var_type);
    final_deref.op0() = tc;
    current_element = final_deref;
  }
  else if (iterable_expr.type().is_array() || iterable_expr.type().is_pointer())
  {
    // For arrays/strings, use direct indexing
    index_exprt array_index(
      iterable_expr, symbol_expr(index_var), loop_var_type);
    current_element = array_index;
  }
  else
  {
    throw std::runtime_error(
      "Cannot index into type: " + iterable_expr.type().id_string());
  }

  code_assignt loop_var_assign(symbol_expr(*loop_var), current_element);
  loop_var_assign.location() = location;
  loop_body.copy_to_operands(loop_var_assign);

  // 7. Handle filter conditions (if present)
  code_blockt conditional_block;

  // 8. Evaluate element expression and append to result
  // Switch context to loop body for all operations
  code_blockt *saved_block = converter_.current_block;
  converter_.current_block = &loop_body;

  // Evaluate element expression - temporaries go to loop_body
  exprt element_expr = converter_.get_expr(elt);

  // Build push call - temporaries also go to loop_body
  exprt push_call = build_push_list_call(result_list, element, element_expr);
  loop_body.copy_to_operands(push_call);

  // Restore context
  converter_.current_block = saved_block;

  // Update type map
  list_type_map[result_list_id].push_back(
    std::make_pair(element_expr.identifier().as_string(), element_expr.type()));

  // If we had filter conditions, wrap append in if statement
  if (generator.contains("ifs") && !generator["ifs"].empty())
  {
    exprt combined_condition = gen_boolean(true);
    for (const auto &if_clause : generator["ifs"])
    {
      exprt if_expr = converter_.get_expr(if_clause);
      if (combined_condition.is_true())
        combined_condition = if_expr;
      else
      {
        exprt and_expr("and", bool_type());
        and_expr.copy_to_operands(combined_condition, if_expr);
        combined_condition = and_expr;
      }
    }

    codet if_stmt;
    if_stmt.set_statement("ifthenelse");
    if_stmt.copy_to_operands(combined_condition, conditional_block);
    if_stmt.location() = location;
    loop_body.copy_to_operands(if_stmt);
  }

  // 9. Increment index: i = i + 1
  exprt increment("+", size_type());
  increment.copy_to_operands(symbol_expr(index_var), gen_one(size_type()));
  code_assignt index_increment(symbol_expr(index_var), increment);
  index_increment.location() = location;
  loop_body.copy_to_operands(index_increment);

  // 10. Create while loop: while (i < length)
  exprt loop_condition("<", bool_type());
  loop_condition.copy_to_operands(symbol_expr(index_var), length_expr);

  codet while_stmt;
  while_stmt.set_statement("while");
  while_stmt.copy_to_operands(loop_condition, loop_body);
  while_stmt.location() = location;
  converter_.add_instruction(while_stmt);

  return symbol_expr(result_list);
}
