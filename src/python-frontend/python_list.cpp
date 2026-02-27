#include <python-frontend/python_list.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_exception_handler.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/string_builder.h>
#include <util/expr.h>
#include <util/type.h>
#include <util/symbol.h>
#include <util/expr_util.h>
#include <util/arith_tools.h>
#include <python-frontend/python_frontend_limits.h>
#include <util/std_code.h>
#include <util/std_expr.h>
#include <util/mp_arith.h>
#include <util/python_types.h>
#include <util/symbolic_types.h>
#include <util/config.h>
#include <string>
#include <functional>

// Default depth for list comparison if option not set
static const int DEFAULT_LIST_COMPARE_DEPTH = 4;

static int get_list_compare_depth()
{
  std::string opt_value =
    config.options.get_option("python-list-compare-depth");
  if (!opt_value.empty())
  {
    try
    {
      int depth = std::stoi(opt_value);
      if (depth > 0)
        return depth;
    }
    catch (...)
    {
    }
  }
  return DEFAULT_LIST_COMPARE_DEPTH;
}

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

  // Copy type info from source list if it's a symbol
  auto copy_type_info_from_expr = [&](const exprt &src_list) {
    if (!src_list.is_symbol())
      return;
    copy_type_map_entries(src_list.identifier().as_string(), dst_id);
  };

  copy_type_info_from_expr(lhs);
  copy_type_info_from_expr(rhs);

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

  locationt location = converter_.get_location_from_decl(element);

  // Check if index is already a constant non-negative value
  if (index.is_constant() && index.type().is_signedbv())
  {
    BigInt idx_value;
    if (!to_integer(index, idx_value) && idx_value >= 0)
    {
      // Index is constant and non-negative, use directly
      side_effect_expr_function_callt list_at_call;
      list_at_call.function() = symbol_expr(*list_at_func_sym);
      list_at_call.arguments().push_back(
        list.type().is_pointer() ? list : address_of_exprt(list));
      list_at_call.arguments().push_back(typecast_exprt(index, size_type()));
      list_at_call.type() = obj_type;
      list_at_call.location() = location;
      return list_at_call;
    }
  }

  // Get list size
  const symbolt *size_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
  assert(size_func && "list_size function not found");

  symbolt &size_var = converter_.create_tmp_symbol(
    element, "$list_size$", size_type(), gen_zero(size_type()));
  code_declt size_decl(symbol_expr(size_var));
  size_decl.location() = location;
  converter_.add_instruction(size_decl);

  code_function_callt size_call;
  size_call.function() = symbol_expr(*size_func);
  size_call.arguments().push_back(
    list.type().is_pointer() ? list : address_of_exprt(list));
  size_call.lhs() = symbol_expr(size_var);
  size_call.type() = size_type();
  size_call.location() = location;
  converter_.add_instruction(size_call);

  // Convert index to size_t for comparison and arithmetic
  exprt index_as_size = typecast_exprt(index, size_type());

  // Create: actual_index = (index < 0) ? (size + index) : index
  exprt is_negative("<", bool_type());
  is_negative.copy_to_operands(index, gen_zero(index.type()));

  // For negative: size + index (since index is negative, this is size - abs(index))
  exprt positive_index("+", size_type());
  positive_index.copy_to_operands(symbol_expr(size_var), index_as_size);

  // Choose between positive conversion or original
  if_exprt converted_index(is_negative, positive_index, index_as_size);
  converted_index.type() = size_type();

  // Use the converted expression directly in the call
  side_effect_expr_function_callt list_at_call;
  list_at_call.function() = symbol_expr(*list_at_func_sym);
  list_at_call.arguments().push_back(
    list.type().is_pointer() ? list : address_of_exprt(list));
  list_at_call.arguments().push_back(converted_index);
  list_at_call.type() = obj_type;
  list_at_call.location() = location;

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
  {
    // Whitespace split: split on any whitespace and collapse runs.
    auto is_space = [](char c) {
      return std::isspace(static_cast<unsigned char>(c)) != 0;
    };

    if (count == 0)
    {
      size_t first = 0;
      while (first < input.size() && is_space(input[first]))
        ++first;

      if (first == input.size())
      {
        nlohmann::json list_node;
        list_node["_type"] = "List";
        list_node["elts"] = nlohmann::json::array();
        converter.copy_location_fields_from_decl(call_node, list_node);
        python_list list(converter, list_node);
        return list.get();
      }

      size_t last = input.size();
      while (last > first && is_space(input[last - 1]))
        --last;

      nlohmann::json list_node;
      list_node["_type"] = "List";
      list_node["elts"] = nlohmann::json::array();
      converter.copy_location_fields_from_decl(call_node, list_node);

      nlohmann::json elem;
      elem["_type"] = "Constant";
      elem["value"] = input.substr(first, last - first);
      converter.copy_location_fields_from_decl(call_node, elem);
      list_node["elts"].push_back(elem);

      python_list list(converter, list_node);
      return list.get();
    }

    std::vector<std::string> parts;
    size_t i = 0;
    const size_t n = input.size();

    auto skip_ws = [&](size_t &idx) {
      while (idx < n && is_space(input[idx]))
        ++idx;
    };

    auto scan_token = [&](size_t &idx) {
      while (idx < n && !is_space(input[idx]))
        ++idx;
    };

    skip_ws(i);
    if (i == n)
    {
      nlohmann::json list_node;
      list_node["_type"] = "List";
      list_node["elts"] = nlohmann::json::array();
      converter.copy_location_fields_from_decl(call_node, list_node);
      python_list list(converter, list_node);
      return list.get();
    }

    long long remaining = count;
    while (i < n)
    {
      size_t start = i;
      scan_token(i);
      parts.push_back(input.substr(start, i - start));

      if (count >= 0 && remaining == 1)
      {
        skip_ws(i);
        if (i < n)
          parts.push_back(input.substr(i));
        break;
      }

      if (count >= 0)
        --remaining;

      skip_ws(i);
      if (i >= n)
        break;
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

exprt python_list::build_split_list(
  python_converter &converter,
  const nlohmann::json &call_node,
  const exprt &input_expr,
  const std::string &separator,
  long long count)
{
  // For symbolic strings, we create a runtime call to __python_str_split
  // This function will handle the splitting at runtime with symbolic constraints

  locationt location = converter.get_location_from_decl(call_node);

  // Create function symbol for __python_str_split if it doesn't exist
  const std::string func_name = "c:@F@__python_str_split";
  const symbolt *func_symbol = converter.symbol_table().find_symbol(func_name);

  if (!func_symbol)
  {
    // Create function type: PyListObject* __python_str_split(char* str, char* sep, int maxsplit)
    code_typet func_type;
    func_type.return_type() = converter.get_type_handler().get_list_type();

    code_typet::argumentt str_arg;
    str_arg.type() = pointer_typet(char_type());
    func_type.arguments().push_back(str_arg);

    code_typet::argumentt sep_arg;
    sep_arg.type() = pointer_typet(char_type());
    func_type.arguments().push_back(sep_arg);

    code_typet::argumentt count_arg;
    count_arg.type() = long_long_int_type();
    func_type.arguments().push_back(count_arg);

    symbolt new_symbol;
    new_symbol.name = func_name;
    new_symbol.id = func_name;
    new_symbol.type = func_type;
    new_symbol.mode = "C";
    new_symbol.module = "python";
    new_symbol.location = location;
    new_symbol.is_extern = true;

    converter.add_symbol(new_symbol);
    func_symbol = converter.symbol_table().find_symbol(func_name);
  }

  // Build arguments for the call
  exprt::operandst args;

  // Argument 1: input string (ensure it's a pointer)
  exprt str_arg = input_expr;
  if (str_arg.type().is_array())
  {
    // Get address of first element
    str_arg = converter.get_string_handler().get_array_base_address(str_arg);
  }
  args.push_back(str_arg);

  // Argument 2: separator string
  std::string sep_to_use = separator.empty() ? "" : separator;
  exprt sep_expr =
    converter.get_string_builder().build_string_literal(sep_to_use);
  if (sep_expr.type().is_array())
  {
    sep_expr = converter.get_string_handler().get_array_base_address(sep_expr);
  }
  args.push_back(sep_expr);

  // Argument 3: maxsplit count
  exprt count_expr = from_integer(count, long_long_int_type());
  args.push_back(count_expr);

  // Create a temp list symbol to hold the split result.
  const typet list_type = converter.get_type_handler().get_list_type();
  symbolt &split_list =
    converter.create_tmp_symbol(call_node, "$split_list$", list_type, exprt());
  code_declt split_decl(symbol_expr(split_list));
  split_decl.location() = location;
  converter.add_instruction(split_decl);

  // Emit the function call with lhs so the list has a stable identifier.
  code_function_callt split_call;
  split_call.function() = symbol_expr(*func_symbol);
  split_call.arguments() = args;
  split_call.lhs() = symbol_expr(split_list);
  split_call.type() = list_type;
  split_call.location() = location;
  converter.add_instruction(split_call);

  // Record element type as string to ensure correct comparisons on parts[i].
  typet elem_type = converter.get_type_handler().build_array(char_type(), 0);
  list_type_map[split_list.id.as_string()].push_back(
    std::make_pair(std::string(), elem_type));

  return symbol_expr(split_list);
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

    // Determine step value (default 1)
    bool has_step =
      slice_node.contains("step") && !slice_node["step"].is_null();
    long long step_val = 1;
    if (has_step)
    {
      const auto &step_node = slice_node["step"];
      if (step_node["_type"] == "UnaryOp" && step_node["op"]["_type"] == "USub")
      {
        step_val =
          -(long long)step_node["operand"]["value"].get<std::int64_t>();
      }
      else if (step_node["_type"] == "Constant")
      {
        step_val = step_node["value"].get<std::int64_t>();
      }
    }
    bool negative_step = (step_val < 0);

    // For pointer types (function parameters), delegate to __python_str_slice
    // which uses __ESBMC_alloca and survives function returns.
    if (array.type().is_pointer() && array.type().subtype() == char_type())
    {
      std::string slice_func_id = "c:@F@__python_str_slice";
      symbolt *slice_func =
        converter_.symbol_table().find_symbol(slice_func_id);
      if (!slice_func)
      {
        // Create the function symbol if it doesn't exist
        symbolt new_symbol;
        new_symbol.name = "__python_str_slice";
        new_symbol.id = slice_func_id;
        new_symbol.mode = "C";
        new_symbol.is_extern = true;

        // char* __python_str_slice(const char*, long long, long long, long long)
        code_typet slice_type;
        typet char_ptr = gen_pointer_type(char_type());
        typet ll_type = signedbv_typet(64);
        slice_type.return_type() = char_ptr;
        slice_type.arguments().push_back(code_typet::argumentt(char_ptr));
        slice_type.arguments().push_back(code_typet::argumentt(ll_type));
        slice_type.arguments().push_back(code_typet::argumentt(ll_type));
        slice_type.arguments().push_back(code_typet::argumentt(ll_type));
        new_symbol.type = slice_type;

        converter_.symbol_table().add(new_symbol);
        slice_func = converter_.symbol_table().find_symbol(slice_func_id);
      }

      // Extract start/end bounds from slice node
      auto get_bound_expr =
        [&](const std::string &name, long long default_val) -> exprt {
        if (!slice_node.contains(name) || slice_node[name].is_null())
          return from_integer(default_val, signedbv_typet(64));

        const auto &bound = slice_node[name];
        if (bound["_type"] == "UnaryOp" && bound["op"]["_type"] == "USub")
        {
          exprt abs_value = converter_.get_expr(bound["operand"]);
          exprt neg("-", signedbv_typet(64));
          neg.copy_to_operands(
            from_integer(0, signedbv_typet(64)),
            typecast_exprt(abs_value, signedbv_typet(64)));
          return neg;
        }
        exprt e = converter_.get_expr(bound);
        e = remove_function_calls_recursive(e, slice_node);
        return typecast_exprt(e, signedbv_typet(64));
      };

      // Defaults: for positive step start=0,end=MAX; for negative step start=MAX,end=MIN
      // We use large sentinel values; __python_str_slice clamps them
      long long start_default = negative_step ? 999999 : 0;
      long long end_default = negative_step ? -999999 : 999999;

      exprt start_expr = get_bound_expr("lower", start_default);
      exprt end_expr = get_bound_expr("upper", end_default);
      exprt step_expr = from_integer(step_val, signedbv_typet(64));

      // Call __python_str_slice(s, start, end, step) as side-effect expression
      side_effect_expr_function_callt slice_call;
      slice_call.function() = symbol_expr(*slice_func);
      slice_call.arguments().push_back(array);
      slice_call.arguments().push_back(start_expr);
      slice_call.arguments().push_back(end_expr);
      slice_call.arguments().push_back(step_expr);
      slice_call.location() = location;
      slice_call.type() = pointer_typet(char_type());

      return slice_call;
    }

    // For array types (local string literals), generate inline loop
    // Determine element type and logical length
    typet elem_type;
    exprt array_len;
    exprt logical_len;

    {
      const array_typet &src_type = to_array_type(array.type());
      elem_type = src_type.subtype();
      array_len = src_type.size();
      // For char arrays (strings), exclude null terminator from logical length
      logical_len = (elem_type == char_type())
                      ? minus_exprt(array_len, gen_one(size_type()))
                      : array_len;
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
        // Clamp to 0 when abs_value > logical_len (avoids unsigned underflow)
        exprt overflow(">", bool_type());
        overflow.copy_to_operands(abs_value, logical_len);
        exprt converted = minus_exprt(logical_len, abs_value);
        return if_exprt(overflow, gen_zero(size_type()), converted);
      }

      exprt e = converter_.get_expr(bound);
      return remove_function_calls_recursive(e, slice_node);
    };

    // Process bounds: defaults depend on step direction
    exprt lower_expr, upper_expr;
    if (negative_step)
    {
      // For negative step: default lower = len-1, default upper = -1 (before 0)
      lower_expr =
        process_bound("lower", minus_exprt(logical_len, gen_one(size_type())));
    }
    else
    {
      lower_expr = process_bound("lower", gen_zero(size_type()));
    }

    // Upper bound
    if (!negative_step)
      upper_expr = process_bound("upper", logical_len);

    // Clamp bounds to [0, logical_len] to match Python semantics.
    if (!negative_step)
    {
      // lower = max(0, min(lower, logical_len))
      exprt lower_ge_len(">=", bool_type());
      lower_ge_len.copy_to_operands(lower_expr, logical_len);
      lower_expr = if_exprt(lower_ge_len, logical_len, lower_expr);
      lower_expr.type() = size_type();

      // upper = max(0, min(upper, logical_len))
      exprt upper_ge_len(">=", bool_type());
      upper_ge_len.copy_to_operands(upper_expr, logical_len);
      upper_expr = if_exprt(upper_ge_len, logical_len, upper_expr);
      upper_expr.type() = size_type();
    }

    // Calculate slice length
    exprt slice_len;
    if (negative_step)
    {
      // For [::-1]: length = lower + 1 (e.g., lower=len-1 → length=len)
      if (
        (!slice_node.contains("lower") || slice_node["lower"].is_null()) &&
        (!slice_node.contains("upper") || slice_node["upper"].is_null()))
      {
        slice_len = logical_len;
      }
      else
      {
        slice_len = plus_exprt(lower_expr, gen_one(size_type()));
      }
    }
    else if (step_val != 1)
    {
      // For step > 1: length = ceil((upper - lower) / step)
      exprt range = minus_exprt(upper_expr, lower_expr);
      exprt step_const = from_integer(step_val, size_type());
      exprt step_minus_one = from_integer(step_val - 1, size_type());
      slice_len = div_exprt(plus_exprt(range, step_minus_one), step_const);
    }
    else
    {
      slice_len = minus_exprt(upper_expr, lower_expr);
    }

    // Create result array type with extra space for null terminator
    plus_exprt result_size(slice_len, gen_one(size_type()));
    array_typet result_type(elem_type, result_size);

    // Create temporary for sliced array
    symbolt &result = converter_.create_tmp_symbol(
      slice_node, "$array_slice$", result_type, exprt());

    code_declt result_decl(symbol_expr(result));
    result_decl.location() = location;
    converter_.add_instruction(result_decl);

    // Create loop: for i = 0; i < slice_len; i++
    symbolt &idx = converter_.create_tmp_symbol(
      slice_node, "$i$", size_type(), gen_zero(size_type()));
    code_assignt idx_init(symbol_expr(idx), gen_zero(size_type()));
    converter_.add_instruction(idx_init);

    exprt cond("<", bool_type());
    cond.copy_to_operands(symbol_expr(idx), slice_len);

    code_blockt body;

    // Compute source index based on step direction
    exprt src_idx;
    if (negative_step)
    {
      // result[i] = array[lower - i] (for step=-1)
      // For other negative steps: result[i] = array[lower - i * |step|]
      if (step_val == -1)
        src_idx = minus_exprt(lower_expr, symbol_expr(idx));
      else
      {
        exprt abs_step = from_integer(-step_val, size_type());
        src_idx =
          minus_exprt(lower_expr, mult_exprt(symbol_expr(idx), abs_step));
      }
    }
    else if (step_val != 1)
    {
      // result[i] = array[lower + i * step]
      exprt step_const = from_integer(step_val, size_type());
      src_idx =
        plus_exprt(lower_expr, mult_exprt(symbol_expr(idx), step_const));
    }
    else
    {
      // result[i] = array[lower + i]
      src_idx = plus_exprt(lower_expr, symbol_expr(idx));
    }

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
    {
      const auto &bound_node = slice_node[bound_name];
      exprt bound_expr = converter_.get_expr(bound_node);

      // Resolve negative index: convert to (list_size + negative_bound)
      bool is_negative = false;

      // UnaryOp USub in the AST (e.g. -1 represented as USub(1))
      if (
        bound_node.contains("_type") && bound_node["_type"] == "UnaryOp" &&
        bound_node.contains("op") && bound_node["op"]["_type"] == "USub")
      {
        is_negative = true;
      }

      if (is_negative)
      {
        // Compute: list_size + bound_expr  (bound_expr is negative)
        const symbolt *size_func =
          converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
        if (!size_func)
          throw std::runtime_error(
            "__ESBMC_list_size not found in symbol table");

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

        exprt bound_as_size = typecast_exprt(bound_expr, size_type());
        exprt resolved("+", size_type());
        resolved.copy_to_operands(symbol_expr(size_sym), bound_as_size);
        return resolved;
      }

      return bound_expr;
    }

    if (is_upper)
    {
      const symbolt *size_func =
        converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
      if (!size_func)
        throw std::runtime_error(
          "__ESBMC_list_size not found in symbol table");

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
  exprt upper_expr = get_list_bound("upper", true);

  // Clamp upper bound to the current list size to match Python slicing
  // semantics (e.g., l[0:100] on a 5-element list).
  {
    const symbolt *size_func =
      converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
    if (!size_func)
      throw std::runtime_error(
        "__ESBMC_list_size not found in symbol table");

    side_effect_expr_function_callt size_call;
    size_call.function() = symbol_expr(*size_func);
    if (array.type().is_pointer())
      size_call.arguments().push_back(array);
    else
      size_call.arguments().push_back(address_of_exprt(array));
    size_call.type() = size_type();
    size_call.location() = location;

    symbolt &size_sym = converter_.create_tmp_symbol(
      list_value_, "$slice_size$", size_type(), exprt());
    code_declt size_decl(symbol_expr(size_sym));
    size_decl.copy_to_operands(size_call);
    converter_.add_instruction(size_decl);

    exprt upper_ge_size(">=", bool_type());
    upper_ge_size.copy_to_operands(upper_expr, symbol_expr(size_sym));
    upper_expr = if_exprt(upper_ge_size, symbol_expr(size_sym), upper_expr);
    upper_expr.type() = size_type();
  }

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
    nlohmann::json list_node;
    if (
      list_value_.contains("value") && list_value_["value"].is_object() &&
      list_value_["value"].contains("id") &&
      list_value_["value"]["id"].is_string())
    {
      list_node = json_utils::get_var_value(
        list_value_["value"]["id"],
        converter_.current_function_name(),
        converter_.ast());
    }

    const size_t lower_bound = slice_node["lower"]["value"].get<size_t>();
    const size_t upper_bound = slice_node["upper"]["value"].get<size_t>();

    // Only update type map for actual lists (not strings or other types)
    if (
      !list_node.is_null() && list_node.contains("value") &&
      list_node["value"].contains("elts") &&
      list_node["value"]["elts"].is_array())
    {
      const size_t elts_size = list_node["value"]["elts"].size();
      const size_t begin = std::min(lower_bound, elts_size);
      const size_t end = std::min(upper_bound, elts_size);

      for (size_t i = begin; i < end; ++i)
      {
        const exprt element =
          converter_.get_expr(list_node["value"]["elts"][i]);
        list_type_map[sliced_list.id.as_string()].push_back(
          std::make_pair(element.identifier().as_string(), element.type()));
      }
    }
  }

  // This handles cases where one or both bounds are null or negative (e.g.
  // numbers[:-1]), or where the source is a function parameter rather than a
  // locally constructed list, so list_type_map has no entries for it.
  const std::string &sliced_id = sliced_list.id.as_string();
  if (list_type_map[sliced_id].empty())
  {
    if (
      list_type_map[sliced_id].empty() && list_value_.contains("value") &&
      list_value_["value"].contains("id"))
    {
      const std::string &param_name =
        list_value_["value"]["id"].get<std::string>();

      nlohmann::json param_node = json_utils::find_var_decl(
        param_name, converter_.current_function_name(), converter_.ast());

      // Only use annotation fallback for function parameters (arg nodes),
      // not for local variable declarations whose element type should come
      // from the type map populated during list construction.
      if (!param_node.is_null() && param_node["_type"] == "arg")
      {
        const typet elem_type = get_elem_type_from_annotation(
          param_node, converter_.get_type_handler());

        if (elem_type != typet())
        {
          list_type_map[sliced_id].push_back(
            std::make_pair(std::string(), elem_type));
        }
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
  if (
    list_value_.contains("value") && list_value_["value"].is_object() &&
    list_value_["value"].contains("id"))
  {
    list_node = json_utils::find_var_decl(
      list_value_["value"]["id"],
      converter_.current_function_name(),
      converter_.ast());
  }

  exprt pos_expr = converter_.get_expr(slice_node);
  pos_expr = converter_.unwrap_optional_if_needed(pos_expr);
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
      // Compute index for compile-time type lookup only.
      // Do NOT overwrite pos_expr: the list may have been mutated
      // (append/insert/extend), so we must resolve the negative index
      // at runtime via build_list_at_call using __ESBMC_list_size.
      index = slice_node["operand"]["value"].get<size_t>();
      index = list_node["value"]["elts"].size() - index;
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
    if (list_node.contains("_type") && list_node["_type"] == "arg")
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
        if (
          list_value_.contains("value") && list_value_["value"].is_object() &&
          list_value_["value"].contains("id") &&
          list_value_["value"]["id"].is_string())
        {
          const nlohmann::json list_value_node = json_utils::get_var_value(
            list_value_["value"]["id"],
            converter_.current_function_name(),
            converter_.ast());

          elem_type = get_elem_type_from_annotation(
            list_value_node, converter_.get_type_handler());
        }
      }
      else
      {
        size_t type_index =
          (!list_node.is_null() && list_node.contains("value") &&
           list_node["value"].is_object() &&
           list_node["value"].contains("_type") &&
           list_node["value"]["_type"] == "BinOp")
            ? 0
            : index;

        try
        {
          elem_type = list_type_map[list_name].at(type_index).second;
        }
        catch (const std::out_of_range &)
        {
          // Do not emit a frontend conversion error for constant OOB indices.
          // The runtime list access model can raise IndexError, which is
          // observable by Python try/except code.

          // Use the known element type for homogeneous dynamic lists.
          if (!list_type_map[list_name].empty())
          {
            elem_type = list_type_map[list_name].back().second;
          }
          else if (
            list_value_.contains("value") &&
            list_value_["value"].contains("id") &&
            list_value_["value"]["id"].is_string())
          {
            // Try annotation fallback for dynamic lists or function parameters
            const nlohmann::json list_value_node = json_utils::get_var_value(
              list_value_["value"]["id"],
              converter_.current_function_name(),
              converter_.ast());

            elem_type = get_elem_type_from_annotation(
              list_value_node, converter_.get_type_handler());
          }

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
          if (
            list_node["value"].contains("_type") &&
            list_node["value"]["_type"] == "Subscript")
          {
            // For ESBMC_iter_0 = d['a'], get element type from dict's actual value
            if (
              list_node["value"].contains("value") &&
              list_node["value"]["value"].is_object() &&
              list_node["value"]["value"].contains("_type") &&
              list_node["value"]["value"]["_type"] == "Name")
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
      const bool has_const_index =
        (slice_node["_type"] == "Constant" && slice_node.contains("value")) ||
        (slice_node["_type"] == "UnaryOp" && slice_node.contains("op") &&
         slice_node["op"]["_type"] == "USub" &&
         slice_node.contains("operand") &&
         slice_node["operand"]["_type"] == "Constant");
      if (has_const_index)
      {
        const locationt l = converter_.get_location_from_decl(list_value_);
        throw std::runtime_error(
          "List out of bounds at " + l.get_file().as_string() +
          " line: " + l.get_line().as_string());
      }

      // Keep historical frontend diagnostic for literal lists with
      // compile-time constant OOB indexing (including nested accesses).
      if (
        array.is_symbol() && !list_node.is_null() &&
        list_node.contains("value") && list_node["value"].is_object() &&
        list_node["value"].contains("elts") &&
        list_node["value"]["elts"].is_array())
      {
        bool has_const_index = false;
        bool negative_index = false;
        size_t index_abs = 0;

        if (slice_node["_type"] == "Constant" && slice_node.contains("value"))
        {
          has_const_index = true;
          index_abs = slice_node["value"].get<size_t>();
        }
        else if (
          slice_node["_type"] == "UnaryOp" && slice_node.contains("op") &&
          slice_node["op"]["_type"] == "USub" &&
          slice_node.contains("operand") &&
          slice_node["operand"]["_type"] == "Constant")
        {
          has_const_index = true;
          negative_index = true;
          index_abs = slice_node["operand"]["value"].get<size_t>();
        }

        if (has_const_index)
        {
          const size_t list_size = list_node["value"]["elts"].size();
          const bool out_of_bounds =
            (!negative_index && index_abs >= list_size) ||
            (negative_index && index_abs > list_size);
          if (out_of_bounds)
          {
            const locationt l = converter_.get_location_from_decl(list_value_);
            throw std::runtime_error(
              "List out of bounds at " + l.get_file().as_string() +
              " line: " + l.get_line().as_string());
          }
        }
      }

      throw std::runtime_error(
        "Invalid list access: could not resolve position or element type");
    }

    // Preserve historical frontend OOB diagnostics only when we can prove we
    // are indexing a stable list literal (not a reassigned/mutated/derived
    // list). Using the type map alone is too broad and causes false positives.
    // Emit an IndexError raise instead of a C++ exception so Python
    // try/except(IndexError) can observe the error.
    if (array.is_symbol())
    {
      const std::string &list_name = array.identifier().as_string();
      auto it = list_type_map.find(list_name);
      if (it != list_type_map.end() && !it->second.empty())
      {
        bool has_const_index = false;
        bool negative_index = false;
        size_t index_abs = 0;

        if (slice_node["_type"] == "Constant" && slice_node.contains("value"))
        {
          has_const_index = true;
          index_abs = slice_node["value"].get<size_t>();
        }
        else if (
          slice_node["_type"] == "UnaryOp" && slice_node.contains("op") &&
          slice_node["op"]["_type"] == "USub" &&
          slice_node.contains("operand") &&
          slice_node["operand"]["_type"] == "Constant")
        {
          has_const_index = true;
          negative_index = true;
          index_abs = slice_node["operand"]["value"].get<size_t>();
        }

        const bool is_slice_derived_var =
          !list_node.is_null() && list_node.contains("value") &&
          list_node["value"].contains("_type") &&
          list_node["value"]["_type"] == "Subscript";

        bool is_stable_list_literal = false;
        if (
          !list_node.is_null() && list_node.contains("value") &&
          list_node["value"].contains("_type") &&
          list_node["value"]["_type"] == "List" &&
          list_node["value"].contains("elts") &&
          list_node["value"]["elts"].is_array())
        {
          // If sizes diverge, the symbol was likely reassigned/mutated.
          is_stable_list_literal =
            it->second.size() == list_node["value"]["elts"].size();
        }

        if (has_const_index && !is_slice_derived_var && is_stable_list_literal)
        {
          const size_t known_size = it->second.size();
          const bool oob = negative_index ? (index_abs > known_size)
                                          : (index_abs >= known_size);
          if (oob)
          {
            exprt raise =
              converter_.get_exception_handler().gen_exception_raise(
                "IndexError", "list index out of range");
            codet throw_code("expression");
            throw_code.operands().push_back(raise);
            converter_.add_instruction(throw_code);
            // Short-circuit: the raise makes the rest unreachable.
            return gen_zero(elem_type);
          }
        }
      }
    }

    // Build list access and cast result
    exprt list_at_call = build_list_at_call(array, pos_expr, list_value_);

    // Extract and dereference PyObject value
    return extract_pyobject_value(list_at_call, elem_type);
  }

  // Handle static string indexing with IndexError on out-of-bounds
  if (array.type().is_array() && array.type().subtype() == char_type())
  {
    exprt idx = pos_expr;
    if (idx.type() != size_type())
      idx = typecast_exprt(idx, size_type());

    // Logical string length excludes the null terminator
    exprt array_size = to_array_type(array.type()).size();
    if (array_size.type() != size_type())
      array_size = typecast_exprt(array_size, size_type());
    exprt one = from_integer(1, size_type());
    exprt str_len = exprt("-", size_type());
    str_len.copy_to_operands(array_size, one);

    // Emit: if (idx >= str_len) throw IndexError("string index out of range")
    exprt oob_cond(">=", bool_type());
    oob_cond.copy_to_operands(idx, str_len);

    exprt raise = converter_.get_exception_handler().gen_exception_raise(
      "IndexError", "string index out of range");
    codet throw_code("expression");
    throw_code.operands().push_back(raise);

    code_ifthenelset guard;
    guard.cond() = oob_cond;
    guard.then_case() = throw_code;
    converter_.add_instruction(guard);

    return index_exprt(array, idx, char_type());
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

  const bool lhs_is_set = lhs_symbol->is_set;
  const bool rhs_is_set = rhs_symbol->is_set;
  if (lhs_is_set || rhs_is_set)
  {
    if (!(lhs_is_set && rhs_is_set))
      return gen_boolean(op == "NotEq");

    symbolt *set_eq_func =
      converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_set_eq");
    if (!set_eq_func)
    {
      symbolt new_symbol;
      new_symbol.name = "__ESBMC_list_set_eq";
      new_symbol.id = "c:@F@__ESBMC_list_set_eq";
      new_symbol.mode = "C";
      new_symbol.is_extern = true;

      code_typet func_type;
      func_type.return_type() = bool_type();
      typet list_ptr = converter_.get_type_handler().get_list_type();
      func_type.arguments().push_back(code_typet::argumentt(list_ptr));
      func_type.arguments().push_back(code_typet::argumentt(list_ptr));
      new_symbol.type = func_type;

      converter_.symbol_table().add(new_symbol);
      set_eq_func =
        converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_set_eq");
    }

    locationt loc = converter_.get_location_from_decl(list_value_);
    symbolt &eq_ret = converter_.create_tmp_symbol(
      list_value_, "set_eq_tmp", bool_type(), gen_boolean(false));
    code_declt eq_ret_decl(symbol_expr(eq_ret));
    converter_.add_instruction(eq_ret_decl);

    code_function_callt set_eq_call;
    set_eq_call.function() = symbol_expr(*set_eq_func);
    set_eq_call.lhs() = symbol_expr(eq_ret);
    set_eq_call.arguments().push_back(
      lhs_symbol->type.is_pointer()
        ? symbol_expr(*lhs_symbol)
        : address_of_exprt(symbol_expr(*lhs_symbol)));
    set_eq_call.arguments().push_back(
      rhs_symbol->type.is_pointer()
        ? symbol_expr(*rhs_symbol)
        : address_of_exprt(symbol_expr(*rhs_symbol)));
    set_eq_call.type() = bool_type();
    set_eq_call.location() = loc;
    converter_.add_instruction(set_eq_call);

    exprt cond("=", bool_type());
    cond.copy_to_operands(symbol_expr(eq_ret));
    if (op == "Eq")
      cond.copy_to_operands(gen_boolean(true));
    else
      cond.copy_to_operands(gen_boolean(false));

    return cond;
  }

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

  // Get max depth from configuration option
  int max_depth = get_list_compare_depth();
  constant_exprt max_depth_expr(size_type());
  max_depth_expr.set_value(
    integer2binary(max_depth, config.ansi_c.address_width));

  code_function_callt list_eq_func_call;
  list_eq_func_call.function() = symbol_expr(*list_eq_func_sym);
  list_eq_func_call.lhs() = symbol_expr(eq_ret);
  // passing arguments
  list_eq_func_call.arguments().push_back(symbol_expr(*lhs_symbol)); // l1
  list_eq_func_call.arguments().push_back(symbol_expr(*rhs_symbol)); // l2
  list_eq_func_call.arguments().push_back(list_type_id);   // list_type_id
  list_eq_func_call.arguments().push_back(max_depth_expr); // max_depth
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
  symbolt *list_symbol = nullptr;

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

      list_symbol = converter_.find_symbol(rhs.identifier().as_string());
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

      list_symbol = converter_.find_symbol(lhs.identifier().as_string());
      assert(list_symbol);

      if (!parse_size_from_symbol(size_var, list_size))
        return create_vla(list_value_, list_symbol, size_var, list_elem);
    }
    else if (rhs.is_constant())
      list_size = binary2integer(rhs.value().c_str(), true);
  }

  if (!list_symbol)
  {
    if (lhs.type() == list_type && lhs.is_symbol())
      list_symbol = converter_.find_symbol(lhs.identifier().as_string());
    else if (rhs.type() == list_type && rhs.is_symbol())
      list_symbol = converter_.find_symbol(rhs.identifier().as_string());
  }
  assert(list_symbol);

  std::string list_id;
  if (converter_.current_lhs && converter_.current_lhs->is_symbol())
    list_id = converter_.current_lhs->identifier().as_string();
  else
    list_id = list_symbol->id.as_string();

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

  exprt actual_list = other_list;

  // Check if other_list is a string (array or pointer to char)
  if (
    (other_list.type().is_array() &&
     other_list.type().subtype() == char_type()) ||
    (other_list.type().is_pointer() &&
     other_list.type().subtype() == char_type()))
  {
    // Convert string to list of single-character strings
    symbolt &temp_list = create_list();

    // Get string length
    exprt str_len;
    if (other_list.type().is_array())
    {
      const array_typet &arr_type = to_array_type(other_list.type());
      // Subtract 1 for null terminator
      str_len = minus_exprt(arr_type.size(), gen_one(size_type()));
    }
    else // pointer type - use strlen
    {
      const symbolt *strlen_symbol =
        converter_.symbol_table().find_symbol("c:@F@strlen");
      if (!strlen_symbol)
        throw std::runtime_error("strlen function not found in symbol table");

      symbolt &strlen_result = converter_.create_tmp_symbol(
        op, "$strlen_result$", size_type(), gen_zero(size_type()));
      code_declt strlen_decl(symbol_expr(strlen_result));
      strlen_decl.location() = location;
      converter_.add_instruction(strlen_decl);

      code_function_callt strlen_call;
      strlen_call.function() = symbol_expr(*strlen_symbol);
      strlen_call.lhs() = symbol_expr(strlen_result);
      strlen_call.arguments().push_back(other_list);
      strlen_call.type() = size_type();
      strlen_call.location() = location;
      converter_.add_instruction(strlen_call);

      str_len = symbol_expr(strlen_result);
    }

    // Create char array
    array_typet char_arr_type(
      char_type(), from_integer(BigInt(2), size_type()));
    symbolt &char_elem =
      converter_.create_tmp_symbol(op, "$char_elem$", char_arr_type, exprt());
    code_declt char_elem_decl(symbol_expr(char_elem));
    char_elem_decl.location() = location;
    converter_.add_instruction(char_elem_decl);

    // Get type hash for char array
    const type_handler type_handler_ = converter_.get_type_handler();
    const std::string elem_type_name =
      type_handler_.type_to_string(char_arr_type);
    constant_exprt type_hash(size_type());
    type_hash.set_value(integer2binary(
      std::hash<std::string>{}(elem_type_name), config.ansi_c.address_width));

    // Create loop index
    symbolt &idx = converter_.create_tmp_symbol(
      op, "$str_idx$", size_type(), gen_zero(size_type()));
    code_assignt idx_init(symbol_expr(idx), gen_zero(size_type()));
    converter_.add_instruction(idx_init);

    // Loop condition: idx < str_len
    exprt loop_cond("<", bool_type());
    loop_cond.copy_to_operands(symbol_expr(idx), str_len);

    code_blockt loop_body;

    // Get character at index: str[idx]
    index_exprt char_at(other_list, symbol_expr(idx), char_type());

    // Update char_elem[0] = str[idx]
    index_exprt elem_0(
      symbol_expr(char_elem), gen_zero(size_type()), char_type());
    code_assignt assign_char(elem_0, char_at);
    assign_char.location() = location;
    loop_body.copy_to_operands(assign_char);

    // Update char_elem[1] = '\0'
    index_exprt elem_1(
      symbol_expr(char_elem), gen_one(size_type()), char_type());
    code_assignt assign_null(elem_1, gen_zero(char_type()));
    assign_null.location() = location;
    loop_body.copy_to_operands(assign_null);

    // Manually construct list_push call to avoid intermediate copy
    const symbolt *push_func_sym =
      converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_push");
    if (!push_func_sym)
      throw std::runtime_error("Push function symbol not found");

    code_function_callt push_call;
    push_call.function() = symbol_expr(*push_func_sym);
    push_call.arguments().push_back(symbol_expr(temp_list)); // list
    push_call.arguments().push_back(
      address_of_exprt(symbol_expr(char_elem))); // &char_elem
    push_call.arguments().push_back(type_hash);  // type hash
    push_call.arguments().push_back(
      from_integer(BigInt(2), size_type())); // size = 2
    push_call.type() = bool_type();
    push_call.location() = location;
    loop_body.copy_to_operands(push_call);

    // Increment index: idx++
    plus_exprt idx_inc(symbol_expr(idx), gen_one(size_type()));
    code_assignt idx_update(symbol_expr(idx), idx_inc);
    loop_body.copy_to_operands(idx_update);

    // Create while loop
    codet while_loop;
    while_loop.set_statement("while");
    while_loop.copy_to_operands(loop_cond, loop_body);
    converter_.add_instruction(while_loop);

    // Update type map for the elements we just added
    list_type_map[temp_list.id.as_string()].push_back(
      std::make_pair(char_elem.id.as_string(), char_arr_type));

    actual_list = symbol_expr(temp_list);
  }

  // Update list_type_map: copy type info from actual_list to list
  const std::string &list_name = list.id.as_string();
  const std::string &other_list_name = actual_list.identifier().as_string();

  // Copy all type entries from actual_list to the end of list
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
  extend_func_call.arguments().push_back(actual_list);
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

  // 2a. Materialize function calls that return lists
  if (
    iterable_expr.is_code() &&
    iterable_expr.get("statement") == "function_call")
  {
    const code_function_callt &call =
      to_code_function_call(to_code(iterable_expr));

    if (call.type() == list_type)
    {
      // Create temporary variable for the list
      symbolt &tmp_var_symbol = converter_.create_tmp_symbol(
        element, "$iter_temp$", list_type, gen_zero(list_type));

      // Declare the temporary
      code_declt tmp_var_decl(symbol_expr(tmp_var_symbol));
      tmp_var_decl.location() = location;
      converter_.add_instruction(tmp_var_decl);

      // Create function call with temp as LHS
      code_function_callt new_call;
      new_call.function() = call.function();
      new_call.arguments() = call.arguments();
      new_call.lhs() = symbol_expr(tmp_var_symbol);
      new_call.type() = list_type;
      new_call.location() = location;
      converter_.add_instruction(new_call);

      // Use the temp variable as the iterable
      iterable_expr = symbol_expr(tmp_var_symbol);
    }
  }
  // Check for empty list early
  else if (iterable_expr.type() == list_type && iterable_expr.is_symbol())
  {
    const std::string &list_id = iterable_expr.identifier().as_string();
    auto type_map_it = list_type_map.find(list_id);
    if (type_map_it == list_type_map.end() || type_map_it->second.empty())
      return symbol_expr(result_list);
  }

  // 3. Create loop variable
  std::string loop_var_name = target["id"].get<std::string>();
  symbol_id loop_var_sid = converter_.create_symbol_id();
  loop_var_sid.set_object(loop_var_name);

  // Infer loop variable type from iterable
  typet loop_var_type;
  if (iterable_expr.type() == list_type)
  {
    // For list iteration, we need to determine the element type from type_map
    loop_var_type = iterable_expr.type(); // default

    if (iterable_expr.is_symbol())
    {
      const std::string &list_id = iterable_expr.identifier().as_string();
      auto type_map_it = list_type_map.find(list_id);
      if (type_map_it != list_type_map.end() && !type_map_it->second.empty())
      {
        // Use the actual element type from type_map
        loop_var_type = type_map_it->second[0].second;
      }
    }
  }
  else if (iterable_expr.type().is_array())
    loop_var_type = iterable_expr.type().subtype();
  else if (iterable_expr.type().is_pointer())
    loop_var_type = iterable_expr.type().subtype();
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

  // Initialize index to 0
  code_assignt index_init(symbol_expr(index_var), gen_zero(size_type()));
  index_init.location() = location;
  converter_.add_instruction(index_init);

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
    // For lists, determine the actual element type from type map
    typet actual_elem_type = loop_var_type;

    // Try to get the actual element type from the list's type map
    if (iterable_expr.is_symbol())
    {
      const std::string &list_id = iterable_expr.identifier().as_string();
      auto type_map_it = list_type_map.find(list_id);
      if (type_map_it != list_type_map.end() && !type_map_it->second.empty())
      {
        // Get the element type from the first entry
        actual_elem_type = type_map_it->second[0].second;
      }
    }

    // Use list_at and extract_pyobject_value for consistent handling
    current_element =
      build_list_at_call(iterable_expr, symbol_expr(index_var), element);
    current_element = extract_pyobject_value(current_element, actual_elem_type);
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

exprt python_list::build_pop_list_call(
  const symbolt &list,
  const exprt &index,
  const nlohmann::json &element)
{
  const locationt location = converter_.get_location_from_decl(element);

  // Find the list_pop C function
  const symbolt *pop_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_pop");
  assert(pop_func && "list_pop function not found in symbol table");

  // Create side-effect function call
  const typet pyobject_ptr_type =
    pointer_typet(converter_.get_type_handler().get_list_element_type());

  side_effect_expr_function_callt pop_call;
  pop_call.function() = symbol_expr(*pop_func);
  pop_call.arguments().push_back(symbol_expr(list));
  pop_call.arguments().push_back(index);
  pop_call.type() = pyobject_ptr_type;
  pop_call.location() = location;

  // Determine the element type from the list's type map
  const std::string &list_id = list.id.as_string();
  typet elem_type;

  // Try to get element type from list_type_map (use last element for default pop)
  auto type_map_it = list_type_map.find(list_id);
  if (type_map_it != list_type_map.end() && !type_map_it->second.empty())
  {
    // Get the last element's type (since default pop() pops from the end)
    size_t last_idx = type_map_it->second.size() - 1;
    elem_type = type_map_it->second[last_idx].second;

    // Remove the popped element from type map to maintain consistency
    type_map_it->second.pop_back();
  }

  // If type map lookup failed, try to infer from list declaration
  if (elem_type == typet())
  {
    std::string list_name = list.name.as_string();
    nlohmann::json list_node = json_utils::find_var_decl(
      list_name, converter_.current_function_name(), converter_.ast());

    if (
      !list_node.is_null() && list_node.contains("value") &&
      list_node["value"].contains("elts") &&
      list_node["value"]["elts"].is_array() &&
      !list_node["value"]["elts"].empty())
    {
      // Get type from last element (default for pop)
      const auto &elts = list_node["value"]["elts"];
      elem_type = converter_.get_expr(elts[elts.size() - 1]).type();
    }
    // Try to get type from annotation (e.g., l: list[int] = [])
    else if (!list_node.is_null() && list_node.contains("annotation"))
    {
      elem_type =
        get_elem_type_from_annotation(list_node, converter_.get_type_handler());
    }
  }

  // If all type inference failed, use a generic fallback type
  // The runtime assertion in __ESBMC_list_pop will catch actual errors
  if (elem_type == typet())
  {
    // Use any_type() for cases such as empty lists with no annotation
    elem_type = any_type();
  }

  // Extract and dereference PyObject value
  return extract_pyobject_value(pop_call, elem_type);
}

exprt python_list::extract_pyobject_value(
  const exprt &pyobject_expr,
  const typet &elem_type)
{
  // Extract value from PyObject: pyobject_expr->value
  member_exprt obj_value(pyobject_expr, "value", pointer_typet(empty_typet()));

  // Dereference the PyObject* to access its members
  {
    exprt &base = obj_value.struct_op();
    exprt deref("dereference");
    deref.type() = base.type().subtype();
    deref.move_to_operands(base);
    base.swap(deref);
  }

  // For array types, return pointer to element type instead of pointer to array
  // The dereference system doesn't support array types as target types
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
    elem_type.is_pointer() &&
    (elem_type.subtype() == char_type() || elem_type.subtype() == bool_type()))
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

typet python_list::check_homogeneous_list_types(
  const std::string &list_id,
  const std::string &func_name)
{
  auto it = list_type_map.find(list_id);

  if (it == list_type_map.end() || it->second.empty())
    return typet();

  const TypeInfo &type_info = it->second;
  size_t list_size = type_info.size();

  // Get the first element's type
  typet elem_type = type_info[0].second;

  // Check whether a type is a string type (char array or char pointer)
  auto is_string_type = [](const typet &t) -> bool {
    return (t.is_array() && t.subtype() == char_type()) ||
           (t.is_pointer() && t.subtype() == char_type());
  };

  // Check all other elements have the same type
  for (size_t i = 1; i < list_size; i++)
  {
    const typet &current_elem_type = type_info[i].second;

    // For string types, all char arrays and char pointers are considered compatible
    if (is_string_type(elem_type) && is_string_type(current_elem_type))
      continue;

    // For non-string types, require exact match
    if (elem_type != current_elem_type)
    {
      throw std::runtime_error(
        "Type mismatch in " + func_name +
        "() call: list contains mixed types. "
        "ESBMC currently requires all elements to have the same type for " +
        func_name + "().");
    }
  }

  return elem_type;
}

exprt python_list::build_list_from_range(
  python_converter &converter,
  const nlohmann::json &range_args,
  const nlohmann::json &element)
{
  // Validate argument count
  if (range_args.empty() || range_args.size() > 3)
    throw std::runtime_error("range() takes 1 to 3 arguments");

  // Extract constant integer, handling UnaryOp for negative numbers
  auto extract_constant =
    [&](const nlohmann::json &arg) -> std::optional<long long> {
    exprt expr = converter.get_expr(arg);

    if (expr.is_constant())
      return binary2integer(expr.value().as_string(), expr.type().is_signedbv())
        .to_int64();

    // Handle UnaryOp (e.g., -1)
    if (expr.id() == "unary-" && expr.operands().size() == 1)
    {
      const exprt &operand = expr.operands()[0];
      if (operand.is_constant())
      {
        long long val =
          binary2integer(
            operand.value().as_string(), operand.type().is_signedbv())
            .to_int64();
        return -val;
      }
    }

    return std::nullopt;
  };

  // Extract all arguments
  std::optional<long long> arg0 = extract_constant(range_args[0]);
  std::optional<long long> arg1;
  std::optional<long long> arg2;

  if (range_args.size() > 1)
    arg1 = extract_constant(range_args[1]);
  if (range_args.size() > 2)
    arg2 = extract_constant(range_args[2]);

  // Check if all required arguments are constant
  const bool all_constant = arg0.has_value() &&
                            (range_args.size() <= 1 || arg1.has_value()) &&
                            (range_args.size() <= 2 || arg2.has_value());

  // Handle symbolic (non-constant) case
  if (!all_constant)
  {
    return handle_symbolic_range(converter, range_args, element);
  }

  // All arguments are constant
  return build_concrete_range(converter, range_args, element, arg0, arg1, arg2);
}

exprt python_list::handle_symbolic_range(
  python_converter &converter,
  const nlohmann::json &range_args,
  const nlohmann::json &element)
{
  if (range_args.size() == 1)
  {
    // range(n) case: create list with symbolic size n
    exprt n_expr = converter.get_expr(range_args[0]);

    // Create an empty list using existing create_list infrastructure
    nlohmann::json list_node;
    list_node["_type"] = "List";
    list_node["elts"] = nlohmann::json::array();
    converter.copy_location_fields_from_decl(element, list_node);

    python_list temp_list(converter, list_node);
    exprt list_expr = temp_list.get();

    // Set symbolic size using helper method
    set_list_symbolic_size(converter, list_expr, n_expr, element);

    return list_expr;
  }

  // For multi-argument symbolic ranges, return empty list
  nlohmann::json empty_list_node;
  empty_list_node["_type"] = "List";
  empty_list_node["elts"] = nlohmann::json::array();
  converter.copy_location_fields_from_decl(element, empty_list_node);
  python_list list(converter, empty_list_node);
  return list.get();
}

void python_list::set_list_symbolic_size(
  python_converter &converter,
  exprt &list_expr,
  const exprt &size_expr,
  const nlohmann::json &element)
{
  if (!list_expr.type().is_pointer())
    return;

  typet pointee_type = list_expr.type().subtype();

  // Follow symbol types to get actual struct type
  if (pointee_type.is_symbol())
    pointee_type = converter.ns.follow(pointee_type);

  if (!pointee_type.is_struct())
    return;

  const struct_typet &struct_type = to_struct_type(pointee_type);

  // Find and update the size member
  for (const auto &comp : struct_type.components())
  {
    if (comp.get_name() == "size")
    {
      // Create assignment: list->size = n
      dereference_exprt deref(list_expr, pointee_type);
      member_exprt size_member(deref, comp.get_name(), comp.type());
      exprt size_value = typecast_exprt(size_expr, comp.type());
      code_assignt size_assignment(size_member, size_value);

      size_assignment.location() = element.contains("lineno")
                                     ? converter.get_location_from_decl(element)
                                     : locationt();

      converter.current_block->operands().push_back(size_assignment);
      break;
    }
  }
}

exprt python_list::build_concrete_range(
  python_converter &converter,
  const nlohmann::json &range_args,
  const nlohmann::json &element,
  const std::optional<long long> &arg0,
  const std::optional<long long> &arg1,
  const std::optional<long long> &arg2)
{
  // Determine start, stop, step based on argument count
  long long start, stop, step;

  switch (range_args.size())
  {
  case 1:
    start = 0;
    stop = arg0.value();
    step = 1;
    break;

  case 2:
    start = arg0.value();
    stop = arg1.value();
    step = 1;
    break;

  case 3:
    start = arg0.value();
    stop = arg1.value();
    step = arg2.value();
    break;

  default:
    throw std::runtime_error("Invalid range argument count");
  }

  // Validate step
  if (step == 0)
    throw std::runtime_error("range() step argument must not be zero");

  // Calculate and validate range size
  long long range_size;
  if (step > 0)
    range_size = std::max(0LL, (stop - start + step - 1) / step);
  else
    range_size = std::max(0LL, (stop - start + step + 1) / step);

  if (range_size > kMaxSequenceExpansion)
  {
    throw std::runtime_error(
      "range() size too large for expansion: " + std::to_string(range_size) +
      " elements (max: " + std::to_string(kMaxSequenceExpansion) + ")");
  }

  // Build the list of elements
  nlohmann::json list_node;
  list_node["_type"] = "List";
  list_node["elts"] = nlohmann::json::array();

  // Generate list elements
  for (long long i = start; (step > 0) ? (i < stop) : (i > stop); i += step)
  {
    nlohmann::json elem;
    elem["_type"] = "Constant";
    elem["value"] = i;
    elem["kind"] = nullptr;
    converter.copy_location_fields_from_decl(element, elem);
    list_node["elts"].push_back(elem);
  }

  converter.copy_location_fields_from_decl(element, list_node);

  python_list list(converter, list_node);
  return list.get();
}

void python_list::copy_type_map_entries(
  const std::string &from_list_id,
  const std::string &to_list_id)
{
  auto it = list_type_map.find(from_list_id);
  if (it != list_type_map.end())
  {
    for (const auto &type_entry : it->second)
      list_type_map[to_list_id].push_back(type_entry);
  }
}

exprt python_list::build_copy_list_call(
  const symbolt &list,
  const nlohmann::json &element)
{
  const locationt location = converter_.get_location_from_decl(element);
  const typet list_type = converter_.get_type_handler().get_list_type();

  // Find the list_copy C function
  const symbolt *copy_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_copy");
  if (!copy_func)
    throw std::runtime_error("list_copy function not found in symbol table");

  // Create the copied list symbol
  symbolt &copied_list =
    converter_.create_tmp_symbol(element, "$list_copy$", list_type, exprt());

  code_declt copied_decl(symbol_expr(copied_list));
  copied_decl.location() = location;
  converter_.add_instruction(copied_decl);

  // Build function call
  code_function_callt copy_call;
  copy_call.function() = symbol_expr(*copy_func);
  copy_call.arguments().push_back(symbol_expr(list));
  copy_call.lhs() = symbol_expr(copied_list);
  copy_call.type() = list_type;
  copy_call.location() = location;
  converter_.add_instruction(copy_call);

  // Copy type information from original list to copied list
  copy_type_map_entries(list.id.as_string(), copied_list.id.as_string());

  return symbol_expr(copied_list);
}

exprt python_list::build_remove_list_call(
  const symbolt &list,
  const nlohmann::json &op,
  const exprt &elem)
{
  list_elem_info elem_info = get_list_element_info(op, elem);

  const symbolt *remove_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_remove");

  if (!remove_func)
    throw std::runtime_error(
      "__ESBMC_list_remove function not found in symbol table");

  exprt element_arg;
  if (
    elem_info.elem_symbol->type.is_pointer() &&
    elem_info.elem_symbol->type.subtype() == char_type())
    element_arg = symbol_expr(*elem_info.elem_symbol);
  else if (elem_info.elem_symbol->type.is_struct())
    element_arg = address_of_exprt(symbol_expr(*elem_info.elem_symbol));
  else
    element_arg = address_of_exprt(symbol_expr(*elem_info.elem_symbol));

  code_function_callt remove_call;
  remove_call.function() = symbol_expr(*remove_func);
  remove_call.arguments().push_back(symbol_expr(list)); // list
  remove_call.arguments().push_back(element_arg);       // &value or ptr
  remove_call.arguments().push_back(
    symbol_expr(*elem_info.elem_type_sym));               // type_id
  remove_call.arguments().push_back(elem_info.elem_size); // size
  remove_call.type() = bool_type();
  remove_call.location() = elem_info.location;

  return converter_.convert_expression_to_code(remove_call);
}

size_t python_list::get_list_type_map_size(const std::string &list_id)
{
  auto it = list_type_map.find(list_id);
  if (it == list_type_map.end())
    return 0;
  return it->second.size();
}
