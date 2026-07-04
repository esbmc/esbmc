#include "python_list_internal.h"

#include <util/c_typecast.h>

using namespace python_expr;
using namespace python_list_detail;

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
  code_assignt hash_assignment(build_symbol(elem_type_sym), hash_value);
  hash_assignment.location() = location;
  converter_.add_instruction(hash_assignment);

  // Create and declare temporary symbol for list element
  symbolt &elem_symbol =
    converter_.create_tmp_symbol(op, "$list_elem$", elem.type(), elem);
  code_declt elem_decl(build_symbol(elem_symbol));
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
    elem_symbol.get_type().is_pointer() &&
    elem_symbol.get_type().subtype() == bool_type())
  {
    elem_size = from_integer(BigInt(0), size_type());
  }
  // For list pointers (PyListObj*), use pointer size
  else if (elem_symbol.get_type() == list_type)
  {
    // This is a pointer to PyListObj: use pointer size
    const size_t pointer_size_bytes = config.ansi_c.pointer_width() / 8;
    elem_size = from_integer(BigInt(pointer_size_bytes), size_type());
  }
  // Handle struct types (such as dictionaries and user-defined classes).
  // The element type may be a symbol_typet reference (e.g. "tag-A") rather
  // than a plain struct_typet, so follow the type before checking is_struct().
  else if (
    elem_symbol.get_type().is_struct() ||
    (elem_symbol.get_type().id() == "symbol" &&
     converter_.name_space().follow(elem_symbol.get_type()).is_struct()))
  {
    const typet &resolved =
      elem_symbol.get_type().is_struct()
        ? elem_symbol.get_type()
        : converter_.name_space().follow(elem_symbol.get_type());

    // Measure the struct by its true byte size via the canonical computation.
    // A hand-rolled component sum mistakes array- and struct-typed components
    // (e.g. a tuple key of strings, whose elements are char[N]) for pointers,
    // over-reporting the size and tripping an out-of-bounds copy in
    // __ESBMC_copy_value. A struct member that is a dynamically- or
    // infinitely-sized array has no static size; type_byte_size throws there,
    // so fall back to pointer width as the legacy summation did.
    BigInt total_size;
    try
    {
      total_size =
        type_byte_size(migrate_type(resolved), &converter_.name_space());
    }
    catch (const array_type2t::array_size_excp &)
    {
      total_size = config.ansi_c.pointer_width() / 8;
    }
    if (total_size == 0)
      total_size = config.ansi_c.pointer_width() / 8;

    elem_size = from_integer(total_size, size_type());
  }
  // For non-char, non-bool pointer types (e.g., Optional[T] stored as T*):
  // pointer_typet has no width() attribute, so we must use pointer_width here
  // rather than falling to the generic else branch which calls width().
  else if (
    elem_symbol.get_type().is_pointer() &&
    elem_symbol.get_type().subtype() != char_type() &&
    elem_symbol.get_type().subtype() != bool_type())
  {
    const size_t pointer_size_bytes = config.ansi_c.pointer_width() / 8;
    elem_size = from_integer(BigInt(pointer_size_bytes), size_type());
  }
  // For string pointers (char*), calculate length at runtime using strlen
  else if (
    elem_symbol.get_type().is_pointer() &&
    elem_symbol.get_type().subtype() == char_type())
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
    code_declt strlen_decl(build_symbol(strlen_result));
    strlen_decl.location() = location;
    converter_.add_instruction(strlen_decl);

    // Call strlen(elem_symbol)
    code_function_callt strlen_call;
    strlen_call.function() = build_symbol(*strlen_symbol);
    strlen_call.lhs() = build_symbol(strlen_result);
    strlen_call.arguments().push_back(build_symbol(elem_symbol));
    strlen_call.type() = size_type();
    strlen_call.location() = location;
    converter_.add_instruction(strlen_call);

    // Add 1 for null terminator: size = strlen(s) + 1. strlen_result is a
    // synthetic size_type symbol, so build the addition in IREP2 (V.3).
    elem_size = build_add(
      build_symbol(strlen_result),
      from_integer(1, strlen_result.get_type()),
      strlen_result.get_type());
  }
  // A char array at the fixed tuple-string-member width may be NUL-padded
  // (#5571): its identity for membership and dict-key comparisons is the
  // content, not the storage width. Store and probe with content-length + 1
  // bytes so a padded 'B' (unpacked from a tuple) and a tight 'B' literal
  // agree on size. The length is a loop-free ITE chain over the fixed width
  // — a strlen() call here would hide the size behind a loop and return
  // garbage under --no-unwinding-assertions with a small --unwind (the
  // github_3560_2_fail false-SUCCESSFUL). Content without a NUL (e.g. a
  // fully symbolic buffer) keeps the full width, the pre-#5571 behaviour.
  else if (
    elem_symbol.get_type() ==
    type_handler_.get_typet("str", tuple_handler::tuple_str_member_size))
  {
    exprt sz =
      from_integer(BigInt(tuple_handler::tuple_str_member_size), size_type());
    for (size_t i = tuple_handler::tuple_str_member_size; i-- > 0;)
    {
      exprt ch =
        build_index(build_symbol(elem_symbol), from_integer(i, index_type()));
      exprt is_nul = equality_exprt(ch, gen_zero(char_type()));
      sz = if_exprt(is_nul, from_integer(i + 1, size_type()), sz);
    }
    elem_size = sz;
  }
  else
  {
    // Handle arrays and other types
    constexpr size_t BITS_PER_BYTE = 8;
    constexpr size_t DEFAULT_SIZE = 1;

    size_t elem_size_bytes = DEFAULT_SIZE;
    try
    {
      if (elem_symbol.get_type().is_array())
      {
        const size_t subtype_size_bits =
          std::stoull(elem.type().subtype().width().as_string(), nullptr, 10);

        const array_typet &array_type =
          static_cast<const array_typet &>(elem_symbol.get_type());

        const size_t array_length =
          std::stoull(array_type.size().value().as_string(), nullptr, 2);

        elem_size_bytes = (array_length * subtype_size_bits) / BITS_PER_BYTE;
      }
      else
      {
        const size_t type_width_bits =
          std::stoull(elem_symbol.get_type().width().as_string(), nullptr, 10);

        elem_size_bytes = type_width_bits / BITS_PER_BYTE;
      }
    }
    catch (std::invalid_argument &)
    {
      elem_size_bytes = DEFAULT_SIZE;
    }

    // A char array with statically-unknown length -- e.g. a string component
    // recovered from a bare tuple[str, ...] annotation with no literal to
    // size it from (#5444) -- is declared with 0 bytes of storage. Reading a
    // real string out of it (e.g. via a runtime strlen) is not just
    // unsizeable, it is a genuine out-of-bounds access on a 0-byte object:
    // ESBMC's pointer-safety model has no bytes to dereference there, so a
    // "recovered" size would report a spurious CWE-125 violation on
    // perfectly valid Python instead of the real property. Refuse cleanly
    // rather than fabricate an unsound size (issue #5444, deferred item 1).
    if (elem_size_bytes == 0)
    {
      throw std::runtime_error("Element size cannot be zero");
    }

    elem_size = from_integer(BigInt(elem_size_bytes), size_type());
  }

  // A bool element is stored via its address and compared byte-wise by the
  // OM, so it must be widened to a long (8 bytes). Doing this only in the push
  // path left every lookup query at bool's native 1-byte size while the stored
  // element was 8 bytes, so __ESBMC_values_equal's size check never matched —
  // bool dict keys, set elements, and list membership/count/dict.get all
  // failed. Normalizing here, the single point every storage and lookup path
  // funnels through, keeps the two sides consistent. The type hash stays the
  // bool hash computed above, so only the stored representation/size widens.
  if (elem.type() == bool_type())
  {
    symbolt &bool_as_long = converter_.create_tmp_symbol(
      op,
      "$bool_as_long$",
      signedbv_typet(config.ansi_c.long_int_width),
      exprt());
    code_declt bool_long_decl(build_symbol(bool_as_long));
    bool_long_decl.copy_to_operands(build_typecast(
      build_symbol(elem_symbol),
      signedbv_typet(config.ansi_c.long_int_width)));
    bool_long_decl.location() = location;
    converter_.add_instruction(bool_long_decl);

    list_elem_info bool_info;
    bool_info.elem_type_sym = &elem_type_sym;
    bool_info.elem_symbol = &bool_as_long;
    bool_info.elem_size =
      from_integer(BigInt(config.ansi_c.long_int_width / 8), size_type());
    bool_info.location = location;
    return bool_info;
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
  const exprt &elem,
  bool enable_float_path)
{
  list_elem_info elem_info = get_list_element_info(op, elem);

  const symbolt *push_func_sym =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_push");

  if (!push_func_sym)
  {
    throw std::runtime_error("Push function symbol not found");
  }

  code_function_callt push_func_call;
  push_func_call.function() = build_symbol(*push_func_sym);
  push_func_call.arguments().push_back(build_symbol(list)); // list

  // For string types (pointer to char), we must pass the pointer value directly
  // For other types (including other pointers such None/bool*), we must pass the address
  exprt element_arg;
  if (is_empty_user_class_object_type(
        elem_info.elem_symbol->get_type(), converter_.name_space()))
  {
    // Python list stores object references. For class objects, store a pointer
    // to the object (not a byte copy of the struct payload).
    typet obj_ptr_type = pointer_typet(elem_info.elem_symbol->get_type());
    symbolt &obj_ptr_sym =
      converter_.create_tmp_symbol(op, "$list_obj_ref$", obj_ptr_type, exprt());

    code_declt obj_ptr_decl(build_symbol(obj_ptr_sym));
    obj_ptr_decl.location() = elem_info.location;
    converter_.add_instruction(obj_ptr_decl);

    code_assignt obj_ptr_assign(
      build_symbol(obj_ptr_sym),
      build_address_of(build_symbol(*elem_info.elem_symbol)));
    obj_ptr_assign.location() = elem_info.location;
    converter_.add_instruction(obj_ptr_assign);

    element_arg = build_address_of(build_symbol(obj_ptr_sym));
    const size_t pointer_size_bytes = config.ansi_c.pointer_width() / 8;
    elem_info.elem_size = from_integer(BigInt(pointer_size_bytes), size_type());
  }
  else if (
    elem_info.elem_symbol->get_type().is_pointer() &&
    elem_info.elem_symbol->get_type().subtype() == char_type())
  {
    // For string type (char*), we must pass the pointer value itself
    element_arg = build_symbol(*elem_info.elem_symbol);
  }
  else if (
    elem_info.elem_symbol->get_type().is_pointer() &&
    elem_info.elem_symbol->get_type().subtype() == bool_type())
  {
    // For None type (_Bool*), pass the pointer value itself
    // This allows direct NULL checks without dereferencing
    element_arg = build_symbol(*elem_info.elem_symbol);
  }
  else if (elem_info.elem_symbol->get_type().is_struct())
  {
    // For structs (dictionaries), pass address of the struct directly
    element_arg = build_address_of(build_symbol(*elem_info.elem_symbol));
  }
  else
  {
    // For all other types, pass the address of the value. A bool element is
    // already widened to a long by get_list_element_info (so storage and every
    // lookup path agree on its 8-byte size), so no bool special-case is needed
    // here.
    element_arg = build_address_of(build_symbol(*elem_info.elem_symbol));
  }

  push_func_call.arguments().push_back(element_arg); // element or &element
  push_func_call.arguments().push_back(
    build_symbol(*elem_info.elem_type_sym));                 // type hash
  push_func_call.arguments().push_back(elem_info.elem_size); // element size

  // float_type_id: when element is float, its type hash == float_type_id so
  // __ESBMC_copy_value uses *(double*) copy in --ir mode (real sort).
  // enable_float_path=false skips this for dict values which use void* comparison.
  exprt float_type_id_arg =
    (enable_float_path && elem_info.elem_symbol->get_type().is_floatbv())
      ? static_cast<exprt>(build_symbol(*elem_info.elem_type_sym))
      : from_integer(BigInt(0), size_type());
  push_func_call.arguments().push_back(float_type_id_arg); // float_type_id

  // ptr_free gates the uint64 fast paths in __ESBMC_copy_value.
  push_func_call.arguments().push_back(from_integer(
    BigInt(
      converter_.get_type_handler().is_pointer_free(
        elem_info.elem_symbol->get_type())
        ? 1
        : 0),
    int_type()));

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

  exprt float_type_id_arg =
    elem_info.elem_symbol->get_type().is_floatbv()
      ? static_cast<exprt>(build_symbol(*elem_info.elem_type_sym))
      : from_integer(BigInt(0), size_type());

  code_function_callt insert_func_call;
  insert_func_call.function() = build_symbol(*insert_func_sym);
  insert_func_call.arguments().push_back(build_symbol(list));
  insert_func_call.arguments().push_back(index);
  insert_func_call.arguments().push_back(
    build_address_of(build_symbol(*elem_info.elem_symbol)));
  insert_func_call.arguments().push_back(
    build_symbol(*elem_info.elem_type_sym));
  insert_func_call.arguments().push_back(elem_info.elem_size);
  insert_func_call.arguments().push_back(float_type_id_arg);
  insert_func_call.arguments().push_back(from_integer(
    BigInt(
      converter_.get_type_handler().is_pointer_free(
        elem_info.elem_symbol->get_type())
        ? 1
        : 0),
    int_type()));
  insert_func_call.type() = bool_type();
  insert_func_call.location() = elem_info.location;

  return converter_.convert_expression_to_code(insert_func_call);
}

void python_list::emit_list_copy(
  const exprt &src,
  const symbolt &dst,
  const nlohmann::json &element)
{
  const locationt loc = converter_.get_location_from_decl(element);

  // Helpers we’ll call from the C model
  const symbolt *size_sym =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
  const symbolt *at_sym =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_at");
  // Shallow per-element append: preserves element value pointers so nested
  // lists are shared (Python shallow-copy semantics) rather than corrupted by
  // a pointee byte-copy (esbmc/esbmc#5102).
  const symbolt *push_obj_sym =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_push_shallow");
  assert(size_sym && at_sym && push_obj_sym);

  // list_size / list_at take `const List*`
  // so pass the address of values.
  auto as_list_ptr = [](const exprt &e) {
    return e.type().is_pointer() ? e : static_cast<exprt>(build_address_of(e));
  };

  // size_t n = list_size(src_list);
  symbolt &n_sym = converter_.create_tmp_symbol(
    element, "$n$", size_type(), gen_zero(size_type()));
  converter_.add_instruction(code_declt(build_symbol(n_sym)));

  code_function_callt get_size;
  get_size.function() = build_symbol(*size_sym);
  get_size.arguments().push_back(as_list_ptr(src));
  get_size.lhs() = build_symbol(n_sym);
  get_size.type() = size_type();
  get_size.location() = loc;
  converter_.add_instruction(get_size);

  // for (size_t i = 0; i < n; ++i) { push_object(dst, list_at(src, i)); }
  symbolt &i_sym = converter_.create_tmp_symbol(
    element, "$i$", size_type(), gen_zero(size_type()));
  converter_.add_instruction(code_declt(build_symbol(i_sym)));

  // i = 0
  converter_.add_instruction(
    code_assignt(build_symbol(i_sym), gen_zero(size_type())));

  // condition: i < n (both $i$/$n$ are size_type — same width)
  exprt cond = build_less_than(build_symbol(i_sym), build_symbol(n_sym));

  // body
  code_blockt body;

  // tmp_obj = list_at(src, i)
  exprt at_call = build_call_expr(
    *at_sym,
    pointer_typet(converter_.get_type_handler().get_list_element_type()),
    {as_list_ptr(src), build_symbol(i_sym)});
  at_call.location() = loc;

  symbolt &tmp_obj = converter_.create_tmp_symbol(
    element,
    "tmp_list_at",
    pointer_typet(converter_.get_type_handler().get_list_element_type()),
    exprt());
  code_declt tmp_obj_decl(build_symbol(tmp_obj));
  tmp_obj_decl.copy_to_operands(at_call);
  body.copy_to_operands(tmp_obj_decl);

  // list_push_shallow(dst_list, tmp_obj, list_type_id): nested-list elements
  // (type_id == list_type_id) keep their inner pointer; scalars are byte-copied
  // independently, so they are not corrupted by a pointee byte-copy (#5102).
  constant_exprt list_type_id_arg(size_type());
  list_type_id_arg.set_value(integer2binary(
    std::hash<std::string>{}(converter_.get_type_handler().type_to_string(
      converter_.get_type_handler().get_list_type())),
    config.ansi_c.address_width));
  exprt push_call = build_call_expr(
    *push_obj_sym,
    bool_type(),
    {build_symbol(dst), build_symbol(tmp_obj), list_type_id_arg});
  push_call.location() = loc;
  body.copy_to_operands(converter_.convert_expression_to_code(push_call));

  // i = i + 1 (V.3: synthetic size_type increment, built in IREP2)
  exprt i_inc =
    build_add(build_symbol(i_sym), gen_one(size_type()), size_type());
  body.copy_to_operands(code_assignt(build_symbol(i_sym), i_inc));

  // while (i < n) { ... }
  codet loop;
  loop.set_statement("while");
  loop.copy_to_operands(cond, body);
  converter_.add_instruction(loop);
}

exprt python_list::build_concat_list_call(
  const exprt &lhs,
  const exprt &rhs,
  const nlohmann::json &element)
{
  symbolt &dst_list = create_list();
  // Copy lhs then rhs
  emit_list_copy(lhs, dst_list, element);
  emit_list_copy(rhs, dst_list, element);

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

  return build_symbol(dst_list);
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
  exprt source_list;
  exprt repeat_expr;
  const nlohmann::json *source_node = nullptr;
  // True when the list operand is a variable (not a literal).
  bool is_variable_list = false;

  [[maybe_unused]] auto is_integer_type = [](const typet &type) {
    return type.is_signedbv() || type.is_unsignedbv();
  };

  auto parse_size_from_symbol = [&](symbolt *size_var, BigInt &out) -> bool {
    if (
      size_var->get_value().is_code() || size_var->get_value().is_nil() ||
      !size_var->get_value().is_constant())
    {
      return false;
    }

    assert(is_integer_type(size_var->get_value().type()));
    out = binary2integer(size_var->get_value().value().c_str(), true);
    return true;
  };

  auto parse_repeat_expr = [&](const exprt &expr, BigInt &out) -> bool {
    if (expr.is_constant())
    {
      assert(is_integer_type(expr.type()));
      out = binary2integer(expr.value().c_str(), true);
      return true;
    }

    if (!expr.is_symbol())
      return false;

    symbolt *size_var =
      converter_.find_symbol(to_symbol_expr(expr).get_identifier().as_string());
    if (!size_var)
      return false;

    return parse_size_from_symbol(size_var, out);
  };

  if (lhs.type() == list_type && rhs.type() != list_type)
  {
    source_list = lhs;
    repeat_expr = rhs;
    source_node = &left_node;
  }
  else if (rhs.type() == list_type && lhs.type() != list_type)
  {
    source_list = rhs;
    repeat_expr = lhs;
    source_node = &right_node;
  }

  // Constant repetition of an existing list should preserve all elements
  // (e.g., [4, 5] * 2 -> [4, 5, 4, 5]) via concrete frontend lowering.
  if (
    source_node && source_list.is_symbol() &&
    parse_repeat_expr(repeat_expr, list_size))
  {
    const int64_t repeat_count = list_size.to_int64();
    if (repeat_count <= 0)
      return build_symbol(create_list());

    std::vector<exprt> source_elems;
    if (source_node->contains("elts") && (*source_node)["elts"].is_array())
    {
      for (const auto &elt : (*source_node)["elts"])
        source_elems.push_back(converter_.get_expr(elt));
    }
    else
    {
      std::string var_name;
      if (
        source_node->contains("_type") && (*source_node)["_type"] == "Name" &&
        source_node->contains("id") && (*source_node)["id"].is_string())
      {
        var_name = (*source_node)["id"].get<std::string>();
      }
      else
      {
        var_name = json_utils::extract_var_name_from_symbol_id(
          source_list.identifier().as_string());
      }

      nlohmann::json var_decl = json_utils::find_var_decl(
        var_name, converter_.current_function_name(), converter_.ast());
      if (
        !var_decl.is_null() && var_decl.contains("value") &&
        var_decl["value"].is_object() && var_decl["value"].contains("_type") &&
        var_decl["value"]["_type"] == "List" &&
        var_decl["value"].contains("elts") &&
        var_decl["value"]["elts"].is_array())
      {
        for (const auto &elt : var_decl["value"]["elts"])
          source_elems.push_back(converter_.get_expr(elt));
      }
    }

    if (!source_elems.empty())
    {
      symbolt &result = create_list();
      const std::string result_id = result.id.as_string();
      locationt location = converter_.get_location_from_decl(list_value_);

      auto materialize_list_elem = [&](const exprt &elem) -> exprt {
        if (elem.is_symbol())
          return elem;

        symbolt &tmp = converter_.create_tmp_symbol(
          list_value_, "$list_rep_elem$", elem.type(), elem);
        code_declt decl(build_symbol(tmp));
        decl.location() = location;
        converter_.add_instruction(decl);

        code_assignt assign(build_symbol(tmp), elem);
        assign.location() = location;
        converter_.add_instruction(assign);
        return build_symbol(tmp);
      };

      for (int64_t i = 0; i < repeat_count; ++i)
      {
        for (const auto &elem : source_elems)
        {
          exprt map_elem = materialize_list_elem(elem);
          converter_.add_instruction(
            build_push_list_call(result, list_value_, map_elem));
          list_type_map[result_id].push_back(
            std::make_pair(map_elem.identifier().as_string(), map_elem.type()));
        }
      }
      return build_symbol(result);
    }
  }

  // Get element expression from list_type_map for a variable list.
  auto elem_from_type_map = [&](const std::string &src_id) -> exprt {
    const std::string elem_id = get_list_element_id(src_id, 0);
    if (elem_id.empty())
    {
      const typet fallback_type = get_list_element_type(src_id, 0);
      if (!fallback_type.is_nil() && !fallback_type.is_empty())
        return gen_zero(fallback_type);
      return exprt();
    }
    symbolt *elem_sym = converter_.find_symbol(elem_id);
    if (!elem_sym)
      return exprt();

    return build_symbol(*elem_sym);
  };

  // Get all element expressions from list_type_map for a variable list.
  auto elems_from_type_map =
    [&](const std::string &src_id) -> std::vector<exprt> {
    std::vector<exprt> elems;
    auto it = list_type_map.find(src_id);
    if (it == list_type_map.end() || it->second.empty())
      return elems;
    for (const auto &entry : it->second)
    {
      if (entry.first.empty())
        return {};
      symbolt *elem_sym = converter_.find_symbol(entry.first);
      if (!elem_sym)
        return {};
      elems.push_back(build_symbol(*elem_sym));
    }
    return elems;
  };

  // Collect source elements for the VLA path: literal `elts`,
  // or the variable list's type map, falling back to a single `fallback` element
  // if neither yields anything.
  auto collect_vla_elems = [&](
                             const nlohmann::json &node,
                             const exprt &list_operand,
                             const exprt &fallback) -> std::vector<exprt> {
    std::vector<exprt> elems;
    if (node.contains("elts") && node["elts"].is_array())
    {
      for (const auto &elt : node["elts"])
        elems.push_back(converter_.get_expr(elt));
    }
    else
      elems = elems_from_type_map(list_operand.identifier().as_string());
    if (elems.empty())
      elems.push_back(fallback);
    return elems;
  };

  // Count on the lhs (e.g.: 3 * [1] or n * lst). The list operand is the rhs.
  if (lhs.type() != list_type)
  {
    // For literal `elts`, defer the elts[0] extraction to the constant-count branch
    // the VLA path re-extracts every element via collect_vla_elems.
    const bool from_elts = right_node.contains("elts");
    if (!from_elts)
    {
      // rhs is a variable list — get element from list_type_map
      list_elem = elem_from_type_map(rhs.identifier().as_string());
      if (list_elem.is_nil())
        return build_symbol(create_list());
      is_variable_list = true;
    }

    if (lhs.is_constant())
    {
      if (from_elts)
        list_elem = converter_.get_expr(right_node["elts"][0]);
      assert(is_integer_type(lhs.type()));
      list_size = binary2integer(lhs.value().c_str(), true);
    }
    else
    {
      // Non-constant count (symbol `n` or compound `m + 1`): repeat at runtime.
      return create_vla(
        list_value_, lhs, collect_vla_elems(right_node, rhs, list_elem));
    }
  }

  // Count on the rhs (e.g.: [1] * 3 or lst * n). The list operand is the lhs.
  if (rhs.type() != list_type)
  {
    const bool from_elts = left_node.contains("elts");
    if (!from_elts)
    {
      // lhs is a variable list — get element from list_type_map
      list_elem = elem_from_type_map(lhs.identifier().as_string());
      if (list_elem.is_nil())
        return build_symbol(create_list());
      is_variable_list = true;
    }

    if (rhs.is_constant())
    {
      if (from_elts)
        list_elem = converter_.get_expr(left_node["elts"][0]);
      assert(is_integer_type(rhs.type()));
      list_size = binary2integer(rhs.value().c_str(), true);
    }
    else
    {
      // Non-constant count (symbol `n` or compound `m + 1`): repeat at runtime.
      return create_vla(
        list_value_, rhs, collect_vla_elems(left_node, lhs, list_elem));
    }
  }

  // For variable lists, allocate a fresh result list so the source is not
  // mutated.  For literal lists the temp symbol created by get() is reused.
  if (is_variable_list)
  {
    symbolt &result = create_list();
    list_symbol = &result;
  }
  else if (!list_symbol)
  {
    if (lhs.type() == list_type && lhs.is_symbol())
      list_symbol = converter_.find_symbol(lhs.identifier().as_string());
    else if (rhs.type() == list_type && rhs.is_symbol())
      list_symbol = converter_.find_symbol(rhs.identifier().as_string());
  }
  assert(list_symbol);

  if (!list_elem.is_symbol() && !list_elem.is_nil())
  {
    locationt location = converter_.get_location_from_decl(list_value_);
    symbolt &tmp = converter_.create_tmp_symbol(
      list_value_, "$list_rep_elem$", list_elem.type(), list_elem);
    code_declt decl(build_symbol(tmp));
    decl.location() = location;
    converter_.add_instruction(decl);
    code_assignt assign(build_symbol(tmp), list_elem);
    assign.location() = location;
    converter_.add_instruction(assign);
    list_elem = build_symbol(tmp);
  }

  std::string list_id;
  if (converter_.current_lhs && converter_.current_lhs->is_symbol())
    list_id = converter_.current_lhs->identifier().as_string();
  else
    list_id = list_symbol->id.as_string();

  // Variable list: emit `list_size` runtime copies of the source
  // to preserve every element value (symbol && literal).
  if (is_variable_list)
  {
    const exprt &src = (lhs.type() == list_type) ? lhs : rhs;
    const int64_t repeat_count = list_size.to_int64();

    for (int64_t i = 0; i < repeat_count; ++i)
      emit_list_copy(src, *list_symbol, list_value_);

    // Mirror the type-map entries.
    // Make sure later element-type lookups see correct types.
    if (src.is_symbol())
    {
      auto it = list_type_map.find(src.identifier().as_string());
      if (it != list_type_map.end())
      {
        const auto src_entries = it->second;
        for (int64_t i = 0; i < repeat_count; ++i)
          for (const auto &entry : src_entries)
            list_type_map[list_id].push_back(entry);
      }
    }

    return build_symbol(*list_symbol);
  }

  // Literal list: first element is already in list_symbol, just push the rest.
  const int64_t push_count = list_size.to_int64() - 1;

  for (int64_t i = 0; i < push_count; ++i)
  {
    converter_.add_instruction(
      build_push_list_call(*list_symbol, list_value_, list_elem));

    list_type_map[list_id].push_back(
      std::make_pair(list_elem.identifier().as_string(), list_elem.type()));
  }

  return build_symbol(*list_symbol);
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

  if (actual_list.is_code() && actual_list.is_function_call())
  {
    const code_function_callt &call_expr =
      to_code_function_call(to_code(actual_list));

    const typet list_type = converter_.get_type_handler().get_list_type();

    symbolt &tmp_list =
      converter_.create_tmp_symbol(op, "$extend_list$", list_type, exprt());
    code_declt tmp_decl(build_symbol(tmp_list));
    tmp_decl.location() = location;
    converter_.add_instruction(tmp_decl);

    code_function_callt func_call;
    func_call.function() = call_expr.function();
    func_call.arguments() = call_expr.arguments();
    func_call.lhs() = build_symbol(tmp_list);
    func_call.type() = list_type;
    func_call.location() = location;
    converter_.add_instruction(func_call);

    actual_list = build_symbol(tmp_list);
  }

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
      // Subtract 1 for null terminator. V.3: build the (size - 1) subtraction
      // in IREP2 with the result type set explicitly to the array-size type
      // (what the legacy 2-arg minus_exprt's downstream inference yields).
      const exprt &arr_size = arr_type.size();
      str_len = build_sub(arr_size, gen_one(size_type()), arr_size.type());
    }
    else // pointer type - use strlen
    {
      const symbolt *strlen_symbol =
        converter_.symbol_table().find_symbol("c:@F@strlen");
      if (!strlen_symbol)
        throw std::runtime_error("strlen function not found in symbol table");

      symbolt &strlen_result = converter_.create_tmp_symbol(
        op, "$strlen_result$", size_type(), gen_zero(size_type()));
      code_declt strlen_decl(build_symbol(strlen_result));
      strlen_decl.location() = location;
      converter_.add_instruction(strlen_decl);

      code_function_callt strlen_call;
      strlen_call.function() = build_symbol(*strlen_symbol);
      strlen_call.lhs() = build_symbol(strlen_result);
      strlen_call.arguments().push_back(other_list);
      strlen_call.type() = size_type();
      strlen_call.location() = location;
      converter_.add_instruction(strlen_call);

      str_len = build_symbol(strlen_result);
    }

    // Create char array
    array_typet char_arr_type(
      char_type(), from_integer(BigInt(2), size_type()));
    symbolt &char_elem =
      converter_.create_tmp_symbol(op, "$char_elem$", char_arr_type, exprt());
    code_declt char_elem_decl(build_symbol(char_elem));
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
    code_assignt idx_init(build_symbol(idx), gen_zero(size_type()));
    converter_.add_instruction(idx_init);

    // Loop condition: idx < str_len, built in IREP2 (V.1k keystone, W). idx
    // (size_type) and str_len can have mismatched widths (str_len may be
    // arr_size.type()-typed), so reconcile with the same
    // c_implicit_typecast_arithmetic clang_cpp_adjust's adjust_expr_rel applies
    // before building lessthan2t.
    exprt idx_op = build_symbol(idx);
    exprt len_op = str_len;
    namespacet ns(converter_.symbol_table());
    c_implicit_typecast_arithmetic(idx_op, len_op, ns);
    exprt loop_cond = build_less_than(idx_op, len_op);

    code_blockt loop_body;

    // Get character at index: str[idx]
    exprt char_at = build_index(other_list, build_symbol(idx), char_type());

    // Update char_elem[0] = str[idx]
    exprt elem_0 =
      build_index(build_symbol(char_elem), gen_zero(size_type()), char_type());
    code_assignt assign_char(elem_0, char_at);
    assign_char.location() = location;
    loop_body.copy_to_operands(assign_char);

    // Update char_elem[1] = '\0'
    exprt elem_1 =
      build_index(build_symbol(char_elem), gen_one(size_type()), char_type());
    code_assignt assign_null(elem_1, gen_zero(char_type()));
    assign_null.location() = location;
    loop_body.copy_to_operands(assign_null);

    // Manually construct list_push call to avoid intermediate copy
    const symbolt *push_func_sym =
      converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_push");
    if (!push_func_sym)
      throw std::runtime_error("Push function symbol not found");

    code_function_callt push_call;
    push_call.function() = build_symbol(*push_func_sym);
    push_call.arguments().push_back(build_symbol(temp_list)); // list
    push_call.arguments().push_back(
      build_address_of(build_symbol(char_elem))); // &char_elem
    push_call.arguments().push_back(type_hash);   // type hash
    push_call.arguments().push_back(
      from_integer(BigInt(2), size_type())); // size = 2
    push_call.arguments().push_back(
      from_integer(BigInt(0), size_type())); // float_type_id (not float)
    push_call.arguments().push_back(
      from_integer(BigInt(1), int_type())); // ptr_free: char element
    push_call.type() = bool_type();
    push_call.location() = location;
    loop_body.copy_to_operands(push_call);

    // Increment index: idx++ (V.3: synthetic size_type increment in IREP2)
    exprt idx_inc =
      build_add(build_symbol(idx), gen_one(size_type()), size_type());
    code_assignt idx_update(build_symbol(idx), idx_inc);
    loop_body.copy_to_operands(idx_update);

    // Create while loop
    codet while_loop;
    while_loop.set_statement("while");
    while_loop.copy_to_operands(loop_cond, loop_body);
    converter_.add_instruction(while_loop);

    // Update type map for the elements we just added
    list_type_map[temp_list.id.as_string()].push_back(
      std::make_pair(char_elem.id.as_string(), char_arr_type));

    actual_list = build_symbol(temp_list);
  }

  // A tuple operand: Python's extend() accepts any iterable. Materialise the
  // tuple's components into a fresh list so the list model sees a
  // PyListObject* rather than the tuple struct, which __ESBMC_list_extend
  // would otherwise dereference out of bounds.
  if (converter_.get_tuple_handler().is_tuple_type(actual_list.type()))
  {
    const typet &other_type = converter_.ns.follow(actual_list.type());
    symbolt &temp_list = create_list();
    const std::string &temp_id = temp_list.id.as_string();
    for (const auto &comp : to_struct_type(other_type).components())
    {
      exprt elem = build_member(actual_list, comp.get_name(), comp.type());
      exprt push = build_push_list_call(temp_list, op, elem);
      converter_.add_instruction(push);
      add_type_info(temp_id, std::string(), comp.type());
    }
    actual_list = build_symbol(temp_list);
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
  extend_func_call.function() = build_symbol(*extend_func_sym);
  extend_func_call.arguments().push_back(build_symbol(list));
  extend_func_call.arguments().push_back(actual_list);
  extend_func_call.type() = empty_typet();
  extend_func_call.location() = location;

  return extend_func_call;
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

  exprt pop_call =
    build_call_expr(*pop_func, pyobject_ptr_type, {build_symbol(list), index});
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

  code_declt copied_decl(build_symbol(copied_list));
  copied_decl.location() = location;
  converter_.add_instruction(copied_decl);

  // Build function call
  code_function_callt copy_call;
  copy_call.function() = build_symbol(*copy_func);
  copy_call.arguments().push_back(build_symbol(list));
  copy_call.lhs() = build_symbol(copied_list);
  copy_call.type() = list_type;
  copy_call.location() = location;
  converter_.add_instruction(copy_call);

  // Copy type information from original list to copied list
  copy_type_map_entries(list.id.as_string(), copied_list.id.as_string());

  return build_symbol(copied_list);
}

exprt python_list::build_shallow_copy_call(
  const exprt &src_list,
  const nlohmann::json &element)
{
  const locationt location = converter_.get_location_from_decl(element);
  const typet list_type = converter_.get_type_handler().get_list_type();

  const symbolt *copy_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_copy_shallow");
  if (!copy_func)
    throw std::runtime_error(
      "__ESBMC_list_copy_shallow not found in symbol table");

  // Materialize a list-returning call into a temporary so it can be passed
  // to the model by value (same pattern as handle_slice_assignment's RHS).
  exprt src = src_list;
  if (src.is_function_call())
  {
    code_function_callt &fcall = static_cast<code_function_callt &>(src);
    if (!fcall.function().type().is_code())
      throw std::runtime_error(
        "build_shallow_copy_call: unsupported callable source");
    const typet ret_type = to_code_type(fcall.function().type()).return_type();
    symbolt &tmp =
      converter_.create_tmp_symbol(element, "$tuple_src$", ret_type, exprt());
    code_declt decl(build_symbol(tmp));
    decl.location() = location;
    converter_.add_instruction(decl);
    fcall.lhs() = build_symbol(tmp);
    converter_.add_instruction(fcall);
    src = build_symbol(tmp);
  }
  else if (src.id() == "sideeffect")
  {
    symbolt &tmp =
      converter_.create_tmp_symbol(element, "$tuple_src$", src.type(), exprt());
    code_declt decl(build_symbol(tmp));
    decl.copy_to_operands(src);
    decl.location() = location;
    converter_.add_instruction(decl);
    src = build_symbol(tmp);
  }

  symbolt &copied =
    converter_.create_tmp_symbol(element, "$tuple_copy$", list_type, exprt());
  code_declt copied_decl(build_symbol(copied));
  copied_decl.location() = location;
  converter_.add_instruction(copied_decl);

  // Same nested-list type id as the other shallow-sharing paths (#5102).
  constant_exprt list_type_id(size_type());
  list_type_id.set_value(integer2binary(
    std::hash<std::string>{}(
      converter_.get_type_handler().type_to_string(list_type)),
    config.ansi_c.address_width));

  code_function_callt copy_call;
  copy_call.function() = build_symbol(*copy_func);
  copy_call.arguments().push_back(src);
  copy_call.arguments().push_back(list_type_id);
  copy_call.lhs() = build_symbol(copied);
  copy_call.type() = list_type;
  copy_call.location() = location;
  converter_.add_instruction(copy_call);

  if (src.is_symbol())
    copy_type_map_entries(src.identifier().as_string(), copied.id.as_string());

  return build_symbol(copied);
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
    elem_info.elem_symbol->get_type().is_pointer() &&
    elem_info.elem_symbol->get_type().subtype() == char_type())
    element_arg = build_symbol(*elem_info.elem_symbol);
  else if (elem_info.elem_symbol->get_type().is_struct())
    element_arg = build_address_of(build_symbol(*elem_info.elem_symbol));
  else
    element_arg = build_address_of(build_symbol(*elem_info.elem_symbol));

  // Raise ValueError from the frontend so Python try/except can catch it.
  symbolt &remove_ret = converter_.create_tmp_symbol(
    op, "remove_ret", bool_type(), migrate_expr_back(gen_false_expr())); // V.3
  code_declt remove_ret_decl(build_symbol(remove_ret));
  converter_.add_instruction(remove_ret_decl);

  code_function_callt remove_call;
  remove_call.function() = build_symbol(*remove_func);
  remove_call.lhs() = build_symbol(remove_ret);
  remove_call.arguments().push_back(build_symbol(list)); // list
  remove_call.arguments().push_back(element_arg);        // &value or ptr
  remove_call.arguments().push_back(
    build_symbol(*elem_info.elem_type_sym));              // type_id
  remove_call.arguments().push_back(elem_info.elem_size); // size
  remove_call.type() = bool_type();
  remove_call.location() = elem_info.location;
  converter_.add_instruction(remove_call);

  exprt raise = converter_.get_exception_handler().gen_exception_raise(
    "ValueError", "list.remove(x): x not in list");
  codet throw_code("expression");
  throw_code.operands().push_back(raise);

  code_ifthenelset guard;
  // V.3: build the "not removed" guard (x not in list) in IREP2.
  expr2tc rr2;
  migrate_expr(build_symbol(remove_ret), rr2);
  guard.cond() = migrate_expr_back(not2tc(rr2));
  guard.then_case() = throw_code;
  guard.location() = elem_info.location;
  return guard;
}
