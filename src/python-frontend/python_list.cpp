#include <python-frontend/python_list.h>
#include <python-frontend/python_converter.h>
#include <util/c_types.h>
#include <python-frontend/python_exception_handler.h>
#include <python-frontend/function_call/expr.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/string/string_builder.h>
#include <python-frontend/type_utils.h>
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
#include <irep2/irep2_utils.h>
#include <util/migrate.h>
#include <util/type_byte_size.h>
#include <string>
#include <functional>

namespace
{
// A dynamically-sized array type (array with a non-constant size) does not
// survive the migrate_type round-trip faithfully -- get_width() throws
// (array_type2t::dyn_sized_array_excp) downstream. The list slice/reverse
// paths build nodes over such types, so the IREP2 helpers below fall back to
// the legacy constructor when the relevant type contains one.
bool contains_dyn_array(const typet &t)
{
  if (t.is_array())
  {
    const array_typet &at = to_array_type(t);
    if (at.size().is_nil() || !at.size().is_constant())
      return true;
    return contains_dyn_array(at.subtype());
  }
  if (t.is_pointer())
    return contains_dyn_array(t.subtype());
  return false;
}

// V.3: IREP2 expression-construction helpers (exact round-trip; behaviour-
// preserving -- migrate_expr already lowers the legacy nodes through these
// same paths downstream). Back-migrated for the legacy adjust/goto-convert
// seam.
exprt build_symbol(const symbolt &sym)
{
  if (contains_dyn_array(sym.get_type()))
    return symbol_expr(sym);
  return migrate_expr_back(symbol_expr2tc(sym));
}

exprt build_typecast(const exprt &from, const typet &t)
{
  if (contains_dyn_array(t) || contains_dyn_array(from.type()))
    return typecast_exprt(from, t);
  expr2tc from2;
  migrate_expr(from, from2);
  exprt result = migrate_expr_back(typecast2tc(migrate_type(t), from2));
  // migrate_type does not round-trip #cpp_type; restore the exact target type
  // so legacy typecast_exprt(from, t) is reproduced faithfully.
  result.type() = t;
  return result;
}

// address_of2t's sources here are lvalues (symbols/members/indices), never a
// constant_int or nested address_of, so no guard is needed beyond dyn-array.
exprt build_address_of(const exprt &obj)
{
  if (contains_dyn_array(obj.type()))
    return address_of_exprt(obj);
  expr2tc obj2;
  migrate_expr(obj, obj2);
  return migrate_expr_back(address_of2tc(obj2->type, obj2));
}

// index2t requires an array/vector/symbol source; a pointer source (e.g. the
// array/pointer iterable branch) and dyn-sized arrays (slice results) fall
// back to the legacy node. dyn-array types are checked first to avoid
// migrating an expr whose type would not round-trip.
exprt build_index(const exprt &arr, const exprt &idx, const typet &t)
{
  if (contains_dyn_array(arr.type()) || contains_dyn_array(t))
    return index_exprt(arr, idx, t);
  expr2tc arr2, idx2;
  migrate_expr(arr, arr2);
  migrate_expr(idx, idx2);
  if (
    is_array_type(arr2->type) || is_vector_type(arr2->type) ||
    is_symbol_type(arr2->type))
  {
    exprt result = migrate_expr_back(index2tc(migrate_type(t), arr2, idx2));
    // migrate_type does not round-trip type attributes such as #cpp_type
    // (load-bearing here: it distinguishes a 1-char string element from an
    // 8-bit int). Restore the exact element type so legacy index_exprt(arr,
    // idx, t) is reproduced faithfully.
    result.type() = t;
    return result;
  }
  return index_exprt(arr, idx, t);
}

// Expression-context call `fn(args...)` returning return_type. If the return
// type or any argument type contains a dyn-sized array (which does not
// round-trip), build the legacy side_effect_expr_function_callt instead.
// Caller sets .location() on the result where it did before.
exprt build_call_expr(
  const symbolt &fn,
  const typet &return_type,
  const std::vector<exprt> &args)
{
  bool dyn = contains_dyn_array(return_type);
  for (const exprt &a : args)
    dyn = dyn || contains_dyn_array(a.type());
  if (dyn)
  {
    side_effect_expr_function_callt call;
    call.function() = build_symbol(fn);
    for (const exprt &a : args)
      call.arguments().push_back(a);
    call.type() = return_type;
    return call;
  }
  std::vector<expr2tc> args2;
  args2.reserve(args.size());
  for (const exprt &a : args)
  {
    expr2tc a2;
    migrate_expr(a, a2);
    args2.push_back(std::move(a2));
  }
  return migrate_expr_back(side_effect_function_call2tc(
    migrate_type(return_type), symbol_expr2tc(fn), args2));
}

// Build (*obj).field : field_type in IREP2, back-migrated once (V.3). `obj` is
// a pointer to a PyObject struct, so the dereferenced struct is the resolved
// member source and member2t's source precondition holds.
exprt build_deref_member(
  const exprt &obj,
  const irep_idt &field,
  const typet &field_type)
{
  expr2tc obj2;
  migrate_expr(obj, obj2);
  expr2tc deref2 = dereference2tc(migrate_type(obj.type().subtype()), obj2);
  return migrate_expr_back(member2tc(migrate_type(field_type), deref2, field));
}

// Dereference `ptr` to a value of type `t` (exact round-trip of the single-arg
// dereference_exprt(t) + op0=ptr form, which sets the result type to `t`
// directly). Back-migrated once (V.3).
exprt build_dereference(const exprt &ptr, const typet &t)
{
  expr2tc ptr2;
  migrate_expr(ptr, ptr2);
  exprt result = migrate_expr_back(dereference2tc(migrate_type(t), ptr2));
  // migrate_type does not round-trip #cpp_type; restore the exact target type
  // so legacy dereference_exprt(t)+op0=ptr is reproduced faithfully.
  result.type() = t;
  return result;
}
} // namespace

// Default depth for list comparison if option not set
static const int DEFAULT_LIST_COMPARE_DEPTH = 4;

static bool is_excluded_struct_tag_for_object_ref(const std::string &tag)
{
  return tag.find("dict_") != std::string::npos ||
         tag.find("tag-dict") != std::string::npos ||
         tag.rfind("tag-Optional_", 0) == 0 || tag.rfind("tag-tuple", 0) == 0 ||
         tag == "__python_dict__";
}

static bool
is_empty_user_class_object_type(const typet &type, const namespacet &ns)
{
  typet resolved = type;
  if (resolved.id() == "symbol")
    resolved = ns.follow(resolved);

  if (!resolved.is_struct())
    return false;

  const std::string tag = to_struct_type(resolved).tag().as_string();
  if (tag.empty())
    return false;

  if (
    tag.find("__ESBMC_") != std::string::npos ||
    tag.rfind("tag-struct __ESBMC_", 0) == 0)
    return false;

  if (is_excluded_struct_tag_for_object_ref(tag))
    return false;

  // Empty user-defined classes (no data fields) should be stored as object
  // references. Non-empty classes keep value-copy semantics in list storage.
  return to_struct_type(resolved).components().empty();
}

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
    if (!ann.contains("slice") || !ann["slice"].is_object())
      return typet();
    const auto &slice = ann["slice"];

    // Simple element: list[int], list[str], ...
    if (slice.contains("id") && slice["id"].is_string())
      return type_handler_.get_typet(slice["id"].get<std::string>());

    // Tuple element: list[Tuple[A, B]] / list[tuple[A, B]]. Build the concrete
    // tuple struct (not the opaque 0-member "Tuple") so subscript/unpack of a
    // W[i] read sees real components; see type_handler::get_typet(json) for why
    // the opaque form crashes the SMT cast-to-struct path.
    if (
      slice.contains("_type") && slice["_type"] == "Subscript" &&
      slice.contains("value") && slice["value"].is_object() &&
      slice["value"].contains("id") && slice["value"]["id"].is_string() &&
      (slice["value"]["id"] == "Tuple" || slice["value"]["id"] == "tuple"))
      return type_handler_.get_typet(slice);

    // Nested container element: list[list[T]], list[dict[K, V]], ...
    // Resolve to the container's own type (list[T] -> __ESBMC_PyListObj*),
    // so the caller treats W[i] as a list and re-routes W[i][j] through
    // __ESBMC_list_at.
    if (
      slice.contains("_type") && slice["_type"] == "Subscript" &&
      slice.contains("value") && slice["value"].is_object() &&
      slice["value"].contains("id") && slice["value"]["id"].is_string())
    {
      return type_handler_.get_typet(slice["value"]["id"].get<std::string>());
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

static typet get_elem_type_from_return_annotation(
  const nlohmann::json &function_node,
  const type_handler &type_handler_)
{
  if (
    !function_node.contains("returns") || !function_node["returns"].is_object())
    return typet();

  nlohmann::json annotation_node;
  annotation_node["annotation"] = function_node["returns"];
  return get_elem_type_from_annotation(annotation_node, type_handler_);
}

static typet infer_elem_type_from_call_return(
  const nlohmann::json &call_node,
  python_converter &converter)
{
  if (
    !call_node.is_object() || call_node["_type"] != "Call" ||
    !call_node.contains("func") || !call_node["func"].is_object())
    return typet();

  const auto &func = call_node["func"];
  const auto &ast = converter.ast();
  const auto &type_handler_ = converter.get_type_handler();

  if (func["_type"] == "Name" && func.contains("id") && func["id"].is_string())
  {
    nlohmann::json function_node =
      json_utils::find_function(ast["body"], func["id"].get<std::string>());
    return get_elem_type_from_return_annotation(function_node, type_handler_);
  }

  if (
    func["_type"] == "Attribute" && func.contains("attr") &&
    func["attr"].is_string() && func.contains("value") &&
    func["value"].is_object() && func["value"]["_type"] == "Name" &&
    func["value"].contains("id") && func["value"]["id"].is_string())
  {
    const std::string recv_name = func["value"]["id"].get<std::string>();
    nlohmann::json recv_decl = json_utils::find_var_decl(
      recv_name, converter.current_function_name(), ast);

    if (
      recv_decl.contains("annotation") && recv_decl["annotation"].is_object() &&
      recv_decl["annotation"].contains("id") &&
      recv_decl["annotation"]["id"].is_string())
    {
      const std::string class_name =
        recv_decl["annotation"]["id"].get<std::string>();
      nlohmann::json class_node =
        json_utils::find_class(ast["body"], class_name);
      if (!class_node.is_null() && class_node.contains("body"))
      {
        nlohmann::json function_node = json_utils::find_function(
          class_node["body"], func["attr"].get<std::string>());
        return get_elem_type_from_return_annotation(
          function_node, type_handler_);
      }
    }
  }

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

    // Add 1 for null terminator: size = strlen(s) + 1
    // Use strlen_result.type to ensure exact type match
    exprt one_const = from_integer(1, strlen_result.get_type());
    elem_size = exprt("+", strlen_result.get_type());
    elem_size.copy_to_operands(build_symbol(strlen_result), one_const);
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
    // For bool types, cast to signed long int before taking address
    // This ensures proper storage and retrieval
    if (elem_info.elem_symbol->get_type() == bool_type())
    {
      symbolt &bool_as_long = converter_.create_tmp_symbol(
        op,
        "$bool_as_long$",
        signedbv_typet(config.ansi_c.long_int_width),
        exprt());

      exprt bool_cast = build_typecast(
        build_symbol(*elem_info.elem_symbol),
        signedbv_typet(config.ansi_c.long_int_width));

      code_declt bool_long_decl(build_symbol(bool_as_long));
      bool_long_decl.copy_to_operands(bool_cast);
      bool_long_decl.location() = elem_info.location;
      converter_.add_instruction(bool_long_decl);

      element_arg = build_address_of(build_symbol(bool_as_long));

      // Update elem_size to match
      elem_info.elem_size =
        from_integer(BigInt(config.ansi_c.long_int_width / 8), size_type());
    }
    else
    {
      // For all other types, we must pass address of the value
      element_arg = build_address_of(build_symbol(*elem_info.elem_symbol));
    }
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

  // condition: i < n
  exprt cond("<", bool_type());
  cond.copy_to_operands(build_symbol(i_sym), build_symbol(n_sym));

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

  // i = i + 1
  plus_exprt i_inc(build_symbol(i_sym), gen_one(size_type()));
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

symbolt &python_list::create_list()
{
  locationt location = converter_.get_location_from_decl(list_value_);
  const type_handler &type_handler = converter_.get_type_handler();

  // Create list symbol
  const typet list_type = type_handler.get_list_type();
  symbolt &list_symbol =
    converter_.create_tmp_symbol(list_value_, "$py_list$", list_type, exprt());

  // Declare list
  code_declt list_decl(build_symbol(list_symbol));
  list_decl.location() = location;
  converter_.add_instruction(list_decl);

  // Initialize list with storage array
  const symbolt *create_func_sym =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_create");
  assert(create_func_sym);

  // Add list_create call to the block
  code_function_callt list_create_func_call;
  list_create_func_call.function() = build_symbol(*create_func_sym);
  list_create_func_call.lhs() = build_symbol(list_symbol);
  list_create_func_call.type() = list_type;
  list_create_func_call.location() = location;
  converter_.add_instruction(list_create_func_call);

  return list_symbol;
}

exprt python_list::get()
{
  symbolt &list_symbol = create_list();

  const std::string &list_id = list_symbol.id.as_string();
  locationt location = converter_.get_location_from_decl(list_value_);

  auto materialize_list_elem = [&](const exprt &elem) -> exprt {
    if (elem.is_symbol())
      return elem;

    // A list element that is itself a function call (e.g. ``[nd(), nd()]``)
    // comes back from get_expr as a code_function_callt (a code statement).
    // Assigning that directly to the temp produces a malformed assignment
    // whose RHS is a call statement; it leaks into the SSA and crashes the
    // SMT backend on a null operand (issue #4699). Normalise it to a
    // side-effect call expression so create_tmp_symbol/code_assignt lower it
    // to a proper function call, exactly as a statement-level ``x = nd()``.
    const exprt value_elem = to_value_expr(elem, converter_.name_space());

    symbolt &tmp = converter_.create_tmp_symbol(
      list_value_, "$list_elem$", value_elem.type(), value_elem);
    code_declt decl(build_symbol(tmp));
    decl.location() = location;
    converter_.add_instruction(decl);

    code_assignt assign(build_symbol(tmp), value_elem);
    assign.location() = location;
    converter_.add_instruction(assign);
    return build_symbol(tmp);
  };

  // Convert every element once, recording whether the literal mixes integer and
  // floating-point values. Python promotes such a list to a homogeneous float
  // list (e.g. [4.0, 3] is [4.0, 3.0]); storing the ints unconverted would leave
  // their float_buf slot unset and read back as a stale float (issue #5156).
  std::vector<exprt> elems;
  elems.reserve(list_value_["elts"].size());
  bool has_int = false, has_float = false;
  for (auto &e : list_value_["elts"])
  {
    // Clear current_lhs so that constructor calls inside list elements
    // (e.g. [A(), B()]) create their own self temp variable instead of
    // inheriting the outer assignment target as self.
    exprt *saved_lhs = converter_.current_lhs;
    converter_.current_lhs = nullptr;
    elems.push_back(converter_.get_expr(e));
    converter_.current_lhs = saved_lhs;

    const typet &t = elems.back().type();
    if (t.is_floatbv())
      has_float = true;
    else if (t.is_signedbv() || t.is_unsignedbv() || t.is_bool())
      has_int = true; // bool is int-like in Python (True == 1)
  }

  const bool promote_ints = has_int && has_float;

  for (exprt &elem : elems)
  {
    // Promote integer (and bool) elements to double in a mixed int/float
    // literal, matching the list[float] annotation Python infers for the mix.
    const typet &t = elem.type();
    if (promote_ints && (t.is_signedbv() || t.is_unsignedbv() || t.is_bool()))
      elem = build_typecast(elem, double_type());

    exprt map_elem = materialize_list_elem(elem);

    exprt list_push_func_call =
      build_push_list_call(list_symbol, list_value_, map_elem);
    converter_.add_instruction(list_push_func_call);
    list_type_map[list_id].push_back(
      std::make_pair(map_elem.identifier().as_string(), map_elem.type()));
  }

  return build_symbol(list_symbol);
}

exprt python_list::build_list_from_exprs(const std::vector<exprt> &elems)
{
  symbolt &list_symbol = create_list();
  const std::string &list_id = list_symbol.id.as_string();

  for (const exprt &elem : elems)
  {
    // build_push_list_call materializes the value into a temp symbol, derives
    // the element type-id from its type, and copies it into the list storage.
    exprt push_call = build_push_list_call(list_symbol, list_value_, elem);
    converter_.add_instruction(push_call);
    list_type_map[list_id].push_back(
      std::make_pair(std::string(), elem.type()));
  }

  return build_symbol(list_symbol);
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

  // Get list size
  const symbolt *size_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
  assert(size_func && "list_size function not found");

  symbolt &size_var = converter_.create_tmp_symbol(
    element, "$list_size$", size_type(), gen_zero(size_type()));
  code_declt size_decl(build_symbol(size_var));
  size_decl.location() = location;
  converter_.add_instruction(size_decl);

  code_function_callt size_call;
  size_call.function() = build_symbol(*size_func);
  size_call.arguments().push_back(
    list.type().is_pointer() ? list : build_address_of(list));
  size_call.lhs() = build_symbol(size_var);
  size_call.type() = size_type();
  size_call.location() = location;
  converter_.add_instruction(size_call);

  // Convert index to size_t for comparison and arithmetic
  exprt index_as_size = build_typecast(index, size_type());

  // Create: actual_index = (index < 0) ? (size + index) : index
  exprt is_negative("<", bool_type());
  is_negative.copy_to_operands(index, gen_zero(index.type()));

  // For negative: size + index (since index is negative, this is size - abs(index))
  exprt positive_index("+", size_type());
  positive_index.copy_to_operands(build_symbol(size_var), index_as_size);

  // Choose between positive conversion or original
  if_exprt converted_index(is_negative, positive_index, index_as_size);
  converted_index.type() = size_type();

  if (!config.options.get_bool_option("no-bounds-check"))
  {
    // Runtime guard only for negative-index normalization. This prevents
    // underflowed indices (e.g., [] [-1]) from reaching the backend while
    // preserving legacy behavior for non-negative accesses.
    exprt oob_cond(">=", bool_type());
    oob_cond.copy_to_operands(converted_index, build_symbol(size_var));
    exprt negative_oob("and", bool_type());
    negative_oob.copy_to_operands(is_negative, oob_cond);

    exprt raise = converter_.get_exception_handler().gen_exception_raise(
      "IndexError", "list index out of range");
    codet throw_code("expression");
    throw_code.operands().push_back(raise);

    code_ifthenelset guard;
    guard.cond() = negative_oob;
    guard.then_case() = throw_code;
    guard.location() = location;
    converter_.add_instruction(guard);
  }

  // Use the converted expression directly in the call
  exprt list_at_call = build_call_expr(
    *list_at_func_sym,
    obj_type,
    {list.type().is_pointer() ? list : build_address_of(list),
     converted_index});
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

      // Remainder kept verbatim: leading whitespace is skipped, but CPython
      // keeps trailing whitespace for split(None, 0)
      // (e.g. '  a  '.split(None, 0) == ['a  ']).
      nlohmann::json list_node;
      list_node["_type"] = "List";
      list_node["elts"] = nlohmann::json::array();
      converter.copy_location_fields_from_decl(call_node, list_node);

      nlohmann::json elem;
      elem["_type"] = "Constant";
      elem["value"] = input.substr(first);
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
    new_symbol.set_type(func_type);
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
  code_declt split_decl(build_symbol(split_list));
  split_decl.location() = location;
  converter.add_instruction(split_decl);

  // Emit the function call with lhs so the list has a stable identifier.
  code_function_callt split_call;
  split_call.function() = build_symbol(*func_symbol);
  split_call.arguments() = args;
  split_call.lhs() = build_symbol(split_list);
  split_call.type() = list_type;
  split_call.location() = location;
  converter.add_instruction(split_call);

  // Record element type as string to ensure correct comparisons on parts[i].
  typet elem_type = converter.get_type_handler().build_array(char_type(), 0);
  list_type_map[split_list.id.as_string()].push_back(
    std::make_pair(std::string(), elem_type));

  return build_symbol(split_list);
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
    call.lhs() = build_symbol(lhs);
    converter_.add_instruction(call);
    return build_symbol(lhs);
  };

  auto res = add_lhs_var_bound(e);
  for (auto &ee : res.operands())
  {
    ee = add_lhs_var_bound(ee);
    remove_function_calls_recursive(ee, node);
  }

  return res;
}

const symbolt &python_list::get_str_slice_sym()
{
  const std::string id = "c:@F@__python_str_slice";
  symbolt *sym = converter_.symbol_table().find_symbol(id);
  if (!sym)
  {
    symbolt new_symbol;
    new_symbol.name = "__python_str_slice";
    new_symbol.id = id;
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
    new_symbol.set_type(slice_type);
    converter_.symbol_table().add(new_symbol);
    sym = converter_.symbol_table().find_symbol(id);
  }
  return *sym;
}

exprt python_list::handle_range_slice(
  const exprt &array,
  const nlohmann::json &slice_node)
{
  const namespacet ns(converter_.symbol_table());
  const typet list_type = converter_.get_type_handler().get_list_type();
  const typet resolved_array_type = ns.follow(array.type());
  const typet resolved_list_type = ns.follow(list_type);

  // Handle regular array/string slicing (not list slicing)
  // String parameters come as pointer-to-char, so handle both arrays and char pointers
  bool is_string_slice = (resolved_array_type != resolved_list_type &&
                          resolved_array_type.is_array()) ||
                         (resolved_array_type.is_pointer() &&
                          resolved_array_type.subtype() == char_type());

  // Determine step value (default 1).
  bool has_step = slice_node.contains("step") && !slice_node["step"].is_null();
  long long step_val = 1;
  bool literal_zero_step = false;
  if (has_step)
  {
    const auto &step_node = slice_node["step"];
    if (step_node["_type"] == "UnaryOp" && step_node["op"]["_type"] == "USub")
    {
      step_val = -(long long)step_node["operand"]["value"].get<std::int64_t>();
    }
    else if (step_node["_type"] == "Constant")
    {
      step_val = step_node["value"].get<std::int64_t>();
    }
    if (step_val == 0)
    {
      literal_zero_step = true;
      step_val = 1; // continue with valid value to keep IR consistent
    }
  }
  // Python raises ValueError on step==0; emit a failing assertion so
  // verification reports it rather than silently producing a slice.
  // code_assertt does not insert assume(false), so the rest of the slice
  // IR still runs with step_val==1 — that is harmless: the violation is
  // already reported by the checker.
  if (literal_zero_step)
  {
    code_assertt step_assert(gen_boolean(false));
    step_assert.location() = converter_.get_location_from_decl(slice_node);
    step_assert.location().comment("ValueError: slice step cannot be zero");
    converter_.add_instruction(step_assert);
  }
  bool negative_step = (step_val < 0);

  if (is_string_slice)
  {
    locationt location = converter_.get_location_from_decl(slice_node);

    // For pointer types (function parameters), delegate to __python_str_slice
    // which uses __ESBMC_alloca and survives function returns.
    if (
      resolved_array_type.is_pointer() &&
      resolved_array_type.subtype() == char_type())
    {
      const symbolt &slice_func_ref = get_str_slice_sym();

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
            build_typecast(abs_value, signedbv_typet(64)));
          return neg;
        }
        exprt e = converter_.get_expr(bound);
        e = remove_function_calls_recursive(e, slice_node);
        return build_typecast(e, signedbv_typet(64));
      };

      // Defaults: for positive step start=0,end=MAX; for negative step start=MAX,end=MIN
      // We use large sentinel values; __python_str_slice clamps them
      long long start_default = negative_step ? 999999 : 0;
      long long end_default = negative_step ? -999999 : 999999;

      exprt start_expr = get_bound_expr("lower", start_default);
      exprt end_expr = get_bound_expr("upper", end_default);
      exprt step_expr = from_integer(step_val, signedbv_typet(64));

      // Call __python_str_slice(s, start, end, step) as side-effect expression
      exprt slice_call = build_call_expr(
        slice_func_ref,
        pointer_typet(char_type()),
        {array, start_expr, end_expr, step_expr});
      slice_call.location() = location;

      return slice_call;
    }

    // For array types (local string literals), generate inline loop
    auto to_size_expr = [&](const exprt &expr) -> exprt {
      if (expr.type() == size_type())
        return expr;
      return build_typecast(expr, size_type());
    };
    auto size_add = [&](const exprt &lhs, const exprt &rhs) -> exprt {
      plus_exprt out(lhs, rhs);
      out.type() = size_type();
      return out;
    };
    auto size_sub = [&](const exprt &lhs, const exprt &rhs) -> exprt {
      minus_exprt out(lhs, rhs);
      out.type() = size_type();
      return out;
    };
    auto size_mul = [&](const exprt &lhs, const exprt &rhs) -> exprt {
      mult_exprt out(lhs, rhs);
      out.type() = size_type();
      return out;
    };
    auto size_div = [&](const exprt &lhs, const exprt &rhs) -> exprt {
      div_exprt out(lhs, rhs);
      out.type() = size_type();
      return out;
    };

    // Determine element type and logical length
    typet elem_type;
    exprt array_len;
    exprt logical_len;

    {
      const array_typet &src_type = to_array_type(resolved_array_type);
      elem_type = src_type.subtype();
      array_len = to_size_expr(src_type.size());
      // For char arrays (strings), exclude null terminator from logical length
      logical_len = (elem_type == char_type())
                      ? size_sub(array_len, gen_one(size_type()))
                      : array_len;
    }

    // Process slice bounds (handles null, negative indices)
    auto process_bound =
      [&](const std::string &bound_name, const exprt &default_value) -> exprt {
      if (!slice_node.contains(bound_name) || slice_node[bound_name].is_null())
        return to_size_expr(default_value);

      const auto &bound = slice_node[bound_name];

      // Check if negative index
      if (bound["_type"] == "UnaryOp" && bound["op"]["_type"] == "USub")
      {
        exprt abs_value = to_size_expr(converter_.get_expr(bound["operand"]));
        // Clamp to 0 when abs_value > logical_len (avoids unsigned underflow)
        exprt overflow(">", bool_type());
        overflow.copy_to_operands(abs_value, logical_len);
        exprt converted = size_sub(logical_len, abs_value);
        return if_exprt(overflow, gen_zero(size_type()), converted);
      }

      exprt e = converter_.get_expr(bound);
      return to_size_expr(remove_function_calls_recursive(e, slice_node));
    };

    // Process bounds: defaults depend on step direction
    exprt lower_expr, upper_expr;
    if (negative_step)
    {
      // For negative step: default lower = len-1, default upper = -1 (before 0)
      lower_expr =
        process_bound("lower", size_sub(logical_len, gen_one(size_type())));
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
        slice_len = size_add(lower_expr, gen_one(size_type()));
      }
    }
    else if (step_val != 1)
    {
      // For step > 1: length = ceil((upper - lower) / step)
      exprt range =
        size_sub(to_size_expr(upper_expr), to_size_expr(lower_expr));
      exprt step_const = from_integer(step_val, size_type());
      exprt step_minus_one = from_integer(step_val - 1, size_type());
      slice_len = size_div(size_add(range, step_minus_one), step_const);
    }
    else
    {
      slice_len = size_sub(to_size_expr(upper_expr), to_size_expr(lower_expr));
    }

    // Create result array type with extra space for null terminator
    plus_exprt result_size(slice_len, gen_one(size_type()));
    result_size.type() = size_type();
    array_typet result_type(elem_type, result_size);

    // Create temporary for sliced array
    symbolt &result = converter_.create_tmp_symbol(
      slice_node, "$array_slice$", result_type, exprt());

    code_declt result_decl(build_symbol(result));
    result_decl.location() = location;
    converter_.add_instruction(result_decl);

    // Create loop: for i = 0; i < slice_len; i++
    symbolt &idx = converter_.create_tmp_symbol(
      slice_node, "$i$", size_type(), gen_zero(size_type()));
    code_assignt idx_init(build_symbol(idx), gen_zero(size_type()));
    converter_.add_instruction(idx_init);

    exprt cond("<", bool_type());
    cond.copy_to_operands(build_symbol(idx), slice_len);

    code_blockt body;

    // Compute source index based on step direction
    exprt src_idx;
    if (negative_step)
    {
      // result[i] = array[lower - i] (for step=-1)
      // For other negative steps: result[i] = array[lower - i * |step|]
      if (step_val == -1)
        src_idx = size_sub(to_size_expr(lower_expr), build_symbol(idx));
      else
      {
        exprt abs_step = from_integer(-step_val, size_type());
        src_idx = size_sub(
          to_size_expr(lower_expr), size_mul(build_symbol(idx), abs_step));
      }
    }
    else if (step_val != 1)
    {
      // result[i] = array[lower + i * step]
      exprt step_const = from_integer(step_val, size_type());
      src_idx = size_add(
        to_size_expr(lower_expr), size_mul(build_symbol(idx), step_const));
    }
    else
    {
      // result[i] = array[lower + i]
      src_idx = size_add(to_size_expr(lower_expr), build_symbol(idx));
    }

    exprt src = build_index(array, src_idx, elem_type);
    exprt dst = build_index(build_symbol(result), build_symbol(idx), elem_type);
    code_assignt assign(dst, src);
    body.copy_to_operands(assign);

    // i++
    plus_exprt incr(build_symbol(idx), gen_one(size_type()));
    code_assignt update(build_symbol(idx), incr);
    body.copy_to_operands(update);

    codet loop;
    loop.set_statement("while");
    loop.copy_to_operands(cond, body);
    converter_.add_instruction(loop);

    // Add null terminator at result[slice_len]
    exprt null_pos = build_index(build_symbol(result), slice_len, elem_type);
    code_assignt add_null(null_pos, gen_zero(elem_type));
    add_null.location() = location;
    converter_.add_instruction(add_null);

    return build_symbol(result);
  }

  // Handle list slicing
  symbolt &sliced_list = create_list();
  const locationt location = converter_.get_location_from_decl(list_value_);

  // Compute list size symbol once. Both bound resolution and (for negative
  // step) the size-1 default reuse this temporary instead of re-emitting
  // __ESBMC_list_size calls.
  const symbolt *size_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
  if (!size_func)
    throw std::runtime_error("__ESBMC_list_size not found in symbol table");

  exprt size_call = build_call_expr(
    *size_func,
    size_type(),
    {array.type().is_pointer() ? array : build_address_of(array)});
  size_call.location() = location;

  symbolt &size_sym = converter_.create_tmp_symbol(
    list_value_, "$slice_size$", size_type(), exprt());
  code_declt size_decl(build_symbol(size_sym));
  size_decl.copy_to_operands(size_call);
  converter_.add_instruction(size_decl);

  // Counter type: signed for negative step (must reach -1 sentinel without
  // unsigned underflow); unsigned otherwise to match the existing IR shape
  // for the common step==1 case.
  const typet signed_t = signed_size_type();
  const typet counter_type = negative_step ? signed_t : size_type();
  auto signed_add = [&](const exprt &lhs, const exprt &rhs) -> exprt {
    plus_exprt out(lhs, rhs);
    out.type() = signed_t;
    return out;
  };
  auto signed_sub = [&](const exprt &lhs, const exprt &rhs) -> exprt {
    minus_exprt out(lhs, rhs);
    out.type() = signed_t;
    return out;
  };

  // Resolve a slice bound following CPython's slice.indices(len) semantics.
  // Returns an expression of `counter_type`. We always work through signed
  // arithmetic so negative bounds (literal or runtime), the -1 stop sentinel,
  // and the clamp tree stay representable.
  auto resolve_bound = [&](const std::string &name, bool is_upper) -> exprt {
    const exprt size_signed = build_typecast(build_symbol(size_sym), signed_t);
    const exprt zero_s = from_integer(0, signed_t);
    const exprt one_s = from_integer(1, signed_t);

    // Step 1: produce a signed `resolved` value.
    //   - missing bound → step-direction default
    //   - present bound → val (signed); if val < 0 add size at runtime so the
    //     fix also covers non-literal negative bounds like xs[5:b:-1].
    exprt resolved;
    if (!slice_node.contains(name) || slice_node[name].is_null())
    {
      if (negative_step)
        resolved = is_upper ? from_integer(-1, signed_t)
                            : signed_sub(size_signed, one_s);
      else
        resolved = is_upper ? size_signed : zero_s;
    }
    else
    {
      exprt val_signed =
        build_typecast(converter_.get_expr(slice_node[name]), signed_t);
      exprt size_plus_val = signed_add(size_signed, val_signed);
      exprt is_neg("<", bool_type());
      is_neg.copy_to_operands(val_signed, zero_s);
      resolved = if_exprt(is_neg, size_plus_val, val_signed);
      resolved.type() = signed_t;
    }

    // Step 2: clamp to the CPython-defined window for this step direction.
    //   pos step: [0, size]                  (start and stop)
    //   neg step start: [-1, size-1]
    //   neg step stop:  [-1, size-1] (stop>=size is harmless — the loop
    //                                terminates immediately — but we clamp
    //                                to keep the expression in a known range)
    const exprt under = negative_step ? from_integer(-1, signed_t) : zero_s;
    const exprt over =
      negative_step ? signed_sub(size_signed, one_s) : size_signed;

    exprt is_under("<", bool_type());
    is_under.copy_to_operands(resolved, under);
    exprt c1 = if_exprt(is_under, under, resolved);
    c1.type() = signed_t;

    exprt is_over(">", bool_type());
    is_over.copy_to_operands(c1, over);
    exprt c2 = if_exprt(is_over, over, c1);
    c2.type() = signed_t;

    return negative_step ? c2 : build_typecast(c2, size_type());
  };

  exprt lower_expr = resolve_bound("lower", false);
  exprt upper_expr = resolve_bound("upper", true);

  // Initialize counter at lower.
  symbolt &counter = converter_.create_tmp_symbol(
    list_value_, "counter", counter_type, lower_expr);
  code_assignt counter_init(build_symbol(counter), lower_expr);
  converter_.add_instruction(counter_init);

  // Loop condition: counter < upper (positive step) or counter > upper
  // (negative step). Negative step uses signed comparison thanks to the
  // signed counter/upper type.
  exprt loop_condition(negative_step ? ">" : "<", bool_type());
  loop_condition.copy_to_operands(build_symbol(counter), upper_expr);

  // Loop body
  code_blockt loop_body;

  exprt index_expr = build_symbol(counter);
  if (negative_step)
    index_expr = build_typecast(index_expr, size_type());

  const exprt list_at_call = build_list_at_call(array, index_expr, list_value_);
  const symbolt &at_result = converter_.create_tmp_symbol(
    list_value_,
    "tmp_list_at",
    pointer_typet(converter_.get_type_handler().get_list_element_type()),
    exprt());

  code_declt at_decl(build_symbol(at_result));
  at_decl.copy_to_operands(list_at_call);
  loop_body.copy_to_operands(at_decl);

  // Shallow append: preserve element value pointers so nested lists survive the
  // slice copy uncorrupted (esbmc/esbmc#5102).
  const symbolt *push_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_push_shallow");
  if (!push_func)
    throw std::runtime_error("Push function symbol not found");

  // Nested-list elements keep their inner pointer; scalars are byte-copied, so
  // the slice copy does not corrupt nested lists (#5102).
  constant_exprt slice_list_type_id(size_type());
  slice_list_type_id.set_value(integer2binary(
    std::hash<std::string>{}(converter_.get_type_handler().type_to_string(
      converter_.get_type_handler().get_list_type())),
    config.ansi_c.address_width));
  exprt push_call = build_call_expr(
    *push_func,
    bool_type(),
    {build_symbol(sliced_list), build_symbol(at_result), slice_list_type_id});
  push_call.location() = location;
  loop_body.copy_to_operands(converter_.convert_expression_to_code(push_call));

  // counter += step_val. step_val is signed; for positive step it's >0 and
  // converts to size_type cleanly via from_integer.
  exprt step_const = from_integer(step_val, counter_type);
  exprt increment("+", counter_type);
  increment.copy_to_operands(build_symbol(counter), step_const);
  code_assignt counter_update(build_symbol(counter), increment);
  loop_body.copy_to_operands(counter_update);

  codet while_loop;
  while_loop.set_statement("while");
  while_loop.copy_to_operands(loop_condition, loop_body);
  converter_.add_instruction(while_loop);

  // Update type map for sliced elements. Element types are taken positionally
  // from the source list literal. This is valid when step == 1 and each bound
  // is either absent (defaulting to the full range) or a non-negative constant
  // literal. Absent bounds cover the idiomatic full copy src[:] and half-open
  // slices like src[1:] / src[:2], whose nested-list element types would
  // otherwise be lost — corrupting reads such as sl[0][0] (esbmc/esbmc#5102).
  // For step != 1 the index mapping is no longer lower + i, and for negative or
  // non-constant bounds the static range is unknown; both keep the generic
  // fallback below.
  bool bounds_usable = true;
  bool has_lower = false;
  bool has_upper = false;
  size_t lower_bound = 0;
  size_t upper_bound = 0;
  for (int which = 0; which < 2; ++which)
  {
    const char *key = which == 0 ? "lower" : "upper";
    bool &present = which == 0 ? has_lower : has_upper;
    size_t &out = which == 0 ? lower_bound : upper_bound;

    present = slice_node.contains(key) && !slice_node[key].is_null();
    if (!present)
      continue; // absent / None: defaults to the list bound below
    const nlohmann::json &b = slice_node[key];
    if (
      b.is_object() && b.contains("_type") && b["_type"] == "Constant" &&
      b.contains("value") && b["value"].is_number_integer() &&
      b["value"].get<long long>() >= 0)
      out = b["value"].get<size_t>();
    else
      bounds_usable = false; // negative or non-constant: not known statically
  }

  if (step_val == 1 && bounds_usable)
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

    // Only update type map for actual lists (not strings or other types)
    if (
      !list_node.is_null() && list_node.contains("value") &&
      list_node["value"].contains("elts") &&
      list_node["value"]["elts"].is_array())
    {
      const size_t elts_size = list_node["value"]["elts"].size();
      const size_t begin = has_lower ? std::min(lower_bound, elts_size) : 0;
      const size_t end =
        has_upper ? std::min(upper_bound, elts_size) : elts_size;

      for (size_t i = begin; i < end; ++i)
      {
        exprt element = converter_.get_expr(list_node["value"]["elts"][i]);
        if (!element.is_symbol())
        {
          symbolt &tmp = converter_.create_tmp_symbol(
            list_value_, "$slice_elem$", element.type(), element);
          code_declt decl(build_symbol(tmp));
          decl.location() = location;
          converter_.add_instruction(decl);

          code_assignt assign(build_symbol(tmp), element);
          assign.location() = location;
          converter_.add_instruction(assign);
          element = build_symbol(tmp);
        }

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

  return build_symbol(sliced_list);
}

void python_list::handle_slice_assignment(
  const exprt &list_expr,
  const nlohmann::json &slice_node,
  const nlohmann::json &value_node)
{
  // The step is restricted to a constant integer literal (or absent) for
  // now; the model receives it as a runtime value and branches on it at
  // solve time, so this could be lifted, but a symbolic step is untested.
  // Mirrors the literal-only step extraction in handle_range_slice, but
  // rejects a non-constant step instead of silently slicing with step 1.
  long long step_val = 1;
  if (slice_node.contains("step") && !slice_node["step"].is_null())
  {
    const auto &step_node = slice_node["step"];
    if (
      step_node["_type"] == "UnaryOp" && step_node["op"]["_type"] == "USub" &&
      step_node["operand"]["_type"] == "Constant" &&
      step_node["operand"]["value"].is_number_integer())
      step_val = -(long long)step_node["operand"]["value"].get<std::int64_t>();
    else if (
      step_node["_type"] == "Constant" &&
      step_node["value"].is_number_integer())
      step_val = step_node["value"].get<std::int64_t>();
    else
      throw std::runtime_error(
        "List slice assignment requires a constant integer step");
  }

  const locationt location = converter_.get_location_from_decl(slice_node);

  // Evaluate the right-hand side; materialize a call (e.g. sorted(...)) into
  // a temporary so its result can be passed to the model by address. get_expr
  // returns calls as code_function_callt, which must be emitted as an
  // instruction with the temporary as its lhs — embedding it as a decl
  // operand would leak a side effect into the GOTO program.
  exprt rhs = converter_.get_expr(value_node);
  if (rhs.is_function_call())
  {
    code_function_callt &fcall = static_cast<code_function_callt &>(rhs);
    if (!fcall.function().type().is_code())
      throw std::runtime_error(
        "Unsupported callable on list slice assignment right-hand side");
    const typet ret_type = to_code_type(fcall.function().type()).return_type();
    symbolt &tmp = converter_.create_tmp_symbol(
      value_node, "$slice_assign_rhs$", ret_type, exprt());
    code_declt decl(build_symbol(tmp));
    decl.location() = location;
    converter_.add_instruction(decl);
    fcall.lhs() = build_symbol(tmp);
    converter_.add_instruction(fcall);
    rhs = build_symbol(tmp);
  }
  else if (rhs.id() == "sideeffect")
  {
    symbolt &tmp = converter_.create_tmp_symbol(
      value_node, "$slice_assign_rhs$", rhs.type(), exprt());
    code_declt decl(build_symbol(tmp));
    decl.copy_to_operands(rhs);
    decl.location() = location;
    converter_.add_instruction(decl);
    rhs = build_symbol(tmp);
  }

  const namespacet ns(converter_.symbol_table());
  const typet list_type = converter_.get_type_handler().get_list_type();
  const typet resolved_list_type = ns.follow(list_type);
  const typet rhs_type = ns.follow(rhs.type());
  const bool rhs_is_list =
    rhs_type == resolved_list_type ||
    (rhs_type.is_pointer() &&
     ns.follow(rhs_type.subtype()) == resolved_list_type);
  if (!rhs_is_list)
    throw std::runtime_error(
      "List slice assignment requires a list right-hand side");

  const typet i64 = signedbv_typet(64);
  auto bound_expr = [&](const char *name, bool &present) -> exprt {
    present = slice_node.contains(name) && !slice_node[name].is_null();
    if (!present)
      return from_integer(0, i64);
    exprt e = converter_.get_expr(slice_node[name]);
    e = remove_function_calls_recursive(e, slice_node);
    return build_typecast(e, i64);
  };
  bool has_lower = false, has_upper = false;
  exprt lower = bound_expr("lower", has_lower);
  exprt upper = bound_expr("upper", has_upper);

  const symbolt *fn =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_slice_assign");
  if (!fn)
    throw std::runtime_error(
      "__ESBMC_list_slice_assign not found in symbol table");

  // Same nested-list type id as the slice-read path: lets the model keep
  // nested-list records shared instead of byte-copying them (#5102).
  constant_exprt list_type_id(size_type());
  list_type_id.set_value(integer2binary(
    std::hash<std::string>{}(
      converter_.get_type_handler().type_to_string(list_type)),
    config.ansi_c.address_width));

  // Statically-known element byte size, passed to the model so its per-element
  // copy uses a compile-time-constant size and takes __ESBMC_copy_value's
  // branch-free fast path, instead of reading the symbolic-index field
  // src->items[k].size (which forces memcpy's per-byte loop to unwind per
  // element — the dominant cost on large slice assignments). Emitted ONLY for
  // an unambiguous fixed-width scalar element type; 0 otherwise, which makes
  // the model fall back to the per-element o->size read (exact prior behaviour,
  // so a missing/ambiguous type can never produce a wrong copy length).
  // Derive the element byte size from the RHS source list's elements so the
  // model copies with a compile-time-constant size (fast path) instead of the
  // symbolic-index field read src->items[k].size. We read the *converted* RHS
  // element type, which is the actual data being stored, so the size is always
  // correct. Only a plain fixed-width scalar (bitvector) is emitted; anything
  // else (pointers, structs, strings, nested lists, non-literal/heterogeneous
  // RHS) keeps elem_size 0, making the model fall back to the per-element
  // o->size read — exact prior behaviour, so a wrong length is impossible.
  auto scalar_width = [](const typet &t) -> size_t {
    if (
      (t.id() == "signedbv" || t.id() == "unsignedbv" || t.id() == "floatbv" ||
       t.id() == "fixedbv") &&
      !t.width().empty())
      return std::stoull(t.width().as_string(), nullptr, 10) / 8;
    return 0;
  };
  size_t elem_size_bytes = 0;
  if (
    value_node.contains("_type") && value_node["_type"] == "List" &&
    value_node.contains("elts") && value_node["elts"].is_array() &&
    !value_node["elts"].empty())
  {
    // RHS is a list literal: size from its (uniform) element types.
    bool uniform = true;
    size_t w = 0;
    for (const auto &elt : value_node["elts"])
    {
      size_t ew = 0;
      try
      {
        ew = scalar_width(converter_.get_expr(elt).type());
      }
      catch (...)
      {
        uniform = false;
        break;
      }
      if (ew == 0 || (w != 0 && ew != w))
      {
        uniform = false;
        break;
      }
      w = ew;
    }
    if (uniform)
      elem_size_bytes = w;
  }
  else if (
    value_node.contains("_type") && value_node["_type"] == "Name" &&
    value_node.contains("id"))
  {
    // RHS is a list variable (includes self-assignment l[...] = l): size from
    // its declared element type.
    const typet et = get_list_element_type(value_node["id"].get<std::string>());
    if (!et.is_nil())
      elem_size_bytes = scalar_width(et);
  }
  constant_exprt elem_size(size_type());
  elem_size.set_value(
    integer2binary(BigInt(elem_size_bytes), config.ansi_c.address_width));

  exprt call = build_call_expr(
    *fn,
    bool_type(),
    {list_expr.type().is_pointer() ? list_expr : build_address_of(list_expr),
     lower,
     from_integer(has_lower ? 1 : 0, int_type()),
     upper,
     from_integer(has_upper ? 1 : 0, int_type()),
     from_integer(step_val, i64),
     rhs.type().is_pointer() ? rhs : build_address_of(rhs),
     list_type_id,
     elem_size});
  call.location() = location;
  converter_.add_instruction(converter_.convert_expression_to_code(call));
}

exprt python_list::handle_index_access(
  const exprt &array,
  const nlohmann::json &slice_node)
{
  const namespacet ns(converter_.symbol_table());
  const typet resolved_array_type = ns.follow(array.type());

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
    // Both compile-time branches below assume the negated operand is a
    // constant literal (a[-1]). For a non-constant operand (a[-i]) the value
    // is only known at runtime, so leave pos_expr (= -i) and index untouched:
    // build_list_at_call normalizes the negative index at runtime via
    // __ESBMC_list_size, and the element-type lookup falls back to element 0,
    // which is correct for the homogeneous lists ESBMC models (#4926).
    const bool operand_is_constant =
      slice_node.contains("operand") &&
      slice_node["operand"]["_type"] == "Constant" &&
      slice_node["operand"].contains("value");

    if (!operand_is_constant)
    {
      // Nothing to do: runtime normalization handles a[-i].
    }
    // For char* (string parameters), skip compile-time normalization: the size
    // is not known statically, so normalization happens at runtime in the
    // char* indexing block below.
    else if (
      !array.type().is_pointer() &&
      (list_node.is_null() || list_node["value"]["_type"] != "List"))
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
  const bool is_char_array = resolved_array_type.is_array() &&
                             resolved_array_type.subtype() == char_type();
  const bool is_char_ptr = resolved_array_type.is_pointer() &&
                           resolved_array_type.subtype() == char_type();

  if (
    (array.type().is_symbol() || array.type().subtype().is_symbol()) &&
    !is_char_array && !is_char_ptr)
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
        if (!type_map_it->second.empty())
        {
          // Homogeneous lists (e.g. list comprehensions) record a single
          // element-type entry; ESBMC models lists as homogeneous, so reuse
          // that entry for any in-structure index.  Without this, a constant
          // outer index >= 1 into a comprehension-built nested list skips the
          // nested-element handling below, the inner element type (float) is
          // lost, and the value reaches the SMT FP encoder as a non-FP sort
          // (get_exponent_width abort / Z3 "rm and fp sorts", #5129).  Literal
          // lists record one entry per element, so for an in-bounds index
          // eff_index == index and behaviour is unchanged.  The runtime value
          // is still read at pos_expr below, so only the static type is taken
          // from the homogeneous entry.
          const size_t eff_index = index < type_map_it->second.size()
                                     ? index
                                     : type_map_it->second.size() - 1;
          const std::string &elem_id = type_map_it->second.at(eff_index).first;
          elem_type = type_map_it->second.at(eff_index).second;

          if (elem_type == converter_.get_type_handler().get_list_type())
          {
            // Nested-list element.  The recorded elem_id names the inner-list
            // symbol, but copy_type_info copies that id verbatim across a
            // function-return boundary (Q = build()), where it is a *callee*
            // frame local — returning build_symbol(elem_id) would reference a
            // symbol that is never assigned in the caller's symex frame, so its
            // value is nondet (float_buf OOB / wrong value, #5103/#5102).
            //
            // Instead, read the inner list pointer at runtime from `array`
            // (valid in every scope: list_at returns the stored pointer, so
            // aliasing/mutation semantics are preserved), bind it to a fresh
            // caller-scope symbol, and copy the inner element-type map onto
            // that symbol so deeper subscripts (Q[i][j]) still resolve
            // int/float.  For parameter-annotation-only entries (elem_id
            // empty) fall through to the dynamic __ESBMC_list_at path below.
            if (!elem_id.empty())
            {
              const locationt loc =
                converter_.get_location_from_decl(list_value_);

              exprt list_at_call =
                build_list_at_call(array, pos_expr, list_value_);
              exprt inner_ptr = extract_pyobject_value(list_at_call, elem_type);

              symbolt &inner_sym = converter_.create_tmp_symbol(
                list_value_, "$nested_list$", elem_type, exprt());
              code_declt inner_decl(build_symbol(inner_sym));
              inner_decl.location() = loc;
              converter_.add_instruction(inner_decl);

              code_assignt inner_assign(build_symbol(inner_sym), inner_ptr);
              inner_assign.location() = loc;
              converter_.add_instruction(inner_assign);

              copy_type_info(elem_id, inner_sym.id.as_string());
              return build_symbol(inner_sym);
            }
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
      if (
        elem_type == typet() && !list_node.is_null() &&
        list_node.contains("value") && list_node["value"].is_object() &&
        list_node["value"].contains("_type") &&
        list_node["value"]["_type"] == "Call")
      {
        elem_type =
          infer_elem_type_from_call_return(list_node["value"], converter_);
        if (elem_type != typet() && array.is_symbol())
        {
          const std::string &list_name = array.identifier().as_string();
          if (list_type_map[list_name].empty())
            list_type_map[list_name].push_back(std::make_pair("", elem_type));
        }
      }

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
                          // (promotion-aware, see infer_literal_element_type).
                          elem_type = infer_literal_element_type(values[i]);
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
            // Infer element type from the literal, accounting for the int->float
            // promotion applied to mixed numeric literals at construction.
            elem_type = infer_literal_element_type(list_node["value"]);
          }
        }
      }
    }

    // For variable indices, prefer compile-time list element type information
    // when available.
    if (elem_type == typet() && array.is_symbol())
    {
      const std::string &list_name = array.identifier().as_string();
      auto type_map_it = list_type_map.find(list_name);
      if (type_map_it != list_type_map.end() && !type_map_it->second.empty())
        elem_type = type_map_it->second.back().second;
    }

    // Nested subscript chains like W[i][j] where the base is a Name with a
    // nested list annotation (e.g. list[list[T]]).  list_node is null in this
    // case because list_value_["value"] is itself a Subscript rather than a
    // Name, so the standard paths above can't see W's annotation.  Walk the
    // Subscript chain down to the base Name and peel that many layers off the
    // annotation, then hand the result to the existing extractor.
    if (elem_type == typet())
    {
      size_t subscript_depth = 0;
      const nlohmann::json *cur = &list_value_;
      while (cur->is_object() && cur->contains("value") &&
             (*cur)["value"].is_object() && (*cur)["value"].contains("_type") &&
             (*cur)["value"]["_type"] == "Subscript")
      {
        ++subscript_depth;
        cur = &(*cur)["value"];
      }
      if (
        subscript_depth > 0 && cur->is_object() && cur->contains("value") &&
        (*cur)["value"].is_object() && (*cur)["value"].contains("id") &&
        (*cur)["value"]["id"].is_string())
      {
        nlohmann::json base_decl = json_utils::find_var_decl(
          (*cur)["value"]["id"].get<std::string>(),
          converter_.current_function_name(),
          converter_.ast());
        if (!base_decl.is_null() && base_decl.contains("annotation"))
        {
          nlohmann::json drilled = base_decl["annotation"];
          for (size_t k = 0; k < subscript_depth; ++k)
          {
            if (
              !drilled.is_object() || !drilled.contains("_type") ||
              drilled["_type"] != "Subscript" || !drilled.contains("slice"))
            {
              drilled = nlohmann::json();
              break;
            }
            drilled = drilled["slice"];
          }
          if (!drilled.is_null())
          {
            nlohmann::json synth;
            synth["annotation"] = drilled;
            elem_type = get_elem_type_from_annotation(
              synth, converter_.get_type_handler());
          }
        }
      }
    }

    // Python allows indexing lists with unknown element type in dynamic code.
    // If we resolved the index but not the element type, treat elements as Any
    // instead of raising a frontend conversion error.
    if (pos_expr != exprt() && elem_type == typet())
      elem_type = any_type();

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
        // If we can prove out-of-bounds from a concrete list literal, raise
        // IndexError in-model so Python try/except(IndexError) can catch it.
        if (
          array.is_symbol() && !list_node.is_null() &&
          list_node.contains("value") && list_node["value"].is_object() &&
          list_node["value"].contains("elts") &&
          list_node["value"]["elts"].is_array())
        {
          bool negative_index = false;
          size_t index_abs = 0;

          if (slice_node["_type"] == "Constant" && slice_node.contains("value"))
          {
            index_abs = slice_node["value"].get<size_t>();
          }
          else
          {
            negative_index = true;
            index_abs = slice_node["operand"]["value"].get<size_t>();
          }

          const size_t list_size = list_node["value"]["elts"].size();
          const bool out_of_bounds =
            (!negative_index && index_abs >= list_size) ||
            (negative_index && index_abs > list_size);

          if (out_of_bounds)
          {
            exprt raise =
              converter_.get_exception_handler().gen_exception_raise(
                "IndexError", "list index out of range");
            codet throw_code("expression");
            throw_code.operands().push_back(raise);
            converter_.add_instruction(throw_code);
            // Unreachable after raise, but return a placeholder expression.
            return gen_zero(size_type());
          }
        }

        // If array is a constant placeholder (e.g., from a chained OOB access),
        // we're in dead code after a prior IndexError. Emit IndexError and return
        // a placeholder rather than crashing the frontend.
        if (!array.is_symbol() && array.is_constant())
        {
          exprt raise = converter_.get_exception_handler().gen_exception_raise(
            "IndexError", "list index out of range");
          codet throw_code("expression");
          throw_code.operands().push_back(raise);
          converter_.add_instruction(throw_code);
          return gen_zero(size_type());
        }

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

    // A non-constant subscript into a heterogeneous int/float list cannot
    // resolve a single static element type from the index, so the code above
    // defaults to element 0's type. That misreads any element of the other
    // numeric kind (e.g. a float stored at index 2 read as an int, #5160).
    // Python promotes int to float in numeric expressions, so treat the read
    // as float and dispatch on the stored type_id at runtime (see
    // extract_pyobject_value), which yields the correct value for both kinds.
    const bool constant_index =
      slice_node["_type"] == "Constant" ||
      (slice_node["_type"] == "UnaryOp" && slice_node.contains("op") &&
       slice_node["op"]["_type"] == "USub" && slice_node.contains("operand") &&
       slice_node["operand"]["_type"] == "Constant");
    const bool mixed_numeric =
      !constant_index && array.is_symbol() &&
      has_mixed_numeric_types(array.identifier().as_string());
    if (mixed_numeric)
      elem_type = double_type();

    // Build list access and cast result
    exprt list_at_call = build_list_at_call(array, pos_expr, list_value_);

    // The mixed-numeric read dereferences the element three times (type_id,
    // float_idx, value), so bind the __ESBMC_list_at result to a temp first and
    // evaluate the access once instead of three times on the hot path.
    if (mixed_numeric)
    {
      const locationt loc = converter_.get_location_from_decl(list_value_);
      symbolt &elem_obj = converter_.create_tmp_symbol(
        list_value_,
        "$list_elem_obj$",
        pointer_typet(converter_.get_type_handler().get_list_element_type()),
        exprt());
      code_declt obj_decl(build_symbol(elem_obj));
      obj_decl.location() = loc;
      converter_.add_instruction(obj_decl);
      code_assignt obj_assign(build_symbol(elem_obj), list_at_call);
      obj_assign.location() = loc;
      converter_.add_instruction(obj_assign);
      list_at_call = build_symbol(elem_obj);
    }

    // Extract and dereference PyObject value
    return extract_pyobject_value(list_at_call, elem_type, mixed_numeric);
  }

  // Handle static string indexing with IndexError on out-of-bounds
  if (is_char_array)
  {
    exprt idx = pos_expr;
    if (idx.type() != size_type())
      idx = build_typecast(idx, size_type());

    // Logical string length excludes the null terminator
    exprt array_size = to_array_type(resolved_array_type).size();
    if (array_size.type() != size_type())
      array_size = build_typecast(array_size, size_type());
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

    // Tag with #cpp_type==char so downstream consumers (notably
    // python_converter::get_python_type_category) can distinguish a 1-char
    // string element from an arbitrary 8-bit int (e.g. dtype=np.int8) without
    // resorting to a fragile width-only heuristic.
    typet char_t = char_type();
    type_utils::set_cpp_type(char_t, "char");
    return build_index(array, idx, char_t);
  }

  // For char* (string function parameter), implement Python single-index
  // semantics: normalize negative indices, raise IndexError on out-of-bounds,
  // then return a fresh single-char null-terminated string via
  // __python_str_slice so the result is never a dangling pointer.
  if (is_char_ptr)
  {
    typet ll_type = signedbv_typet(64);
    locationt loc = converter_.get_location_from_decl(list_value_);

    // --- 1. strlen(array) → len_sym ---
    const symbolt *strlen_sym =
      converter_.symbol_table().find_symbol("c:@F@strlen");
    if (!strlen_sym)
      throw std::runtime_error("strlen not found in symbol table");

    symbolt &len_sym = converter_.create_tmp_symbol(
      list_value_, "$str_len$", size_type(), gen_zero(size_type()));
    code_declt len_decl(build_symbol(len_sym));
    len_decl.location() = loc;
    converter_.add_instruction(len_decl);

    code_function_callt strlen_call;
    strlen_call.function() = build_symbol(*strlen_sym);
    strlen_call.lhs() = build_symbol(len_sym);
    strlen_call.arguments().push_back(array);
    strlen_call.type() = size_type();
    strlen_call.location() = loc;
    converter_.add_instruction(strlen_call);

    // --- 2. idx_sym = (long long)pos_expr ---
    symbolt &idx_sym = converter_.create_tmp_symbol(
      list_value_, "$str_idx$", ll_type, gen_zero(ll_type));
    code_declt idx_decl(build_symbol(idx_sym));
    idx_decl.location() = loc;
    converter_.add_instruction(idx_decl);

    code_assignt idx_init(
      build_symbol(idx_sym), build_typecast(pos_expr, ll_type));
    idx_init.location() = loc;
    converter_.add_instruction(idx_init);

    // --- 3. Normalize negative index: if (idx < 0) idx += (ll)len ---
    exprt idx_lt_zero("<", bool_type());
    idx_lt_zero.copy_to_operands(
      build_symbol(idx_sym), from_integer(0, ll_type));

    exprt idx_plus_len("+", ll_type);
    idx_plus_len.copy_to_operands(
      build_symbol(idx_sym), build_typecast(build_symbol(len_sym), ll_type));

    code_assignt normalize(build_symbol(idx_sym), idx_plus_len);
    normalize.location() = loc;

    code_ifthenelset norm_guard;
    norm_guard.cond() = idx_lt_zero;
    norm_guard.then_case() = normalize;
    norm_guard.location() = loc;
    converter_.add_instruction(norm_guard);

    // --- 4. OOB check: if (idx < 0 || idx >= (ll)len) raise IndexError ---
    exprt still_neg("<", bool_type());
    still_neg.copy_to_operands(build_symbol(idx_sym), from_integer(0, ll_type));

    exprt idx_ge_len(">=", bool_type());
    idx_ge_len.copy_to_operands(
      build_symbol(idx_sym), build_typecast(build_symbol(len_sym), ll_type));

    exprt raise = converter_.get_exception_handler().gen_exception_raise(
      "IndexError", "string index out of range");
    codet throw_code("expression");
    throw_code.operands().push_back(raise);

    code_ifthenelset oob_guard;
    oob_guard.cond() = or_exprt(still_neg, idx_ge_len);
    oob_guard.then_case() = throw_code;
    oob_guard.location() = loc;
    converter_.add_instruction(oob_guard);

    // --- 5. __python_str_slice(array, idx, idx+1, 1) ---
    // idx is now a normalized, in-bounds positive index; the slice helper
    // produces a fresh alloca'd single-char null-terminated string.
    exprt end_expr("+", ll_type);
    end_expr.copy_to_operands(build_symbol(idx_sym), from_integer(1, ll_type));

    exprt slice_call = build_call_expr(
      get_str_slice_sym(),
      gen_pointer_type(char_type()),
      {array, build_symbol(idx_sym), end_expr, from_integer(1, ll_type)});
    slice_call.location() = loc;
    return slice_call;
  }

  // Handle static arrays
  return build_index(array, pos_expr, array.type().subtype());
}

void python_list::get_list_type_flags(
  const std::string &list_id,
  const type_handler &th,
  int &type_flag,
  size_t &float_type_id)
{
  type_flag = 0;
  float_type_id = 0;

  bool has_float = false;
  bool has_int = false;
  bool is_string = false;

  size_t map_size = python_list::get_list_type_map_size(list_id);
  for (size_t k = 0; k < map_size; ++k)
  {
    const typet elem_type = python_list::get_list_element_type(list_id, k);
    if (elem_type.is_floatbv())
    {
      if (!has_float)
      {
        float_type_id = std::hash<std::string>{}(th.type_to_string(elem_type));
        has_float = true;
      }
    }
    else if (
      (elem_type.is_pointer() && elem_type.subtype() == char_type()) ||
      (elem_type.is_array() && elem_type.subtype() == char_type()))
    {
      is_string = true;
    }
    else
      has_int = true;
  }

  if (is_string)
    type_flag = 2;
  else if (has_float && has_int)
    type_flag = 3;
  else if (has_float)
    type_flag = 1;
  else
    type_flag = 0;
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

      code_declt temp_decl(build_symbol(temp_sym));
      temp_decl.location() = converter_.get_location_from_decl(list_value_);
      converter_.add_instruction(temp_decl);

      code_assignt temp_assign(build_symbol(temp_sym), member);
      temp_assign.location() = converter_.get_location_from_decl(list_value_);
      converter_.add_instruction(temp_assign);

      return build_symbol(temp_sym);
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
  // Note: Python set ordering (< as strict subset, <= as subset-or-equal)
  // is not yet implemented here.  Ordering operators on sets currently
  // fall through to the Eq/NotEq path and will return incorrect results.
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
      new_symbol.set_type(func_type);

      converter_.symbol_table().add(new_symbol);
      set_eq_func =
        converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_set_eq");
    }

    locationt loc = converter_.get_location_from_decl(list_value_);
    symbolt &eq_ret = converter_.create_tmp_symbol(
      list_value_, "set_eq_tmp", bool_type(), gen_boolean(false));
    code_declt eq_ret_decl(build_symbol(eq_ret));
    converter_.add_instruction(eq_ret_decl);

    code_function_callt set_eq_call;
    set_eq_call.function() = build_symbol(*set_eq_func);
    set_eq_call.lhs() = build_symbol(eq_ret);
    set_eq_call.arguments().push_back(
      lhs_symbol->get_type().is_pointer()
        ? build_symbol(*lhs_symbol)
        : build_address_of(build_symbol(*lhs_symbol)));
    set_eq_call.arguments().push_back(
      rhs_symbol->get_type().is_pointer()
        ? build_symbol(*rhs_symbol)
        : build_address_of(build_symbol(*rhs_symbol)));
    set_eq_call.type() = bool_type();
    set_eq_call.location() = loc;
    converter_.add_instruction(set_eq_call);

    exprt cond("=", bool_type());
    cond.copy_to_operands(build_symbol(eq_ret));
    if (op == "Eq")
      cond.copy_to_operands(gen_boolean(true));
    else
      cond.copy_to_operands(gen_boolean(false));

    return cond;
  }

  // Fast path for list equality/inequality when we have concrete type-map
  // entries for both operands. This avoids __ESBMC_list_eq loops.
  if (op == "Eq" || op == "NotEq")
  {
    auto resolve_map_id = [&](const symbolt *sym) -> std::string {
      const std::string direct_id = sym->id.as_string();
      auto has_map = [&](const std::string &id) {
        auto it = list_type_map.find(id);
        return it != list_type_map.end() && !it->second.empty();
      };

      if (has_map(direct_id))
        return direct_id;

      const exprt &v = sym->get_value();
      if (v.is_symbol())
      {
        const std::string alias_id = v.identifier().as_string();
        if (has_map(alias_id))
          return alias_id;
      }
      else if (
        v.id() == "typecast" && !v.operands().empty() && v.op0().is_symbol())
      {
        const std::string alias_id = v.op0().identifier().as_string();
        if (has_map(alias_id))
          return alias_id;
      }

      return direct_id;
    };

    const std::string lhs_id = resolve_map_id(lhs_symbol);
    const std::string rhs_id = resolve_map_id(rhs_symbol);
    auto is_numeric = [](const typet &t) {
      return t.is_signedbv() || t.is_unsignedbv() || t.is_floatbv();
    };
    auto is_bool = [](const typet &t) { return t == bool_type(); };

    auto is_concrete_map = [&](const std::string &list_id) -> bool {
      auto it = list_type_map.find(list_id);
      if (it == list_type_map.end() || it->second.empty())
        return false;

      for (const auto &entry : it->second)
      {
        if (entry.first.empty())
          return false;

        const symbolt *elem_sym = converter_.find_symbol(entry.first);
        if (!elem_sym)
          return false;
        if (!(is_numeric(elem_sym->get_type()) ||
              is_bool(elem_sym->get_type())))
          return false;
      }
      return true;
    };

    auto has_mixed_int_float = [&](const std::string &list_id) -> bool {
      auto it = list_type_map.find(list_id);
      if (it == list_type_map.end() || it->second.empty())
        return false;

      bool has_int = false;
      bool has_float = false;
      for (const auto &entry : it->second)
      {
        const symbolt *elem_sym = converter_.find_symbol(entry.first);
        const typet t = elem_sym ? elem_sym->get_type() : entry.second;
        if (t.is_floatbv())
          has_float = true;
        else if (t.is_signedbv() || t.is_unsignedbv())
          has_int = true;
      }
      return has_int && has_float;
    };

    if (is_concrete_map(lhs_id) && is_concrete_map(rhs_id))
    {
      if (has_mixed_int_float(lhs_id) || has_mixed_int_float(rhs_id))
      {
        // For mixed int/float lists, runtime operations (e.g., sort) can
        // reorder heterogeneous elements and invalidate index->type mapping.
        // Keep the structural runtime comparator for soundness.
      }
      else
      {
        const size_t lhs_n = get_list_type_map_size(lhs_id);
        const size_t rhs_n = get_list_type_map_size(rhs_id);

        if (lhs_n == rhs_n && lhs_n <= 64)
        {
          exprt all_equal = gen_boolean(true);
          bool comparable = true;

          for (size_t i = 0; i < lhs_n; ++i)
          {
            const std::string lhs_elem_id = get_list_element_id(lhs_id, i);
            const std::string rhs_elem_id = get_list_element_id(rhs_id, i);
            const symbolt *lhs_elem_sym = converter_.find_symbol(lhs_elem_id);
            const symbolt *rhs_elem_sym = converter_.find_symbol(rhs_elem_id);
            const typet lhs_elem_type = lhs_elem_sym
                                          ? lhs_elem_sym->get_type()
                                          : get_list_element_type(lhs_id, i);
            const typet rhs_elem_type = rhs_elem_sym
                                          ? rhs_elem_sym->get_type()
                                          : get_list_element_type(rhs_id, i);
            if (lhs_elem_type.is_nil() || rhs_elem_type.is_nil())
            {
              comparable = false;
              break;
            }

            const exprt idx = from_integer(BigInt(i), size_type());
            exprt lhs_at =
              build_list_at_call(build_symbol(*lhs_symbol), idx, list_value_);
            exprt rhs_at =
              build_list_at_call(build_symbol(*rhs_symbol), idx, list_value_);
            exprt lhs_val = extract_pyobject_value(lhs_at, lhs_elem_type);
            exprt rhs_val = extract_pyobject_value(rhs_at, rhs_elem_type);

            exprt eq_elem;
            if (lhs_elem_type == rhs_elem_type && is_numeric(lhs_elem_type))
            {
              eq_elem = equality_exprt(lhs_val, rhs_val);
            }
            else if (lhs_elem_type == rhs_elem_type && is_bool(lhs_elem_type))
            {
              eq_elem = equality_exprt(lhs_val, rhs_val);
            }
            else if (is_numeric(lhs_elem_type) && is_numeric(rhs_elem_type))
            {
              lhs_val = build_typecast(lhs_val, double_type());
              rhs_val = build_typecast(rhs_val, double_type());
              eq_elem = equality_exprt(lhs_val, rhs_val);
            }
            else
            {
              comparable = false;
              break;
            }

            exprt and_expr("and", bool_type());
            and_expr.copy_to_operands(all_equal, eq_elem);
            all_equal = and_expr;
          }

          if (comparable)
          {
            if (op == "NotEq")
              return not_exprt(all_equal);
            return all_equal;
          }
        }
      }
    }
  }

  // ── Ordering operators: Lt, LtE, Gt, GtE ──────────────────────────────────
  // Implemented via __ESBMC_list_lt (lexicographic less-than):
  //   Lt  : list_lt(l1, l2)
  //   LtE : !list_lt(l2, l1)    (i.e. !(l1 > l2))
  //   Gt  : list_lt(l2, l1)
  //   GtE : !list_lt(l1, l2)    (i.e. !(l1 < l2))
  if (op == "Lt" || op == "LtE" || op == "Gt" || op == "GtE")
  {
    // Look up or register the __ESBMC_list_lt symbol.
    const symbolt *list_lt_func_sym =
      converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_lt");
    if (!list_lt_func_sym)
    {
      symbolt new_symbol;
      new_symbol.name = "__ESBMC_list_lt";
      new_symbol.id = "c:@F@__ESBMC_list_lt";
      new_symbol.mode = "C";
      new_symbol.is_extern = true;

      code_typet func_type;
      func_type.return_type() = bool_type();
      typet list_ptr = converter_.get_type_handler().get_list_type();
      func_type.arguments().push_back(code_typet::argumentt(list_ptr));
      func_type.arguments().push_back(code_typet::argumentt(list_ptr));
      func_type.arguments().push_back(code_typet::argumentt(int_type()));
      func_type.arguments().push_back(code_typet::argumentt(size_type()));
      new_symbol.set_type(func_type);

      converter_.symbol_table().add(new_symbol);
      list_lt_func_sym =
        converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_lt");
    }
    assert(list_lt_func_sym);

    // Determine element type flags from both lists and merge them so that
    // cross-type comparisons like [1,2] < [1.0,2.0] are handled correctly.
    int type_flag_lhs = 0, type_flag_rhs = 0;
    size_t float_type_id_lhs = 0, float_type_id_rhs = 0;
    get_list_type_flags(
      lhs_symbol->id.as_string(),
      converter_.get_type_handler(),
      type_flag_lhs,
      float_type_id_lhs);
    get_list_type_flags(
      rhs_symbol->id.as_string(),
      converter_.get_type_handler(),
      type_flag_rhs,
      float_type_id_rhs);

    const bool lhs_has_float = (type_flag_lhs == 1 || type_flag_lhs == 3);
    const bool lhs_has_int = (type_flag_lhs == 0 || type_flag_lhs == 3);
    const bool rhs_has_float = (type_flag_rhs == 1 || type_flag_rhs == 3);
    const bool rhs_has_int = (type_flag_rhs == 0 || type_flag_rhs == 3);
    const bool is_string = (type_flag_lhs == 2 || type_flag_rhs == 2);
    const bool has_float = lhs_has_float || rhs_has_float;
    const bool has_int = lhs_has_int || rhs_has_int;
    const size_t float_type_id =
      float_type_id_lhs ? float_type_id_lhs : float_type_id_rhs;

    int type_flag;
    if (is_string)
      type_flag = 2;
    else if (has_float && has_int)
      type_flag = 3;
    else if (has_float)
      type_flag = 1;
    else
      type_flag = 0;

    // Emit: lt_ret = __ESBMC_list_lt(a, b, type_flag, float_type_id)
    // Derivations (total order):
    //   Lt  : list_lt(l1, l2) == true   → no swap, check true
    //   LtE : !list_lt(l2, l1) == true  → swap,    check false
    //   Gt  : list_lt(l2, l1) == true   → swap,    check true
    //   GtE : !list_lt(l1, l2) == true  → no swap, check false
    const bool swap = (op == "LtE" || op == "Gt");
    const symbolt *a_sym = swap ? rhs_symbol : lhs_symbol;
    const symbolt *b_sym = swap ? lhs_symbol : rhs_symbol;

    symbolt &lt_ret = converter_.create_tmp_symbol(
      list_value_, "lt_tmp", bool_type(), gen_boolean(false));
    code_declt lt_ret_decl(build_symbol(lt_ret));
    converter_.add_instruction(lt_ret_decl);

    code_function_callt lt_call;
    lt_call.function() = build_symbol(*list_lt_func_sym);
    lt_call.lhs() = build_symbol(lt_ret);
    lt_call.arguments().push_back(build_symbol(*a_sym));
    lt_call.arguments().push_back(build_symbol(*b_sym));
    lt_call.arguments().push_back(from_integer(type_flag, int_type()));
    lt_call.arguments().push_back(from_integer(float_type_id, size_type()));
    lt_call.type() = bool_type();
    lt_call.location() = converter_.get_location_from_decl(list_value_);
    converter_.add_instruction(lt_call);

    // Lt / Gt → lt_ret must be true; LtE / GtE → lt_ret must be false
    exprt cond("=", bool_type());
    cond.copy_to_operands(build_symbol(lt_ret));
    if (op == "Lt" || op == "Gt")
      cond.copy_to_operands(gen_boolean(true));
    else
      cond.copy_to_operands(gen_boolean(false));

    return cond;
  }

  // ── Equality operators: Eq, NotEq ─────────────────────────────────────────

  // Compute list type_id for nested list detection
  const typet &list_type = l1.type();
  const std::string list_type_name =
    converter_.get_type_handler().type_to_string(list_type);
  constant_exprt list_type_id(size_type());
  list_type_id.set_value(integer2binary(
    std::hash<std::string>{}(list_type_name), config.ansi_c.address_width));

  symbolt &eq_ret = converter_.create_tmp_symbol(
    list_value_, "eq_tmp", bool_type(), gen_boolean(false));
  code_declt eq_ret_decl(build_symbol(eq_ret));
  converter_.add_instruction(eq_ret_decl);

  // Get max depth from configuration option
  int max_depth = get_list_compare_depth();
  constant_exprt max_depth_expr(size_type());
  max_depth_expr.set_value(
    integer2binary(max_depth, config.ansi_c.address_width));

  // Merge the float element type_id from both operands so that mixed int/float
  // elements compare numerically (Python's 1 == 1.0), as list_lt already does.
  int type_flag_lhs = 0, type_flag_rhs = 0;
  size_t float_type_id_lhs = 0, float_type_id_rhs = 0;
  get_list_type_flags(
    lhs_symbol->id.as_string(),
    converter_.get_type_handler(),
    type_flag_lhs,
    float_type_id_lhs);
  get_list_type_flags(
    rhs_symbol->id.as_string(),
    converter_.get_type_handler(),
    type_flag_rhs,
    float_type_id_rhs);
  const size_t float_type_id =
    float_type_id_lhs ? float_type_id_lhs : float_type_id_rhs;

  // Statically-known element byte size for the primitive comparison, so the
  // model's __ESBMC_values_equal takes its branch-free fast path instead of
  // memcmp's symbolic-size byte loop (the dominant cost when comparing large
  // lists, e.g. `assert l == [...]`). Emitted only when both operands' first
  // element is the same fixed-width scalar; 0 otherwise, which makes the model
  // fall back to the per-element a->size read (exact prior behaviour).
  size_t eq_elem_size_bytes = 0;
  {
    auto scalar_width = [](const typet &t) -> size_t {
      if (
        (t.id() == "signedbv" || t.id() == "unsignedbv" ||
         t.id() == "floatbv" || t.id() == "fixedbv") &&
        !t.width().empty())
        return std::stoull(t.width().as_string(), nullptr, 10) / 8;
      return 0;
    };
    const typet lt =
      get_list_element_type(converted_l1.identifier().as_string(), 0);
    const typet rt =
      get_list_element_type(converted_l2.identifier().as_string(), 0);
    size_t lw = lt.is_nil() ? 0 : scalar_width(lt);
    size_t rw = rt.is_nil() ? 0 : scalar_width(rt);
    if (lw != 0 && lw == rw)
      eq_elem_size_bytes = lw;
  }

  code_function_callt list_eq_func_call;
  list_eq_func_call.function() = build_symbol(*list_eq_func_sym);
  list_eq_func_call.lhs() = build_symbol(eq_ret);
  // passing arguments
  list_eq_func_call.arguments().push_back(build_symbol(*lhs_symbol)); // l1
  list_eq_func_call.arguments().push_back(build_symbol(*rhs_symbol)); // l2
  list_eq_func_call.arguments().push_back(list_type_id);   // list_type_id
  list_eq_func_call.arguments().push_back(max_depth_expr); // max_depth
  list_eq_func_call.arguments().push_back(
    from_integer(float_type_id, size_type())); // float_type_id
  list_eq_func_call.arguments().push_back(
    from_integer(eq_elem_size_bytes, size_type())); // elem_size
  list_eq_func_call.type() = bool_type();
  list_eq_func_call.location() = converter_.get_location_from_decl(list_value_);
  converter_.add_instruction(list_eq_func_call);

  exprt cond("=", bool_type());
  cond.copy_to_operands(build_symbol(eq_ret));
  if (op == "Eq")
    cond.copy_to_operands(gen_boolean(true));
  else
    cond.copy_to_operands(gen_boolean(false));

  return cond;
}

exprt python_list::create_vla(
  const nlohmann::json &element,
  const exprt &count,
  const std::vector<exprt> &list_elems)
{
  locationt location = converter_.get_location_from_decl(element);

  // Fresh result list: a literal source (e.g. [0]) already holds its initial
  // element, so pushing onto it would yield count + 1 elements.
  symbolt &result = create_list();

  // Materialise the count (which may be a compound expression such as m + 1)
  // into an int symbol to use as the loop bound.
  exprt bound_value =
    (count.type() == int_type()) ? count : build_typecast(count, int_type());
  symbolt &bound = converter_.create_tmp_symbol(
    element, "$list_rep_count$", int_type(), exprt());
  code_declt bound_decl(build_symbol(bound));
  bound_decl.location() = location;
  converter_.add_instruction(bound_decl);
  code_assignt bound_assign(build_symbol(bound), bound_value);
  bound_assign.location() = location;
  converter_.add_instruction(bound_assign);

  // counter = 0
  symbolt &counter = converter_.create_tmp_symbol(
    element, "counter", int_type(), gen_zero(int_type()));
  code_assignt counter_code(build_symbol(counter), gen_zero(int_type()));
  counter_code.location() = location;
  converter_.add_instruction(counter_code);

  // while (counter < bound) { push each elem in order; counter += 1; }
  exprt cond("<", bool_type());
  cond.operands().push_back(build_symbol(counter));
  cond.operands().push_back(build_symbol(bound));

  code_blockt then;
  for (const auto &list_elem : list_elems)
    then.copy_to_operands(build_push_list_call(result, element, list_elem));

  exprt incr("+", int_type());
  incr.copy_to_operands(build_symbol(counter));
  incr.copy_to_operands(gen_one(int_type()));
  code_assignt update(build_symbol(counter), incr);
  then.copy_to_operands(update);

  codet while_cod;
  while_cod.set_statement("while");
  while_cod.copy_to_operands(cond, then);
  converter_.add_instruction(while_cod);

  // Record one type-map entry per source element for index-based type lookups.
  auto &result_types = list_type_map[result.id.as_string()];
  for (const auto &list_elem : list_elems)
    result_types.push_back(std::make_pair(std::string(), list_elem.type()));

  return build_symbol(result);
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
  code_declt contains_ret_decl(build_symbol(contains_ret));
  converter_.add_instruction(contains_ret_decl);

  // Build the function call as a statement
  code_function_callt contains_call;
  contains_call.function() = build_symbol(*list_contains_func);
  contains_call.lhs() = build_symbol(contains_ret);

  // Pass the list directly
  contains_call.arguments().push_back(list);

  // For pointer types (e.g., string parameters), use the pointer directly
  // For value types, take the address
  exprt item_arg;
  if (item_info.elem_symbol->get_type().is_pointer())
  {
    // String parameters are pointers - use the pointer value directly
    item_arg = build_symbol(*item_info.elem_symbol);
  }
  else
  {
    // For arrays or other value types, take the address
    item_arg = build_address_of(build_symbol(*item_info.elem_symbol));
  }

  contains_call.arguments().push_back(item_arg);

  // For void/char pointers from iteration, use stored type info from list
  exprt type_hash = build_symbol(*item_info.elem_type_sym);
  exprt elem_size = item_info.elem_size;

  // void* items (e.g. a loop variable over a string list) need the stored
  // char-array type_id and runtime length recovered from list_type_map.
  // The lookup is keyed by symbol name, so non-symbol receivers cannot carry
  // void* elements; skipping them is sound.
  if (
    item_info.elem_symbol->get_type() == pointer_typet(empty_typet()) &&
    list.is_symbol())
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
            code_declt strlen_decl(build_symbol(strlen_result));
            strlen_decl.location() = item_info.location;
            converter_.add_instruction(strlen_decl);

            code_function_callt strlen_call;
            strlen_call.function() = build_symbol(*strlen_symbol);
            strlen_call.lhs() = build_symbol(strlen_result);
            strlen_call.arguments().push_back(
              build_symbol(*item_info.elem_symbol));
            strlen_call.type() = size_type();
            strlen_call.location() = item_info.location;
            converter_.add_instruction(strlen_call);

            // Add 1 for null terminator: size = strlen(s) + 1
            exprt one_const = from_integer(1, strlen_result.get_type());
            elem_size = exprt("+", strlen_result.get_type());
            elem_size.copy_to_operands(build_symbol(strlen_result), one_const);
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
  result.copy_to_operands(build_symbol(contains_ret));
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

    // Loop condition: idx < str_len
    exprt loop_cond("<", bool_type());
    loop_cond.copy_to_operands(build_symbol(idx), str_len);

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

    // Increment index: idx++
    plus_exprt idx_inc(build_symbol(idx), gen_one(size_type()));
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

std::string
python_list::get_list_element_id(const std::string &list_id, size_t index)
{
  auto it = list_type_map.find(list_id);
  if (it == list_type_map.end() || index >= it->second.size())
    return {};
  return it->second[index].first;
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
      code_declt tmp_var_decl(build_symbol(tmp_var_symbol));
      tmp_var_decl.location() = location;
      converter_.add_instruction(tmp_var_decl);

      // Create function call with temp as LHS
      code_function_callt new_call;
      new_call.function() = call.function();
      new_call.arguments() = call.arguments();
      new_call.lhs() = build_symbol(tmp_var_symbol);
      new_call.type() = list_type;
      new_call.location() = location;
      converter_.add_instruction(new_call);

      // Use the temp variable as the iterable
      iterable_expr = build_symbol(tmp_var_symbol);
    }
  }
  // Check for empty list early
  else if (iterable_expr.type() == list_type && iterable_expr.is_symbol())
  {
    const std::string &list_id = iterable_expr.identifier().as_string();
    auto type_map_it = list_type_map.find(list_id);
    if (type_map_it == list_type_map.end() || type_map_it->second.empty())
      return build_symbol(result_list);
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

  code_declt index_decl(build_symbol(index_var));
  index_decl.location() = location;
  converter_.add_instruction(index_decl);

  // Initialize index to 0
  code_assignt index_init(build_symbol(index_var), gen_zero(size_type()));
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

    code_declt length_decl(build_symbol(length_var));
    length_decl.location() = location;
    converter_.add_instruction(length_decl);

    code_function_callt size_call;
    size_call.function() = build_symbol(*size_func);
    size_call.arguments().push_back(
      iterable_expr.type().is_pointer() ? iterable_expr
                                        : build_address_of(iterable_expr));
    size_call.lhs() = build_symbol(length_var);
    size_call.type() = size_type();
    size_call.location() = location;
    converter_.add_instruction(size_call);

    length_expr = build_symbol(length_var);
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

      code_declt length_decl(build_symbol(length_var));
      length_decl.location() = location;
      converter_.add_instruction(length_decl);

      code_function_callt strlen_call;
      strlen_call.function() = build_symbol(*strlen_func);
      strlen_call.arguments().push_back(iterable_expr);
      strlen_call.lhs() = build_symbol(length_var);
      strlen_call.type() = size_type();
      strlen_call.location() = location;
      converter_.add_instruction(strlen_call);

      length_expr = build_symbol(length_var);
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
      build_list_at_call(iterable_expr, build_symbol(index_var), element);
    current_element = extract_pyobject_value(current_element, actual_elem_type);
  }
  else if (iterable_expr.type().is_array() || iterable_expr.type().is_pointer())
  {
    // For arrays/strings, use direct indexing
    exprt array_index =
      build_index(iterable_expr, build_symbol(index_var), loop_var_type);
    current_element = array_index;
  }
  else
  {
    throw std::runtime_error(
      "Cannot index into type: " + iterable_expr.type().id_string());
  }

  code_assignt loop_var_assign(build_symbol(*loop_var), current_element);
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
  increment.copy_to_operands(build_symbol(index_var), gen_one(size_type()));
  code_assignt index_increment(build_symbol(index_var), increment);
  index_increment.location() = location;
  loop_body.copy_to_operands(index_increment);

  // 10. Create while loop: while (i < length)
  exprt loop_condition("<", bool_type());
  loop_condition.copy_to_operands(build_symbol(index_var), length_expr);

  codet while_stmt;
  while_stmt.set_statement("while");
  while_stmt.copy_to_operands(loop_condition, loop_body);
  while_stmt.location() = location;
  converter_.add_instruction(while_stmt);

  return build_symbol(result_list);
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

exprt python_list::extract_pyobject_value(
  const exprt &pyobject_expr,
  const typet &elem_type,
  bool mixed_numeric)
{
  // For float types, read __ESBMC_float_buf[item->float_idx].
  // This avoids the void*→integer truncation in --ir mode: float_idx is a size_t
  // (no sort mismatch in BV mode), and float_buf is a typed global double array
  // (real-sorted in --ir mode), so the array read gives the correct real value.
  if (elem_type.is_floatbv())
  {
    // Helper: build (*pyobject_expr).field for a given field/type.
    auto member_of = [&](const char *field, const typet &ftype) -> exprt {
      return build_deref_member(pyobject_expr, field, ftype);
    };

    // Look up __ESBMC_float_buf global (static in list.c, but still in symbol table)
    const symbolt *fbuf_sym =
      converter_.symbol_table().find_symbol("c:list.c@__ESBMC_float_buf");
    assert(fbuf_sym && "could not find __ESBMC_float_buf symbol");

    // Build __ESBMC_float_buf[item->float_idx]
    exprt float_idx_member = member_of("float_idx", size_type());
    exprt float_val =
      build_index(build_symbol(*fbuf_sym), float_idx_member, elem_type);

    if (!mixed_numeric)
      return float_val;

    // Dynamic index into a heterogeneous int/float list: the element may be an
    // int whose value lives behind item->value (not in float_buf), so reading
    // float_buf unconditionally would misread it (#5160). Dispatch on the
    // stored type_id: float elements come from float_buf, int elements are
    // promoted to double from their 8-byte payload. The float type_id is the
    // same hash the push path stamps onto float elements (float_type_id).
    const size_t float_type_id = std::hash<std::string>{}(
      converter_.get_type_handler().type_to_string(elem_type));

    // (double)*(long long *)item->value
    exprt value_member = member_of("value", pointer_typet(empty_typet()));
    exprt as_int_ptr =
      build_typecast(value_member, pointer_typet(long_long_int_type()));
    exprt int_val = build_dereference(as_int_ptr, long_long_int_type());
    exprt int_as_float = build_typecast(int_val, elem_type);

    // item->type_id == float_type_id ? float_buf[float_idx] : (double)int
    equality_exprt is_float(
      member_of("type_id", size_type()),
      from_integer(float_type_id, size_type()));
    return if_exprt(is_float, float_val, int_as_float);
  }

  // Extract value from PyObject: (*pyobject_expr).value
  exprt obj_value =
    build_deref_member(pyobject_expr, "value", pointer_typet(empty_typet()));

  // For array types, return pointer to element type instead of pointer to array
  // The dereference system doesn't support array types as target types
  if (elem_type.is_array())
  {
    const array_typet &arr_type = to_array_type(elem_type);
    // Cast to pointer to element type (e.g., char* instead of char[2]*)
    return build_typecast(obj_value, pointer_typet(arr_type.subtype()));
  }

  // For char* strings and None (_Bool*), the void* already contains the pointer value
  // For all other types, the void* contains a pointer to the value
  if (
    elem_type.is_pointer() &&
    (elem_type.subtype() == char_type() || elem_type.subtype() == bool_type()))
  {
    // String and None case: cast void* directly to the pointer type (no dereference needed)
    return build_typecast(obj_value, elem_type);
  }
  else
  {
    // All other types: cast void* to pointer-to-type, then dereference
    exprt tc = build_typecast(obj_value, pointer_typet(elem_type));
    return build_dereference(tc, elem_type);
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

  // Scan all elements to detect mixed int/float
  bool has_int = elem_type.is_signedbv() || elem_type.is_unsignedbv();
  bool has_float = elem_type.is_floatbv();

  for (size_t i = 1; i < list_size; i++)
  {
    const typet &current_elem_type = type_info[i].second;

    // For string types, all char arrays and char pointers are considered compatible
    if (is_string_type(elem_type) && is_string_type(current_elem_type))
      continue;

    if (current_elem_type.is_floatbv())
      has_float = true;
    else if (
      current_elem_type.is_signedbv() || current_elem_type.is_unsignedbv())
      has_int = true;

    // Only int<->float mixing is allowed (Python promotes int to float).
    // Any other mismatch — including different-width or signed/unsigned integers
    // — is an error.
    bool int_float_mix =
      (elem_type.is_floatbv() && (current_elem_type.is_signedbv() ||
                                  current_elem_type.is_unsignedbv())) ||
      ((elem_type.is_signedbv() || elem_type.is_unsignedbv()) &&
       current_elem_type.is_floatbv());
    if (elem_type != current_elem_type && !int_float_mix)
    {
      throw std::runtime_error(
        "Type mismatch in " + func_name +
        "() call: list contains mixed types. "
        "ESBMC currently requires all elements to have the same type for " +
        func_name + "().");
    }
  }

  // Mixed int and float: Python promotes int to float for comparisons
  if (has_int && has_float)
    return double_type();

  return elem_type;
}

bool python_list::has_mixed_numeric_types(const std::string &list_id)
{
  auto it = list_type_map.find(list_id);
  if (it == list_type_map.end() || it->second.empty())
    return false;
  bool has_int = false, has_float = false;
  for (const auto &elem : it->second)
  {
    if (elem.second.is_floatbv())
      has_float = true;
    else if (elem.second.is_signedbv() || elem.second.is_unsignedbv())
      has_int = true;
  }
  return has_int && has_float;
}

typet python_list::infer_literal_element_type(
  const nlohmann::json &list_literal)
{
  nlohmann::json first_elem = json_utils::get_list_element(list_literal, 0);
  if (first_elem.is_null() || first_elem.empty())
    return typet();

  const type_handler &th = converter_.get_type_handler();

  // A heterogeneous int/float literal is promoted to a homogeneous double list
  // at construction (python_list::get, promote_ints), so every element is a
  // double in __ESBMC_float_buf. Read it as a double regardless of which
  // element the index selects; the first element's int type misreads the bits.
  if (
    list_literal["_type"] == "List" && list_literal.contains("elts") &&
    list_literal["elts"].is_array())
  {
    bool has_int = false, has_float = false;
    for (const auto &e : list_literal["elts"])
    {
      const typet t = th.get_typet(e);
      if (t.is_floatbv())
        has_float = true;
      else if (t.is_signedbv() || t.is_unsignedbv() || t.is_bool())
        has_int = true;
    }
    if (has_int && has_float)
      return double_type();
  }

  return th.get_typet(first_elem);
}

typet python_list::numeric_element_type(const std::string &list_id)
{
  auto it = list_type_map.find(list_id);
  if (it == list_type_map.end() || it->second.empty())
    return typet();

  bool has_float = false;
  const typet first = it->second[0].second;
  for (const auto &elem : it->second)
  {
    const typet &t = elem.second;
    if (t.is_floatbv())
      has_float = true;
    else if (!(t.is_signedbv() || t.is_unsignedbv()))
      return typet(); // non-numeric element: not a numeric list
  }

  // int/float mix (or all-float): Python promotes to float, read as double.
  if (has_float)
    return double_type();

  // All integers: require one shared integer type for a sound single-type read.
  for (const auto &elem : it->second)
    if (elem.second != first)
      return typet();
  return first;
}

exprt python_list::build_min_max_for_mixed_numeric(
  const exprt &list_arg,
  const std::string &list_id,
  const std::string &func_name,
  irep_idt comparison_op)
{
  const TypeInfo &type_info = list_type_map.at(list_id);
  size_t n = type_info.size();

  if (n == 0)
    throw std::runtime_error(func_name + "() arg is an empty sequence");

  pointer_typet obj_ptr_type(
    converter_.get_type_handler().get_list_element_type());
  const typet double_t = double_type();
  locationt loc = converter_.get_location_from_decl(list_value_);

  // Declare a temp symbol, emit its declaration, and return its symbol_expr.
  auto make_tmp = [&](const std::string &name, const typet &type) -> exprt {
    symbolt &sym =
      converter_.create_tmp_symbol(list_value_, name, type, exprt());
    code_declt decl(build_symbol(sym));
    decl.location() = loc;
    converter_.add_instruction(decl);
    return build_symbol(sym);
  };

  // Access element i from the list and return it promoted to double.
  auto get_elem_as_double = [&](size_t i) -> exprt {
    typet orig_type = type_info[i].second;
    exprt list_at = build_list_at_call(
      list_arg, from_integer(BigInt(i), size_type()), list_value_);
    exprt obj = make_tmp("$list_obj$", obj_ptr_type);
    code_assignt assign_obj(obj, list_at);
    assign_obj.location() = loc;
    converter_.add_instruction(assign_obj);
    exprt val = extract_pyobject_value(obj, orig_type);
    return orig_type.is_floatbv() ? val : build_typecast(val, double_t);
  };

  exprt result = make_tmp("$minmax_result$", double_t);
  code_assignt init(result, get_elem_as_double(0));
  init.location() = loc;
  converter_.add_instruction(init);

  for (size_t i = 1; i < n; i++)
  {
    exprt elem = make_tmp("$minmax_elem$", double_t);
    code_assignt assign_elem(elem, get_elem_as_double(i));
    assign_elem.location() = loc;
    converter_.add_instruction(assign_elem);

    exprt condition(comparison_op, bool_type());
    condition.copy_to_operands(elem, result);
    code_ifthenelset ite;
    ite.cond() = condition;
    ite.then_case() = code_assignt(result, elem);
    ite.location() = loc;
    converter_.add_instruction(ite);
  }

  return result;
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
      exprt size_value = build_typecast(size_expr, comp.type());
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

exprt python_list::build_set_membership_call(
  const symbolt &set,
  const nlohmann::json &op,
  const exprt &elem,
  const std::string &method_name)
{
  const std::string c_func = "c:@F@__ESBMC_set_" + method_name;
  const symbolt *func = converter_.symbol_table().find_symbol(c_func);
  if (!func)
    throw std::runtime_error(c_func + " function not found in symbol table");

  list_elem_info elem_info = get_list_element_info(op, elem);

  exprt element_arg;
  if (
    elem_info.elem_symbol->get_type().is_pointer() &&
    elem_info.elem_symbol->get_type().subtype() == char_type())
    element_arg = build_symbol(*elem_info.elem_symbol);
  else
    element_arg = build_address_of(build_symbol(*elem_info.elem_symbol));

  code_function_callt call;
  call.function() = build_symbol(*func);
  call.arguments().push_back(build_symbol(set));
  call.arguments().push_back(element_arg);
  call.arguments().push_back(build_symbol(*elem_info.elem_type_sym));
  call.arguments().push_back(elem_info.elem_size);
  call.type() = bool_type();
  call.location() = elem_info.location;

  // Track the new element's compile-time type info so subsequent ops
  // (`elem in set`, set comparisons) recognise it.
  if (method_name == "add")
    add_type_info(
      set.id.as_string(),
      elem_info.elem_symbol->id.as_string(),
      elem_info.elem_symbol->get_type());

  return converter_.convert_expression_to_code(call);
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

// list.count(x) / list.index(x): both pass (list, &value, type_id, size) to a
// C model that walks the list comparing elements (same plumbing as remove), and
// return a size_t value. `func_id` selects the model; the only difference is
// index() asserts the element is present (handled inside the model).
exprt python_list::build_count_index_list_call(
  const symbolt &list,
  const nlohmann::json &op,
  const exprt &elem,
  const std::string &func_id)
{
  list_elem_info elem_info = get_list_element_info(op, elem);

  const symbolt *func = converter_.symbol_table().find_symbol(func_id);
  if (!func)
    throw std::runtime_error(func_id + " function not found in symbol table");

  exprt element_arg;
  if (
    elem_info.elem_symbol->get_type().is_pointer() &&
    elem_info.elem_symbol->get_type().subtype() == char_type())
    element_arg = build_symbol(*elem_info.elem_symbol);
  else
    element_arg = build_address_of(build_symbol(*elem_info.elem_symbol));

  // Args: (list, &value-or-ptr, type_id, size) — same shape as the remove call.
  exprt call = build_call_expr(
    *func,
    size_type(),
    {build_symbol(list),
     element_arg,
     build_symbol(*elem_info.elem_type_sym),
     elem_info.elem_size});
  call.location() = elem_info.location;

  return call;
}

exprt python_list::build_count_list_call(
  const symbolt &list,
  const nlohmann::json &op,
  const exprt &elem)
{
  return build_count_index_list_call(list, op, elem, "c:@F@__ESBMC_list_count");
}

exprt python_list::build_index_list_call(
  const symbolt &list,
  const nlohmann::json &op,
  const exprt &elem)
{
  return build_count_index_list_call(list, op, elem, "c:@F@__ESBMC_list_index");
}

size_t python_list::get_list_type_map_size(const std::string &list_id)
{
  auto it = list_type_map.find(list_id);
  if (it == list_type_map.end())
    return 0;
  return it->second.size();
}

void python_list::reverse_type_info(const std::string &list_id)
{
  auto it = list_type_map.find(list_id);
  if (it == list_type_map.end() || it->second.size() <= 1)
    return;
  std::reverse(it->second.begin(), it->second.end());
}

void python_list::handle_list_var_unpacking(
  const nlohmann::json &ast_node,
  const nlohmann::json &target,
  const exprt &list_expr,
  codet &target_block)
{
  const auto &targets = target["elts"];
  const locationt loc = converter_.get_location_from_decl(ast_node);

  // Find starred target index (-1 if none)
  int star_idx = -1;
  for (size_t i = 0; i < targets.size(); i++)
  {
    if (targets[i]["_type"] == "Starred")
    {
      star_idx = static_cast<int>(i);
      break;
    }
  }

  const size_t before_star =
    (star_idx >= 0) ? static_cast<size_t>(star_idx) : targets.size();
  const size_t after_star =
    (star_idx >= 0) ? targets.size() - static_cast<size_t>(star_idx) - 1 : 0;

  // Get element type from list_type_map or from the variable's annotation
  typet elem_type;
  if (list_expr.is_symbol())
  {
    const std::string &list_id = list_expr.identifier().as_string();
    auto it = list_type_map.find(list_id);
    if (it != list_type_map.end() && !it->second.empty())
      elem_type = it->second[0].second;
  }
  if (elem_type == typet() && ast_node["value"].contains("id"))
  {
    const std::string &var_name = ast_node["value"]["id"].get<std::string>();
    nlohmann::json decl = json_utils::find_var_decl(
      var_name, converter_.current_function_name(), converter_.ast());
    elem_type =
      get_elem_type_from_annotation(decl, converter_.get_type_handler());
  }

  // Subscript-chain fallback: e.g. `w, v = items[i-1]` where the RHS is a
  // Subscript whose base Name carries a list[list[T]]-style annotation.
  // Walk down the chain and peel one extra annotation layer (the unpacked
  // tuple/list is the element of the innermost subscript).
  if (
    elem_type == typet() && ast_node["value"].is_object() &&
    ast_node["value"].contains("_type") &&
    ast_node["value"]["_type"] == "Subscript")
  {
    size_t subscript_depth = 0;
    const nlohmann::json *cur = &ast_node["value"];
    while (cur->is_object() && cur->contains("value") &&
           (*cur)["value"].is_object() && (*cur)["value"].contains("_type") &&
           (*cur)["value"]["_type"] == "Subscript")
    {
      ++subscript_depth;
      cur = &(*cur)["value"];
    }
    if (
      cur->is_object() && cur->contains("value") &&
      (*cur)["value"].is_object() && (*cur)["value"].contains("id") &&
      (*cur)["value"]["id"].is_string())
    {
      const std::string base_name = (*cur)["value"]["id"].get<std::string>();
      nlohmann::json base_decl = json_utils::find_var_decl(
        base_name, converter_.current_function_name(), converter_.ast());
      if (!base_decl.is_null() && base_decl.contains("annotation"))
      {
        nlohmann::json drilled = base_decl["annotation"];
        // Peel one layer per Subscript node plus one more for the unpacked
        // element itself.
        for (size_t k = 0; k <= subscript_depth; ++k)
        {
          if (
            !drilled.is_object() || !drilled.contains("_type") ||
            drilled["_type"] != "Subscript" || !drilled.contains("slice"))
          {
            drilled = nlohmann::json();
            break;
          }
          drilled = drilled["slice"];
        }
        if (!drilled.is_null())
        {
          nlohmann::json synth;
          synth["annotation"] = drilled;
          elem_type =
            get_elem_type_from_annotation(synth, converter_.get_type_handler());
        }
      }
    }
  }

  // Final fallback: treat elements as Any rather than aborting the conversion.
  // Mirrors the subscript-read path which falls back to any_type() when the
  // element type cannot be inferred (see `python_list::get_expr`).
  if (elem_type == typet())
    elem_type = any_type();

  // Helper: find or create a variable symbol and assign an expression to it
  auto assign_to_target = [&](
                            const nlohmann::json &tgt_node, const exprt &val) {
    if (tgt_node["_type"] != "Name")
      throw std::runtime_error(
        "List unpacking only supports simple names, not " +
        tgt_node["_type"].get<std::string>());

    const std::string var_name = tgt_node["id"].get<std::string>();
    symbol_id var_sid = converter_.create_symbol_id();
    var_sid.set_object(var_name);
    symbolt *var_symbol = converter_.find_symbol(var_sid.to_string());
    if (!var_symbol)
    {
      symbolt new_symbol = converter_.create_symbol(
        loc.get_file().as_string(),
        var_name,
        var_sid.to_string(),
        loc,
        val.type());
      new_symbol.lvalue = true;
      new_symbol.file_local = true;
      new_symbol.is_extern = false;
      var_symbol = converter_.symbol_table().move_symbol_to_context(new_symbol);
    }
    code_assignt assign(build_symbol(*var_symbol), val);
    assign.location() = loc;
    target_block.copy_to_operands(assign);
  };

  // Assign targets before the star using concrete indices
  for (size_t i = 0; i < before_star; i++)
  {
    exprt idx = from_integer(i, size_type());
    exprt list_at = build_list_at_call(list_expr, idx, list_value_);
    exprt val = extract_pyobject_value(list_at, elem_type);
    assign_to_target(targets[i], val);
  }

  // Handle starred target: collect remaining elements into a new list
  if (star_idx >= 0)
  {
    const auto &star_value = targets[static_cast<size_t>(star_idx)]["value"];
    if (star_value["_type"] != "Name")
      throw std::runtime_error(
        "Starred unpacking only supports simple names, not " +
        star_value["_type"].get<std::string>());

    // Create new list for the starred variable
    symbolt &star_list = create_list();

    // Compute source list size once
    const symbolt *size_func =
      converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
    assert(size_func);

    symbolt &size_var = converter_.create_tmp_symbol(
      list_value_, "$unpack_size$", size_type(), gen_zero(size_type()));
    code_declt size_decl(build_symbol(size_var));
    target_block.copy_to_operands(size_decl);

    code_function_callt size_call;
    size_call.function() = build_symbol(*size_func);
    size_call.arguments().push_back(
      list_expr.type().is_pointer() ? list_expr : build_address_of(list_expr));
    size_call.lhs() = build_symbol(size_var);
    size_call.type() = size_type();
    size_call.location() = loc;
    target_block.copy_to_operands(size_call);

    // upper = size - after_star
    exprt upper_expr;
    if (after_star > 0)
    {
      upper_expr = exprt("-", size_type());
      upper_expr.copy_to_operands(
        build_symbol(size_var), from_integer(after_star, size_type()));
    }
    else
    {
      upper_expr = build_symbol(size_var);
    }

    // Loop: for loop_idx = before_star; loop_idx < upper; loop_idx++
    symbolt &loop_idx = converter_.create_tmp_symbol(
      list_value_, "$i$", size_type(), gen_zero(size_type()));
    code_assignt idx_init(
      build_symbol(loop_idx), from_integer(before_star, size_type()));
    target_block.copy_to_operands(idx_init);

    exprt loop_cond("<", bool_type());
    loop_cond.copy_to_operands(build_symbol(loop_idx), upper_expr);

    code_blockt loop_body;

    // tmp_at = __ESBMC_list_at(list_expr, loop_idx)
    const exprt at_call =
      build_list_at_call(list_expr, build_symbol(loop_idx), list_value_);
    symbolt &tmp_at = converter_.create_tmp_symbol(
      list_value_,
      "tmp_unpack_at",
      pointer_typet(converter_.get_type_handler().get_list_element_type()),
      exprt());
    code_declt tmp_at_decl(build_symbol(tmp_at));
    tmp_at_decl.copy_to_operands(at_call);
    loop_body.copy_to_operands(tmp_at_decl);

    // __ESBMC_list_push_shallow(star_list, tmp_at): preserve element value
    // pointers so nested lists survive the unpack copy uncorrupted (#5102).
    const symbolt *push_obj_func =
      converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_push_shallow");
    assert(push_obj_func);

    // Nested-list elements keep their inner pointer; scalars are byte-copied,
    // so the unpack copy does not corrupt nested lists (#5102).
    constant_exprt star_list_type_id(size_type());
    star_list_type_id.set_value(integer2binary(
      std::hash<std::string>{}(converter_.get_type_handler().type_to_string(
        converter_.get_type_handler().get_list_type())),
      config.ansi_c.address_width));
    exprt push_call = build_call_expr(
      *push_obj_func,
      bool_type(),
      {build_symbol(star_list), build_symbol(tmp_at), star_list_type_id});
    push_call.location() = loc;
    loop_body.copy_to_operands(
      converter_.convert_expression_to_code(push_call));

    // loop_idx++
    exprt inc("+", size_type());
    inc.copy_to_operands(build_symbol(loop_idx), gen_one(size_type()));
    code_assignt inc_assign(build_symbol(loop_idx), inc);
    loop_body.copy_to_operands(inc_assign);

    codet while_loop;
    while_loop.set_statement("while");
    while_loop.copy_to_operands(loop_cond, loop_body);
    target_block.copy_to_operands(while_loop);

    // Record element type for the starred list
    python_list::add_type_info_entry(star_list.id.as_string(), "", elem_type);

    // Assign the new list to the starred variable and register its type info
    assign_to_target(star_value, build_symbol(star_list));
    // Also register the starred variable's own symbol id so subsequent list
    // method calls (e.g., rest.append(x)) can look up the element type.
    {
      const std::string var_name = star_value["id"].get<std::string>();
      symbol_id var_sid = converter_.create_symbol_id();
      var_sid.set_object(var_name);
      python_list::add_type_info_entry(var_sid.to_string(), "", elem_type);
    }

    // Assign targets after the star using size_var
    for (size_t j = 0; j < after_star; j++)
    {
      size_t target_idx = static_cast<size_t>(star_idx) + 1 + j;
      // index = size - after_star + j
      exprt after_idx = exprt("-", size_type());
      after_idx.copy_to_operands(
        build_symbol(size_var), from_integer(after_star - j, size_type()));
      exprt list_at = build_list_at_call(list_expr, after_idx, list_value_);
      exprt val = extract_pyobject_value(list_at, elem_type);
      assign_to_target(targets[target_idx], val);
    }
  }
}
