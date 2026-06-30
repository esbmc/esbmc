#include <python-frontend/type_handler.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/type_utils.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/tuple_handler.h>
#include <python-frontend/python_typechecking.h>
#include <python-frontend/symbol_id.h>
#include <util/arith_tools.h>
#include <util/config.h>
#include <util/context.h>
#include <util/c_types.h>
#include <util/message.h>
#include <util/migrate.h>
#include <util/python_types.h>
#include <irep2/irep2_utils.h>

#include <regex>

namespace
{
// 512-bit signed bitvector for Python int under --ir. Roughly 154 decimal
// digits; covers factorial(170), 64-byte from_bytes round-trips, bit_length
// on values up to 2^509. Under int-encoding the SMT layer drops widths and
// the value space is unbounded — the width here only sizes constant-fold
// intermediates. Issue #4642.
constexpr unsigned kPythonBignumWidth = 512;

// The bit_length OM (`src/python-frontend/models/int.py`) caps its loop at
// `length < kPythonBitLengthCap` so symex terminates without `--unwind`
// (issue #4756). For soundness the cap must be at least
// kPythonBignumWidth: any input representable in IntWide has bit_length
// strictly less than kPythonBignumWidth, so the loop exits via `n == 0`
// and never via the cap. If kPythonBignumWidth grows past the cap, the OM
// silently underreports — bump kPythonBitLengthCap and the OM literal
// together (the OM literal cannot read this constant; it is FLAIL-mangled
// before the C++ side is touched).
constexpr unsigned kPythonBitLengthCap = 512;
static_assert(
  kPythonBignumWidth <= kPythonBitLengthCap,
  "bit_length OM cap (models/int.py) must cover the full Python int width");

// Phase 4.3 seam (Part IV §5/§6): lower an internally-built IREP2 type to the
// legacy `typet` the symbol table and shared downstream passes consume,
// re-attaching the `#cpp_type` hint IREP2 cannot carry (F-P5). The elementary
// builders construct `type2tc` via typed factories and pass through here, so
// the legacy bytes reaching `create_symbol` stay byte-identical to before.
typet lower_to_seam(const type2tc &t, const irep_idt &cpp_type = irep_idt())
{
  typet legacy = migrate_type_back(t);
  if (!cpp_type.empty())
    type_utils::set_cpp_type(legacy, cpp_type);
  return legacy;
}
} // namespace

unsigned type_handler::python_int_width()
{
  return config.options.get_bool_option("int-encoding")
           ? kPythonBignumWidth
           : static_cast<unsigned>(config.ansi_c.long_long_int_width);
}

typet type_handler::python_int_typet()
{
  if (config.options.get_bool_option("int-encoding"))
    return signedbv_typet(kPythonBignumWidth);
  return long_long_int_type();
}

type_handler::type_handler(const python_converter &converter)
  : converter_(converter)
{
}

exprt type_handler::get_expr_helper(const nlohmann::json &json) const
{
  // This is safe because get_expr doesn't modify the converter's logical state
  return const_cast<python_converter &>(converter_).get_expr(json);
}

bool type_handler::is_pointer_free(const typet &t) const
{
  namespacet ns(converter_.symbol_table());
  typet ty = (t.id() == "symbol") ? ns.follow(t) : t;

  if (ty.is_pointer())
    return false;
  if (ty.is_array())
    return is_pointer_free(ty.subtype());
  if (ty.is_struct() || ty.is_union())
  {
    for (const auto &comp : to_struct_union_type(ty).components())
      if (!is_pointer_free(comp.type()))
        return false;
    return true;
  }
  return true; // primitives
}

bool type_handler::is_constructor_call(const nlohmann::json &json) const
{
  if (
    !json.contains("_type") || json["_type"] != "Call" ||
    !json.contains("func") || !json["func"].is_object())
    return false;

  const auto &func = json["func"];
  if (!func.contains("_type") || !func["_type"].is_string())
    return false;

  std::string func_name;
  if (func["_type"] == "Attribute")
  {
    if (!func.contains("attr") || !func["attr"].is_string())
      return false;
    func_name = func["attr"].get<std::string>();
  }
  else
  {
    if (!func.contains("id") || !func["id"].is_string())
      return false;
    func_name = func["id"].get<std::string>();
  }

  if (func_name == "__init__")
    return true;

  if (type_utils::is_builtin_type(func_name))
    return false;

  /* The statement is a constructor call if the function call on the
   * rhs corresponds to the name of a class. */

  // First, check if the class is defined in the AST (handles forward references)
  // example: class Foo: -> "Bar":
  // Bar is a class here defined later
  if (json_utils::is_class(func_name, converter_.ast()))
    return true;

  // Then check the symbol table for already-processed classes
  bool is_ctor_call = false;

  const contextt &symbol_table = converter_.symbol_table();

  symbol_table.foreach_operand([&](const symbolt &s) {
    if (s.get_type().id() == "struct" && s.name == func_name)
    {
      is_ctor_call = true;
      return;
    }
  });

  return is_ctor_call;
}

/// This utility maps internal ESBMC types to their corresponding Python type strings
std::string type_handler::type_to_string(const typet &t) const
{
  if (t == double_type())
    return "float";

  // Both the default int64 lowering and the --ir bignum widening
  // produced by python_int_typet() name the same Python type. Method
  // dispatch (e.g. `(2 ** 64).bit_length()`) keys off this string, so
  // both widths must map to "int". Issue #1964 / #4642.
  if (t == long_long_int_type() || t == python_int_typet())
    return "int";

  if (t == long_long_uint_type())
    return "uint64";

  if (t == bool_type())
    return "bool";

  if (t == uint256_type())
    return "uint256";

  if (t.is_array())
  {
    const array_typet &arr_type = to_array_type(t); // Safer than static_cast

    const typet &elem_type = arr_type.subtype();

    if (elem_type == char_type())
      return "str";

    if (elem_type == int_type())
      return "bytes";

    // Handle nested arrays (e.g., list of strings)
    if (elem_type.is_array())
      return type_to_string(elem_type);
  }

  if (t.is_pointer() && t.subtype() == char_type())
    return "str";

  return "";
}

std::string type_handler::get_python_type_name(const typet &t) const
{
  if (is_complex_type(t))
    return "complex";
  if (t.is_bool())
    return "bool";
  if (t.is_floatbv())
    return "float";
  if (t.is_signedbv() || t.is_unsignedbv())
    return "int";
  if ((t.is_array() || t.is_pointer()) && t.subtype() == char_type())
    return "str";
  if (t.id() == "symbol")
  {
    std::string tag = t.get_string("identifier");
    return (tag.rfind("tag-", 0) == 0) ? tag.substr(4) : tag;
  }
  return type_to_string(t);
}

std::string type_handler::get_var_type(const std::string &var_name) const
{
  nlohmann::json ref = json_utils::find_var_decl(
    var_name, converter_.current_function_name(), converter_.ast());

  if (ref.empty())
    return std::string();

  if (!ref.contains("annotation") || ref["annotation"].is_null())
    return std::string();

  const auto &annotation = ref["annotation"];

  // Handle simple type annotations: int, str, list, etc.
  if (annotation.is_object() && annotation.contains("id"))
    return annotation["id"].get<std::string>();

  // Handle subscripted types: List[str], Optional[int], etc.
  if (
    annotation.is_object() && annotation.contains("_type") &&
    annotation["_type"] == "Subscript" && annotation.contains("value") &&
    annotation["value"].is_object() && annotation["value"].contains("id"))
    return annotation["value"]["id"];

  // Handle Union types (e.g., list[str] | None, str | int)
  // Union is represented as BinOp with BitOr operator
  if (
    annotation.is_object() && annotation.contains("_type") &&
    annotation["_type"] == "BinOp" && annotation.contains("op") &&
    annotation["op"]["_type"] == "BitOr")
  {
    // For Union types, extract the non-None type
    // Check left side first
    if (annotation.contains("left"))
    {
      const auto &left = annotation["left"];

      // Skip None type on the left
      if (!(left.contains("_type") && left["_type"] == "Constant" &&
            left.contains("value") && left["value"].is_null()))
      {
        // Recursively extract type from left side
        if (left.contains("id"))
          return left["id"].get<std::string>();

        // Handle subscripted types on left: list[str] | None
        if (
          left["_type"] == "Subscript" && left.contains("value") &&
          left["value"].contains("id"))
          return left["value"]["id"].get<std::string>();
      }
    }

    // If left was None, check right side
    if (annotation.contains("right"))
    {
      const auto &right = annotation["right"];

      if (!(right.contains("_type") && right["_type"] == "Constant" &&
            right.contains("value") && right["value"].is_null()))
      {
        if (right.contains("id"))
          return right["id"].get<std::string>();

        if (
          right["_type"] == "Subscript" && right.contains("value") &&
          right["value"].contains("id"))
          return right["value"]["id"].get<std::string>();
      }
    }
  }

  return std::string();
}

std::string
type_handler::get_var_classname(const nlohmann::json &value_node) const
{
  if (!value_node.contains("_type") || value_node["_type"] != "Name")
    return "";

  const std::string var_name = value_node["id"].get<std::string>();
  auto get_class_name = [this](const std::string &name) -> std::string {
    if (name.empty())
      return "";

    const std::string class_name =
      (name.rfind("tag-", 0) == 0) ? name.substr(4) : name;
    const std::string class_tag = "tag-" + class_name;
    return converter_.symbol_table().find_symbol(class_tag) ? class_name : "";
  };

  symbol_id var_sid(
    converter_.python_file(),
    converter_.current_classname(),
    converter_.current_function_name());
  var_sid.set_object(var_name);

  symbolt *var_symbol = converter_.find_symbol(var_sid.to_string());
  if (!var_symbol)
    var_symbol = converter_.find_symbol(var_sid.global_to_string());

  if (var_symbol)
  {
    const auto annotation_types =
      converter_.get_typechecker().get_annotation_types(
        var_symbol->id.as_string());
    if (!annotation_types.empty())
    {
      typet ann_type = annotation_types.front();
      if (ann_type.is_pointer())
        ann_type = ann_type.subtype();
      if (ann_type.is_struct())
        return get_class_name(to_struct_type(ann_type).tag().as_string());
    }
  }

  return get_class_name(get_var_type(var_name));
}

/// Check if two types are compatible for list homogeneity
/// This considers strings of different lengths as the same type
bool type_handler::are_types_compatible(const typet &t1, const typet &t2) const
{
  // Exact match
  if (t1 == t2)
    return true;

  // Both are character arrays (strings) - consider them compatible
  if (t1.is_array() && t2.is_array())
  {
    const array_typet &arr1 = to_array_type(t1);
    const array_typet &arr2 = to_array_type(t2);

    // If subtypes match, consider them compatible regardless of size
    if (arr1.subtype() == arr2.subtype())
      return true;
  }

  return false;
}

/// Get a normalized/canonical type for list element type inference
/// This ensures all strings use the same representative type regardless of length
typet type_handler::get_canonical_string_type(const typet &t) const
{
  // For string types (char arrays), return a canonical string type
  if (t.is_array())
  {
    const array_typet &arr_type = to_array_type(t);
    if (arr_type.subtype() == char_type())
    {
      // Return a canonical string type (size 0 array indicates variable length string)
      return build_array(char_type(), 0);
    }
  }

  return t;
}

/// This method creates a `typet` representing a statically sized array.
/// It is typically used to model Python sequences like strings and byte arrays
typet type_handler::build_array(const typet &sub_type, const size_t size) const
{
  // Phase 4.3 (Part IV §5): build the array type IREP2-internal and lower it
  // back to the legacy `typet` at the seam. The element type arrives as a
  // legacy `typet` (from the elementary builders / callers), so migrate it in;
  // the size is a constant of the platform word-width unsignedbv (`size_type`),
  // matching the legacy `constant_exprt` byte-for-byte after back-migration.
  const type2tc subtype = migrate_type(sub_type);
  const type2tc size_t_type = migrate_type(size_type());
  const expr2tc array_size = constant_int2tc(size_t_type, BigInt(size));
  return lower_to_seam(array_type2tc(subtype, array_size, false));
}

std::vector<int> type_handler::get_array_type_shape(const typet &type) const
{
  // If the type is not an array, return an empty shape.
  if (!type.is_array())
    return {};

  // Since type is an array, cast it to array_typet.
  const auto &arr_type = static_cast<const array_typet &>(type);
  std::vector<int> shape{
    std::stoi(arr_type.size().value().as_string(), nullptr, 2)};

  // Recursively append dimensions from the subtype.
  auto sub_shape = get_array_type_shape(type.subtype());
  shape.insert(shape.end(), sub_shape.begin(), sub_shape.end());

  return shape;
}

/// Convert a Python AST type to an ESBMC internal irep type.
/// This function maps high-level Python types (from AST) to low-level internal
/// ESBMC representations using `typet`. It supports core built-in types
///
/// References:
/// - Python 3 type system: https://docs.python.org/3/library/stdtypes.html
/// - ESBMC irep type system: src/util/type.h
typet type_handler::get_typet(const std::string &ast_type, size_t type_size)
  const
{
  if (ast_type == "Any")
    return any_type();

  // Handle empty string - return empty type without error message
  // This can occur when no type annotation is provided or for internal use
  if (ast_type.empty())
    return empty_typet();

  // Handle generic type annotations such as list[T], dict[K,V], etc.
  // Extract base type and let existing code handle it
  size_t bracket_pos = ast_type.find('[');
  if (bracket_pos != std::string::npos)
  {
    std::string base_type = ast_type.substr(0, bracket_pos);
    return get_typet(base_type, type_size);
  }

  // type: represents Python type objects (int, str, float, bool, etc.)
  // Type objects are used for dynamic type checking and introspection
  if (ast_type == "type")
    return build_array(char_type(), type_size > 0 ? type_size : 10);

  // Typing module types should be treated as transparent
  // These are type hints only and don't enforce runtime type checking
  if (ast_type == "BinaryIO" || ast_type == "TextIO" || ast_type == "IO")
    return any_type();

  // object — top type in Python; accept any value
  if (ast_type == "object")
    return any_type();

  // NoneType — represents Python's None value, modelled as a pointer-width
  // unsigned integer (the legacy pointer_type() helper). Built IREP2-internal
  // and lowered at the seam (Phase 4.3, Part IV §5).
  if (ast_type == "NoneType")
    return lower_to_seam(unsignedbv_type2tc(config.ansi_c.pointer_width()));

  // Optional[T] - when type string is just "Optional" without inner type
  // This can occur during type inference. Same pointer-width unsigned integer
  // placeholder as NoneType, lowered at the seam.
  if (ast_type == "Optional")
    return lower_to_seam(unsignedbv_type2tc(config.ansi_c.pointer_width()));

  // Callable: represents function/callable types
  // Return a pointer to a generic no-argument code type (function pointer),
  // built IREP2-internal and lowered at the seam (Phase 4.3, Part IV §5). Empty
  // argument/name vectors satisfy code_type2t's args.size()==argument_names
  // .size() invariant (irep2_type.h).
  if (ast_type == "Callable")
  {
    const type2tc code_t = code_type2tc(
      std::vector<type2tc>{},
      get_empty_type(),
      std::vector<irep_idt>{},
      /*ellipsis=*/false);
    return lower_to_seam(pointer_type2tc(code_t));
  }

  // Python float type: IEEE 754 double-precision mapping
  // Python floats are implemented using C double (IEEE 754 double-precision)
  // as per Python documentation. This ensures proper precision, range, and
  // compatibility with Python's numeric type promotion (int -> float -> complex).
  if (ast_type == "float")
    return lower_to_seam(double_type2());

  // int — arbitrarily large integers
  // We approximate using 64-bit signed integer here.
  if (ast_type == "int" || ast_type == "GeneralizedIndex")
    // FIXME: Support bignum for true Python semantics
    return lower_to_seam(signedbv_type2tc(config.ansi_c.long_long_int_width));

  // Internal alias for operational models that need a width-polymorphic
  // signed integer parameter — accepts both narrow and wide Python int
  // values without truncating at the call boundary. Currently used by
  // models/int.py::bit_length so that `int(2 ** 64).bit_length()` works
  // under --ir without forcing every other `int`-typed callsite into the
  // wider representation (which would cascade into the list/dict/set OM,
  // see #4653). Issue #1964 / #4642.
  if (ast_type == "IntWide")
    return python_int_typet();

  // Unsigned integers used in domains like Ethereum or system modeling
  if (
    ast_type == "uint" || ast_type == "uint64" || ast_type == "Epoch" ||
    ast_type == "Slot")
    return lower_to_seam(unsignedbv_type2tc(config.ansi_c.long_long_int_width));

  // bool — represents True/False
  if (ast_type == "bool")
    return lower_to_seam(get_bool_type());

  // slice — Python's slice() builtin, modelled as __ESBMC_PySliceObj.
  // Only resolve to the slice struct if the operational model is loaded;
  // otherwise fall through so early type-handler queries don't assert.
  if (ast_type == "slice")
  {
    const symbolt *sym =
      converter_.symbol_table().find_symbol("tag-struct __ESBMC_PySliceObj");
    if (sym)
      return symbol_typet(sym->id);
  }

  if (ast_type == "complex")
  {
    const char *complex_type_id = "tag-complex";
    contextt &symbol_table = converter_.symbol_table();

    if (!symbol_table.find_symbol(complex_type_id))
    {
      symbolt type_symbol;
      type_symbol.id = complex_type_id;
      type_symbol.name = "complex";
      type_symbol.set_type(get_complex_struct_type());
      type_symbol.mode = "Python";
      type_symbol.is_type = true;
      symbol_table.move_symbol_to_context(type_symbol);
    }

    return get_complex_struct_type();
  }

  // Custom large unsigned integer types (used in Ethereum, BLS, etc.)
  if (ast_type == "uint256" || ast_type == "BLSFieldElement")
    return lower_to_seam(unsignedbv_type2tc(256));

  // bytes — immutable sequences of bytes
  // Here modeled as array of signed integers (8-bit).
  if (ast_type == "bytes")
  {
    // TODO: Refactor to model using unsigned/signed char
    return build_array(long_long_int_type(), type_size);
  }

  // bytearray — the mutable counterpart of bytes — is not modeled. Reject it
  // with a clean diagnostic: the unsupported-type fall-through below returns an
  // empty_typet() that later crashes symex with an uncaught
  // type2t::symbolic_type_excp (SIGABRT) on item assignment / bytearray(n).
  if (ast_type == "bytearray")
    throw std::runtime_error(
      "bytearray is not supported; use bytes for an immutable byte sequence");

  // divmod() — returns tuple of (quotient, remainder)
  // The return type is determined dynamically based on operands
  // Let handle_divmod create the proper type
  if (ast_type == "divmod")
  {
    // Return empty type - the actual tuple type will be created
    // in handle_divmod based on the operand types
    return empty_typet();
  }

  // ord(): Converts a 1-character string to its Unicode code point (as integer)
  if (ast_type == "ord")
    return long_long_int_type();

  // abs(): Return the absolute value of a number
  if (ast_type == "abs")
    return long_long_int_type();

  // str/string: immutable sequences of Unicode characters
  // chr(): returns a 1-character string
  // bin(): returns string representation of integer in binary
  // hex(): returns string representation of integer in hex
  // oct(): Converts an integer to a lowercase octal string
  if (
    ast_type == "str" || ast_type == "string" || ast_type == "chr" ||
    ast_type == "bin" || ast_type == "hex" || ast_type == "oct")
  {
    if (type_size == 1)
    {
      // 8-bit char built IREP2-internal; #cpp_type "char" is re-attached at the
      // seam for C-backend compatibility (F-P5 — IREP2 cannot carry it).
      const type2tc char_t = config.ansi_c.char_is_unsigned
                               ? unsignedbv_type2tc(config.ansi_c.char_width)
                               : signedbv_type2tc(config.ansi_c.char_width);
      return lower_to_seam(char_t, "char");
    }
    return build_array(char_type(), type_size); // Array of characters
  }

  // all(): Return True if all elements are truthy (returns bool)
  if (ast_type == "all")
    return bool_type();

  // any(): Return True if any element is truthy (returns bool)
  if (ast_type == "any")
    return bool_type();

  // tuple — handle tuple type annotations
  // For generic "tuple" without element types, return empty type
  // so the actual type is inferred from the tuple value
  if (ast_type == "tuple")
    return empty_typet();

  // The capitalised typing-module aliases (typing.List/Dict/Set) must resolve
  // to the same builtin collection types as their lowercase forms. Otherwise a
  // nested annotation like List[List[float]] types its element as a bogus
  // "tag-List" struct that no list/dict/set machinery recognises — len(A[0])
  // then misroutes to strlen() and aborts on a struct/pointer mismatch (#5162;
  // Dict/Set crash identically). A user may legally shadow these names with
  // their own class (e.g. a hand-rolled `class List`, #3728), and that class
  // must win. is_class() is unusable as the guard because it also matches the
  // typing OM's own class definitions, which are imported in exactly the alias
  // case we want to rewrite — so scan only the user's own module body.
  auto typing_alias = [&](const char *alias) {
    return ast_type == alias &&
           json_utils::find_class(converter_.ast()["body"], alias).is_null();
  };

  // list/range — range objects are stored as lists in ESBMC's model
  if (ast_type == "list" || ast_type == "range" || typing_alias("List"))
    return get_list_type();

  // dict — handle dict type annotations
  // For generic "dict" without key/value types, return empty type
  // so the actual type is inferred from the dictionary literal
  if (ast_type == "dict" || typing_alias("Dict"))
    return get_dict_type();

  // Reuse list infrastructure for simplicity for now
  if (ast_type == "set" || typing_alias("Set"))
    return get_list_type();

  // Custom user-defined types / classes
  if (
    json_utils::is_class(ast_type, converter_.ast()) ||
    type_utils::is_python_exceptions(ast_type))
    return symbol_typet("tag-" + ast_type);

  // Check if it's a defined class in the AST
  bool is_defined = json_utils::is_class(ast_type, converter_.ast());

  // Look up the type in the symbol table
  if (!is_defined)
  {
    symbolt *s = converter_.find_symbol(std::string("tag-" + ast_type));
    if (s)
      return s->get_type();
  }

  // Check if it's a built-in type (handles tuple, list, dict, etc.)
  if (!is_defined)
    is_defined = type_utils::is_builtin_type(ast_type);

  // Check if it's imported
  if (!is_defined)
    is_defined = converter_.is_imported_module(ast_type);

  if (!is_defined)
  {
    // Imported free functions (e.g. "from dataclasses import replace") can
    // appear in call expressions and should not be treated as unknown type
    // names during inference.
    const std::string import_probe =
      "py:" + converter_.python_file() + "@" + ast_type;
    if (converter_.find_imported_symbol(import_probe) != nullptr)
      return any_type();

    const nlohmann::json &decl =
      json_utils::find_var_decl(ast_type, "", converter_.ast());
    if (!decl.empty() && decl.contains("value") && decl["value"].is_object())
    {
      const nlohmann::json &value = decl["value"];
      // TypeVar(...) has no concrete type(treat as Any)
      if (
        value.contains("_type") && value["_type"] == "Call" &&
        value.contains("func") && value["func"].is_object() &&
        value["func"].contains("id") && value["func"]["id"] == "TypeVar")
        return any_type();
      // Handle simple alias: X = T to T
      // the preprocessor rewrites
      // X = NewType('X', T) to X = T
      if (
        value.contains("_type") && value["_type"] == "Name" &&
        value.contains("id"))
      {
        const std::string &target = value["id"];
        if (target != ast_type && type_utils::is_builtin_type(target))
          return get_typet(target, type_size);
        if (
          target != ast_type && json_utils::is_class(target, converter_.ast()))
          return symbol_typet("tag-" + target);
      }
    }
  }

  // If still not found, it's a NameError
  if (!is_defined)
  {
    throw std::runtime_error(
      "NameError: name '" + ast_type + "' is not defined");
  }

  // Otherwise, it's an unsupported/unhandled type - log warning and continue
  log_warning("Unknown or unsupported AST type: {}", ast_type);

  return empty_typet();
}

typet type_handler::get_typet_from_call_func(const nlohmann::json &func) const
{
  std::string func_name;
  if (func.contains("id"))
  {
    func_name = func["id"].get<std::string>();
    if (type_utils::is_builtin_type(func_name))
      return get_typet(func_name);
    // User-defined class constructor: A() -> struct type tag-A
    if (json_utils::is_class(func_name, converter_.ast()))
      return symbol_typet("tag-" + func_name);
  }
  else if (func["_type"] == "Attribute" && func.contains("attr"))
  {
    func_name = func["attr"].get<std::string>();
    if (func_name == "randint" || func_name == "randrange")
      return long_long_int_type();
    if (func_name == "random" || func_name == "uniform")
      return lower_to_seam(double_type2());
    if (type_utils::is_builtin_type(func_name))
      return get_typet(func_name);
  }

  const std::string nondet_prefix = "nondet_";
  if (!func_name.empty() && func_name.rfind(nondet_prefix, 0) == 0)
  {
    typet t = get_typet(func_name.substr(nondet_prefix.size()));
    if (t != empty_typet())
      return t;
  }

  throw std::runtime_error("Invalid type");
}

typet type_handler::get_typet(const nlohmann::json &elem) const
{
  // Handle null/empty values
  if (elem.is_null())
    return empty_typet();

  // Handle primitive types
  if (elem.is_number_integer() || elem.is_number_unsigned())
    return long_long_int_type();
  else if (elem.is_boolean())
    return bool_type();
  else if (elem.is_number_float())
    return lower_to_seam(double_type2());
  else if (elem.is_string())
  {
    size_t str_size = elem.get<std::string>().size();
    if (str_size > 1)
      str_size += 1;
    return build_array(char_type(), str_size);
  }

  // Handle nested value object
  if (elem.is_object())
  {
    // Tuple annotation subscript: Tuple[int, str] / tuple[int, str]. Build the
    // concrete tuple struct (matching the tag-tuple_* type a tuple literal
    // produces) so a list element annotated list[Tuple[...]] resolves to real
    // components. This must precede the wrapper-node unwrap below: a Subscript
    // node also carries a "value" key (the subscripted name), so the unwrap
    // would otherwise recurse into the bare "Tuple" name and resolve it to the
    // opaque 0-member "Tuple" symbol type — which crashes the SMT
    // cast-to-struct path when a function returns such an element.
    if (
      elem["_type"] == "Subscript" && elem.contains("value") &&
      elem["value"].is_object() && elem["value"].contains("id") &&
      elem["value"]["id"].is_string() &&
      (elem["value"]["id"] == "Tuple" || elem["value"]["id"] == "tuple") &&
      elem.contains("slice"))
      return converter_.get_tuple_handler().get_tuple_type_from_annotation(
        elem);

    // Recursive delegation for wrapper node
    if (elem.contains("value"))
      return get_typet(elem["value"]);

    // Handle Python AST UnaryOp node (e.g., -1, +1, ~1, not x)
    if (elem["_type"] == "UnaryOp" && elem.contains("operand"))
    {
      // For unary operations, the result type is typically the same as the operand type
      return get_typet(elem["operand"]);
    }

    // Handle Python AST BinOp node (e.g., 1 + 0, a * b). Reuse
    // get_operand_type, which recursively resolves the operand types, then
    // map the type name back to a typet. Without this, an expression-derived
    // binding used as a list element aborted (issue #4909).
    if (elem["_type"] == "BinOp")
      return get_typet(get_operand_type(elem));

    // Handle Python AST List node
    if (elem["_type"] == "List" && elem.contains("elts"))
    {
      const auto &elements = elem["elts"];
      if (elements.empty())
        return build_array(long_long_int_type(), 0);

      typet subtype = get_typet(elements[0]);
      return build_array(subtype, elements.size());
    }

    // Handle Python AST Tuple node
    // Converts tuple expressions such as (1, 2) or ("hello", 42, 3.14) to struct types
    if (elem["_type"] == "Tuple" && elem.contains("elts"))
    {
      struct_typet tuple_type;
      const auto &elements = elem["elts"];

      // Build a struct component for each tuple element
      for (size_t i = 0; i < elements.size(); i++)
      {
        // Recursively get the type of each element
        typet elem_type = get_typet(elements[i]);

        // Create a named component for this element
        // Component names follow pattern: element_0, element_1, element_2, ...
        std::string comp_name = "element_" + std::to_string(i);
        struct_typet::componentt comp(comp_name, elem_type);
        tuple_type.components().push_back(comp);
      }

      return tuple_type;
    }

    // Handle Dict
    if (elem["_type"] == "Dict" && elem.contains("keys"))
      return get_dict_type(elem);
  }

  if (elem.is_array())
  {
    if (elem.empty())
      return build_array(long_long_int_type(), 0);

    typet subtype = get_typet(elem[0]);
    return build_array(subtype, elem.size());
  }

  if (
    elem["_type"] == "Call" && type_utils::is_builtin_type(elem["func"]["id"]))
  {
    return get_typet(elem["func"]["id"].get<std::string>());
  }

  if (elem["_type"] == "Name")
  {
    const nlohmann::json &var = json_utils::find_var_decl(
      elem["id"], converter_.current_function_name(), converter_.ast());

    if (!var.empty() && var.contains("value") && !var["value"].is_null())
    {
      // Resolve the type of the binding's right-hand side. Dispatch on the
      // whole RHS node rather than assuming it is a Constant wrapper: a derived
      // binding such as `X = Y` (alias) or `X = 1 + 0` (expression) has no
      // nested "value" key, so the old `var["value"]["value"]` aborted. See
      // issue #4909.
      if (var["value"]["_type"] != "Call")
        return get_typet(var["value"]);

      if (var["value"].contains("func"))
        return get_typet_from_call_func(var["value"]["func"]);

      throw std::runtime_error("Invalid type");
    }
    return empty_typet();
  }

  if (elem["_type"] == "Call")
    return get_typet_from_call_func(elem["func"]);

  throw std::runtime_error("Invalid type");
}

bool type_handler::has_multiple_types(const nlohmann::json &container) const
{
  if (container.empty())
    return false;

  // Helper lambda that leverages existing get_typet method
  auto get_element_type = [this](const nlohmann::json &element) -> typet {
    try
    {
      typet elem_type = get_typet(element);
      // For array types, we want the element type, not the container type
      return elem_type.is_array() ? elem_type.subtype() : elem_type;
    }
    catch (const std::exception &)
    {
      log_warning("Failed to determine element type in has_multiple_types");
      return empty_typet();
    }
  };

  // Get canonical type of first element
  typet canonical_first_type =
    get_canonical_string_type(get_element_type(container[0]));

  if (canonical_first_type == empty_typet())
    return false; // Couldn't determine type, assume homogeneous

  // Check all elements for type compatibility
  for (const auto &element : container)
  {
    // Handle nested lists recursively
    if (
      element["_type"] == "List" && element.contains("elts") &&
      !element["elts"].empty())
    {
      if (has_multiple_types(element["elts"]))
        return true;
    }

    // Check type compatibility
    typet element_type = get_canonical_string_type(get_element_type(element));
    if (
      element_type != empty_typet() &&
      !are_types_compatible(canonical_first_type, element_type))
      return true;
  }

  return false;
}

typet type_handler::get_list_type(const nlohmann::json &list_value) const
{
  if (
    list_value.is_null() ||
    (list_value.contains("elts") && list_value["elts"].empty()))
  {
    // For empty containers, return the list type pointer
    // The actual element type will be determined when elements are added
    return get_list_type();
  }

  if (list_value["_type"] == "arg" && list_value.contains("annotation"))
  {
    // Handle case where annotation is directly a Subscript (e.g., List['Action'])
    if (list_value["annotation"]["_type"] == "Subscript")
    {
      const nlohmann::json &slice = list_value["annotation"]["slice"];
      typet t;

      if (slice.contains("id"))
      {
        // Regular identifier like List[int]
        t = get_typet(slice["id"].get<std::string>());
      }
      else if (slice["_type"] == "Constant" && slice.contains("value"))
      {
        // String constant like List['Action'] (forward reference)
        std::string type_string = slice["value"].get<std::string>();
        t = get_typet(type_utils::remove_quotes(type_string));
      }
      else if (
        slice["_type"] == "Subscript" && slice.contains("value") &&
        slice["value"].is_object() && slice["value"].contains("id") &&
        slice["value"]["id"].is_string())
      {
        // Nested container like list[list[T]] or list[dict[K, V]] — resolve
        // to the inner container's own type so subsequent subscripts route
        // through the right element-access primitive.
        t = get_typet(slice["value"]["id"].get<std::string>());
      }
      else
        t = empty_typet();
      // Phase 4.3 (Part IV §5): build the pointer type IREP2-internal and lower
      // at the seam. The pointee arrives as a legacy typet, so migrate it in.
      return lower_to_seam(pointer_type2tc(migrate_type(t)));
    }

    // Check if the nested structure exists before accessing
    if (
      list_value["annotation"].contains("value") &&
      list_value["annotation"]["value"].contains("id"))
    {
      [[maybe_unused]] const nlohmann::json &type_ann =
        list_value["annotation"]["value"]["id"];
      assert(type_ann == "list" || type_ann == "List");
      typet t =
        get_typet(list_value["annotation"]["slice"]["id"].get<std::string>());
      // Phase 4.3 (Part IV §5): pointer built IREP2-internal, lowered at seam.
      return lower_to_seam(pointer_type2tc(migrate_type(t)));
    }
  }

  if (list_value["_type"] == "List") // Get list value type from elements
  {
    const nlohmann::json &elts = list_value["elts"];

    if (!has_multiple_types(elts)) // All elements have the same type
    {
      typet subtype;

      if (elts[0]["_type"] == "Constant" || elts[0]["_type"] == "UnaryOp")
      { // One-dimensional list
        // Retrieve the type of the first element
        const auto &elem = (elts[0]["_type"] == "UnaryOp")
                             ? elts[0]["operand"]["value"]
                             : elts[0]["value"];
        subtype = get_typet(elem);
      }
      else
      {
        // Get sub-array type from multi-dimensional list
        if (elts[0]["_type"] == "Call")
        {
          if (type_utils::is_builtin_type(elts[0]["func"]["id"]))
            subtype = get_typet(elts[0]["func"]["id"].get<std::string>());
        }
        else if (elts[0].contains("elts"))
          subtype = get_typet(elts[0]["elts"]);
        else
        {
          // Handle other element types directly
          subtype = get_typet(elts[0]);
        }
      }

      return build_array(subtype, elts.size());
    }
    throw std::runtime_error("Multiple type lists are not supported yet");
  }

  if (list_value["_type"] == "Call") // Get list type from function return type
  {
    symbol_id sid(
      converter_.python_file(),
      converter_.current_classname(),
      converter_.current_function_name());

    if (list_value["func"]["_type"] == "Attribute")
      sid.set_function(list_value["func"]["attr"]);
    else
      sid.set_function(list_value["func"]["id"]);

    symbolt *func_symbol = converter_.find_symbol(sid.to_string());

    assert(func_symbol);
    return static_cast<const code_typet &>(func_symbol->get_type())
      .return_type();
  }

  if (list_value.contains("_type") && list_value["_type"] == "BinOp")
  {
    // Handle cases like x = [0] * 5
    if (list_value["op"]["_type"] == "Mult")
    {
      exprt left_expr = get_expr_helper(list_value["left"]);
      exprt right_expr = get_expr_helper(list_value["right"]);

      typet list_type = (left_expr.is_symbol()) ? left_expr.type().subtype()
                                                : right_expr.type().subtype();
      exprt size = (left_expr.is_symbol()) ? right_expr : left_expr;
      // Phase 4.3 (Part IV §5): build the array type IREP2-internal and lower
      // at the seam. Unlike build_array the size here is a runtime expression
      // (e.g. `x = [0] * n`), so it is migrated in rather than a constant.
      expr2tc array_size;
      migrate_expr(size, array_size);
      return lower_to_seam(
        array_type2tc(migrate_type(list_type), array_size, false));
    }
  }

  return typet();
}

const typet type_handler::get_list_type() const
{
  static const symbolt *list_type_symbol = nullptr;
  const char *list_type_id = "tag-struct __ESBMC_PyListObj";
  list_type_symbol = converter_.symbol_table().find_symbol(list_type_id);
  assert(list_type_symbol);
  // Phase 4.3 (Part IV §5): pointer-to-symbol built IREP2-internal (symbol type
  // constructed natively, no legacy round-trip) and lowered at the seam.
  return lower_to_seam(pointer_type2tc(symbol_type2tc(list_type_symbol->id)));
}

typet type_handler::get_list_element_type() const
{
  static const symbolt *type = nullptr;
  const char *type_id = "tag-struct __ESBMC_PyObj";
  type = converter_.symbol_table().find_symbol(type_id);
  assert(type);
  return symbol_typet(type->id);
}

typet type_handler::get_slice_type() const
{
  const symbolt *slice_type_symbol =
    converter_.symbol_table().find_symbol("tag-struct __ESBMC_PySliceObj");
  assert(slice_type_symbol);
  return symbol_typet(slice_type_symbol->id);
}

/// This method inspects the JSON representation of a Python operand node and attempts to
/// infer its type based on its AST node type (`_type`). It currently supports variable
/// names, constants (literals), and list subscripts. This type information is used for
/// symbolic execution or translation within ESBMC.
std::string type_handler::get_operand_type(const nlohmann::json &operand) const
{
  // Handle variable reference (e.g., `x`)
  if (
    operand.contains("_type") && operand["_type"] == "Name" &&
    operand.contains("id"))
    return get_var_type(operand["id"]);

  // Handle constant/literal values (e.g., 42, "hello", True, 3.14)
  else if (operand["_type"] == "Constant" && operand.contains("value"))
  {
    const auto &value = operand["value"];
    if (value.is_string())
      return "str";
    else if (value.is_number_integer() || value.is_number_unsigned())
      return "int";
    else if (value.is_boolean())
      return "bool";
    else if (value.is_number_float())
      return "float";
  }
  else if (operand["_type"] == "Set")
  {
    return "set";
  }
  else if (operand["_type"] == "BinOp")
  {
    std::string lhs_type = get_operand_type(operand["left"]);
    std::string rhs_type = get_operand_type(operand["right"]);

    if (!lhs_type.empty() && lhs_type == rhs_type)
      return lhs_type;

    if (!lhs_type.empty())
      return lhs_type;
    if (!rhs_type.empty())
      return rhs_type;
  }
  else if (
    operand["_type"] == "Attribute" && operand.contains("attr") &&
    operand.contains("value"))
  {
    const std::string attr_name = operand["attr"].get<std::string>();

    auto find_annotated_attr_type =
      [&](const nlohmann::json &class_node) -> std::string {
      if (class_node.empty() || !class_node.contains("body"))
        return std::string();

      for (const auto &member : class_node["body"])
      {
        if (
          member["_type"] != "FunctionDef" || !member.contains("name") ||
          member["name"] != "__init__" || !member.contains("body"))
          continue;

        for (const auto &stmt : member["body"])
        {
          if (
            stmt["_type"] == "AnnAssign" && stmt.contains("target") &&
            stmt["target"].contains("_type") &&
            stmt["target"]["_type"] == "Attribute" &&
            stmt["target"].contains("attr") &&
            stmt["target"]["attr"] == attr_name &&
            stmt["target"].contains("value") &&
            stmt["target"]["value"].contains("_type") &&
            stmt["target"]["value"]["_type"] == "Name" &&
            stmt["target"]["value"].contains("id") &&
            stmt["target"]["value"]["id"] == "self" &&
            stmt.contains("annotation") && stmt["annotation"].contains("id"))
            return stmt["annotation"]["id"].get<std::string>();
        }
      }

      return std::string();
    };

    const auto &attr_value = operand["value"];
    if (
      attr_value.contains("_type") && attr_value["_type"] == "Name" &&
      attr_value.contains("id") && attr_value["id"] == "self")
    {
      const auto self_class_node = json_utils::find_class(
        converter_.ast()["body"], converter_.current_classname());
      std::string self_attr_type = find_annotated_attr_type(self_class_node);
      if (!self_attr_type.empty())
        return self_attr_type;
    }

    std::string obj_type = get_operand_type(attr_value);
    if (!obj_type.empty())
    {
      const auto class_node =
        json_utils::find_class(converter_.ast()["body"], obj_type);
      std::string attr_type = find_annotated_attr_type(class_node);
      if (!attr_type.empty())
        return attr_type;
    }
  }

  // Handle call expressions: constructor calls like A() and method calls like B().g()
  else if (operand["_type"] == "Call" && operand.contains("func"))
  {
    const auto &func = operand["func"];
    // Direct constructor call: A() — return the class name as the type
    if (func["_type"] == "Name" && func.contains("id"))
      return func["id"].get<std::string>();
    // Method call: obj.method() — infer the return type from the class definition
    if (
      func["_type"] == "Attribute" && func.contains("attr") &&
      func.contains("value"))
    {
      std::string obj_type = get_operand_type(func["value"]);
      if (!obj_type.empty())
      {
        std::string method_name = func["attr"].get<std::string>();
        const auto &ast = converter_.ast();
        nlohmann::json class_node =
          json_utils::find_class(ast["body"], obj_type);
        if (class_node.empty() || !class_node.contains("body"))
          return std::string();
        for (const auto &member : class_node["body"])
        {
          if (
            member["_type"] == "FunctionDef" && member["name"] == method_name &&
            member.contains("returns") && !member["returns"].is_null() &&
            member["returns"].contains("id"))
            return member["returns"]["id"].get<std::string>();
        }
      }
    }
  }

  // Handle list subscript (e.g., `mylist[0]`)
  else if (
    operand["_type"] == "Subscript" && operand.contains("value") &&
    get_operand_type(operand["value"]) == "list")
  {
    const auto &list_expr = operand["value"];
    if (list_expr.contains("id"))
    {
      std::string list_id = list_expr["id"].get<std::string>();

      // Find the declaration of the list variable
      nlohmann::json list_node = json_utils::find_var_decl(
        list_id, converter_.current_function_name(), converter_.ast());

      // Get the type of the list and return the subtype (element type)
      array_typet list_type = get_list_type(list_node["value"]);
      return type_to_string(list_type.subtype());
    }
  }

  // If no known type can be determined, issue a warning and return std::string()
  log_warning(
    "type_handler::get_operand_type: unable to determine operand type for AST "
    "node: {}",
    operand.dump(2));
  return std::string();
}

bool type_handler::is_2d_array(const nlohmann::json &arr) const
{
  return arr.contains("_type") && arr["_type"] == "List" &&
         arr.contains("elts") && !arr["elts"].empty() &&
         arr["elts"][0].is_object() && arr["elts"][0].contains("elts");
}

// Add this method to the type_handler class
int type_handler::get_array_dimensions(const nlohmann::json &arr) const
{
  if (!arr.is_object() || arr["_type"] != "List" || !arr.contains("elts"))
    return 0;

  if (arr["elts"].empty())
    return 1; // Empty array is considered 1D

  // Check the first element to determine nesting depth
  const auto &first_elem = arr["elts"][0];

  if (!first_elem.is_object())
    return 1;

  if (first_elem["_type"] == "List")
  {
    // Recursive case: this is a nested array
    return 1 + get_array_dimensions(first_elem);
  }
  else
  {
    // Base case: first element is not a list, so this is 1D
    return 1;
  }
}

size_t type_handler::get_type_width(const typet &type) const
{
  // First try to parse width directly
  std::string width_str = type.width().c_str();
  if (!width_str.empty())
  {
    try
    {
      return std::stoi(width_str);
    }
    catch (const std::exception &)
    {
      // Fall through to type name inference
    }
  }

  // If width is empty or parsing failed, try to infer from type name
  std::string type_str = type.width().as_string();
  if (!type_str.empty())
  {
    // Handle common Python/ESBMC type mappings
    if (type_str == "int32")
      return 32;
    else if (type_str == "int")
      return 64;
    else if (type_str == "int64" || type_str == "long")
      return 64;
    else if (type_str == "int16" || type_str == "short")
      return 16;
    else if (type_str == "int8" || type_str == "char")
      return 8;
    else if (type_str == "float32")
      return 32;
    else if (type_str == "float")
      return 64;
    else if (type_str == "double" || type_str == "float64")
      return 64;
    else if (type_str == "bool")
      return 1;

    // Try to extract number from string like "int32", "uint64", etc.
    std::regex width_regex(R"(\d+)");
    std::smatch match;
    if (std::regex_search(type_str, match, width_regex))
    {
      try
      {
        return std::stoi(match.str());
      }
      catch (const std::exception &)
      {
        // Fall through to default
      }
    }
  }

  // Default to 32 for unknown types
  return 32;
}

typet type_handler::build_optional_type(const typet &base_type)
{
  // Create a struct with two fields:
  // 1. is_none: bool - indicates if value is None
  // 2. value: T - the actual value when not None

  struct_typet optional_type;
  optional_type.tag("tag-Optional_" + base_type.to_string());
  set_python_aggregate_kind(optional_type, "optional");

  // Add is_none field
  struct_typet::componentt is_none_field("is_none", "is_none", bool_type());
  is_none_field.set_access("public");
  optional_type.components().push_back(is_none_field);

  // Add value field
  struct_typet::componentt value_field("value", "value", base_type);
  value_field.set_access("public");
  optional_type.components().push_back(value_field);

  return optional_type;
}

bool type_handler::class_derives_from(
  const std::string &class_name,
  const std::string &expected_base) const
{
  if (class_name == expected_base)
    return true;

  const auto &ast = converter_.ast();
  const auto class_node = json_utils::find_class(ast["body"], class_name);
  if (class_node.empty() || !class_node.contains("bases"))
    return false;

  for (const auto &base : class_node["bases"])
  {
    std::string base_name;
    if (base.contains("_type") && base["_type"] == "Name")
      base_name = base["id"].get<std::string>();
    else if (base.contains("_type") && base["_type"] == "Attribute")
      base_name = base["attr"].get<std::string>();
    if (!base_name.empty() && class_derives_from(base_name, expected_base))
      return true;
  }
  return false;
}

const typet type_handler::get_dict_type() const
{
  return converter_.dict_handler_->get_dict_struct_type();
}

typet type_handler::get_dict_type(const nlohmann::json &dict_value) const
{
  std::string dict_str = dict_value.dump(2);
  log_debug("type_handler", "get_dict_type - dict_value: {}", dict_str.c_str());

  // For now, return the generic dict type
  // In the future, this could infer specific key/value types
  return get_dict_type();
}
