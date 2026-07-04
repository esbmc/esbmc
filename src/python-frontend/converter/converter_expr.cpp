#include <python-frontend/converter/converter_internal.h>
#include <python-frontend/convert_float_literal.h>
#include <python-frontend/function_call/expr.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/python_annotation.h>
#include <python-frontend/python_consteval.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_dict_handler.h>
#include <python-frontend/python_exception_handler.h>
#include <python-frontend/python_expr_builder.h>
#include <python-frontend/python_int_overflow.h>
#include <python-frontend/python_lambda.h>
#include <python-frontend/python_list.h>
#include <python-frontend/python_math.h>
#include <python-frontend/string/string_builder.h>
#include <python-frontend/string/string_handler.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/tuple_handler.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/type_utils.h>
#include <irep2/irep2_utils.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/encoding.h>
#include <util/expr_util.h>
#include <util/message.h>
#include <util/migrate.h>
#include <util/python_types.h>
#include <util/std_code.h>
#include <util/string_constant.h>

using namespace json_utils;

static bool contains_cpp_throw(const exprt &expr)
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

static exprt build_shape_tuple_expr(
  python_converter &converter,
  const std::vector<exprt> &dims)
{
  std::vector<typet> element_types(dims.size(), int_type());
  struct_typet tuple_type =
    converter.get_tuple_handler().create_tuple_struct_type(element_types);
  // V.3: build the tuple value in IREP2, back-migrating once. The operands are
  // already-built int_type() dimension exprs, so a constant_struct2t over them
  // round-trips exactly through migrate. Re-attach the full struct type
  // afterwards: migrate_type drops the frontend-only #python_aggregate_kind
  // marker that the in/membership dispatch reads (python_aggregate_kind) with no
  // tag fallback — mirroring tuple_handler::get_tuple_expr.
  std::vector<expr2tc> members;
  members.reserve(dims.size());
  for (const exprt &d : dims)
  {
    expr2tc d2;
    migrate_expr(d, d2);
    members.push_back(std::move(d2));
  }
  exprt tuple_expr =
    migrate_expr_back(constant_struct2tc(migrate_type(tuple_type), members));
  tuple_expr.type() = tuple_type;
  return tuple_expr;
}

static nlohmann::json normalize_bool_index_node(const nlohmann::json &node)
{
  if (
    node.contains("_type") && node["_type"] == "Constant" &&
    node.contains("value") && node["value"].is_boolean())
  {
    nlohmann::json converted = node;
    converted["value"] = node["value"].get<bool>() ? 1 : 0;
    return converted;
  }
  return node;
}

static void throw_numpy_multidim_index_error(
  python_converter &converter,
  const nlohmann::json &element)
{
  std::ostringstream msg;
  msg << "TypeError: multi-dimensional indexing (a[i, j, ...]) is not "
         "supported; numpy arrays are modelled as 1D lists";
  const locationt loc = converter.get_location_from_decl(element);
  if (!loc.is_nil())
    msg << " at " << loc.get_file() << ":" << loc.get_line();
  throw std::runtime_error(msg.str());
}

// True for a bare `:` slice, i.e. Slice(lower=None, upper=None, step=None).
// Used to recognize the two 2-D slicing patterns this frontend supports,
// `a[:, j]` and `a[i, :]`; any other slice (bounded or stepped) inside a
// tuple subscript is still rejected by throw_numpy_multidim_index_error.
static bool is_full_slice_node(const nlohmann::json &node)
{
  if (!(node.contains("_type") && node["_type"] == "Slice"))
    return false;
  auto is_absent = [&](const char *key) {
    return !node.contains(key) || node[key].is_null();
  };
  return is_absent("lower") && is_absent("upper") && is_absent("step");
}

static void throw_numpy_too_many_indices_error(
  python_converter &converter,
  const nlohmann::json &element,
  std::size_t num_indices)
{
  std::ostringstream msg;
  msg << "IndexError: too many indices for array: array has fewer "
         "dimensions than the "
      << num_indices << " indices given";
  const locationt loc = converter.get_location_from_decl(element);
  if (!loc.is_nil())
    msg << " at " << loc.get_file() << ":" << loc.get_line();
  throw std::runtime_error(msg.str());
}

class get_expr_depth_guard
{
public:
  explicit get_expr_depth_guard(python_converter &converter)
    : converter(converter)
  {
    constexpr std::size_t kMaxGetExprDepth = 512;
    if (++converter.get_expr_depth_ > kMaxGetExprDepth)
    {
      --converter.get_expr_depth_;
      throw std::runtime_error(
        "TypeError: Python expression nesting exceeded recursion limit of " +
        std::to_string(kMaxGetExprDepth));
    }
  }

  ~get_expr_depth_guard()
  {
    --converter.get_expr_depth_;
  }

private:
  python_converter &converter;
};

// True when `method_name` is a method decorated with @property in `class_name`
// or one of its (transitive) base classes. Reading `obj.prop` then invokes the
// getter. A same-named non-property method in a derived class shadows a base
// property (Python MRO), so a match that is not @property stops the search.
static bool is_property_method(
  const nlohmann::json &ast_body,
  const std::string &class_name,
  const std::string &method_name)
{
  const nlohmann::json cls = json_utils::find_class(ast_body, class_name);
  if (cls.empty() || !cls.contains("body"))
    return false;

  for (const auto &stmt : cls["body"])
  {
    if (
      stmt.value("_type", std::string()) != "FunctionDef" ||
      stmt.value("name", std::string()) != method_name)
      continue;
    if (stmt.contains("decorator_list"))
      for (const auto &dec : stmt["decorator_list"])
        if (
          dec.value("_type", std::string()) == "Name" &&
          dec.value("id", std::string()) == "property")
          return true;
    return false; // method exists but is not a property: shadows any base
  }

  if (cls.contains("bases"))
    for (const auto &base : cls["bases"])
      if (
        base.contains("id") &&
        is_property_method(
          ast_body, base["id"].template get<std::string>(), method_name))
        return true;

  return false;
}

static ExpressionType get_expression_type(const nlohmann::json &element)
{
  // Return UNKNOWN if the expected "_type" field is missing
  if (!element.contains("_type"))
    return ExpressionType::UNKNOWN;

  // Map of Python AST "_type" strings to internal expression categories
  static const std::unordered_map<std::string, ExpressionType> type_map = {
    {"UnaryOp", ExpressionType::UNARY_OPERATION},
    {"BinOp", ExpressionType::BINARY_OPERATION},
    {"Compare",
     ExpressionType::BINARY_OPERATION}, // Comparison treated as binary op
    {"BoolOp", ExpressionType::LOGICAL_OPERATION},
    {"Constant", ExpressionType::LITERAL},
    {"Name", ExpressionType::VARIABLE_REF},
    {"Attribute",
     ExpressionType::VARIABLE_REF}, // Both treated as variable references
    {"Call", ExpressionType::FUNC_CALL},
    {"IfExp", ExpressionType::IF_EXPR},
    {"Subscript", ExpressionType::SUBSCRIPT},
    {"List", ExpressionType::LIST},
    {"Set", ExpressionType::LIST},
    {"GeneratorExp", ExpressionType::LIST},
    {"Lambda", ExpressionType::FUNC_CALL},
    {"JoinedStr", ExpressionType::FSTRING},
    {"Tuple", ExpressionType::TUPLE},
    {"Slice", ExpressionType::SLICE},
    {"Dict", ExpressionType::LITERAL},
    {"DictComp", ExpressionType::LITERAL}};

  const auto &type = element["_type"];
  auto it = type_map.find(type);
  if (it != type_map.end())
    return it->second;

  // If the type is not recognized, return UNKNOWN
  return ExpressionType::UNKNOWN;
}

exprt python_converter::make_char_array_expr(
  const std::vector<unsigned char> &string_literal,
  const typet &t)
{
  exprt expr = gen_zero(t);
  const typet &char_type = t.subtype();

  for (size_t i = 0; i < string_literal.size(); ++i)
  {
    uint8_t ch = string_literal[i];
    exprt char_value = constant_exprt(
      integer2binary(BigInt(ch), bv_width(char_type)),
      integer2string(BigInt(ch)),
      char_type);
    expr.operands().at(i) = char_value;
  }

  return expr;
}
/// Convert Python AST literal to expression.
/// Handles integers, booleans, floats, chars, strings, and byte literals.
/// Example: {"_type": "Constant", "value": 42} -> integer constant expr
exprt python_converter::get_literal(const nlohmann::json &element)
{
  const auto &annotated_node =
    (element["_type"] == "UnaryOp") ? element["operand"] : element;

  // Handle Python complex constants emitted by parser annotations.
  // This must run before generic string handling because complex constants
  // may carry a string-like "value" in the serialized AST.
  if (
    annotated_node.contains("esbmc_type_annotation") &&
    annotated_node["esbmc_type_annotation"] == "complex")
  {
    double real = annotated_node.value("real_value", 0.0);
    double imag = annotated_node.value("imag_value", 0.0);

    // UnaryOp(USub, Constant(complex)) must preserve the sign.
    if (
      element.contains("_type") && element["_type"] == "UnaryOp" &&
      element.contains("op") && element["op"].contains("_type"))
    {
      const std::string op_type = element["op"]["_type"].get<std::string>();
      if (op_type == "USub")
      {
        real = -real;
        imag = -imag;
      }
    }

    return make_complex(
      from_double(real, double_type()), from_double(imag, double_type()));
  }

  // Determine the source of the literal's value.
  const auto &value = (element["_type"] == "UnaryOp")
                        ? element["operand"]["value"]
                        : element["value"];

  // Bignum literal tagged by parser.py (issue #4642). Checked before the null
  // branch below because parser.py replaces tagged literals' `value` with null
  // so direct readers (consteval, funcall pre-scan, f-strings) bail via their
  // `is_number_integer()` guard instead of silently folding bignum to 0. Until
  // arbitrary-precision int lands in the irep2 type system, reject the literal
  // explicitly rather than letting nlohmann::json silently truncate values
  // above uint64 to double or letting from_integer wrap values in [2^63, 2^64)
  // into negative int64. UnaryOp(USub, Constant(_bigint=...)) is dispatched
  // through convert_unop → get_expr(operand) → get_literal(operand), so the
  // trap fires from the recursive Constant entry and needs no special case.
  if (element.contains("_bigint"))
  {
    const std::string &digits = element["_bigint"].get<std::string>();
    throw python_int_overflow_excp(
      "Python int overflow: literal " + digits +
      " does not fit in 64-bit int. ESBMC approximates Python int as a "
      "fixed-width bitvector; arbitrary-precision int support is tracked in "
      "issue #4642.");
  }

  // Handle None literals (null values)
  if (value.is_null())
  {
    // Create a null pointer expression to represent NoneType
    constant_exprt null_expr(none_type());
    null_expr.set_value("NULL");
    return null_expr;
  }

  // Handle integer literals (int)
  if (value.is_number_integer())
    return from_integer(value.get<long long>(), long_long_int_type());

  // Handle boolean literals (True/False)
  // V.3: build the bool constant in IREP2, back-migrated for the legacy seam.
  if (value.is_boolean())
    return migrate_expr_back(
      value.get<bool>() ? gen_true_expr() : gen_false_expr());

  // Handle floating-point literals (float)
  if (value.is_number_float())
  {
    exprt expr;
    convert_float_literal(
      value.dump(), expr); // `value.dump()` converts it to string
    return expr;
  }

  if (!value.is_string())
    return exprt(); // Not a string, no handling

  const std::string &str_val = value.get<std::string>();

  // Handle string or byte literals
  typet t = current_element_type;
  std::vector<uint8_t> string_literal;

  if (is_bytes_literal(element))
  {
    std::vector<uint8_t> bytes;
    if (element.contains("encoded_bytes"))
      bytes = base64_decode(element["encoded_bytes"].get<std::string>());
    else
      bytes.assign(str_val.begin(), str_val.end());

    return string_builder_->build_raw_byte_array(bytes);
  }
  else
  {
    // Strings are null-terminated
    return string_builder_->build_string_literal(str_val);
  }

  return make_char_array_expr(string_literal, t);
}

// Detect bytes literals
bool python_converter::is_bytes_literal(const nlohmann::json &element)
{
  // Check if element has encoded_bytes field (explicit bytes)
  if (element.contains("encoded_bytes"))
    return true;

  // Check if element has bytes type annotation
  if (
    element.contains("annotation") && element["annotation"].contains("id") &&
    element["annotation"]["id"] == "bytes")
    return true;

  // Check if element has a parent context indicating bytes
  if (element.contains("kind") && element["kind"] == "bytes")
    return true;

  // Check if this is part of a bytes assignment/initialization
  if (current_element_type.id() == "bytes")
    return true;

  // Check if this is an array of uint8 (bytes representation)
  if (current_element_type.id() == "array")
  {
    const typet &subtype = current_element_type.subtype();
    if (subtype.id() == "unsignedbv")
    {
      // Convert irep_idt width to integer
      const irep_idt &width_str = subtype.width();
      try
      {
        int width = std::stoi(width_str.as_string());
        if (width == 8)
          return true;
      }
      catch (const std::exception &)
      {
        // If conversion fails, continue with other checks
      }
    }
  }

  return false;
}

exprt python_converter::get_lambda_expr(const nlohmann::json &element)
{
  return lambda_handler_->get_lambda_expr(element);
}

exprt python_converter::get_named_expr(const nlohmann::json &element)
{
  // PEP 572 walrus `(target := value)`: assign value to target, then evaluate
  // to the assigned value. Lower to a plain assignment emitted into the current
  // block and read the target back, so the existing assignment machinery
  // (symbol creation, type inference, flow tracking) is reused unchanged.
  const nlohmann::json &target = element["target"];

  // No block to emit into (e.g. a type-inference pre-pass): evaluate the value
  // without binding rather than dropping a half-formed declaration.
  if (!current_block)
    return get_expr(element["value"]);

  nlohmann::json assign = nlohmann::json::object();
  assign["_type"] = "Assign";
  assign["targets"] = nlohmann::json::array({target});
  assign["value"] = element["value"];
  copy_location_fields_from_decl(element, assign);

  // get_var_assign mutates converter state (current_lhs, is_converting_rhs,
  // is_converting_lhs, current_element_type); save/restore so the enclosing
  // expression conversion is unaffected. flow_class_map_ is intentionally
  // persistent (flow tracking) and left as get_var_assign updates it.
  exprt *saved_lhs = current_lhs;
  bool saved_rhs = is_converting_rhs;
  bool saved_lhs_flag = is_converting_lhs;
  typet saved_elem_type = current_element_type;
  get_var_assign(assign, *current_block);
  current_lhs = saved_lhs;
  is_converting_rhs = saved_rhs;
  is_converting_lhs = saved_lhs_flag;
  current_element_type = saved_elem_type;

  return get_expr(target);
}

exprt python_converter::get_expr(const nlohmann::json &element)
{
  get_expr_depth_guard depth_guard(*this);

  // Walrus operator `(target := value)` — assign as a side effect, evaluate to
  // the bound value. Handled before the type switch since NamedExpr is not in
  // the expression-type map.
  if (element.is_object() && element.value("_type", "") == "NamedExpr")
    return get_named_expr(element);

  // A walrus in a ternary branch (`a if c else b`) is evaluated conditionally,
  // but get_named_expr binds unconditionally. Refuse with a clean diagnostic
  // here (before type inference) rather than return an unsound verdict. The
  // ternary test is always evaluated, so a walrus there stays supported.
  if (
    element.is_object() && element.value("_type", "") == "IfExp" &&
    ((element.contains("body") && contains_named_expr(element["body"])) ||
     (element.contains("orelse") && contains_named_expr(element["orelse"]))))
    throw std::runtime_error(
      "Walrus operator ':=' in a conditional expression is not supported");

  exprt expr;

  ExpressionType type = get_expression_type(element);

  switch (type)
  {
  case ExpressionType::UNARY_OPERATION:
  {
    expr = get_unary_operator_expr(element);
    break;
  }
  case ExpressionType::BINARY_OPERATION:
  {
    expr = get_binary_operator_expr(element);
    break;
  }
  case ExpressionType::LOGICAL_OPERATION:
  {
    expr = get_logical_operator_expr(element);
    break;
  }
  case ExpressionType::LITERAL:
  {
    if (dict_handler_->is_dict_literal(element))
    {
      expr = dict_handler_->get_dict_literal(element);
      break;
    }

    if (element["_type"] == "DictComp")
    {
      expr = dict_handler_->get_dict_comprehension(element);
      break;
    }

    expr = get_literal(element);
    break;
  }
  case ExpressionType::LIST:
  {
    // For now, treat set literals such as lists
    // Store elements in order they appear (order doesn't matter for sets)
    if (element["_type"] == "Set")
    {
      python_set set_handler(*this, element);
      expr = set_handler.get();
      break;
    }

    // Handle generator expressions
    if (element["_type"] == "GeneratorExp")
    {
      python_list list(*this, element);
      expr = list.handle_comprehension(element);
      break;
    }

    // Check if we should use static arrays (for numpy and similar operations)
    if (build_static_lists)
    {
      typet size = type_handler_.get_typet(element["elts"]);
      expr = get_static_array(element, size);
      break;
    }

    // List handling (dynamic lists)
    python_list list(*this, element);
    expr = list.get();
    break;
  }
  case ExpressionType::VARIABLE_REF:
  {
    std::string var_name;
    bool is_class_attr = false;
    if (element["_type"] == "Name")
    {
      var_name = element["id"].get<std::string>();
      // Handle type identifiers (int, str, float, bool, etc.)
      if (type_utils::is_type_identifier(var_name))
      {
        // Create a string constant containing the type name
        std::string type_name = var_name;
        typet str_type =
          type_handler_.build_array(char_type(), type_name.size() + 1);
        constant_exprt type_str(type_name, type_name, str_type);
        expr = type_str;
        break;
      }
    }
    else if (element["_type"] == "Attribute")
    {
      // Resolve `<base>.<attr>` after unwrapping Optional[T] / pointer-to-struct
      // / complex types. Returns nil if the attribute cannot be resolved.
      auto resolve_member_on_base =
        [this](exprt base_expr, const std::string &attr_name) -> exprt {
        typet base_type = base_expr.type();
        if (base_type.is_pointer())
          base_type = base_type.subtype();
        if (base_type.id() == "symbol")
          base_type = ns.follow(base_type);

        // Unwrap Optional[T] before attribute access.
        if (base_type.is_struct())
        {
          const struct_typet &opt_st = to_struct_type(base_type);
          const std::string &tag = opt_st.tag().as_string();
          if (
            tag.rfind("tag-Optional_", 0) == 0 &&
            opt_st.has_component("value") && !opt_st.has_component(attr_name))
          {
            const typet &inner_raw = opt_st.get_component("value").type();
            // V.3: build the (optional) dereference of the base directly in
            // IREP2 instead of staging a legacy "dereference" node and
            // re-migrating it; byte-identical (migrate of the legacy node is
            // exactly dereference2tc(migrate_type(subtype), migrate(base))).
            expr2tc ob2;
            migrate_expr(base_expr, ob2);
            if (base_expr.type().is_pointer())
              ob2 =
                dereference2tc(migrate_type(base_expr.type().subtype()), ob2);
            base_expr = migrate_expr_back(
              member2tc(migrate_type(inner_raw), ob2, "value"));
            base_type = inner_raw;
            if (base_type.is_pointer())
              base_type = base_type.subtype();
            if (base_type.id() == "symbol")
              base_type = ns.follow(base_type);
          }
        }

        // Unwrap pointer-to-struct so the struct component lookup succeeds.
        if (base_type.is_pointer())
        {
          typet pointed_to = base_type.subtype();
          if (pointed_to.id() == "symbol")
            pointed_to = ns.follow(pointed_to);
          if (pointed_to.is_struct())
            base_type = pointed_to;
        }

        // Delegate complex attribute access (.real, .imag) to the handler.
        if (is_complex_type(base_type))
        {
          exprt result =
            complex_handler_.handle_attribute_access(base_expr, attr_name);
          if (!result.is_nil())
            return result;
        }

        // NumPy baseline support: expose `.shape` for modelled arrays/lists.
        // - C arrays: shape is extracted from nested array dimensions.
        // - ESBMC runtime list model: shape is a 1D tuple (len(list),).
        if (attr_name == "shape")
        {
          if (base_type.is_array())
          {
            std::vector<int> dims =
              type_handler_.get_array_type_shape(base_type);
            std::vector<exprt> dim_exprs;
            dim_exprs.reserve(dims.size());
            for (int dim : dims)
              dim_exprs.push_back(from_integer(dim, int_type()));
            return build_shape_tuple_expr(*this, dim_exprs);
          }

          const typet list_type = type_handler_.get_list_type();
          if (
            base_type == list_type || (base_expr.type().is_pointer() &&
                                       base_expr.type().subtype() == list_type))
          {
            const symbolt *size_func =
              symbol_table_.find_symbol("c:@F@__ESBMC_list_size");
            if (!size_func)
              throw std::runtime_error(
                "__ESBMC_list_size not found for list shape access");

            // (int)__ESBMC_list_size(&base_expr), built in IREP2 (V.3).
            expr2tc base2;
            migrate_expr(base_expr, base2);
            if (!is_pointer_type(base2->type))
              base2 = address_of2tc(base2->type, base2);
            expr2tc size_call = side_effect_function_call2tc(
              migrate_type(size_type()), symbol_expr2tc(*size_func), {base2});
            exprt list_len = migrate_expr_back(
              typecast2tc(migrate_type(int_type()), size_call));
            return build_shape_tuple_expr(*this, {list_len});
          }
        }

        if (base_type.is_struct())
        {
          const struct_typet &struct_type = to_struct_type(base_type);
          if (struct_type.has_component(attr_name))
          {
            const typet &attr_type =
              struct_type.get_component(attr_name).type();
            typet clean_type = clean_attribute_type(attr_type);
            // V.1k step-2 hypothesis: build member2t with the (possibly
            // symbol-typed) source permitted by the step-1 relaxation, then
            // back-migrate to the legacy body so the EXISTING adjust + goto
            // -convert resolves it as today. No converter-side ns.follow.
            // V.3: the (optional) base dereference is built directly in IREP2
            // rather than staged as a legacy "dereference" node and re-migrated
            // (byte-identical to migrate of that node).
            expr2tc src2;
            migrate_expr(base_expr, src2);
            if (base_expr.type().is_pointer())
              src2 =
                dereference2tc(migrate_type(base_expr.type().subtype()), src2);
            return migrate_expr_back(
              member2tc(migrate_type(clean_type), src2, attr_name));
          }
        }

        return nil_exprt();
      };

      // Handle nested attribute chain (e.g., self.b.a)
      if (element["value"]["_type"] == "Attribute")
      {
        exprt base_expr = get_expr(element["value"]);
        const std::string &attr_name = element["attr"].get<std::string>();

        typet base_type = base_expr.type();
        if (base_type.is_pointer())
          base_type = base_type.subtype();
        if (base_type.id() == "symbol")
          base_type = ns.follow(base_type);

        // Handle enum member attribute access before the general struct path.
        // e.g., TrafficLight.RED.value  -> the integer value of the member
        // e.g., TrafficLight.RED.name   -> a constant char-array string literal
        //
        // We handle this first so that .name always returns a constant array
        // (enabling reliable constant-folding in string comparisons) even after
        // enum members have been re-typed as their enclosing struct.
        if (
          element["value"].is_object() && element["value"].contains("value") &&
          element["value"]["value"].is_object() &&
          element["value"]["value"].contains("_type") &&
          element["value"]["value"]["_type"] == "Name" &&
          element["value"].contains("attr"))
        {
          const std::string class_name =
            element["value"]["value"]["id"].get<std::string>();
          const std::string member_name =
            element["value"]["attr"].get<std::string>();

          if (python_frontend::is_enum_class(class_name, *ast_json))
          {
            if (attr_name == "value")
            {
              // If base_expr is already the enum struct, extract the value
              // component; otherwise base_expr is the raw int value itself.
              if (base_type.is_struct())
              {
                const struct_typet &st = to_struct_type(base_type);
                if (st.has_component("value"))
                {
                  const typet &vt =
                    clean_attribute_type(st.get_component("value").type());
                  expr2tc bv2;
                  migrate_expr(base_expr, bv2);
                  expr = migrate_expr_back(
                    member2tc(migrate_type(vt), bv2, "value"));
                  break;
                }
              }
              expr = base_expr;
              break;
            }
            if (attr_name == "name")
            {
              // Return a constant char-array so that string comparisons can be
              // resolved at compile time via compare_constants_internal.
              expr = string_builder_->build_string_literal(member_name);
              break;
            }
          }
        }

        exprt resolved = resolve_member_on_base(base_expr, attr_name);

        // Flow-sensitive class tracking (#4771/#4772): the usage-site scanner
        // left this attribute as any_type() (void*) because it was assigned
        // values of different classes; resolve_member_on_base can't find the
        // field on a void* base. If the base lvalue was last assigned a known
        // class at an unconditional top-level point, cast the base to that
        // class's struct and retry, so last-write-wins layout is used here.
        if (resolved.is_nil())
        {
          const std::string bp = flow_lvalue_path(element["value"]);
          auto it =
            bp.empty() ? flow_class_map_.end() : flow_class_map_.find(bp);
          if (it != flow_class_map_.end())
          {
            // (tag-Cls*)base_expr, built in IREP2 (V.3).
            const typet cast_t =
              gen_pointer_type(symbol_typet("tag-" + it->second));
            expr2tc base2;
            migrate_expr(base_expr, base2);
            exprt cast =
              migrate_expr_back(typecast2tc(migrate_type(cast_t), base2));
            cast.type() = cast_t; // restore #cpp_type that migrate_type drops
            resolved = resolve_member_on_base(cast, attr_name);
          }
        }

        if (!resolved.is_nil())
        {
          expr = resolved;
          break;
        }

        log_error("Cannot resolve nested attribute: {}", attr_name);
        abort();
      }
      else if (element["value"]["_type"] == "Name")
      {
        var_name = element["value"]["id"].get<std::string>();
      }
      else if (element["value"]["_type"] == "Subscript")
      {
        // Attribute access on a subscript result, e.g. `d[key].attr`.
        exprt base_expr = get_expr(element["value"]);
        const std::string &attr_name = element["attr"].get<std::string>();

        exprt resolved = resolve_member_on_base(base_expr, attr_name);
        if (!resolved.is_nil())
        {
          expr = resolved;
          break;
        }

        log_error(
          "Cannot resolve attribute '{}' on subscript result", attr_name);
        abort();
      }
      else
      {
        log_error(
          "Unsupported Attribute value type: {}",
          element["value"]["_type"].get<std::string>());
        abort();
      }

      // Handle module attribute access (e.g., math.inf) — unless the module
      // name is shadowed by a local binding in the current scope (e.g. a
      // parameter named `node` when `from node import Node` is in scope).
      // Python's scoping rules give precedence to the local binding.
      auto name_resolves_to_symbol = [&](const std::string &name) {
        symbol_id sid = create_symbol_id();
        sid.set_object(name);
        if (find_symbol(sid.to_string()))
          return true;

        // Function/method bodies can access module-level names via fallback.
        sid.set_function("");
        if (find_symbol(sid.to_string()))
          return true;

        // Also probe the root-global form used by some frontend-generated IDs.
        if (find_symbol(sid.global_to_string()))
          return true;

        return false;
      };

      if (is_imported_module(var_name) && !name_resolves_to_symbol(var_name))
      {
        std::string attr_name = element["attr"].get<std::string>();
        std::string module_path = imported_modules[var_name];

        // Construct symbol ID for module member: py:module_path@member_name
        symbol_id module_sid(module_path, "", "");
        module_sid.set_object(attr_name);

        symbolt *symbol = find_symbol(module_sid.to_string());
        if (!symbol)
        {
          log_error(
            "Module member '{}' not found in module '{}'", attr_name, var_name);
          abort();
        }

        expr = symbol_expr(*symbol);
        break;
      }

      if (is_class(var_name, *ast_json))
      {
        // Found a class attribute
        var_name = "C@" + var_name;
        is_class_attr = true;
      }
    }

    assert(!var_name.empty());

    symbol_id sid = create_symbol_id();
    sid.set_object(var_name);

    if (element.contains("attr") && is_class_attr)
    {
      sid.set_attribute(element["attr"].get<std::string>());
      sid.set_function("");
    }

    std::string sid_str = sid.to_string();

    symbolt *symbol = nullptr;
    if (!(symbol = find_symbol(sid_str)))
    {
      // Fallback for global variables accessed inside functions or class methods
      if (!is_class_attr && element["_type"] == "Name")
      {
        sid.set_function(""); // remove function scope
        sid_str = sid.to_string();
        symbol = find_symbol(sid_str);
        if (!symbol)
        {
          // also try module-level global (strips class scope too)
          symbol = find_symbol(sid.global_to_string());
        }
      }
      if (!symbol)
      {
        // Check if this Name refers to a function
        if (!is_class_attr && element["_type"] == "Name")
        {
          if (
            symbolt *nested_func_symbol = find_nested_function_symbol(var_name))
          {
            expr = symbol_expr(*nested_func_symbol);
            break;
          }

          symbol_id func_sid(current_python_file, "", var_name);
          symbolt *func_symbol =
            symbol_table_.find_symbol(func_sid.to_string());
          if (func_symbol && func_symbol->get_type().is_code())
          {
            expr = symbol_expr(*func_symbol);
            break;
          }

          // A bare class name used as a value, e.g. `register(SomeClass)` or
          // `create_publisher(topic, Twist)` -- passing the class object itself
          // as an argument. Python classes are first-class objects, but ESBMC
          // has no first-class type value, so model it as an opaque nondet
          // placeholder. Inert uses (storing or forwarding the class) then
          // convert instead of aborting; constructing through such a forwarded
          // value is not modelled.
          if (is_class(var_name, *ast_json))
          {
            expr = side_effect_expr_nondett(any_type());
            expr.location() = get_location_from_decl(element);
            break;
          }
        }
        locationt location = get_location_from_decl(element);
        std::ostringstream error_msg;
        if (!current_func_name_.empty())
        {
          // Variable referenced inside a function
          error_msg << "Variable '" << var_name
                    << "' is not defined in function '" << current_func_name_
                    << "'";
          if (!location.get_line().empty())
            error_msg << " at line " << location.get_line();
          error_msg << ".";
        }
        else
        {
          // Variable referenced at global scope
          error_msg << "Variable '" << var_name << "' is not defined";
          if (!location.get_line().empty())
            error_msg << " at line " << location.get_line();
          error_msg << ".";
        }
        log_error("{}", error_msg.str());
        abort();
      }
    }

    // Straight-line dynamic retyping (#4770, #4774): if this variable was
    // reassigned across the numeric<->string boundary, get_var_assign minted a
    // fresh symbol of the new type and recorded it here. Redirect the load to
    // that symbol so it observes the variable's current type and value.
    if (!is_class_attr)
    {
      auto alias = retype_aliases_.find(symbol->id.as_string());
      if (alias != retype_aliases_.end())
      {
        if (symbolt *retyped = symbol_table_.find_symbol(alias->second))
          symbol = retyped;
      }
    }

    expr = symbol_expr(*symbol);

    // If the looked-up symbol is an enum class attribute with int type,
    // wrap it in the proper enum struct expression so callers that expect
    // the enum class type (e.g. function parameters) receive a struct value.
    if (
      is_class_attr && symbol->get_type().is_signedbv() &&
      element.contains("attr"))
    {
      std::string cn = var_name;
      if (cn.starts_with("C@"))
        cn = cn.substr(2);
      if (python_frontend::is_enum_class(cn, *ast_json))
      {
        const std::string mname = element["attr"].get<std::string>();
        expr = make_enum_member_struct_expr(*symbol, cn, mname);
      }
    }

    // Get instance attribute
    if (!is_class_attr && element["_type"] == "Attribute")
    {
      const std::string &attr_name = element["attr"].get<std::string>();

      if (attr_name == "shape")
      {
        typet sym_type = symbol->get_type();
        if (sym_type.is_pointer())
          sym_type = sym_type.subtype();
        if (sym_type.id() == "symbol")
          sym_type = ns.follow(sym_type);

        if (sym_type.is_array())
        {
          std::vector<int> dims = type_handler_.get_array_type_shape(sym_type);
          std::vector<exprt> dim_exprs;
          dim_exprs.reserve(dims.size());
          for (int dim : dims)
            dim_exprs.push_back(from_integer(dim, int_type()));
          expr = build_shape_tuple_expr(*this, dim_exprs);
          break;
        }

        const typet list_type = type_handler_.get_list_type();
        if (
          sym_type == list_type || (symbol->get_type().is_pointer() &&
                                    symbol->get_type().subtype() == list_type))
        {
          const symbolt *size_func =
            symbol_table_.find_symbol("c:@F@__ESBMC_list_size");
          if (!size_func)
            throw std::runtime_error(
              "__ESBMC_list_size not found for list shape access");

          // (int)__ESBMC_list_size(&expr), built in IREP2 (V.3).
          expr2tc base2;
          migrate_expr(expr, base2);
          if (!is_pointer_type(base2->type))
            base2 = address_of2tc(base2->type, base2);
          expr2tc size_call = side_effect_function_call2tc(
            migrate_type(size_type()), symbol_expr2tc(*size_func), {base2});
          exprt list_len =
            migrate_expr_back(typecast2tc(migrate_type(int_type()), size_call));
          expr = build_shape_tuple_expr(*this, {list_len});
          break;
        }
      }

      // Delegate complex attribute access (.real, .imag) to the handler.
      if (is_complex_type(symbol->get_type()))
      {
        exprt result =
          complex_handler_.handle_attribute_access(expr, attr_name);
        if (!result.is_nil())
        {
          expr = result;
          break;
        }
      }

      // Numeric-tower properties on int/float (attribute access returning a
      // known constant). A real number is its own real part with a zero
      // imaginary part; CPython additionally exposes int as the ratio n/1 via
      // numerator/denominator. float has no numerator/denominator (it is not a
      // Rational), so those fall through to the AttributeError below.
      {
        const typet &num_t = symbol->get_type();
        const bool is_int = num_t.is_signedbv() || num_t.is_unsignedbv();
        const bool is_float = num_t.is_floatbv();
        if (is_int || is_float)
        {
          if (attr_name == "real" || (is_int && attr_name == "numerator"))
            break; // value is unchanged — expr already holds it
          if (attr_name == "imag")
          {
            expr = is_float ? from_double(0.0, num_t) : from_integer(0, num_t);
            break;
          }
          if (is_int && attr_name == "denominator")
          {
            expr = from_integer(1, num_t);
            break;
          }
        }
      }

      // Get object type name from symbol. e.g.: tag-MyClass
      std::string obj_type_name;
      const typet &symbol_type = (symbol->get_type().is_pointer())
                                   ? symbol->get_type().subtype()
                                   : symbol->get_type();

      // Handle union types
      if (symbol_type.is_array() && symbol_type.subtype() == char_type())
      {
        // For union types, we need to infer which concrete type to use.
        // Strategy: Look for isinstance checks in the current scope to determine
        // the expected type, or search for classes that have this attribute.

        symbolt *target_class_symbol = nullptr;

        // Search all class types in the symbol table to find one that has this attribute
        symbol_table_.foreach_operand_in_order([&](const symbolt &s) {
          if (target_class_symbol)
            return; // Already found

          if (s.id.as_string().find("tag-") == 0 && s.get_type().is_struct())
          {
            const struct_typet &struct_type = to_struct_type(s.get_type());
            if (struct_type.has_component(attr_name))
              target_class_symbol = const_cast<symbolt *>(&s);
          }
        });

        if (!target_class_symbol)
        {
          throw std::runtime_error(
            "Cannot access attribute '" + attr_name +
            "' on union type: no class with this attribute found");
        }

        // V.3: build the char* -> target_class* cast, the dereference, and
        // the member access entirely in IREP2, back-migrated once at the
        // legacy seam. Keeping the typecast and dereference as expr2tc avoids
        // the back-migrate/re-migrate round-trip the staged legacy form paid.
        typet target_ptr_type =
          gen_pointer_type(target_class_symbol->get_type());
        expr2tc expr2;
        migrate_expr(expr, expr2);
        expr2tc casted2 = typecast2tc(migrate_type(target_ptr_type), expr2);

        // Dereference to get the object
        expr2tc deref2 = dereference2tc(
          migrate_type(target_class_symbol->get_type()), casted2);

        // Access the member on the object
        const struct_typet &target_struct =
          to_struct_type(target_class_symbol->get_type());
        const typet &attr_type = target_struct.get_component(attr_name).type();
        typet clean_type = clean_attribute_type(attr_type);

        expr = migrate_expr_back(
          member2tc(migrate_type(clean_type), deref2, attr_name));
        break;
      }

      if (symbol_type.id() == "struct")
      {
        // Struct types store class name in "tag" field
        const struct_typet &struct_type = to_struct_type(symbol_type);
        obj_type_name = "tag-" + struct_type.tag().as_string();
      }
      else
      {
        // Search named_sub for identifier
        for (const auto &it : symbol_type.get_named_sub())
        {
          if (it.first == "identifier")
            obj_type_name = it.second.id_string();
        }
      }

      // Get class definition from symbols table.
      symbolt *class_symbol = obj_type_name.empty()
                                ? nullptr
                                : symbol_table_.find_symbol(obj_type_name);
      if (!class_symbol)
      {
        std::string fallback_class_id;
        symbol_table_.foreach_operand_in_order([&](const symbolt &s) {
          if (!fallback_class_id.empty())
            return;
          if (s.id.as_string().find("tag-") == 0 && s.get_type().is_struct())
          {
            const struct_typet &st = to_struct_type(s.get_type());
            if (st.has_component(attr_name))
              fallback_class_id = s.id.as_string();
          }
        });
        if (!fallback_class_id.empty())
          class_symbol = symbol_table_.find_symbol(fallback_class_id);
      }
      if (!class_symbol)
      {
        // Surface a Python-level AttributeError naming the attribute and the
        // location of the access. The previous "Class '' not found" message
        // leaked an internal lookup vocabulary and was emitted whenever an
        // attribute was read on a value whose symbol type is not a class
        // struct -- e.g. ``a.shape`` on a numpy array (which the frontend
        // models as a plain list), or any other non-class object.
        const std::string base_name =
          element.contains("value") && element["value"].contains("id")
            ? element["value"]["id"].get<std::string>()
            : std::string();
        std::ostringstream msg;
        msg << "AttributeError: '";
        if (!base_name.empty())
          msg << base_name;
        else
          msg << "object";
        msg << "' has no attribute '" << attr_name << "'";
        const locationt loc = get_location_from_decl(element);
        if (!loc.is_nil())
          msg << " at " << loc.get_file() << ":" << loc.get_line();
        throw std::runtime_error(msg.str());
      }

      // Read-modify-set: we may push_back a new component into the class
      // type, so own it locally and set it back at the end.
      typet class_symbol_type = class_symbol->get_type();
      struct_typet &class_type = static_cast<struct_typet &>(class_symbol_type);
      auto build_member_expr_from_class = [&](const typet &attr_type) -> exprt {
        typet clean_type = clean_attribute_type(attr_type);
        exprt base = symbol_expr(*symbol);
        typet base_type = base.type();
        if (base_type.id() == "symbol")
          base_type = ns.follow(base_type);

        bool points_to_struct = false;
        if (base_type.is_pointer())
        {
          typet pointee = base_type.subtype();
          if (pointee.id() == "symbol")
            pointee = ns.follow(pointee);
          points_to_struct = pointee.is_struct() || pointee.is_union();
        }

        if (!(base_type.is_struct() || base_type.is_union() ||
              points_to_struct))
        {
          // (class_type*)base, built in IREP2 (V.3).
          const typet bt = gen_pointer_type(class_type);
          expr2tc base2;
          migrate_expr(base, base2);
          base = migrate_expr_back(typecast2tc(migrate_type(bt), base2));
          base.type() = bt; // restore #cpp_type that migrate_type drops
        }

        // V.3: dereference (when base is a pointer) and member access built in
        // IREP2; exact round-trip of the legacy dereference + member_exprt.
        expr2tc b2;
        migrate_expr(base, b2);
        if (base.type().is_pointer())
          b2 = dereference2tc(migrate_type(base.type().subtype()), b2);
        return migrate_expr_back(
          member2tc(migrate_type(clean_type), b2, attr_name));
      };

      if (is_converting_lhs)
      {
        // Add member in the class if not exists
        if (!class_type.has_component(attr_name))
        {
          struct_typet::componentt comp = python_frontend::build_component(
            class_type.tag().as_string(), attr_name, current_element_type);
          class_type.components().push_back(comp);
          // Persist the mutation back to the symbol (read-modify-set).
          class_symbol->set_type(class_symbol_type);
        }

        // Register instance attribute for both regular and normalized keys
        register_instance_attribute(
          symbol->id.as_string(),
          attr_name,
          var_name,
          class_type.tag().as_string());
      }

      // Check if this specific instance has explicitly set this attribute
      bool instance_has_attr = is_instance_attribute(
        symbol->id.as_string(),
        attr_name,
        var_name,
        class_type.tag().as_string());

      // For LHS (writing): always use instance member and register it
      if (is_converting_lhs && class_type.has_component(attr_name))
      {
        const typet &attr_type = class_type.get_component(attr_name).type();
        expr = build_member_expr_from_class(attr_type);

        // Register as instance attribute
        register_instance_attribute(
          symbol->id.as_string(),
          attr_name,
          var_name,
          class_type.tag().as_string());
      }
      // For RHS (reading): use instance member if explicitly set OR if symbol is a parameter
      // This allows parameter objects like 'f: Foo' to access instance attributes
      else if (
        !is_converting_lhs && class_type.has_component(attr_name) &&
        (instance_has_attr || symbol->is_parameter ||
         is_complex_type(class_type)))
      {
        const typet &attr_type = class_type.get_component(attr_name).type();
        expr = build_member_expr_from_class(attr_type);
      }
      // Otherwise use class attribute
      else
      {
        sid.set_function("");
        sid.set_class(extract_class_name_from_tag(obj_type_name));
        sid.set_object(attr_name);
        symbolt *class_attr_symbol = symbol_table_.find_symbol(sid.to_string());

        if (!class_attr_symbol)
        {
          // Not found in the direct class — walk base classes (Python MRO).
          const std::string derived_name =
            extract_class_name_from_tag(obj_type_name);
          auto class_node =
            json_utils::find_class((*ast_json)["body"], derived_name);
          if (!class_node.empty() && class_node.contains("bases"))
          {
            for (const auto &base_node : class_node["bases"])
            {
              if (!base_node.contains("id"))
                continue;
              symbol_id base_sid = create_symbol_id();
              base_sid.set_function("");
              base_sid.set_class(base_node["id"].get<std::string>());
              base_sid.set_object(attr_name);
              class_attr_symbol =
                symbol_table_.find_symbol(base_sid.to_string());
              if (class_attr_symbol)
                break;
            }
          }
        }

        if (!class_attr_symbol)
        {
          // No class-level symbol: attribute was set per-instance (e.g. in
          // __init__).  This happens when the object comes from a list element
          // or other expression that bypasses instance_has_attr registration.
          // Fall back to the struct member if the component exists.
          if (class_type.has_component(attr_name))
          {
            const typet &attr_type = class_type.get_component(attr_name).type();
            expr = build_member_expr_from_class(attr_type);
          }
          else if (is_property_method(
                     (*ast_json)["body"],
                     extract_class_name_from_tag(obj_type_name),
                     attr_name))
          {
            // Reading a @property: invoke its getter. Rewrite `obj.attr` to a
            // call `obj.attr()` and convert that, reusing the method-call
            // machinery (the @property decorator is otherwise ignored, so the
            // getter is a plain self-method returning the value).
            nlohmann::json call_node;
            call_node["_type"] = "Call";
            call_node["func"] = element;
            call_node["args"] = nlohmann::json::array();
            call_node["keywords"] = nlohmann::json::array();
            copy_location_fields_from_decl(element, call_node);
            expr = get_expr(call_node);
          }
          else
          {
            throw std::runtime_error(
              "Attribute \"" + attr_name + "\" not found");
          }
        }
        else
          expr = symbol_expr(*class_attr_symbol);
      }
    }

    // Tracks global reads within a function
    if (
      element["_type"] == "Name" &&
      sid.to_string().find("@C") == std::string::npos &&
      sid.to_string().find("@F") != std::string::npos && is_right &&
      !symbol_table_.find_symbol(sid.to_string().c_str()))
    {
      local_loads.push_back(sid.to_string());
    }
    break;
  }
  case ExpressionType::FUNC_CALL:
  {
    // Check if this is a lambda expression
    if (element["_type"] == "Lambda")
      expr = get_lambda_expr(element);
    else
      expr = get_function_call(element);
    break;
  }
  // Ternary operator
  case ExpressionType::IF_EXPR:
  {
    expr = get_conditional_stm(element);
    break;
  }
  case ExpressionType::SUBSCRIPT:
  {
    exprt array = get_expr(element["value"]);

    // If evaluating the base raised (e.g. bin()/hex()/oct() on a non-constant
    // integer returns a cpp-throw side effect), propagate the exception rather
    // than attempting to index it. Slicing a thrown exception is meaningless
    // and would build an address_of over a non-array operand, crashing the
    // converter.
    if (array.is_nil() || contains_cpp_throw(array))
    {
      expr = array;
      break;
    }

    // An inline list-returning call used directly as a subscript base --
    // e.g. sorted(words)[0] -- comes back as a code_function_callt statement.
    // Indexing it builds __ESBMC_list_size over the call statement, whose
    // operand type is empty, aborting symex ("got empty, expected pointer").
    // Materialise the call into a temporary first, mirroring the assigned
    // path (s = sorted(words); s[0]). See #4807.
    if (current_block)
      array =
        materialize_list_function_call(array, element["value"], *current_block);

    const nlohmann::json &slice = element["slice"];
    typet array_type = ns.follow(array.type());

    // A fully-constant string slice (e.g. "abcdef"[5:0:-1]) folds at
    // conversion time. The runtime string-slice path mishandles a negative
    // step — wrong content and an unconstrained length — whereas consteval's
    // slice computation matches CPython. Gate on the converted str type (a
    // bytes literal decodes to the same Constant JSON as str, and its runtime
    // slice is a separate concern), so only a genuine constant str slice folds
    // and anything non-constant falls through to the existing runtime path.
    if (
      ast_json && slice.is_object() && slice.value("_type", "") == "Slice" &&
      type_utils::is_string_type(array_type))
    {
      python_consteval slice_evaluator(*ast_json);
      if (auto folded = slice_evaluator.try_eval_global_expr(element))
        if (folded->kind == PyConstValue::STRING)
        {
          expr = string_builder_->build_string_literal(folded->string_val);
          break;
        }
    }

    // Unwrap pointer-to-dict so d[key] reaches the dict handler when
    // d is held by pointer (e.g. dict-of-class-value via the symbol table).
    typet array_type_for_dict = array_type;
    bool array_is_dict_pointer = false;
    if (array_type_for_dict.is_pointer())
    {
      typet pointed = ns.follow(array_type_for_dict.subtype());
      if (pointed.is_struct() && dict_handler_->is_dict_type(pointed))
      {
        array_type_for_dict = pointed;
        array_is_dict_pointer = true;
      }
    }
    // Handle tuple subscripting - tuples are structs, not arrays.
    // Unwrap pointer-to-tuple so key[i] reaches the tuple handler when the
    // tuple arrives by pointer (e.g. a tuple-of-slices coerced to a pointer
    // parameter for __getitem__, see GitHub #4539).
    if (array_type.is_pointer())
    {
      typet pointed = ns.follow(array_type.subtype());
      if (pointed.is_struct() && tuple_handler_->is_tuple_type(pointed))
      {
        // V.1k keystone (D): deref of the tuple pointer built in IREP2.
        // build_dereference restores #cpp_type and falls back to legacy for
        // dyn-array pointees, so it reproduces the legacy node exactly.
        array = python_expr::build_dereference(array, pointed);
        array_type = pointed;
      }
    }
    if (tuple_handler_->is_tuple_type(array_type))
    {
      expr = tuple_handler_->handle_tuple_subscript(array, slice, element);
      break;
    }

    // Handle dictionary subscript with type inference from annotations
    if (
      array_type_for_dict.is_struct() &&
      dict_handler_->is_dict_type(array_type_for_dict))
    {
      // Dereference once so the handler operates on the dict struct.
      if (array_is_dict_pointer)
      {
        dereference_exprt deref(array, array_type_for_dict);
        deref.type() = array_type_for_dict;
        array = std::move(deref);
      }

      // Try to resolve the expected return type from the dict's type annotation
      typet expected_type =
        dict_handler_->resolve_expected_type_for_dict_subscript(array);

      // Pass the expected type to the dict handler
      // If empty, the handler will use its default heuristics
      expr = dict_handler_->handle_dict_subscript(array, slice, expected_type);
      break;
    }

    const typet list_type = type_handler_.get_list_type();
    const bool array_is_runtime_list =
      array_type == list_type ||
      (array_type.is_pointer() && ns.follow(array_type.subtype()) == list_type);
    const bool array_is_builtin_array = array_type.is_array();
    const bool tuple_index_targets_list_model =
      array_is_runtime_list || array_is_builtin_array;

    // Multi-dimensional indexing ``a[i, j, k, ...]`` for list/array-backed
    // models: lower to chained single-axis indexing `a[i][j][k]...`, one
    // axis at a time. Mixed slice/index tuples (e.g. `a[1:2, 0]`) are not
    // supported by the chained-list model and are rejected explicitly.
    if (
      tuple_index_targets_list_model && slice.contains("_type") &&
      slice["_type"] == "Tuple")
    {
      if (
        slice.contains("elts") && slice["elts"].is_array() &&
        !slice["elts"].empty())
      {
        std::vector<nlohmann::json> idx_nodes;
        idx_nodes.reserve(slice["elts"].size());
        bool has_slice_dim = false;
        for (const auto &raw_idx : slice["elts"])
        {
          const nlohmann::json idx = normalize_bool_index_node(raw_idx);
          if (idx.contains("_type") && idx["_type"] == "Slice")
            has_slice_dim = true;
          idx_nodes.push_back(idx);
        }
        if (has_slice_dim)
        {
          // 2-D slicing: a[:, j] (column select) and a[i, :] (row select,
          // equivalent to chained a[i][:]). Any other slice/index tuple
          // combination (partial bounds, both dims sliced, 3+ dims, ...)
          // stays unsupported.
          if (
            idx_nodes.size() == 2 && is_full_slice_node(idx_nodes[0]) !=
                                       is_full_slice_node(idx_nodes[1]))
          {
            python_list list(*this, element);
            if (is_full_slice_node(idx_nodes[0]))
              expr = list.build_column_select(array, idx_nodes[1], element);
            else
            {
              exprt current = list.index(array, idx_nodes[0]);
              if (!contains_cpp_throw(current))
                current = list.index(current, idx_nodes[1]);
              expr = current;
            }
            break;
          }

          throw_numpy_multidim_index_error(*this, element);
        }

        python_list list(*this, element);
        exprt current = array;
        for (std::size_t axis = 0; axis < idx_nodes.size(); ++axis)
        {
          if (axis > 0)
          {
            const typet current_type = ns.follow(current.type());
            const bool current_is_list_like = current_type.is_array() ||
                                              current_type.is_pointer() ||
                                              current_type == list_type;
            if (!current_is_list_like)
              throw_numpy_too_many_indices_error(
                *this, element, idx_nodes.size());
          }

          current = list.index(current, idx_nodes[axis]);
          if (contains_cpp_throw(current))
            break;
        }

        expr = current;
        break;
      }

      throw_numpy_multidim_index_error(*this, element);
    }

    // Fancy/integer-array indexing ``a[[0, 2]]``: a literal index list
    // selects elements by position, resolved and bounds-checked at
    // conversion time (see python_list::build_fancy_index).
    if (
      tuple_index_targets_list_model && slice.contains("_type") &&
      slice["_type"] == "List" && slice.contains("elts") &&
      slice["elts"].is_array())
    {
      std::vector<nlohmann::json> idx_elts(
        slice["elts"].begin(), slice["elts"].end());
      python_list list(*this, element);
      expr = list.build_fancy_index(array, idx_elts, element);
      break;
    }

    // Boolean-mask indexing ``a[mask]``: when the index is a bare variable
    // reference whose static type is a bool array, filter `array` at
    // runtime to the elements where `mask` is True (NumPy fancy indexing).
    // Restricted to a simple Name index so the type can be checked without
    // re-evaluating a side-effecting expression later in this function.
    if (
      tuple_index_targets_list_model && slice.contains("_type") &&
      slice["_type"] == "Name")
    {
      exprt mask_candidate = get_expr(slice);
      if (!contains_cpp_throw(mask_candidate))
      {
        const typet mask_type = ns.follow(mask_candidate.type());
        if (mask_type.is_array())
        {
          if (ns.follow(mask_type.subtype()).is_bool())
          {
            python_list list(*this, element);
            expr = list.build_bool_mask_index(array, mask_candidate, element);
            break;
          }

          // Fancy/integer-array indexing through a variable (as opposed to
          // a literal index list, a[[0, 2]]) is not modelled yet; give an
          // explicit error instead of falling through to the generic
          // scalar-index path below, whose "not str" message is written
          // for a different mistake (string indices) and would be
          // confusing here.
          std::ostringstream msg;
          msg << "TypeError: fancy indexing with a non-boolean array is "
                 "not supported; only boolean-mask indexing (a[mask]) and "
                 "literal integer-list indexing (a[[i, j, ...]]) are "
                 "supported for array indices";
          const locationt loc = get_location_from_decl(element);
          if (!loc.is_nil())
            msg << " at " << loc.get_file() << ":" << loc.get_line();
          throw std::runtime_error(msg.str());
        }
      }
    }

    // Handle object subscripting through __getitem__:
    //   obj[key] -> obj.__getitem__(key)
    if (has_dunder_method(element["value"], "__getitem__"))
    {
      nlohmann::json args = nlohmann::json::array();
      args.push_back(slice);
      nlohmann::json call_node =
        build_dunder_call(element["value"], "__getitem__", args, element);
      expr = get_function_call(call_node);
      break;
    }

    // Reject subscripting scalar numeric / boolean values up front. CPython
    // raises "TypeError: 'int'/'float'/'bool' object is not subscriptable";
    // the list handler below silently produces an unresolved-type value that
    // later trips the binop "Unsupported comparison with unresolved operand
    // type" error far from the actual mistake (e.g. when the user expects
    // a tuple from a Counter.most_common() result and chains a second
    // subscript onto an int).
    if (
      !array_type.is_array() && !array_type.is_pointer() &&
      !array_type.is_struct() &&
      (array_type.is_signedbv() || array_type.is_unsignedbv() ||
       array_type.is_floatbv() || array_type.is_bool()))
    {
      std::string type_name;
      if (array_type.is_bool())
        type_name = "bool";
      else if (array_type.is_floatbv())
        type_name = "float";
      else
        type_name = "int";

      std::ostringstream msg;
      msg << "TypeError: '" << type_name << "' object is not subscriptable";
      const locationt loc = get_location_from_decl(element);
      if (!loc.is_nil())
        msg << " at " << loc.get_file() << ":" << loc.get_line();
      throw std::runtime_error(msg.str());
    }

    // Handle regular array/list subscripting
    python_list list(*this, element);
    expr = list.index(array, normalize_bool_index_node(slice));
    break;
  }
  case ExpressionType::FSTRING:
    expr = string_handler_.get_fstring_expr(element);
    break;
  case ExpressionType::TUPLE:
    return get_tuple_expr(element);
  case ExpressionType::SLICE:
    return build_slice_object(element);
  default:
  {
    std::ostringstream oss;
    oss << "Unsupported expression ";
    if (element.contains("_type"))
      oss << element["_type"].get<std::string>();

    if (element.contains("lineno"))
      oss << " at line " << element["lineno"].template get<int>();

    throw std::runtime_error(oss.str());
  }
  }

  return expr;
}

exprt python_converter::get_tuple_expr(const nlohmann::json &element)
{
  return tuple_handler_->get_tuple_expr(element);
}

// Shared lowering for Slice AST nodes and slice() builtin calls. Materialises
// a constant PySliceObject (defined in c2goto/library/python/python_types.h)
// whose has_* flags distinguish missing/None components from explicit zeros.
// Components are filled by name so that compiler-inserted alignment padding
// (which appears as an anonymous trailing member) gets a default zero value.
static exprt make_slice_struct_expr(
  python_converter &conv,
  const nlohmann::json *lower,
  const nlohmann::json *upper,
  const nlohmann::json *step,
  const nlohmann::json &source_node)
{
  const namespacet ns(conv.symbol_table());
  // Use the followed struct_typet (rather than the symbol_typet) so that
  // downstream call-site coercion sees `arg.type().is_struct()` and converts
  // the rvalue to an address when the callee parameter is pointer-typed.
  const struct_typet &struct_type =
    to_struct_type(ns.follow(conv.get_type_handler().get_slice_type()));

  // Unspecified bounds (`:`) are modelled as nondeterministic values rather
  // than literal zeros so that user code reading `sl.start` / `sl.stop` /
  // `sl.step` cannot mistake a bare slice for an explicit `0:0`. The
  // companion `has_start` / `has_stop` / `has_step` flags remain the
  // authoritative "was a bound supplied?" signal; `sl.start is None` is
  // lowered to a check of those flags in converter_compare.cpp
  // (try_lower_slice_member_is_none). See github #4543.
  auto lower_int =
    [&](const nlohmann::json *node, const typet &field_type) -> exprt {
    if (!node || node->is_null())
      return side_effect_expr_nondett(field_type);
    exprt value = conv.get_expr(*node);
    if (value.type() != field_type)
    {
      // (field_type)value, built in IREP2 (V.3).
      expr2tc v2;
      migrate_expr(value, v2);
      value = migrate_expr_back(typecast2tc(migrate_type(field_type), v2));
      value.type() = field_type; // restore #cpp_type that migrate_type drops
    }
    return value;
  };

  auto present_flag = [](const nlohmann::json *node) {
    return node && !node->is_null();
  };

  // V.3: build the PySliceObject value in IREP2. Each member is built exactly
  // as the legacy struct_exprt did -- start/stop/step (a value, nondet, or an
  // IREP2 typecast from lower_int), the has_* flags, and a zero for the
  // anonymous trailing padding member -- then migrated; the constant_struct2t
  // over those members back-migrates to the same struct value. Re-attach the
  // followed struct type afterwards: migrate_type_back does not reproduce the
  // component #cpp_type attributes / padding layout of the C model struct.
  std::vector<expr2tc> members;
  members.reserve(struct_type.components().size());
  for (const auto &component : struct_type.components())
  {
    const std::string name = component.get_name().as_string();
    exprt member;
    if (name == "start")
      member = lower_int(lower, component.type());
    else if (name == "stop")
      member = lower_int(upper, component.type());
    else if (name == "step")
      member = lower_int(step, component.type());
    else if (name == "has_start")
      member = from_integer(present_flag(lower) ? 1 : 0, component.type());
    else if (name == "has_stop")
      member = from_integer(present_flag(upper) ? 1 : 0, component.type());
    else if (name == "has_step")
      member = from_integer(present_flag(step) ? 1 : 0, component.type());
    else
      member = gen_zero(component.type());
    expr2tc m2;
    migrate_expr(member, m2);
    members.push_back(std::move(m2));
  }

  exprt slice_expr =
    migrate_expr_back(constant_struct2tc(migrate_type(struct_type), members));
  slice_expr.type() = struct_type;

  if (source_node.contains("lineno"))
    slice_expr.location() = conv.get_location_from_decl(source_node);

  return slice_expr;
}

exprt python_converter::build_slice_object(const nlohmann::json &slice_node)
{
  assert(slice_node.contains("_type") && slice_node["_type"] == "Slice");
  const nlohmann::json *lower =
    slice_node.contains("lower") ? &slice_node["lower"] : nullptr;
  const nlohmann::json *upper =
    slice_node.contains("upper") ? &slice_node["upper"] : nullptr;
  const nlohmann::json *step =
    slice_node.contains("step") ? &slice_node["step"] : nullptr;
  return make_slice_struct_expr(*this, lower, upper, step, slice_node);
}

exprt python_converter::build_slice_from_args(
  const nlohmann::json &args,
  const nlohmann::json &source_node)
{
  // slice(stop) / slice(start, stop) / slice(start, stop, step). Mirror
  // CPython: with one argument, the single value is stop; start is None.
  const nlohmann::json *lower = nullptr;
  const nlohmann::json *upper = nullptr;
  const nlohmann::json *step = nullptr;
  if (args.size() == 1)
  {
    upper = &args[0];
  }
  else if (args.size() == 2)
  {
    lower = &args[0];
    upper = &args[1];
  }
  else if (args.size() == 3)
  {
    lower = &args[0];
    upper = &args[1];
    step = &args[2];
  }
  else
  {
    throw std::runtime_error(
      "TypeError: slice expected 1 to 3 arguments, got " +
      std::to_string(args.size()));
  }
  return make_slice_struct_expr(*this, lower, upper, step, source_node);
}
