#include <python-frontend/char_utils.h>
#include <python-frontend/complex_handler.h>
#include <python-frontend/converter/converter_internal.h>
#include <python-frontend/convert_float_literal.h>
#include <python-frontend/function_call_builder.h>
#include <python-frontend/python_consteval.h>
#include <python-frontend/function_call_expr.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/module_locator.h>
#include <python-frontend/python_annotation.h>
#include <python-frontend/python_class_builder.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_dict_handler.h>
#include <python-frontend/python_exception_handler.h>
#include <python-frontend/python_lambda.h>
#include <python-frontend/python_list.h>
#include <python-frontend/python_typechecking.h>
#include <python-frontend/string_builder.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/tuple_handler.h>
#include <python-frontend/type_utils.h>
#include <util/arith_tools.h>
#include <util/base_type.h>
#include <util/c_typecast.h>
#include <util/c_types.h>
#include <util/encoding.h>
#include <util/expr_util.h>
#include <util/irep.h>
#include <util/message.h>
#include <util/python_types.h>
#include <util/std_code.h>
#include <util/string_constant.h>
#include <util/symbolic_types.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <regex>
#include <stdexcept>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <boost/filesystem.hpp>

using namespace json_utils;
namespace fs = boost::filesystem;


std::string
python_converter::get_op(const std::string &op, const typet &type) const
{
  return python_frontend::map_operator(op, type);
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
  if (value.is_boolean())
    return gen_boolean(value.get<bool>());

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

exprt python_converter::get_expr(const nlohmann::json &element)
{
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
            exprt optional_base = base_expr;
            if (optional_base.type().is_pointer())
            {
              exprt deref("dereference");
              deref.type() = optional_base.type().subtype();
              deref.move_to_operands(optional_base);
              optional_base = std::move(deref);
            }
            base_expr = member_exprt(optional_base, "value", inner_raw);
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

        if (base_type.is_struct())
        {
          const struct_typet &struct_type = to_struct_type(base_type);
          if (struct_type.has_component(attr_name))
          {
            const typet &attr_type =
              struct_type.get_component(attr_name).type();
            typet clean_type = clean_attribute_type(attr_type);
            exprt member_base = base_expr;
            if (member_base.type().is_pointer())
            {
              exprt deref("dereference");
              deref.type() = member_base.type().subtype();
              deref.move_to_operands(member_base);
              member_base = std::move(deref);
            }
            return member_exprt(member_base, attr_name, clean_type);
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
                  expr = member_exprt(base_expr, "value", vt);
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

      // Handle module attribute access (e.g., math.inf)
      if (is_imported_module(var_name))
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
          if (func_symbol && func_symbol->type.is_code())
          {
            expr = symbol_expr(*func_symbol);
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

    expr = symbol_expr(*symbol);

    // If the looked-up symbol is an enum class attribute with int type,
    // wrap it in the proper enum struct expression so callers that expect
    // the enum class type (e.g. function parameters) receive a struct value.
    if (is_class_attr && symbol->type.is_signedbv() && element.contains("attr"))
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

      // Delegate complex attribute access (.real, .imag) to the handler.
      if (is_complex_type(symbol->type))
      {
        exprt result =
          complex_handler_.handle_attribute_access(expr, attr_name);
        if (!result.is_nil())
        {
          expr = result;
          break;
        }
      }

      // Get object type name from symbol. e.g.: tag-MyClass
      std::string obj_type_name;
      const typet &symbol_type =
        (symbol->type.is_pointer()) ? symbol->type.subtype() : symbol->type;

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

          if (s.id.as_string().find("tag-") == 0 && s.type.is_struct())
          {
            const struct_typet &struct_type = to_struct_type(s.type);
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

        // Create a typecast from char* to target_class*
        typet target_ptr_type = gen_pointer_type(target_class_symbol->type);
        exprt casted_expr = typecast_exprt(expr, target_ptr_type);

        // Dereference to get the object
        exprt deref_expr("dereference", target_class_symbol->type);
        deref_expr.copy_to_operands(casted_expr);

        // Access the member on the object
        const struct_typet &target_struct =
          to_struct_type(target_class_symbol->type);
        const typet &attr_type = target_struct.get_component(attr_name).type();
        typet clean_type = clean_attribute_type(attr_type);

        member_exprt member_expr(deref_expr, attr_name, clean_type);
        expr = member_expr;
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
          if (s.id.as_string().find("tag-") == 0 && s.type.is_struct())
          {
            const struct_typet &st = to_struct_type(s.type);
            if (st.has_component(attr_name))
              fallback_class_id = s.id.as_string();
          }
        });
        if (!fallback_class_id.empty())
          class_symbol = symbol_table_.find_symbol(fallback_class_id);
      }
      if (!class_symbol)
      {
        throw std::runtime_error("Class \"" + obj_type_name + "\" not found");
      }

      struct_typet &class_type =
        static_cast<struct_typet &>(class_symbol->type);
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
          base = typecast_exprt(base, gen_pointer_type(class_type));

        if (base.type().is_pointer())
        {
          exprt deref("dereference");
          deref.type() = base.type().subtype();
          deref.move_to_operands(base);
          base = std::move(deref);
        }

        return member_exprt(base, attr_name, clean_type);
      };

      if (is_converting_lhs)
      {
        // Add member in the class if not exists
        if (!class_type.has_component(attr_name))
        {
          struct_typet::componentt comp = python_frontend::build_component(
            class_type.tag().as_string(), attr_name, current_element_type);
          class_type.components().push_back(comp);
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
    const nlohmann::json &slice = element["slice"];
    typet array_type = ns.follow(array.type());

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
        array = dereference_exprt(array, pointed);
        array.type() = pointed;
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

    // Handle regular array/list subscripting
    python_list list(*this, element);
    expr = list.index(array, slice);
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

  auto lower_int =
    [&](const nlohmann::json *node, const typet &field_type) -> exprt {
    if (!node || node->is_null())
      return gen_zero(field_type);
    exprt value = conv.get_expr(*node);
    if (value.type() != field_type)
      value = typecast_exprt(value, field_type);
    return value;
  };

  auto present_flag = [](const nlohmann::json *node) {
    return node && !node->is_null();
  };

  struct_exprt slice_expr(struct_type);
  for (const auto &component : struct_type.components())
  {
    const std::string name = component.get_name().as_string();
    if (name == "start")
      slice_expr.operands().push_back(lower_int(lower, component.type()));
    else if (name == "stop")
      slice_expr.operands().push_back(lower_int(upper, component.type()));
    else if (name == "step")
      slice_expr.operands().push_back(lower_int(step, component.type()));
    else if (name == "has_start")
      slice_expr.operands().push_back(
        from_integer(present_flag(lower) ? 1 : 0, component.type()));
    else if (name == "has_stop")
      slice_expr.operands().push_back(
        from_integer(present_flag(upper) ? 1 : 0, component.type()));
    else if (name == "has_step")
      slice_expr.operands().push_back(
        from_integer(present_flag(step) ? 1 : 0, component.type()));
    else
      slice_expr.operands().push_back(gen_zero(component.type()));
  }

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

size_t python_converter::get_type_size(const nlohmann::json &ast_node)
{
  size_t type_size = 0;

  // Handle lambda functions - they don't have a meaningful size
  if (
    ast_node.contains("value") && ast_node["value"].contains("_type") &&
    ast_node["value"]["_type"] == "Lambda")
    return 0;

  if (ast_node.contains("value") && ast_node["value"].contains("value"))
  {
    // Handle bytes literals
    if (
      ast_node.contains("annotation") &&
      ast_node["annotation"].contains("id") &&
      ast_node["annotation"]["id"] == "bytes")
    {
      if (ast_node["value"].contains("encoded_bytes"))
      {
        const std::string &str =
          ast_node["value"]["encoded_bytes"].get<std::string>();
        std::vector<uint8_t> decoded = base64_decode(str);
        type_size = decoded.size();
      }
      else if (ast_node["value"]["value"].is_string())
      {
        // Direct bytes literal such as b'A'
        type_size = ast_node["value"]["value"].get<std::string>().size();
      }
    }
    else if (ast_node["value"]["value"].is_string())
      type_size = ast_node["value"]["value"].get<std::string>().size();
  }
  else if (
    ast_node["value"].contains("args") &&
    ast_node["value"]["args"].is_array() &&
    ast_node["value"]["args"].size() > 0 &&
    ast_node["value"]["args"][0].contains("value") &&
    ast_node["value"]["args"][0]["value"].is_string())
  {
    type_size = ast_node["value"]["args"][0]["value"].get<std::string>().size();
  }
  else if (
    ast_node["value"].contains("_type") && ast_node["value"]["_type"] == "List")
  {
    type_size = ast_node["value"]["elts"].size();
  }
  // Handle cases where size cannot be determined from AST structure
  else if (
    ast_node["value"].contains("value") &&
    ast_node["value"]["value"].is_string())
  {
    // Fallback for direct string values
    type_size = ast_node["value"]["value"].get<std::string>().size();
  }

  return type_size;
}

symbolt python_converter::create_return_temp_variable(
  const typet &return_type,
  const locationt &location,
  const std::string &func_name)
{
  static int temp_counter = 0;
  temp_counter++;

  symbol_id temp_sid = create_symbol_id();
  std::string temp_name =
    "return_value$_" + func_name + "$" + std::to_string(temp_counter);
  temp_sid.set_object(temp_name);

  symbolt temp_symbol;
  temp_symbol.id = temp_sid.to_string();
  temp_symbol.name = temp_sid.to_string();
  temp_symbol.type = return_type;
  temp_symbol.lvalue = true;
  temp_symbol.static_lifetime = false;
  temp_symbol.location = location;
  temp_symbol.mode = "Python";
  temp_symbol.module = location.get_file().as_string();
  temp_symbol.file_local = true;
  temp_symbol.is_extern = false;

  return temp_symbol;
}

const nlohmann::json &get_return_statement(const nlohmann::json &function)
{
  for (const auto &stmt : function["body"])
  {
    if (python_frontend::get_statement_type(stmt) == StatementType::RETURN)
      return stmt;
  }

  throw std::runtime_error(
    "Function " + function["name"].get<std::string>() +
    " has no return statement");
}



bool python_converter::function_has_missing_return_paths(
  const nlohmann::json &function_node)
{
  const auto &body = function_node["body"];
  if (body.empty())
    return true;

  // Check if the last statement is a return
  const auto &last_stmt = body.back();
  if (last_stmt["_type"] == "Return")
    return false;

  // Check for if-else structures at the end
  if (last_stmt["_type"] == "If")
  {
    // Check if both if and else branches have returns
    bool if_has_return = false;
    bool else_has_return = false;

    // Check if branch
    if (!last_stmt["body"].empty())
    {
      const auto &if_last = last_stmt["body"].back();
      if_has_return = (if_last["_type"] == "Return");
    }

    // Check else branch
    if (last_stmt.contains("orelse") && !last_stmt["orelse"].empty())
    {
      const auto &else_last = last_stmt["orelse"].back();
      else_has_return = (else_last["_type"] == "Return");
    }

    return !(if_has_return && else_has_return);
  }

  return true; // No explicit return found
}

TypeFlags
python_converter::infer_types_from_returns(const nlohmann::json &function_body)
{
  TypeFlags flags;

  std::function<void(const nlohmann::json &)> scan = [&](const nlohmann::json
                                                           &body) {
    for (const auto &stmt : body)
    {
      if (stmt["_type"] == "Return" && stmt["value"].is_null())
      {
        // Bare "return" (no value) is semantically "return None"
        flags.has_none = true;
      }
      else if (stmt["_type"] == "Return" && !stmt["value"].is_null())
      {
        const auto &val = stmt["value"];

        if (val["_type"] == "Constant")
        {
          const auto &constant_val = val["value"];
          if (constant_val.is_number_float())
            flags.has_float = true;
          else if (constant_val.is_number_integer())
            flags.has_int = true;
          else if (constant_val.is_boolean())
            flags.has_bool = true;
          else if (constant_val.is_null())
            flags.has_none = true;
          else
          {
            std::string type_name = constant_val.is_string()   ? "string"
                                    : constant_val.is_object() ? "object"
                                    : constant_val.is_array()  ? "array"
                                                               : "unknown";
            throw std::runtime_error(
              "Unsupported return type '" + type_name + "' detected");
          }
        }
        else if (val["_type"] == "BinOp" || val["_type"] == "UnaryOp")
        {
          flags.has_float = true; // Default for expressions
        }
        else if (val["_type"] == "Call")
        {
          // For return <func_call>(), look up the called function's returns
          // to infer the value type being propagated through the call
          const auto &func = val["func"];
          bool resolved = false;
          if (func.contains("id") && ast_json)
          {
            std::string called_name = func["id"].get<std::string>();
            const auto &module_body = (*ast_json)["body"];
            for (const auto &item : module_body)
            {
              if (item["_type"] == "FunctionDef" && item["name"] == called_name)
              {
                // Scan the called function's return statements directly
                // (one level only to avoid infinite recursion)
                for (const auto &s : item["body"])
                {
                  if (
                    s["_type"] == "Return" && !s["value"].is_null() &&
                    s["value"]["_type"] == "Constant" &&
                    !s["value"]["value"].is_null())
                  {
                    const auto &cv = s["value"]["value"];
                    if (cv.is_number_float())
                      flags.has_float = true;
                    else if (cv.is_number_integer())
                      flags.has_int = true;
                    else if (cv.is_boolean())
                      flags.has_bool = true;
                    resolved = true;
                  }
                }
                break;
              }
            }
          }
          if (!resolved)
            flags.has_int = true; // Default to int for unresolvable calls
        }
        else if (val["_type"] == "Name")
        {
          // return <variable> — indicates a value return of unknown type
          flags.has_int = true;
        }
      }

      if (stmt.contains("body") && stmt["body"].is_array())
        scan(stmt["body"]);
      if (stmt.contains("orelse") && stmt["orelse"].is_array())
        scan(stmt["orelse"]);
    }
  };

  scan(function_body);
  return flags;
}

// Return true if 'param_name' has any attribute written (x.attr = ...)
// anywhere in 'body' (recursive scan over nested blocks).
static bool param_is_mutated_in_body(
  const std::string &param_name,
  const nlohmann::json &body)
{
  if (!body.is_array())
    return false;

  for (const auto &stmt : body)
  {
    if (!stmt.is_object())
      continue;

    const std::string &stype =
      stmt.contains("_type") ? stmt["_type"].get<std::string>() : "";

    // x.attr = value  (plain assignment)
    if (stype == "Assign" && stmt.contains("targets"))
    {
      for (const auto &tgt : stmt["targets"])
      {
        if (
          tgt.contains("_type") && tgt["_type"] == "Attribute" &&
          tgt.contains("value") && tgt["value"].contains("_type") &&
          tgt["value"]["_type"] == "Name" && tgt["value"].contains("id") &&
          tgt["value"]["id"] == param_name)
          return true;
      }
    }
    // x.attr: T = value  (annotated assignment)
    else if (stype == "AnnAssign" && stmt.contains("target"))
    {
      const auto &tgt = stmt["target"];
      if (
        tgt.contains("_type") && tgt["_type"] == "Attribute" &&
        tgt.contains("value") && tgt["value"].contains("_type") &&
        tgt["value"]["_type"] == "Name" && tgt["value"].contains("id") &&
        tgt["value"]["id"] == param_name)
        return true;
    }

    // Recurse into nested blocks (if/while/for bodies, else branches)
    for (const char *key : {"body", "orelse", "handlers", "finalbody"})
    {
      if (stmt.contains(key) && stmt[key].is_array())
      {
        if (param_is_mutated_in_body(param_name, stmt[key]))
          return true;
      }
    }
  }
  return false;
}

static bool node_uses_param_as_list_like(
  const std::string &param_name,
  const nlohmann::json &node)
{
  if (!node.is_object())
    return false;

  if (node.contains("_type") && node["_type"].is_string())
  {
    const std::string node_type = node["_type"].get<std::string>();

    // x[i]
    if (
      node_type == "Subscript" && node.contains("value") &&
      node["value"].is_object() && node["value"].value("_type", "") == "Name" &&
      node["value"].value("id", "") == param_name)
      return true;

    if (node_type == "Call")
    {
      // len(x)
      if (
        node.contains("func") && node["func"].is_object() &&
        node["func"].value("_type", "") == "Name" &&
        node["func"].value("id", "") == "len" && node.contains("args") &&
        node["args"].is_array() && !node["args"].empty() &&
        node["args"][0].is_object() &&
        node["args"][0].value("_type", "") == "Name" &&
        node["args"][0].value("id", "") == param_name)
      {
        return true;
      }

      // x.append(...), x.pop(...), ...
      if (
        node.contains("func") && node["func"].is_object() &&
        node["func"].value("_type", "") == "Attribute" &&
        node["func"].contains("value") && node["func"]["value"].is_object() &&
        node["func"]["value"].value("_type", "") == "Name" &&
        node["func"]["value"].value("id", "") == param_name &&
        node["func"].contains("attr") && node["func"]["attr"].is_string())
      {
        const std::string attr = node["func"]["attr"].get<std::string>();
        if (
          attr == "append" || attr == "extend" || attr == "insert" ||
          attr == "pop" || attr == "remove" || attr == "clear" ||
          attr == "sort" || attr == "reverse")
          return true;
      }
    }
  }

  for (auto it = node.begin(); it != node.end(); ++it)
  {
    const auto &child = it.value();
    if (child.is_object())
    {
      if (node_uses_param_as_list_like(param_name, child))
        return true;
    }
    else if (child.is_array())
    {
      for (const auto &elem : child)
      {
        if (elem.is_object() && node_uses_param_as_list_like(param_name, elem))
          return true;
      }
    }
  }

  return false;
}

static bool param_is_list_like_in_body(
  const std::string &param_name,
  const nlohmann::json &body)
{
  if (!body.is_array())
    return false;

  for (const auto &stmt : body)
  {
    if (node_uses_param_as_list_like(param_name, stmt))
      return true;
  }

  return false;
}

size_t python_converter::register_function_argument(
  const nlohmann::json &element,
  code_typet &type,
  const symbol_id &id,
  const locationt &location,
  bool is_keyword_only)
{
  (void)is_keyword_only;

  // Extract the argument name and resolve its type from the annotation.
  // Special cases: `self` and `cls` are modelled as pointers to the current class
  std::string arg_name = element["arg"].get<std::string>();
  typet arg_type;

  if (arg_name == "self")
    arg_type = gen_pointer_type(type_handler_.get_typet(current_class_name_));
  else if (arg_name == "cls")
    arg_type = any_type();
  else
  {
    if (!element.contains("annotation") || element["annotation"].is_null())
    {
      // Python does not require type annotations; treat unannotated parameters
      // as Any (void*) to follow Python semantics.
      arg_type = any_type();
    }
    else
      arg_type = get_type_from_annotation(element["annotation"], element);
  }

  // Arrays are converted to pointers so that the backend receives the same
  // representation regardless of how the parameter is declared.
  if (arg_type.is_array())
    arg_type = gen_pointer_type(arg_type.subtype());

  assert(arg_type != typet());

  code_typet::argumentt arg;
  arg.type() = arg_type;
  arg.cmt_base_name(arg_name);

  // Build a unique identifier for the parameter. The identifier mirrors the
  // scheme used elsewhere in the converter (function-id@parameter-name)
  std::string arg_id = id.to_string() + "@" + arg_name;
  arg.cmt_identifier(arg_id);
  arg.identifier(arg_id);
  arg.location() = get_location_from_decl(element);

  type.arguments().push_back(arg);
  size_t inserted_index = type.arguments().size() - 1;

  // Materialise a symbol for the parameter so that subsequent passes (e.g.
  // attribute access on instances) can resolve it.
  symbolt param_symbol = create_symbol(
    location.get_file().as_string(),
    arg_name,
    arg_id,
    arg.location(),
    arg_type);
  param_symbol.lvalue = true;
  param_symbol.is_parameter = true;
  param_symbol.file_local = true;
  param_symbol.static_lifetime = false;
  param_symbol.is_extern = false;
  symbol_table_.add(param_symbol);
  symbolt *stored_param = symbol_table_.find_symbol(arg_id);
  if (
    stored_param != nullptr && element.contains("annotation") &&
    !element["annotation"].is_null())
  {
    get_typechecker().cache_annotation_types(
      *stored_param, element["annotation"]);

    if (
      element["annotation"].contains("_type") &&
      element["annotation"]["_type"] == "Subscript" &&
      element["annotation"].contains("value") &&
      element["annotation"]["value"].contains("id"))
    {
      const std::string container_name =
        element["annotation"]["value"]["id"].get<std::string>();
      if (container_name == "List" || container_name == "list")
      {
        typet elem_type = type_handler_.get_list_type(element).subtype();
        if (!elem_type.is_empty())
          python_list::add_type_info_entry(arg_id, "", elem_type);
      }
    }
  }

  // If the parameter is class-typed (e.g. Foo), copy instance attributes from
  // the class’ synthetic `self` symbol so method bodies can access members via
  // this parameter.
  if (arg_name != "self" && arg_name != "cls")
  {
    typet base_type = arg_type.is_pointer() ? arg_type.subtype() : arg_type;
    if (base_type.id() == "symbol")
      base_type = ns.follow(base_type);

    if (base_type.is_struct())
    {
      const struct_typet &struct_type = to_struct_type(base_type);
      std::string class_tag = struct_type.tag().as_string();

      std::string class_name = extract_class_name_from_tag(class_tag);

      symbol_id self_sid(
        location.get_file().as_string(), class_name, class_name);
      self_sid.set_object("self");

      copy_instance_attributes(self_sid.to_string(), arg_id);

      std::string normalized_key = create_normalized_self_key(class_tag);
      copy_instance_attributes(normalized_key, arg_id);
    }
  }

  return inserted_index;
}

void python_converter::process_function_arguments(
  const nlohmann::json &function_node,
  code_typet &type,
  const symbol_id &id,
  const locationt &location)
{
  std::vector<size_t> positional_indices;
  std::vector<size_t> kwonly_indices;

  // Extract args node to avoid repeated access
  const nlohmann::json &args_node = function_node["args"];

  // Process regular arguments
  for (const nlohmann::json &element : args_node["args"])
  {
    size_t index =
      register_function_argument(element, type, id, location, false);
    positional_indices.push_back(index);
  }

  // Process keyword-only arguments (parameters after * separator)
  if (args_node.contains("kwonlyargs") && !args_node["kwonlyargs"].is_null())
  {
    for (const nlohmann::json &element : args_node["kwonlyargs"])
    {
      size_t index =
        register_function_argument(element, type, id, location, true);
      kwonly_indices.push_back(index);
    }
  }

  if (
    args_node.contains("defaults") && args_node["defaults"].is_array() &&
    !args_node["defaults"].empty() && !positional_indices.empty())
  {
    const auto &defaults = args_node["defaults"];
    size_t defaults_count = defaults.size();

    if (defaults_count <= positional_indices.size())
    {
      for (size_t i = 0; i < defaults_count; ++i)
      {
        size_t positional_index =
          positional_indices[positional_indices.size() - defaults_count + i];
        if (!defaults[i].is_null())
        {
          exprt default_expr = get_expr(defaults[i]);
          type.arguments()[positional_index].default_value() = default_expr;

          // If the default is a function pointer and the parameter was
          // annotated as Any (void*), upgrade the parameter type to match.
          // This enables indirect-call resolution for function-alias defaults
          // like def h(op=g) where g = f (a named function).
          if (
            default_expr.type().is_pointer() &&
            default_expr.type().subtype().is_code())
          {
            auto &param_arg = type.arguments()[positional_index];
            if (param_arg.type() == any_type())
            {
              param_arg.type() = default_expr.type();
              std::string param_id = param_arg.cmt_identifier().as_string();
              if (!param_id.empty())
              {
                symbolt *param_sym = symbol_table_.find_symbol(param_id);
                if (param_sym)
                  param_sym->type = default_expr.type();
              }
            }
          }
        }
      }
    }
  }

  if (
    args_node.contains("kw_defaults") && args_node["kw_defaults"].is_array() &&
    args_node["kw_defaults"].size() == kwonly_indices.size())
  {
    const auto &kw_defaults = args_node["kw_defaults"];
    for (size_t i = 0; i < kw_defaults.size(); ++i)
    {
      if (!kw_defaults[i].is_null())
      {
        exprt default_expr = get_expr(kw_defaults[i]);
        type.arguments()[kwonly_indices[i]].default_value() = default_expr;
      }
    }
  }

  // Python object reference semantics: if a non-enum class parameter is
  // mutated inside the function (x.attr = ...), model it as a pointer so
  // that mutations are visible to the caller (same as 'self' for methods).
  if (!function_node.contains("body"))
    return;
  const nlohmann::json &body = function_node["body"];

  // Refine unannotated Any parameters to list model type when body usage
  // clearly matches list semantics (len(x), x[i], list mutator methods).
  // Restrict this refinement to functions from the main source file to avoid
  // affecting imported module internals.
  if (location.get_file().as_string() == main_python_file)
  {
    for (auto &param_arg : type.arguments())
    {
      const std::string param_name = param_arg.get_base_name().as_string();
      if (param_name == "self" || param_name == "cls" || param_name.empty())
        continue;

      if (param_arg.type() != any_type())
        continue;

      if (!param_is_list_like_in_body(param_name, body))
        continue;

      typet list_t = type_handler_.get_list_type();
      param_arg.type() = list_t;

      const std::string param_id = param_arg.cmt_identifier().as_string();
      if (!param_id.empty())
      {
        symbolt *param_sym = symbol_table_.find_symbol(param_id);
        if (param_sym)
          param_sym->type = list_t;
      }
    }
  }

  for (auto &param_arg : type.arguments())
  {
    const std::string param_name = param_arg.get_base_name().as_string();
    if (param_name == "self" || param_name == "cls" || param_name.empty())
      continue;

    // Only applies to user-defined (non-enum) class-typed parameters.
    typet ptype = param_arg.type();
    if (ptype.id() == "symbol")
      ptype = ns.follow(ptype);
    if (!ptype.is_struct())
      continue;
    const std::string class_tag = to_struct_type(ptype).tag().as_string();
    const std::string class_name = extract_class_name_from_tag(class_tag);
    if (
      !json_utils::is_class(class_name, *ast_json) ||
      python_frontend::is_enum_class(class_name, *ast_json))
      continue;

    // Check whether the function body mutates this parameter.
    if (!param_is_mutated_in_body(param_name, body))
      continue;

    // Upgrade the parameter to a pointer and update the parameter symbol.
    typet ptr_type = gen_pointer_type(param_arg.type());
    param_arg.type() = ptr_type;
    const std::string param_id = param_arg.cmt_identifier().as_string();
    if (!param_id.empty())
    {
      symbolt *param_sym = symbol_table_.find_symbol(param_id);
      if (param_sym)
        param_sym->type = ptr_type;
    }
  }
}

void python_converter::validate_return_paths(
  const nlohmann::json &function_node,
  const code_typet &type,
  exprt &function_body)
{
  // Skip validation for void returns and constructors
  if (
    type.return_type().is_empty() ||
    type.return_type().id() == typet::t_empty ||
    type.return_type().id() == "constructor" ||
    !function_has_missing_return_paths(function_node))
  {
    return;
  }

  locationt loc = get_location_from_decl(function_node);

  code_assertt missing_return_assert;
  missing_return_assert.assertion() = gen_boolean(false);
  missing_return_assert.location() = loc;
  missing_return_assert.location().comment(
    "Missing return statement detected in function '" + current_func_name_ +
    "'");

  function_body.copy_to_operands(missing_return_assert);
}

typet python_converter::infer_return_type_from_body(const nlohmann::json &body)
{
  auto infer_constant_type = [](const nlohmann::json &constant_value) -> typet {
    if (constant_value.is_number_float())
      return double_type();
    if (constant_value.is_number_integer())
      return long_long_int_type();
    if (constant_value.is_boolean())
      return bool_type();
    if (constant_value.is_string())
      return gen_pointer_type(char_type());
    if (constant_value.is_null())
      return none_type();
    return empty_typet();
  };

  for (const auto &stmt : body)
  {
    if (stmt["_type"] == "Return" && !stmt["value"].is_null())
    {
      const auto &ret_val = stmt["value"];

      // `return self` in a method: surface the class's struct value type so
      // callers can assign to a `Class`-typed local. Without this, fallback
      // inference picks up `Class *` from `self` and the call-site assignment
      // becomes a pointer-to-struct mismatch that trips an assertion in
      // value_set::make_member on later member access. See GitHub #4514.
      if (
        ret_val["_type"] == "Name" && ret_val.contains("id") &&
        ret_val["id"] == "self" && !current_class_name_.empty())
        return type_handler_.get_typet(current_class_name_);

      // If returning a tuple, infer its type
      if (ret_val["_type"] == "Tuple")
        return tuple_handler_->get_tuple_expr(ret_val).type();

      // Constant returns (including strings)
      if (ret_val["_type"] == "Constant" && ret_val.contains("value"))
      {
        typet inferred = infer_constant_type(ret_val["value"]);
        if (!inferred.is_empty())
          return inferred;
      }

      // Heuristic: return dict.get(key, default) -> infer from default literal.
      if (
        ret_val["_type"] == "Call" && ret_val.contains("func") &&
        ret_val["func"].contains("_type") &&
        ret_val["func"]["_type"] == "Attribute" &&
        ret_val["func"].contains("attr") && ret_val["func"]["attr"] == "get" &&
        ret_val.contains("args") && ret_val["args"].is_array() &&
        ret_val["args"].size() >= 2)
      {
        const auto &default_arg = ret_val["args"][1];
        if (
          default_arg.contains("_type") && default_arg["_type"] == "Constant" &&
          default_arg.contains("value"))
        {
          typet inferred = infer_constant_type(default_arg["value"]);
          if (!inferred.is_empty())
            return inferred;
        }
      }
    }
  }

  return empty_typet();
}

void python_converter::get_function_definition(
  const nlohmann::json &function_node)
{
  // Function return type
  code_typet type;
  const nlohmann::json &return_node = function_node["returns"];

  // Determine return type
  if (
    return_node.is_null() ||
    (return_node["_type"] == "Constant" && return_node["value"].is_null()))
  {
    type.return_type() = empty_typet();
  }
  else if (return_node.contains("id") || return_node["_type"] == "Subscript")
  {
    const nlohmann::json &return_type = (return_node["_type"] == "Subscript")
                                          ? return_node["value"]["id"]
                                          : return_node["id"];

    if (return_type == "Any")
    {
      // Infer type from return statements
      TypeFlags flags = infer_types_from_returns(function_node["body"]);
      type.return_type() = type_utils::select_widest_type(flags, double_type());

      if (!flags.has_float && !flags.has_int && !flags.has_bool)
        log_warning("Default to double since no type could be inferred");
    }
    else if (return_type == "Union")
    {
      // Extract Union member types
      TypeFlags flags = type_utils::extract_union_types(return_node["slice"]);
      type.return_type() = type_utils::select_widest_type(flags, any_type());

      if (!flags.has_float && !flags.has_int && !flags.has_bool)
        log_warning("Union with no recognized types, defaulting to pointer");
    }
    else if (return_type == "list" || return_type == "List")
    {
      type.return_type() = type_handler_.get_list_type();
    }
    else if (return_type == "dict" || return_type == "Dict")
    {
      type.return_type() = dict_handler_->get_dict_struct_type();
    }
    else if (return_type == "str")
    {
      // String return types should be pointers, not arrays
      type.return_type() = gen_pointer_type(char_type());
    }
    else if (
      (return_type == "Tuple" || return_type == "tuple") &&
      return_node["_type"] == "Subscript")
    {
      type.return_type() =
        tuple_handler_->get_tuple_type_from_annotation(return_node);
    }
    else
    {
      type.return_type() =
        type_handler_.get_typet(return_type.get<std::string>());
    }
  }
  else if (return_node["_type"] == "BinOp")
  {
    // Handle PEP 604 union syntax: int | bool
    TypeFlags flags = type_utils::extract_binop_union_types(return_node);
    type.return_type() = type_utils::select_widest_type(flags, any_type());

    if (!flags.has_float && !flags.has_int && !flags.has_bool)
      log_warning("Union with no recognized types, defaulting to pointer");
  }
  else if (return_node["_type"] == "Tuple")
  {
    // Handle tuple return types such as (int, str)
    // TODO: we must still handle tuple types!
    type.return_type() = type_handler_.get_typet(std::string("tuple"));
  }
  else if (return_node["_type"] == "Constant" || return_node["_type"] == "Str")
  {
    std::string type_string =
      type_utils::remove_quotes(return_node["value"].get<std::string>());
    if (type_string == "str")
      type.return_type() = gen_pointer_type(char_type());
    else
      type.return_type() = type_handler_.get_typet(type_string);
  }
  else
    throw std::runtime_error("Return type undefined");

  // Setup function context
  const std::string caller_func_name = current_func_name_;

  locationt location = get_location_from_decl(function_node);

  current_element_type = type.return_type();
  std::string func_name = function_node["name"].get<std::string>();

  // __init__() is renamed to Classname()
  if (func_name == "__init__")
  {
    func_name = current_class_name_;
    type.return_type() = typet("constructor");
  }

  // If we are inside another function, create a nested name
  if (!caller_func_name.empty())
  {
    current_func_name_ = caller_func_name + "@F@" + func_name;
  }
  else
  {
    current_func_name_ = func_name;
  }

  scope_stack_.push_back("@F@" + func_name);

  symbol_id id = create_symbol_id();

  std::string module_name =
    current_python_file.substr(0, current_python_file.find_last_of("."));

  // Process function arguments
  process_function_arguments(function_node, type, id, location);

  // Create and register function symbol
  symbolt symbol = create_symbol(
    module_name, current_func_name_, id.to_string(), location, type);
  symbol.lvalue = true;
  symbol.is_extern = false;
  symbol.file_local = false;

  symbolt *added_symbol = symbol_table_.move_symbol_to_context(symbol);

  // Pre-scan: detect mixed value+None returns and upgrade return type to
  // Optional so None checks work correctly at runtime.
  // This applies even when the function has an explicit return annotation:
  // Python does not enforce annotations, so `-> int` with `return None` in
  // the body must be modelled as Optional[int].
  auto body_has_none_return = [](const nlohmann::json &body) -> bool {
    std::function<bool(const nlohmann::json &)> scan =
      [&](const nlohmann::json &stmts) -> bool {
      for (const auto &s : stmts)
      {
        if (s["_type"] == "Return")
        {
          if (s["value"].is_null())
            return true;
          if (
            s["value"]["_type"] == "Constant" && s["value"]["value"].is_null())
            return true;
        }
        if (s.contains("body") && s["body"].is_array() && scan(s["body"]))
          return true;
        if (s.contains("orelse") && s["orelse"].is_array() && scan(s["orelse"]))
          return true;
      }
      return false;
    };
    return scan(body);
  };

  bool already_optional =
    type.return_type().is_struct() && to_struct_type(type.return_type())
                                        .tag()
                                        .as_string()
                                        .starts_with("tag-Optional_");
  if (!already_optional && body_has_none_return(function_node["body"]))
  {
    if (type.return_type().is_empty())
    {
      // Unannotated function: need full type inference to pick value_type
      TypeFlags return_flags = infer_types_from_returns(function_node["body"]);
      bool has_value_return =
        return_flags.has_int || return_flags.has_float || return_flags.has_bool;
      if (has_value_return)
      {
        typet value_type =
          type_utils::select_widest_type(return_flags, long_long_int_type());
        typet optional_type = type_handler_.build_optional_type(value_type);
        type.return_type() = optional_type;
        current_element_type = optional_type;
        added_symbol->type = type;
      }
    }
    else
    {
      // Explicitly-annotated function (e.g., -> int) with return None paths:
      // upgrade the annotated type to Optional[annotated_type].
      typet optional_type =
        type_handler_.build_optional_type(type.return_type());
      type.return_type() = optional_type;
      current_element_type = optional_type;
      added_symbol->type = type;
    }
  }

  // For unannotated functions, attempt AST-based return inference before body
  // conversion so return expressions are typed in the right context.
  if (type.return_type().is_empty())
  {
    typet inferred_type = infer_return_type_from_body(function_node["body"]);
    if (!inferred_type.is_empty())
    {
      type.return_type() = inferred_type;
      current_element_type = inferred_type;
      added_symbol->type = type;
    }
  }

  // Save function return type for use in get_return_statements
  typet saved_func_return_type = current_func_return_type_;
  current_func_return_type_ = type.return_type();

  // Process function body
  exprt function_body = get_block(function_node["body"]);

  // Restore saved function return type (for nested function defs)
  current_func_return_type_ = saved_func_return_type;

  // If return type is empty/unannotated, try to infer from return statements
  if (type.return_type().is_empty())
  {
    typet inferred_type = infer_return_type_from_body(function_node["body"]);
    if (!inferred_type.is_empty())
    {
      type.return_type() = inferred_type;
      added_symbol->type = type; // Update the symbol's type
    }
  }

  // If return type is still empty, scan the converted GOTO body for RETURN
  // instructions with typed values. This handles indirect calls through
  // function-pointer parameters (e.g., "return op(1,1)" where op defaults
  // to a typed function pointer).
  if (type.return_type().is_empty())
  {
    for (const auto &instr : function_body.operands())
    {
      if (!instr.is_code())
        continue;
      const codet &code_instr = to_code(instr);
      if (code_instr.get_statement() == "return")
      {
        const code_returnt &ret = to_code_return(code_instr);
        if (ret.has_return_value())
        {
          const typet &ret_type = ret.return_value().type();
          if (!ret_type.is_empty())
          {
            type.return_type() = ret_type;
            added_symbol->type = type;
            break;
          }
        }
      }
    }
  }

  // Inject runtime checks for annotated parameters
  if (type_assertions_enabled())
    get_typechecker().inject_parameter_type_assertions(
      function_node, id, type, function_body);

  // Add ESBMC_Hide label for models/imports
  if (is_loading_models || is_importing_module)
  {
    code_labelt esbmc_hide;
    esbmc_hide.set_label("__ESBMC_HIDE");
    esbmc_hide.code() = code_skipt();
    function_body.operands().insert(
      function_body.operands().begin(), esbmc_hide);
  }

  // Validate return paths
  validate_return_paths(function_node, type, function_body);

  added_symbol->value = function_body;

  scope_stack_.pop_back();

  // Restore caller function name
  current_func_name_ = caller_func_name;
}



python_converter::python_converter(
  contextt &_context,
  const nlohmann::json *ast,
  const global_scope &gs)
  : symbol_table_(_context),
    ast_json(ast),
    global_scope_(gs),
    type_handler_(*this),
    string_builder_(new string_builder(*this, &string_handler_)),
    sym_generator_("python_converter::"),
    ns(_context),
    current_func_name_(""),
    current_class_name_(""),
    current_block(nullptr),
    current_lhs(nullptr),
    string_handler_(*this, symbol_table_, type_handler_, string_builder_),
    math_handler_(*this, symbol_table_, type_handler_),
    complex_handler_(*this, symbol_table_, type_handler_),
    tuple_handler_(new tuple_handler(*this, type_handler_)),
    dict_handler_(new python_dict_handler(*this, symbol_table_, type_handler_)),
    typechecker_(new python_typechecking(*this)),
    lambda_handler_(new python_lambda(*this, _context, type_handler_)),
    exception_handler_(new python_exception_handler(*this, type_handler_))
{
}

python_converter::~python_converter()
{
  delete string_builder_;
  delete tuple_handler_;
  delete dict_handler_;
  delete typechecker_;
  delete lambda_handler_;
  delete exception_handler_;
}

python_typechecking &python_converter::get_typechecker()
{
  return *typechecker_;
}

const python_typechecking &python_converter::get_typechecker() const
{
  return *typechecker_;
}

string_builder &python_converter::get_string_builder()
{
  if (!string_builder_)
  {
    string_builder_ = new string_builder(*this, &string_handler_);
    string_handler_.set_string_builder(string_builder_);
  }
  return *string_builder_;
}

static void add_global_static_variable(
  contextt &ctx,
  const typet t,
  const std::string &name)
{
  std::string id = "c:@" + name;
  symbolt symbol;
  symbol.mode = "C";
  symbol.type = std::move(t);
  symbol.name = name;
  symbol.id = id;

  symbol.lvalue = true;
  symbol.static_lifetime = true;
  symbol.is_extern = false;
  symbol.file_local = false;
  symbol.value = gen_zero(t, true);
  symbol.value.zero_initializer(true);

  symbolt *added_symbol = ctx.move_symbol_to_context(symbol);
  assert(added_symbol);
}

void python_converter::load_c_intrisics(code_blockt &)
{
  // Add symbols required by the C models
  // __ESBMC_rounding_mode is pulled in indirectly via fesetround in cprover_library.cpp

  auto type1 = array_typet(bool_type(), exprt("infinity"));
  add_global_static_variable(symbol_table_, type1, "__ESBMC_alloc");
  add_global_static_variable(symbol_table_, type1, "__ESBMC_is_dynamic");

  auto type2 = array_typet(size_type(), exprt("infinity"));
  add_global_static_variable(symbol_table_, type2, "__ESBMC_alloc_size");
}

///  Only addresses __name__; other Python built-ins such as
/// __file__, __doc__, __package__ are unsupported
void python_converter::create_builtin_symbols()
{
  // Create __name__ symbol
  symbol_id name_sid(current_python_file, "", "");
  name_sid.set_object("__name__");

  locationt location;
  location.set_file(current_python_file.c_str());
  location.set_line(1);

  std::string module_name =
    current_python_file.substr(0, current_python_file.find_last_of("."));

  // Determine the value of __name__ based on whether this is the main module or imported
  std::string name_value;
  if (current_python_file == main_python_file)
    name_value = "__main__";
  else
  {
    // Extract module name from filename (e.g., "/path/to/other.py" -> "other")
    size_t last_slash = current_python_file.find_last_of("/\\");
    size_t last_dot = current_python_file.find_last_of(".");
    if (
      last_slash != std::string::npos && last_dot != std::string::npos &&
      last_dot > last_slash)
    {
      name_value =
        current_python_file.substr(last_slash + 1, last_dot - last_slash - 1);
    }
    else if (last_dot != std::string::npos)
      name_value = current_python_file.substr(0, last_dot);
    else
      name_value = current_python_file;
  }

  typet string_type =
    type_handler_.build_array(char_type(), name_value.size() + 1);

  // Create the symbol
  symbolt name_symbol = create_symbol(
    module_name, "__name__", name_sid.to_string(), location, string_type);

  name_symbol.lvalue = true;
  name_symbol.static_lifetime = true;
  name_symbol.is_extern = false;
  name_symbol.file_local = false;

  // Set the value
  exprt name_expr = gen_zero(string_type);
  const typet &char_type_ref = string_type.subtype();

  for (size_t i = 0; i < name_value.size(); ++i)
  {
    uint8_t ch = name_value[i];
    exprt char_value = constant_exprt(
      integer2binary(BigInt(ch), bv_width(char_type_ref)),
      integer2string(BigInt(ch)),
      char_type_ref);
    name_expr.operands().at(i) = char_value;
  }

  // Add null terminator
  exprt null_char = constant_exprt(
    integer2binary(BigInt(0), bv_width(char_type_ref)),
    integer2string(BigInt(0)),
    char_type_ref);
  name_expr.operands().at(name_value.size()) = null_char;

  name_symbol.value = name_expr;

  // Add to symbol table
  symbol_table_.add(name_symbol);
}

bool python_converter::import_module_into_block(
  const nlohmann::json &import_node,
  module_locator &locator,
  code_blockt &block)
{
  const std::string &module_name = (import_node["_type"] == "ImportFrom")
                                     ? import_node["module"]
                                     : import_node["names"][0]["name"];

  if (imported_modules.find(module_name) != imported_modules.end())
    return true;

  std::ifstream imported_file = locator.open_module_file(module_name);
  if (!imported_file.is_open())
    return false;

  nlohmann::json nested_module_json;
  imported_file >> nested_module_json;

  current_python_file = nested_module_json["filename"].get<std::string>();
  imported_modules.emplace(module_name, current_python_file);

  // Process nested imports first.
  process_module_imports(nested_module_json, locator, block);

  // Then process this module's definitions.
  create_builtin_symbols();
  python_annotation<nlohmann::json> imported_annotator(
    nested_module_json, const_cast<global_scope &>(global_scope_));
  imported_annotator.add_type_annotation();

  exprt imported_code = with_ast(&nested_module_json, [&]() {
    return get_block(nested_module_json["body"]);
  });

  convert_expression_to_code(imported_code);

  // Add imported module code.
  block.copy_to_operands(imported_code);
  return true;
}

void python_converter::process_module_imports(
  const nlohmann::json &module_ast,
  module_locator &locator,
  code_blockt &block)
{
  // Process imports in this module first (depth-first)
  for (const auto &elem : module_ast["body"])
  {
    if (elem["_type"] == "ImportFrom" || elem["_type"] == "Import")
    {
      std::string saved_file = current_python_file;
      import_module_into_block(elem, locator, block);
      current_python_file = saved_file;
    }
  }
}

void python_converter::convert()
{
  main_python_file = (*ast_json)["filename"].get<std::string>();
  current_python_file = main_python_file;

  // Create built-in symbols for main module (__name__ = "__main__")
  create_builtin_symbols();

  // Block to accumulate model library code
  code_blockt models_block;

  if (!config.options.get_bool_option("no-library"))
  {
    // Load operational models
    const std::string &ast_output_dir =
      (*ast_json)["ast_output_dir"].get<std::string>();
    std::list<std::string> model_files = {
      "builtins",
      "range",
      "int",
      "consensus",
      "random",
      "exceptions",
      "datetime",
      "nondet"};
    std::list<std::string> model_folders = {"os", "numpy"};

    for (const auto &folder : model_folders)
    {
      append_models_from_directory(model_files, ast_output_dir + "/" + folder);
    }

    is_loading_models = true;

    for (const auto &file : model_files)
    {
      std::stringstream model_path;
      model_path << ast_output_dir << "/" << file << ".json";

      std::ifstream model_file(model_path.str());
      nlohmann::json model_json;
      model_file >> model_json;
      model_file.close();

      size_t pos = file.rfind("/");
      if (pos != std::string::npos)
      {
        std::string filename = file.substr(pos + 1);
        if (imported_modules.find(filename) != imported_modules.end())
          current_python_file = imported_modules[filename];
      }

      exprt model_code =
        with_ast(&model_json, [&]() { return get_block((*ast_json)["body"]); });

      convert_expression_to_code(model_code);

      // Accumulate model code
      models_block.copy_to_operands(model_code);
      current_python_file = main_python_file;
    }
    is_loading_models = false;
  }

  // Create a block to hold intrinsic assignments and load C intrinsics
  code_blockt intrinsic_block;
  load_c_intrisics(intrinsic_block);

  // Pre-register module-level variable symbols so class methods can reference
  // globals declared later in the file (Python LEGB rule).
  preregister_global_variables((*ast_json)["body"]);

  // Variables to hold user code and initialization code
  codet user_code;
  code_blockt init_code;

  // Handle --function option
  const std::string function = config.options.get_option("function");
  if (!function.empty())
  {
    /* If the user passes --function, we add only a call to the
     * respective function in __ESBMC_main instead of entire Python program
     */

    nlohmann::json function_node;
    // Find function node in AST
    for (const auto &element : (*ast_json)["body"])
    {
      if (element["_type"] == "FunctionDef" && element["name"] == function)
      {
        function_node = element;
        break;
      }
    }

    if (function_node.empty())
      throw std::runtime_error("Function " + function + " not found");

    code_blockt block;

    // Add intrinsic assignments first
    block.copy_to_operands(intrinsic_block);

    // Convert classes referenced by the function
    for (const auto &clazz : global_scope_.classes())
    {
      const auto &class_node = find_class((*ast_json)["body"], clazz);
      get_class_definition(class_node, block);
      current_class_name_.clear();
    }

    // Convert only the global variables referenced by the function
    for (const auto &global_var : global_scope_.variables())
    {
      const auto &var_node = find_var_decl(global_var, "", *ast_json);
      get_var_assign(var_node, block);
    }

    // Convert function arguments types
    for (const auto &arg : function_node["args"]["args"])
    {
      // Check if annotation exists and is not null before accessing "id"
      if (
        arg.contains("annotation") && !arg["annotation"].is_null() &&
        arg["annotation"].contains("id"))
      {
        auto node = find_class((*ast_json)["body"], arg["annotation"]["id"]);
        if (!node.empty())
          get_class_definition(node, block);
      }
    }

    // Convert a single function
    get_function_definition(function_node);

    // Get function symbol
    symbol_id sid = create_symbol_id();
    sid.set_function(function);
    symbolt *symbol = symbol_table_.find_symbol(sid.to_string());

    if (!symbol)
      throw std::runtime_error("Symbol " + sid.to_string() + " not found");

    // Create function call
    code_function_callt call;
    call.location() = symbol->location;
    call.function() = symbol_expr(*symbol);

    const code_typet::argumentst &arguments =
      to_code_type(symbol->type).arguments();

    // Function args are nondet values
    for (const code_typet::argumentt &arg : arguments)
    {
      exprt arg_value = exprt("sideeffect", arg.type());
      arg_value.statement("nondet");
      call.arguments().push_back(arg_value);
    }

    convert_expression_to_code(call);
    convert_expression_to_code(block);

    // Prepare user code: class definitions + function call
    code_blockt user_code_body;
    user_code_body.copy_to_operands(block);
    user_code_body.copy_to_operands(call);
    user_code.swap(user_code_body);

    // Add models to init code
    if (!models_block.operands().empty())
      init_code.copy_to_operands(models_block);
  }
  else
  {
    // Convert imported modules
    module_locator locator((*ast_json)["ast_output_dir"].get<std::string>());

    // Accumulate all imports
    code_blockt all_imports_block;

    for (const auto &elem : (*ast_json)["body"])
    {
      if (elem["_type"] == "ImportFrom" || elem["_type"] == "Import")
      {
        if (elem.value("module_not_found", false))
        {
          const std::string module_name = (elem["_type"] == "ImportFrom")
                                            ? elem["module"]
                                            : elem["names"][0]["name"];
          log_warning("skipping unresolvable import: {}", module_name);
          continue;
        }
        is_importing_module = true;
        if (!import_module_into_block(elem, locator, all_imports_block))
        {
          const std::string &module_name = (elem["_type"] == "ImportFrom")
                                             ? elem["module"]
                                             : elem["names"][0]["name"];
          throw std::runtime_error(
            "Cannot open file: " + locator.module_path(module_name));
        }
      }
    }

    // Do the same for imports that appear directly inside functions.
    for (const auto &elem : (*ast_json)["body"])
    {
      if (
        elem["_type"] != "FunctionDef" || !elem.contains("body") ||
        !elem["body"].is_array())
        continue;

      for (const auto &stmt : elem["body"])
      {
        if (stmt["_type"] != "ImportFrom" && stmt["_type"] != "Import")
          continue;

        is_importing_module = true;
        if (!import_module_into_block(stmt, locator, all_imports_block))
        {
          const std::string &module_name = (stmt["_type"] == "ImportFrom")
                                             ? stmt["module"]
                                             : stmt["names"][0]["name"];
          throw std::runtime_error(
            "Cannot open file: " + locator.module_path(module_name));
        }
      }
    }

    is_importing_module = false;
    current_python_file = main_python_file;

    // Convert main statements
    exprt main_block = get_block((*ast_json)["body"]);
    user_code = convert_expression_to_code(main_block);

    // Prepare initialization code: models + intrinsics + imports
    if (!models_block.operands().empty())
      init_code.copy_to_operands(models_block);
    init_code.copy_to_operands(intrinsic_block);
    if (!all_imports_block.operands().empty())
      init_code.copy_to_operands(all_imports_block);
  }

  /*
   * Create three-function architecture for coverage support (similar to Solidity Frontend):
   *
   * 1. python_init
   *    - Contains models, intrinsics, and imports initialization
   *    - Marked with __ESBMC_HIDE label to exclude from coverage statistics
   *    - Only created if there is initialization code
   *
   * 2. python_user_main
   *    - Contains only user code from the main module
   *    - This is what gets analyzed for branch/decision/assertion coverage
   *
   * 3. __ESBMC_main
   *    - Entry point for ESBMC verification
   *    - Initializes static lifetime variables
   *    - Calls python_init() if it exists
   *    - Calls python_user_main()
   *
   * This architecture ensures that coverage analysis only counts user code,
   * not initialization/library code, making Python behave consistently with C.
   */
  if (!init_code.operands().empty())
  {
    code_typet init_type;
    init_type.return_type() = empty_typet();

    symbolt init_symbol;
    init_symbol.id = "python_init";
    init_symbol.name = "python_init";
    init_symbol.type = init_type;
    init_symbol.lvalue = true;
    init_symbol.is_extern = false;
    init_symbol.file_local = false;
    init_symbol.location = get_location_from_decl(*ast_json);

    // Add __ESBMC_HIDE label to hide from coverage
    code_labelt esbmc_hide;
    esbmc_hide.set_label("__ESBMC_HIDE");
    esbmc_hide.code() = code_skipt();

    code_blockt init_body;
    init_body.copy_to_operands(esbmc_hide);
    init_body.copy_to_operands(init_code);
    init_symbol.value.swap(init_body);

    if (symbol_table_.move(init_symbol))
    {
      throw std::runtime_error("The python_init function is already defined");
    }
  }

  // Create python_user_main function containing only user code
  code_typet user_main_type;
  user_main_type.return_type() = empty_typet();

  symbolt user_main_symbol;
  user_main_symbol.id = "python_user_main";
  user_main_symbol.name = "python_user_main";
  user_main_symbol.type = user_main_type;
  user_main_symbol.lvalue = true;
  user_main_symbol.is_extern = false;
  user_main_symbol.file_local = false;
  user_main_symbol.location = get_location_from_decl(*ast_json);
  user_main_symbol.value = user_code;

  if (symbol_table_.move(user_main_symbol))
  {
    throw std::runtime_error(
      "The python_user_main function is already defined");
  }

  // Create __ESBMC_main that initializes and calls user code
  code_typet main_type;
  main_type.return_type() = empty_typet();

  symbolt main_symbol;
  main_symbol.id = "__ESBMC_main";
  main_symbol.name = "__ESBMC_main";
  main_symbol.type = main_type;
  main_symbol.lvalue = true;
  main_symbol.is_extern = false;
  main_symbol.file_local = false;
  main_symbol.location = get_location_from_decl(*ast_json);

  code_blockt main_body;

  // 1. Initialize static lifetime variables
  symbol_table_.foreach_operand_in_order([&main_body](const symbolt &s) {
    if (s.static_lifetime && !s.value.is_nil() && !s.type.is_code())
    {
      code_assignt assign(symbol_expr(s), s.value);
      assign.location() = s.location;
      main_body.copy_to_operands(assign);
    }
  });

  // 2. Call python_init for initialization
  if (!init_code.operands().empty())
  {
    const symbolt *init_sym = symbol_table_.find_symbol("python_init");
    if (init_sym)
    {
      code_function_callt init_call;
      init_call.function() = symbol_expr(*init_sym);
      main_body.copy_to_operands(init_call);
    }
  }

  // 3. Call python_user_main
  const symbolt *user_main_sym = symbol_table_.find_symbol("python_user_main");
  if (!user_main_sym)
  {
    throw std::runtime_error("python_user_main symbol not found after move");
  }

  code_function_callt user_main_call;
  user_main_call.function() = symbol_expr(*user_main_sym);
  main_body.copy_to_operands(user_main_call);

  main_symbol.value.swap(main_body);

  if (symbol_table_.move(main_symbol))
  {
    throw std::runtime_error(
      "The main function is already defined in another module");
  }
}

