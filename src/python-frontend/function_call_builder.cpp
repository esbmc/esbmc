#include <python-frontend/function_call_builder.h>
#include <python-frontend/function_call_expr.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/numpy_call_expr.h>
#include <python-frontend/python_list.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/string_builder.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/type_utils.h>
#include <util/arith_tools.h>
#include <util/message.h>

#include <boost/algorithm/string/predicate.hpp>
#include <climits>
#include <optional>

static size_t utf8_codepoint_count(const std::string &text)
{
  size_t count = 0;
  for (unsigned char c : text)
  {
    if ((c & 0xC0) != 0x80)
      ++count;
  }
  return count;
}

bool function_call_builder::is_nondet_str_call(const nlohmann::json &node) const
{
  return node.contains("_type") && node["_type"] == "Call" &&
         node.contains("func") && node["func"].contains("_type") &&
         node["func"]["_type"] == "Name" && node["func"].contains("id") &&
         node["func"]["id"] == "nondet_str";
}

bool function_call_builder::is_symbolic_string(const nlohmann::json &node) const
{
  if (is_nondet_str_call(node))
    return true;

  if (node.contains("_type") && node["_type"] == "Name" && node.contains("id"))
  {
    const std::string var_name = node["id"].get<std::string>();
    nlohmann::json var_value = json_utils::get_var_value(
      var_name, converter_.get_current_func_name(), converter_.get_ast_json());

    if (
      !var_value.empty() && var_value.contains("value") &&
      is_nondet_str_call(var_value["value"]))
      return true;
  }

  return false;
}

const std::string kGetObjectSize = "__ESBMC_get_object_size";
const std::string kStrlen = "strlen";
const std::string kEsbmcAssume = "__ESBMC_assume";
const std::string kVerifierAssume = "__VERIFIER_assume";
const std::string kLoopInvariant = "__loop_invariant";
const std::string kEsbmcLoopInvariant = "__ESBMC_loop_invariant";
const std::string kEsbmcCover = "__ESBMC_cover";

function_call_builder::function_call_builder(
  python_converter &converter,
  const nlohmann::json &call)
  : converter_(converter), call_(call)
{
}

bool function_call_builder::is_cover_call(const symbol_id &function_id) const
{
  const std::string &func_name = function_id.get_function();
  return (func_name == kEsbmcCover);
}

bool function_call_builder::is_numpy_call(const symbol_id &function_id) const
{
  if (type_utils::is_builtin_type(function_id.get_function()))
    return false;

  const std::string &filename = function_id.get_filename();

  return boost::algorithm::ends_with(filename, "/models/numpy.py") ||
         filename.find("/numpy/linalg") != std::string::npos;
}

bool function_call_builder::is_assume_call(const symbol_id &function_id) const
{
  const std::string &func_name = function_id.get_function();
  return (func_name == kEsbmcAssume || func_name == kVerifierAssume);
}

bool function_call_builder::is_len_call(const symbol_id &function_id) const
{
  const std::string &func_name = function_id.get_function();
  return func_name == kGetObjectSize || func_name == kStrlen;
}

symbol_id function_call_builder::build_function_id() const
{
  const std::string &python_file = converter_.python_file();
  const std::string &current_class_name = converter_.current_classname();
  const std::string &current_function_name = converter_.current_function_name();
  const auto &ast = converter_.ast();
  type_handler th(converter_);

  bool is_member_function_call = false;

  const auto &func_json = call_["func"];

  const std::string &func_type = func_json["_type"];

  std::string func_name, obj_name, class_name;

  symbol_id function_id(python_file, current_class_name, current_function_name);

  if (func_type == "Name")
  {
    func_name = func_json["id"];

    // Map Python loop invariant name to ESBMC internal name
    if (func_name == kLoopInvariant)
      func_name = kEsbmcLoopInvariant;
  }
  else if (func_type == "Attribute") // Handling obj_name.func_name() calls
  {
    is_member_function_call = true;
    func_name = func_json["attr"];

    // Get object name
    if (func_json["value"]["_type"] == "Attribute")
    {
      /* Handle nested attribute chains (e.g., self.f.foo(), a.b.c.method())
       *
       * When calling a method through an attribute chain, we need to determine
       * the class type of the intermediate object. For example, in self.f.foo(),
       * we need to know that 'f' has type Foo to correctly resolve Foo.foo().
       *
       * Strategy: Recursively walk the attribute chain from left to right,
       * resolving each component's type by looking up struct members in the
       * symbol table, until we reach the final object whose class we need.
       */

      std::function<typet(const nlohmann::json &)> resolve_attr_type =
        [&](const nlohmann::json &node) -> typet {
        if (node["_type"] == "Name")
        {
          // Base case: resolve variable to its type
          std::string name = node["id"].get<std::string>();

          // Skip module names - let the module handling logic deal with them
          if (converter_.is_imported_module(name))
            return typet();

          symbol_id var_sid(
            python_file, current_class_name, current_function_name);
          var_sid.set_object(name);
          symbolt *var_symbol = converter_.find_symbol(var_sid.to_string());
          return var_symbol ? var_symbol->type : typet();
        }
        else if (node["_type"] == "Attribute")
        {
          // Recursive case: resolve base, then look up member
          typet base_type = resolve_attr_type(node["value"]);
          if (base_type.id().empty())
            return typet();

          // Normalize type (dereference pointers, follow symbol types)
          if (base_type.is_pointer())
            base_type = base_type.subtype();
          if (base_type.id() == "symbol")
            base_type = converter_.ns.follow(base_type);

          // Look up the member in the struct
          if (base_type.is_struct())
          {
            const struct_typet &struct_type = to_struct_type(base_type);
            std::string attr = node["attr"].get<std::string>();
            return struct_type.has_component(attr)
                     ? struct_type.get_component(attr).type()
                     : typet();
          }
          return typet();
        }
        return typet();
      };

      // Resolve the full attribute chain type
      typet obj_type = resolve_attr_type(func_json["value"]);

      if (!obj_type.id().empty())
      {
        // Normalize the resolved type
        if (obj_type.is_pointer())
          obj_type = obj_type.subtype();
        if (obj_type.id() == "symbol")
          obj_type = converter_.ns.follow(obj_type);

        // Extract class name from struct type
        if (obj_type.is_struct())
        {
          std::string tag = to_struct_type(obj_type).tag().as_string();
          obj_name = (tag.find("tag-") == 0) ? tag.substr(4) : tag;
        }
        else
        {
          obj_name = th.type_to_string(obj_type);
        }
      }
      else
      {
        // Fallback: use the direct attribute name
        // For module.Class.method(), we want "Class" as the object name
        obj_name = func_json["value"]["attr"].get<std::string>();
      }
    }
    else if (
      func_json["value"]["_type"] == "Constant" &&
      func_json["value"]["value"].is_string())
    {
      obj_name = "str";
    }
    else if (func_json["value"]["_type"] == "BinOp")
    {
      std::string lhs_type = th.get_operand_type(func_json["value"]["left"]);
      std::string rhs_type = th.get_operand_type(func_json["value"]["right"]);

      assert(lhs_type == rhs_type);

      obj_name = lhs_type;
    }
    else if (func_json["value"]["_type"] == "Call")
    {
      obj_name = func_json["value"]["func"]["id"];
      if (obj_name == "nondet_str")
        obj_name = "str";
      if (obj_name == "super")
      {
        symbolt *base_class_func = converter_.find_function_in_base_classes(
          current_class_name, function_id.to_string(), func_name, false);
        if (base_class_func)
        {
          return symbol_id::from_string(base_class_func->id.as_string());
        }
      }
    }
    else
    {
      if (
        func_json["value"]["_type"] == "Name" &&
        func_json["value"].contains("id"))
        obj_name = func_json["value"]["id"];
      else
        obj_name = "str";
    }

    obj_name = json_utils::get_object_alias(ast, obj_name);

    if (
      !json_utils::is_class(obj_name, ast) &&
      converter_.is_imported_module(obj_name))
    {
      const auto &module_path = converter_.get_imported_module_path(obj_name);

      function_id =
        symbol_id(module_path, current_class_name, current_function_name);

      is_member_function_call = false;
    }
  }

  // build symbol_id
  if (func_name == "len")
  {
    const auto &arg = call_["args"][0];
    func_name = kStrlen;

    // Special case: single character string literals should return 1 directly
    if (arg["_type"] == "Constant" && arg["value"].is_string())
    {
      std::string str_val = arg["value"].get<std::string>();
      if (str_val.size() == 1)
      {
        // For single character strings, we'll handle this specially in build()
        // by returning 1 directly instead of calling a C function
        func_name = "__ESBMC_len_single_char";
        function_id.clear();
        function_id.set_prefix("esbmc:");
        function_id.set_function(func_name);
        return function_id;
      }
    }

    // Handle len(range(...))
    if (arg["_type"] == "Call")
    {
      if (
        arg.contains("func") && arg["func"].contains("id") &&
        arg["func"]["id"] == "range")
        func_name = kGetObjectSize;
      else if (
        arg.contains("func") && arg["func"].contains("id") &&
        arg["func"]["id"] == "set")
        func_name = kGetObjectSize;
    }
    else if (arg["_type"] == "List")
      func_name = kGetObjectSize;
    else if (arg["_type"] == "Name")
    {
      const std::string &var_type = th.get_var_type(arg["id"]);
      symbol_id var_sid(python_file, current_class_name, current_function_name);
      var_sid.set_object(arg["id"].get<std::string>());
      symbolt *var_symbol = converter_.find_symbol(var_sid.to_string());

      // Check if this is a tuple by looking up the variable's type
      if (var_type == "tuple" || var_type.empty())
      {
        if (var_symbol && var_symbol->type.id() == "struct")
        {
          const struct_typet &struct_type = to_struct_type(var_symbol->type);

          // Check if this is a tuple by examining the tag
          if (struct_type.tag().as_string().find("tag-tuple") == 0)
          {
            // Mark this as a tuple len() call
            func_name = "__ESBMC_len_tuple";
            function_id.clear();
            function_id.set_prefix("esbmc:");
            function_id.set_function(func_name);
            return function_id;
          }
        }
      }
      if (var_type == "dict")
      {
        // Mark this as a dict len() call
        func_name = "__ESBMC_len_dict";
        function_id.clear();
        function_id.set_prefix("esbmc:");
        function_id.set_function(func_name);
        return function_id;
      }
      else if (
        var_type == "bytes" || var_type == "list" || var_type == "List" ||
        var_type == "set" || var_type == "Sequence" || var_type == "range")
      {
        func_name = kGetObjectSize;
      }
      else if (var_type.empty() && var_symbol)
      {
        typet actual_type = var_symbol->type;
        if (actual_type.is_pointer())
          actual_type = actual_type.subtype();
        if (actual_type.id() == "symbol")
          actual_type = converter_.ns.follow(actual_type);

        if (actual_type.is_struct())
        {
          const struct_typet &struct_type = to_struct_type(actual_type);
          std::string tag = struct_type.tag().as_string();
          if (tag.find("__ESBMC_PyListObj") != std::string::npos)
            func_name = kGetObjectSize;
        }
      }
      else if (var_type == "str" || var_type.empty())
      {
        // For string types (including optional strings), always use strlen
        // This handles both str and Optional[str] (pointer to array) cases
        func_name = kStrlen;

        // Check if this is a single character by looking up the variable
        if (
          var_symbol && var_symbol->value.is_constant() &&
          (var_symbol->value.type().is_unsignedbv() ||
           var_symbol->value.type().is_signedbv()))
        {
          // This is a single character variable
          func_name = "__ESBMC_len_single_char";
          function_id.clear();
          function_id.set_prefix("esbmc:");
          function_id.set_function(func_name);
          return function_id;
        }
      }
    }
    function_id.clear();
    function_id.set_prefix("c:");
  }
  else if (type_utils::is_builtin_type(obj_name))
  {
    class_name = obj_name;
    function_id = symbol_id(python_file, class_name, func_name);
  }
  else if (is_assume_call(function_id))
  {
    function_id.clear();
  }
  else if (is_cover_call(function_id))
  {
    function_id.clear();
  }

  // Insert class name in the symbol id
  if (obj_name == "super")
  {
    class_name = current_class_name;
  }
  else if (th.is_constructor_call(call_))
  {
    class_name = func_name;
  }
  else if (is_member_function_call)
  {
    if (
      type_utils::is_builtin_type(obj_name) ||
      json_utils::is_class(obj_name, ast))
    {
      class_name = obj_name;
    }
    else
    {
      // Look up variable type from symbol table instead of AST
      symbol_id var_sid(python_file, current_class_name, current_function_name);
      var_sid.set_object(obj_name);
      symbolt *var_symbol = converter_.find_symbol(var_sid.to_string());

      if (!var_symbol)
        throw std::runtime_error("Variable " + obj_name + " not found");

      // Extract class name from the type, following symbol references
      typet var_type = var_symbol->type.is_pointer()
                         ? var_symbol->type.subtype()
                         : var_symbol->type;

      // Follow symbol type references using the converter's namespace
      var_type = converter_.ns.follow(var_type);

      if (var_type.is_struct())
      {
        const struct_typet &struct_type = to_struct_type(var_type);
        class_name = struct_type.tag().as_string();
      }
      else
        class_name = th.type_to_string(var_type);
    }
  }

  if (!class_name.empty())
  {
    function_id.set_class(class_name);
  }

  function_id.set_function(func_name);

  // Check if this is a nested function call
  if (func_type == "Name" && !current_function_name.empty())
  {
    // Self-recursive call (e.g., bar() inside foo@F@bar)
    if (current_function_name.ends_with("@F@" + func_name))
    {
      function_id.set_function(current_function_name);
      return function_id;
    }

    // Walk nesting chain from deepest to shallowest
    for (std::string ctx = current_function_name; !ctx.empty();)
    {
      std::string nested_id = ctx + "@F@" + func_name;
      symbol_id sid(python_file, current_class_name, nested_id);

      if (converter_.symbol_table().find_symbol(sid.to_string()))
      {
        function_id.set_function(nested_id);
        return function_id;
      }

      size_t pos = ctx.rfind("@F@");
      ctx = (pos != std::string::npos) ? ctx.substr(0, pos) : "";
    }
  }

  return function_id;
}

exprt function_call_builder::build() const
{
  symbol_id function_id = build_function_id();

  if (is_len_call(function_id) && !call_["args"].empty())
  {
    auto const_string_len_from_symbol =
      [this](const std::string &name) -> std::optional<BigInt> {
      // Be conservative with named symbols; only __name__ is known constant.
      if (name != "__name__")
        return std::nullopt;

      std::string name_value;
      if (converter_.python_file() == converter_.main_python_filename())
        name_value = "__main__";
      else
      {
        const std::string &file = converter_.python_file();
        size_t last_slash = file.find_last_of("/\\");
        size_t last_dot = file.find_last_of(".");
        if (
          last_slash != std::string::npos && last_dot != std::string::npos &&
          last_dot > last_slash)
        {
          name_value = file.substr(last_slash + 1, last_dot - last_slash - 1);
        }
        else if (last_dot != std::string::npos)
          name_value = file.substr(0, last_dot);
        else
          name_value = file;
      }

      return BigInt(name_value.size());
    };

    auto joinedstr_len =
      [&const_string_len_from_symbol](
        const nlohmann::json &joined) -> std::optional<BigInt> {
      if (!joined.contains("values") || !joined["values"].is_array())
        return std::nullopt;

      BigInt total = BigInt(0);

      for (const auto &part : joined["values"])
      {
        if (
          part["_type"] == "Constant" && part.contains("value") &&
          part["value"].is_string())
        {
          const std::string text = part["value"].get<std::string>();
          total += BigInt(utf8_codepoint_count(text));
          continue;
        }

        if (part["_type"] == "FormattedValue" && part.contains("value"))
        {
          const auto &value = part["value"];
          if (value["_type"] == "Name" && value.contains("id"))
          {
            if (
              auto len =
                const_string_len_from_symbol(value["id"].get<std::string>()))
            {
              total += *len;
              continue;
            }
          }

          // Cannot prove constant length
          return std::nullopt;
        }

        // Unsupported part type for constant length folding
        return std::nullopt;
      }

      return total;
    };

    // Fast path for literal strings: len("...") is known at compile time.
    if (
      call_["args"][0].contains("_type") &&
      call_["args"][0]["_type"] == "Constant" &&
      call_["args"][0].contains("value") &&
      call_["args"][0]["value"].is_string())
    {
      const std::string text = call_["args"][0]["value"].get<std::string>();
      return from_integer(BigInt(utf8_codepoint_count(text)), size_type());
    }

    exprt arg_expr = converter_.get_expr(call_["args"][0]);

    if (arg_expr.type().is_signedbv() || arg_expr.type().is_unsignedbv())
      return from_integer(1, long_long_int_type());

    // If the argument is a named variable initialized with a constant string
    // or f-string, compute its length from the initializer to avoid strlen.
    if (
      call_["args"][0].contains("_type") &&
      call_["args"][0]["_type"] == "Name" && call_["args"][0].contains("id"))
    {
      const std::string var_name = call_["args"][0]["id"].get<std::string>();
      bool has_augassign = false;
      auto count_assignments = [&](
                                 const nlohmann::json &node,
                                 const std::string &name,
                                 auto &&self) -> int {
        int count = 0;
        if (!node.is_object() && !node.is_array())
          return 0;

        if (node.is_object() && node.contains("_type"))
        {
          const std::string type = node["_type"].get<std::string>();
          if (type == "Assign" && node.contains("targets"))
          {
            for (const auto &tgt : node["targets"])
            {
              if (
                tgt.contains("_type") && tgt["_type"] == "Name" &&
                tgt.contains("id") && tgt["id"] == name)
                count++;
            }
          }
          else if (type == "AnnAssign" && node.contains("target"))
          {
            const auto &tgt = node["target"];
            if (
              tgt.contains("_type") && tgt["_type"] == "Name" &&
              tgt.contains("id") && tgt["id"] == name)
              count++;
          }
          else if (type == "AugAssign" && node.contains("target"))
          {
            const auto &tgt = node["target"];
            if (
              tgt.contains("_type") && tgt["_type"] == "Name" &&
              tgt.contains("id") && tgt["id"] == name)
            {
              has_augassign = true;
              count++;
            }
          }
        }

        if (node.is_array())
        {
          for (const auto &elem : node)
            count += self(elem, name, self);
        }
        else if (node.is_object())
        {
          for (const auto &item : node.items())
            count += self(item.value(), name, self);
        }

        return count;
      };

      int assign_count = 0;
      const nlohmann::json &ast = converter_.get_ast_json();
      if (converter_.get_current_func_name().empty())
        assign_count =
          count_assignments(ast["body"], var_name, count_assignments);
      else
      {
        std::vector<std::string> function_path =
          json_utils::split_function_path(converter_.get_current_func_name());
        nlohmann::json func_node =
          json_utils::find_function_by_path(ast, function_path);
        if (!func_node.empty() && func_node.contains("body"))
          assign_count =
            count_assignments(func_node["body"], var_name, count_assignments);
      }

      // Only fold len() for variables assigned exactly once and never augmented.
      if (assign_count == 1 && !has_augassign)
      {
        nlohmann::json var_value = json_utils::get_var_value(
          var_name,
          converter_.get_current_func_name(),
          converter_.get_ast_json());

        if (!var_value.empty() && var_value.contains("value"))
        {
          if (
            var_value["value"].contains("_type") &&
            var_value["value"]["_type"] == "JoinedStr")
          {
            if (auto len = joinedstr_len(var_value["value"]))
              return from_integer(*len, size_type());
          }
          else if (
            var_value["value"].contains("_type") &&
            var_value["value"]["_type"] == "Constant" &&
            var_value["value"].contains("value") &&
            var_value["value"]["value"].is_string())
          {
            const std::string text =
              var_value["value"]["value"].get<std::string>();
            return from_integer(BigInt(utf8_codepoint_count(text)), size_type());
          }
        }
      }
    }

    // If this is a symbol with a constant string array value, compute length
    // directly from the array size to avoid strlen unwinding.
    if (arg_expr.is_symbol())
    {
      symbolt *arg_symbol = converter_.find_symbol(
        to_symbol_expr(arg_expr).get_identifier().as_string());
      if (
        arg_symbol && arg_symbol->value.is_not_nil() &&
        arg_symbol->value.type().is_array() && arg_symbol->value.is_constant())
      {
        const array_typet &arr_type = to_array_type(arg_symbol->value.type());
        if (
          type_utils::is_char_type(arr_type.subtype()) &&
          arr_type.size().is_constant())
        {
          BigInt sz;
          if (!to_integer(arr_type.size(), sz) && sz > 0)
            return from_integer(sz - 1, size_type());
        }
      }
    }

    // If this is a fixed-size char array, compute length at compile time
    // to avoid strlen unwinding.
    typet actual_type = arg_expr.type();
    if (actual_type.is_pointer())
      actual_type = actual_type.subtype();
    if (actual_type.id() == "symbol")
      actual_type = converter_.ns.follow(actual_type);

    if (actual_type.id() == "array")
    {
      const array_typet &arr_type = to_array_type(actual_type);
      if (
        type_utils::is_char_type(arr_type.subtype()) &&
        arr_type.size().is_constant())
      {
        BigInt sz;
        if (!to_integer(arr_type.size(), sz) && sz > 0)
          return from_integer(sz - 1, size_type());
      }
    }
  }

  // Special handling for single character len() calls
  if (function_id.get_function() == "__ESBMC_len_single_char")
    return from_integer(1, long_long_int_type());

  if (function_id.get_function() == "__ESBMC_len_tuple")
  {
    const auto &arg = call_["args"][0];
    exprt obj_expr = converter_.get_expr(arg);

    if (obj_expr.type().id() == "struct")
    {
      const struct_typet &struct_type = to_struct_type(obj_expr.type());
      size_t tuple_len = struct_type.components().size();
      return from_integer(tuple_len, size_type());
    }

    // Fallback
    return from_integer(0, size_type());
  }

  if (function_id.get_function() == "__ESBMC_len_dict")
  {
    const auto &arg = call_["args"][0];
    exprt obj_expr = converter_.get_expr(arg);

    // Check actual type: could be dict or list (e.g., from d.keys())
    typet actual_type = obj_expr.type();
    if (actual_type.is_pointer())
      actual_type = actual_type.subtype();
    if (actual_type.id() == "symbol")
      actual_type = converter_.ns.follow(actual_type);

    // If it's actually a list, call list_size directly
    if (actual_type.is_struct())
    {
      const struct_typet &struct_type = to_struct_type(actual_type);
      std::string tag = struct_type.tag().as_string();
      if (tag.find("__ESBMC_PyListObj") != std::string::npos)
      {
        // It's a list, not a dict: call list_size on it directly
        code_typet list_size_type;
        list_size_type.return_type() = size_type();
        code_typet::argumentt arg_type;
        arg_type.type() = pointer_typet(struct_type);
        list_size_type.arguments().push_back(arg_type);

        symbol_exprt list_size_func("c:@F@__ESBMC_list_size", list_size_type);

        side_effect_expr_function_callt call_expr(size_type());
        call_expr.function() = list_size_func;
        call_expr.arguments().push_back(obj_expr);

        return call_expr;
      }
    }

    // It's genuinely a dict: get the keys member
    typet keys_type = pointer_typet(struct_typet());
    member_exprt keys_member(obj_expr, "keys", keys_type);

    // Create the list_get_size function symbol
    code_typet list_get_size_type;
    list_get_size_type.return_type() = size_type();
    code_typet::argumentt arg_type;
    arg_type.type() = keys_type;
    list_get_size_type.arguments().push_back(arg_type);

    symbol_exprt list_get_size_func(
      "c:@F@__ESBMC_list_size", list_get_size_type);

    // Create the function call
    side_effect_expr_function_callt call_expr(size_type());
    call_expr.function() = list_get_size_func;
    call_expr.arguments().push_back(keys_member);

    return call_expr;
  }

  // Special handling for assume calls: convert to code_assume instead of function call
  if (is_assume_call(function_id))
  {
    if (call_["args"].empty())
      throw std::runtime_error("__ESBMC_assume requires one boolean argument");

    exprt condition = converter_.get_expr(call_["args"][0]);
    if (!condition.type().is_bool())
      condition = typecast_exprt(condition, bool_type());

    // Create code_assume statement
    codet assume_code("assume");
    assume_code.copy_to_operands(condition);
    assume_code.location() = converter_.get_location_from_decl(call_);

    return assume_code;
  }

  // cover calls convert to code_assert statement
  // cover semantics: cover(cond) behaves as assert(!cond)
  // - failure (counterexample) means the condition is satisfiable
  // - success (proof) means the condition is not satisfiable
  if (is_cover_call(function_id))
  {
    if (call_["args"].empty())
      throw std::runtime_error("__ESBMC_cover requires one boolean argument");

    exprt condition = converter_.get_expr(call_["args"][0]);

    // Negate the condition: cover(cond) = assert(!cond)
    exprt negated_condition = gen_not(condition);

    // Create code_assert statement with cover property
    code_assertt cover_code(negated_condition);
    locationt loc = converter_.get_location_from_decl(call_);
    cover_code.location() = loc;

    return cover_code;
  }

  if (call_["func"]["_type"] == "Attribute")
  {
    std::string method_name = call_["func"]["attr"].get<std::string>();

    if (method_name == "startswith")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (call_["args"].size() != 1)
        throw std::runtime_error("startswith() requires exactly one argument");

      exprt prefix_arg = converter_.get_expr(call_["args"][0]);
      locationt loc = converter_.get_location_from_decl(call_);

      return converter_.get_string_handler().handle_string_startswith(
        obj_expr, prefix_arg, loc);
    }

    if (method_name == "capitalize")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);
      if (!call_["args"].empty())
        throw std::runtime_error("capitalize() takes no arguments");

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_capitalize(
        obj_expr, loc);
    }

    if (method_name == "title")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);
      if (!call_["args"].empty())
        throw std::runtime_error("title() takes no arguments");

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_title(obj_expr, loc);
    }

    if (method_name == "swapcase")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);
      if (!call_["args"].empty())
        throw std::runtime_error("swapcase() takes no arguments");

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_swapcase(
        obj_expr, loc);
    }

    if (method_name == "casefold")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);
      if (!call_["args"].empty())
        throw std::runtime_error("casefold() takes no arguments");

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_casefold(
        obj_expr, loc);
    }

    if (method_name == "endswith")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (call_["args"].size() != 1)
        throw std::runtime_error("endswith() requires exactly one argument");

      exprt suffix_arg = converter_.get_expr(call_["args"][0]);
      locationt loc = converter_.get_location_from_decl(call_);

      return converter_.get_string_handler().handle_string_endswith(
        obj_expr, suffix_arg, loc);
    }

    if (method_name == "removeprefix")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (call_["args"].size() != 1)
        throw std::runtime_error("removeprefix() requires one argument");

      exprt prefix_arg = converter_.get_expr(call_["args"][0]);
      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_removeprefix(
        obj_expr, prefix_arg, loc);
    }

    if (method_name == "removesuffix")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (call_["args"].size() != 1)
        throw std::runtime_error("removesuffix() requires one argument");

      exprt suffix_arg = converter_.get_expr(call_["args"][0]);
      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_removesuffix(
        obj_expr, suffix_arg, loc);
    }

    if (method_name == "isdigit")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (!call_["args"].empty())
        throw std::runtime_error("isdigit() takes no arguments");

      locationt loc = converter_.get_location_from_decl(call_);

      return converter_.get_string_handler().handle_string_isdigit(
        obj_expr, loc);
    }

    if (method_name == "isalnum")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (!call_["args"].empty())
        throw std::runtime_error("isalnum() takes no arguments");

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_isalnum(
        obj_expr, loc);
    }

    if (method_name == "isupper")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (!call_["args"].empty())
        throw std::runtime_error("isupper() takes no arguments");

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_isupper(
        obj_expr, loc);
    }

    if (method_name == "isnumeric")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (!call_["args"].empty())
        throw std::runtime_error("isnumeric() takes no arguments");

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_isnumeric(
        obj_expr, loc);
    }

    if (method_name == "isidentifier")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (!call_["args"].empty())
        throw std::runtime_error("isidentifier() takes no arguments");

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_isidentifier(
        obj_expr, loc);
    }

    if (method_name == "islower")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);
      if (!call_["args"].empty())
        throw std::runtime_error("islower() takes no arguments");

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_islower(
        obj_expr, loc);
    }

    if (method_name == "lower")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);
      if (!call_["args"].empty())
        throw std::runtime_error("lower() takes no arguments");

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_lower(obj_expr, loc);
    }

    if (method_name == "rfind")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (call_["args"].size() < 1 || call_["args"].size() > 3)
        throw std::runtime_error("rfind() requires one to three arguments");

      exprt find_arg = converter_.get_expr(call_["args"][0]);
      locationt loc = converter_.get_location_from_decl(call_);

      if (call_["args"].size() == 1)
      {
        return converter_.get_string_handler().handle_string_rfind(
          obj_expr, find_arg, loc);
      }

      exprt start_arg = converter_.get_expr(call_["args"][1]);
      exprt end_arg = from_integer(INT_MIN, int_type());
      if (call_["args"].size() == 3)
        end_arg = converter_.get_expr(call_["args"][2]);

      return converter_.get_string_handler().handle_string_rfind_range(
        obj_expr, find_arg, start_arg, end_arg, loc);
    }
    if (method_name == "upper")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);
      if (!call_["args"].empty())
        throw std::runtime_error("upper() takes no arguments");

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_upper(obj_expr, loc);
    }
    if (method_name == "index")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (call_["args"].size() < 1 || call_["args"].size() > 3)
        throw std::runtime_error("index() requires one to three arguments");

      exprt find_arg = converter_.get_expr(call_["args"][0]);
      locationt loc = converter_.get_location_from_decl(call_);

      if (call_["args"].size() == 1)
      {
        return converter_.get_string_handler().handle_string_index(
          call_, obj_expr, find_arg, loc);
      }

      exprt start_arg = converter_.get_expr(call_["args"][1]);
      exprt end_arg = from_integer(INT_MIN, int_type());
      if (call_["args"].size() == 3)
        end_arg = converter_.get_expr(call_["args"][2]);

      return converter_.get_string_handler().handle_string_index_range(
        call_, obj_expr, find_arg, start_arg, end_arg, loc);
    }

    if (method_name == "find")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (call_["args"].size() < 1 || call_["args"].size() > 3)
        throw std::runtime_error("find() requires one to three arguments");

      exprt find_arg = converter_.get_expr(call_["args"][0]);
      locationt loc = converter_.get_location_from_decl(call_);

      if (call_["args"].size() == 1)
      {
        return converter_.get_string_handler().handle_string_find(
          obj_expr, find_arg, loc);
      }

      exprt start_arg = converter_.get_expr(call_["args"][1]);
      exprt end_arg = from_integer(INT_MIN, int_type());
      if (call_["args"].size() == 3)
        end_arg = converter_.get_expr(call_["args"][2]);

      return converter_.get_string_handler().handle_string_find_range(
        obj_expr, find_arg, start_arg, end_arg, loc);
    }

    if (method_name == "isalpha")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);
      if (!call_["args"].empty())
        throw std::runtime_error("isalpha() takes no arguments");

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_isalpha(
        obj_expr, loc);
    }

    if (method_name == "isspace")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);
      if (!call_["args"].empty())
        throw std::runtime_error("isspace() takes no arguments");

      locationt loc = converter_.get_location_from_decl(call_);

      // Check if this is a single character (from iteration) or a string
      if (obj_expr.type().is_unsignedbv() || obj_expr.type().is_signedbv())
      {
        // Single character - use C's isspace function
        return converter_.get_string_handler().handle_char_isspace(
          obj_expr, loc);
      }
      else
      {
        // String variable - use the string version
        return converter_.get_string_handler().handle_string_isspace(
          obj_expr, loc);
      }
    }

    if (method_name == "lstrip")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      // lstrip() takes optional chars argument
      exprt chars_arg = nil_exprt();
      if (!call_["args"].empty())
        chars_arg = converter_.get_expr(call_["args"][0]);

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_lstrip(
        obj_expr, chars_arg, loc);
    }

    if (method_name == "rstrip")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      // rstrip() takes optional chars argument
      exprt chars_arg = nil_exprt();
      if (!call_["args"].empty())
        chars_arg = converter_.get_expr(call_["args"][0]);

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_rstrip(
        obj_expr, chars_arg, loc);
    }

    if (method_name == "strip")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      // strip() takes optional chars argument
      exprt chars_arg = nil_exprt();
      if (!call_["args"].empty())
        chars_arg = converter_.get_expr(call_["args"][0]);

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_strip(
        obj_expr, chars_arg, loc);
    }

    if (method_name == "center")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (call_["args"].size() < 1 || call_["args"].size() > 2)
        throw std::runtime_error("center() requires one or two arguments");

      exprt width_arg = converter_.get_expr(call_["args"][0]);
      exprt fill_arg = nil_exprt();
      if (call_["args"].size() == 2)
        fill_arg = converter_.get_expr(call_["args"][1]);

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_center(
        obj_expr, width_arg, fill_arg, loc);
    }

    if (method_name == "ljust")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (call_["args"].size() < 1 || call_["args"].size() > 2)
        throw std::runtime_error("ljust() requires one or two arguments");

      exprt width_arg = converter_.get_expr(call_["args"][0]);
      exprt fill_arg = nil_exprt();
      if (call_["args"].size() == 2)
        fill_arg = converter_.get_expr(call_["args"][1]);

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_ljust(
        obj_expr, width_arg, fill_arg, loc);
    }

    if (method_name == "rjust")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (call_["args"].size() < 1 || call_["args"].size() > 2)
        throw std::runtime_error("rjust() requires one or two arguments");

      exprt width_arg = converter_.get_expr(call_["args"][0]);
      exprt fill_arg = nil_exprt();
      if (call_["args"].size() == 2)
        fill_arg = converter_.get_expr(call_["args"][1]);

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_rjust(
        obj_expr, width_arg, fill_arg, loc);
    }

    if (method_name == "zfill")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (call_["args"].size() != 1)
        throw std::runtime_error("zfill() requires one argument");

      exprt width_arg = converter_.get_expr(call_["args"][0]);
      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_zfill(
        obj_expr, width_arg, loc);
    }

    if (method_name == "expandtabs")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (call_["args"].size() > 1)
        throw std::runtime_error("expandtabs() takes zero or one argument");

      exprt tabsize_arg = nil_exprt();
      if (call_["args"].size() == 1)
        tabsize_arg = converter_.get_expr(call_["args"][0]);

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_expandtabs(
        obj_expr, tabsize_arg, loc);
    }

    if (method_name == "replace")
    {
      if (call_["args"].size() != 2 && call_["args"].size() != 3)
        throw std::runtime_error(
          "replace() requires two or three arguments in minimal support");

      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);
      exprt old_arg = converter_.get_expr(call_["args"][0]);
      exprt new_arg = converter_.get_expr(call_["args"][1]);

      exprt count_expr = from_integer(-1, int_type());
      if (call_["args"].size() == 3)
      {
        long long count_value = 0;
        if (!json_utils::extract_constant_integer(
              call_["args"][2],
              converter_.get_current_func_name(),
              converter_.get_ast_json(),
              count_value))
        {
          throw std::runtime_error(
            "replace() only supports constant count in minimal support");
        }
        count_expr = from_integer(count_value, int_type());
      }

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_replace(
        obj_expr, old_arg, new_arg, count_expr, loc);
    }

    if (method_name == "count")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (call_["args"].size() < 1 || call_["args"].size() > 3)
        throw std::runtime_error("count() requires one to three arguments");

      exprt sub_arg = converter_.get_expr(call_["args"][0]);
      exprt start_arg = nil_exprt();
      exprt end_arg = nil_exprt();
      if (call_["args"].size() >= 2)
        start_arg = converter_.get_expr(call_["args"][1]);
      if (call_["args"].size() == 3)
        end_arg = converter_.get_expr(call_["args"][2]);

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_count(
        obj_expr, sub_arg, start_arg, end_arg, loc);
    }

    if (method_name == "splitlines")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (!call_["args"].empty())
        throw std::runtime_error("splitlines() takes no arguments");

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_splitlines(
        call_, obj_expr, loc);
    }

    if (method_name == "partition")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (call_["args"].size() != 1)
        throw std::runtime_error("partition() requires one argument");

      exprt sep_arg = converter_.get_expr(call_["args"][0]);
      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_partition(
        obj_expr, sep_arg, loc);
    }

    if (method_name == "format")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);
      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_format(
        call_, obj_expr, loc);
    }

    if (method_name == "format_map")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);
      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_format_map(
        call_, obj_expr, loc);
    }
    if (method_name == "split")
    {
      if (call_["args"].size() > 2)
        throw std::runtime_error(
          "split() requires zero, one, or two arguments in minimal support");

      auto is_none_literal = [](const nlohmann::json &node) {
        if (
          node.contains("_type") && node["_type"] == "Constant" &&
          node.contains("value") && node["value"].is_null())
        {
          return true;
        }
        if (
          node.contains("_type") && node["_type"] == "Name" &&
          node.contains("id") && node["id"].is_string() && node["id"] == "None")
        {
          return true;
        }
        return false;
      };

      auto find_keyword =
        [&](const std::string &name) -> const nlohmann::json * {
        if (!call_.contains("keywords") || !call_["keywords"].is_array())
          return nullptr;
        for (const auto &kw : call_["keywords"])
        {
          if (
            kw.contains("arg") && kw["arg"].is_string() && kw["arg"] == name &&
            kw.contains("value"))
          {
            return &kw["value"];
          }
        }
        return nullptr;
      };

      std::string separator;
      if (call_["args"].empty())
      {
        const nlohmann::json *sep_kw = find_keyword("sep");
        if (sep_kw == nullptr || is_none_literal(*sep_kw))
          separator = "";
        else if (!string_handler::extract_constant_string(
                   *sep_kw, converter_, separator))
        {
          // If separator is not constant, use empty string (whitespace split)
          separator = "";
        }
      }
      else if (is_none_literal(call_["args"][0]))
      {
        separator = "";
      }
      else
      {
        if (!string_handler::extract_constant_string(
              call_["args"][0], converter_, separator))
        {
          // If separator is not constant, use empty string (whitespace split)
          separator = "";
        }
      }

      long long count = -1;
      const nlohmann::json *count_node = nullptr;
      if (call_["args"].size() == 2)
      {
        count_node = &call_["args"][1];
      }
      else
      {
        const nlohmann::json *count_kw = find_keyword("maxsplit");
        if (count_kw != nullptr)
          count_node = count_kw;
      }

      if (count_node != nullptr)
      {
        if (!json_utils::extract_constant_integer(
              *count_node,
              converter_.get_current_func_name(),
              converter_.get_ast_json(),
              count))
        {
          // If count is not constant, use -1 (split all)
          count = -1;
        }
      }

      std::string input;
      if (!string_handler::extract_constant_string(
            call_["func"]["value"], converter_, input))
      {
        // For symbolic strings, we need to handle split() differently
        // For now, we'll try to extract the expression and work with it
        exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

        // Build split result with symbolic string
        return python_list::build_split_list(
          converter_, call_, obj_expr, separator, count);
      }

      return python_list::build_split_list(
        converter_, call_, input, separator, count);
    }
  }

  // Add len function to symbol table
  if (is_len_call(function_id))
  {
    const auto &symbol_table = converter_.symbol_table();
    const std::string &func_symbol_id = function_id.to_string();

    if (symbol_table.find_symbol(func_symbol_id.c_str()) == nullptr)
    {
      code_typet code_type;
      code_type.return_type() = long_long_int_type();
      code_type.arguments().push_back(pointer_typet(empty_typet()));

      const std::string &python_file = converter_.python_file();
      const std::string &func_name = function_id.get_function();
      locationt location = converter_.get_location_from_decl(call_);

      symbolt symbol = converter_.create_symbol(
        python_file, func_name, func_symbol_id, location, code_type);

      converter_.add_symbol(symbol);
    }
  }

  // Add loop invariant symbol to symbol table
  if (function_id.get_function() == kEsbmcLoopInvariant)
  {
    const auto &symbol_table = converter_.symbol_table();
    const std::string &func_symbol_id = function_id.to_string();

    if (symbol_table.find_symbol(func_symbol_id.c_str()) == nullptr)
    {
      code_typet code_type;
      code_type.return_type() = empty_typet();
      code_typet::argumentt arg;
      arg.type() = bool_type();
      code_type.arguments().push_back(arg);

      const std::string &python_file = converter_.python_file();
      const std::string &func_name = function_id.get_function();
      locationt location = converter_.get_location_from_decl(call_);

      symbolt symbol = converter_.create_symbol(
        python_file, func_name, func_symbol_id, location, code_type);

      converter_.add_symbol(symbol);
    }
  }

  // Handle NumPy functions
  if (is_numpy_call(function_id))
  {
    // Adjust the function ID when reusing functions from the C models
    if (type_utils::is_c_model_func(function_id.get_function()))
    {
      function_id.set_prefix("c:");
      function_id.set_filename("");
    }

    numpy_call_expr numpy_call(function_id, call_, converter_);
    return numpy_call.get();
  }

  function_call_expr call_expr(function_id, call_, converter_);
  return call_expr.get();
}
