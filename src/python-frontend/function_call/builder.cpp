#include <python-frontend/function_call/builder.h>
#include <python-frontend/function_call/expr.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/numpy_call_expr.h>
#include <python-frontend/python_list.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/type_utils.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/message.h>
#include <util/std_expr.h>
#include <python-frontend/python_expr_builder.h>

#include <boost/algorithm/string/predicate.hpp>
#include <optional>

using namespace python_expr;

static typet normalize_pylist_candidate_type(typet type, const namespacet &ns)
{
  if (type.is_pointer())
    type = type.subtype();
  if (type.id() == "symbol")
    type = ns.follow(type);
  return type;
}

static std::optional<struct_typet>
try_get_pylist_struct_type(const typet &type, const namespacet &ns)
{
  typet actual_type = normalize_pylist_candidate_type(type, ns);
  if (!actual_type.is_struct())
    return std::nullopt;

  const struct_typet &struct_type = to_struct_type(actual_type);
  const std::string tag = struct_type.tag().as_string();
  if (tag.find("__ESBMC_PyListObj") == std::string::npos)
    return std::nullopt;

  return struct_type;
}

static bool is_pylist_object_type(const typet &type, const namespacet &ns)
{
  return try_get_pylist_struct_type(type, ns).has_value();
}

const std::string kGetObjectSize = "__ESBMC_get_object_size";
const std::string kStrlen = "strlen";
const std::string kEsbmcAssume = "__ESBMC_assume";
const std::string kVerifierAssume = "__VERIFIER_assume";
const std::string kEsbmcAssert = "__ESBMC_assert";
const std::string kEsbmcUnreachable = "__ESBMC_unreachable";
const std::string kLoopInvariant = "__loop_invariant";
const std::string kEsbmcLoopInvariant = "__ESBMC_loop_invariant";
const std::string kEsbmcCover = "__ESBMC_cover";
const std::string kEsbmcAtomicBegin = "__ESBMC_atomic_begin";
const std::string kEsbmcAtomicEnd = "__ESBMC_atomic_end";
const std::string kEsbmcSpawnThread = "__ESBMC_spawn_thread";
const std::string kPytInitTid = "__pyt_init_tid";
const std::string kPytJoin = "__pyt_join";
const std::string kPytTerminate = "__pyt_terminate";
const std::string kPyLockBlockAndCheck = "__ESBMC_pylock_block_and_check";

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

bool function_call_builder::is_assert_call(const symbol_id &function_id) const
{
  return function_id.get_function() == kEsbmcAssert;
}

bool function_call_builder::is_unreachable_call(
  const symbol_id &function_id) const
{
  return function_id.get_function() == kEsbmcUnreachable;
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

      // Tracks the dotted module path being built during type resolution.
      // Set as a side-effect in the Name base case and extended in Attribute
      // nodes when type resolution fails, so that pkg.mod4.MyClass() can be
      // recognised as module "pkg.mod4" without a separate traversal pass.
      std::string module_path_candidate;

      std::function<typet(const nlohmann::json &)> resolve_attr_type =
        [&](const nlohmann::json &node) -> typet {
        if (node["_type"] == "Name")
        {
          // Base case: resolve variable to its type
          std::string name = node.value("id", "");
          if (name.empty())
            return typet();

          // Track the name for dotted-path reconstruction (used when type
          // resolution fails and we need to check for a module path).
          module_path_candidate = name;

          // Skip module names - let the module handling logic deal with them
          if (converter_.is_imported_module(name))
            return typet();

          symbol_id var_sid(
            python_file, current_class_name, current_function_name);
          var_sid.set_object(name);
          symbolt *var_symbol = converter_.find_symbol(var_sid.to_string());
          return var_symbol ? var_symbol->get_type() : typet();
        }
        else if (node["_type"] == "Attribute")
        {
          // Recursive case: resolve base, then look up member
          typet base_type = resolve_attr_type(node["value"]);
          std::string attr = node.value("attr", "");
          if (attr.empty())
            return typet();
          if (base_type.id().empty())
          {
            // Type resolution failed for the base — extend the dotted path
            // candidate so the caller can check is_imported_module on the
            // full path (e.g., "pkg" + "mod4" -> "pkg.mod4").
            if (!module_path_candidate.empty())
              module_path_candidate += "." + attr;
            return typet();
          }

          // Normalize type (dereference pointers, follow symbol types)
          if (base_type.is_pointer())
            base_type = base_type.subtype();
          if (base_type.id() == "symbol")
            base_type = converter_.ns.follow(base_type);

          // Look up the member in the struct
          if (base_type.is_struct())
          {
            const struct_typet &struct_type = to_struct_type(base_type);
            if (struct_type.has_component(attr))
              return struct_type.get_component(attr).type();
            // Not an instance member — check for class-level symbol
            // (e.g. mutable_attr = [] declared in the class body).
            symbol_id cls_sid(python_file, struct_type.tag().as_string(), "");
            cls_sid.set_object(attr);
            symbolt *cls_sym = converter_.find_symbol(cls_sid.to_string());
            if (cls_sym)
              return cls_sym->get_type();
            return typet();
          }
          return typet();
        }
        return typet();
      };

      // Resolve the full attribute chain type
      typet obj_type = resolve_attr_type(func_json["value"]);

      if (!obj_type.id().empty())
      {
        // If the resolved type is a list (e.g. class-level mutable_attr = []),
        // map directly to the "list" builtin name.
        if (is_pylist_object_type(obj_type, converter_.ns))
        {
          obj_name = "list";
        }
        else
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
      }
      else
      {
        // Type resolution failed. module_path_candidate was built as a
        // side-effect of resolve_attr_type; check whether it names a module
        // that is directly registered (e.g., "pkg.mod4" for pkg.mod4.MyClass()).
        // We require get_imported_module_path() to be non-empty rather than
        // just is_imported_module(), because is_imported_module() can return
        // true based solely on a JSON file existing (is_module fallback), even
        // for submodules like "os.path" that are registered under their stem
        // ("path") rather than their dotted name. If the dotted name has no
        // direct path mapping, fall back to the last attribute component
        // (e.g., "path" for os.path.exists()) which IS in imported_modules.
        if (
          !module_path_candidate.empty() &&
          !converter_.get_imported_module_path(module_path_candidate).empty())
          obj_name = module_path_candidate;
        else
          obj_name = func_json["value"].value("attr", "");
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
      const auto &inner_call = func_json["value"];
      if (
        inner_call.contains("func") && inner_call["func"].contains("_type") &&
        inner_call["func"]["_type"] == "Name" &&
        inner_call["func"].contains("id") &&
        inner_call["func"]["id"].is_string())
      {
        obj_name = inner_call["func"]["id"].get<std::string>();
      }
      else
      {
        // Nested calls (e.g., u.encode(...).decode(...)) do not always have
        // func.id. Use inferred call result type instead of indexing JSON
        // fields that may be null.
        obj_name = th.get_operand_type(inner_call);
      }

      if (obj_name.empty())
        obj_name = "str";

      if (obj_name == "nondet_str")
        obj_name = "str";

      if (obj_name == "super")
      {
        // For __init__, use is_ctor=true: parent's __init__ is stored under
        // the class name (e.g. Vehicle::Vehicle), not the literal "__init__".
        bool is_init_call = (func_name == "__init__");
        symbolt *base_class_func = converter_.find_function_in_base_classes(
          current_class_name, function_id.to_string(), func_name, is_init_call);
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
      else if (
        arg.contains("func") && arg["func"].contains("id") &&
        arg["func"]["id"] == "dict")
      {
        // len(dict(...))
        // route to dict-aware size handler.
        func_name = "__ESBMC_len_dict";
        function_id.clear();
        function_id.set_prefix("esbmc:");
        function_id.set_function(func_name);
        return function_id;
      }
    }
    else if (arg["_type"] == "List")
      func_name = kGetObjectSize;
    else if (arg["_type"] == "Name")
    {
      const std::string &var_type = th.get_var_type(arg["id"]);
      const std::string var_name = arg["id"].get<std::string>();
      symbol_id var_sid(python_file, current_class_name, current_function_name);
      var_sid.set_object(var_name);
      symbolt *var_symbol = converter_.find_symbol(var_sid.to_string());

      auto symbol_points_to_list = [&](const symbolt *sym) -> bool {
        return sym && is_pylist_object_type(sym->get_type(), converter_.ns);
      };

      const bool var_symbol_is_list = symbol_points_to_list(var_symbol);

      // A dict-typed symbol may carry no "dict" frontend annotation: a dict
      // comprehension ({k: v for ...}) or a dict-returning call assigns the
      // __python_dict__ struct directly without recording the type in the
      // annotation map. Detect the struct type on the symbol so len() routes
      // to the dict-size handler rather than falling through to strlen, which
      // would read the struct's bytes as a C string and yield a wrong size.
      const bool var_symbol_is_dict =
        var_symbol && converter_.get_dict_handler()->is_dict_type(
                        converter_.ns.follow(var_symbol->get_type()));

      auto is_list_slice_assignment = [&]() -> bool {
        nlohmann::json var_value = json_utils::get_var_value(
          var_name,
          converter_.get_current_func_name(),
          converter_.get_ast_json());
        if (
          var_value.empty() || !var_value.contains("value") ||
          !var_value["value"].is_object())
          return false;

        const auto &value_node = var_value["value"];
        if (
          !value_node.contains("_type") || value_node["_type"] != "Subscript" ||
          !value_node.contains("slice") || !value_node["slice"].is_object() ||
          !value_node["slice"].contains("_type") ||
          value_node["slice"]["_type"] != "Slice" ||
          !value_node.contains("value") || !value_node["value"].is_object() ||
          !value_node["value"].contains("_type"))
          return false;

        const auto &base = value_node["value"];
        if (base["_type"] == "List")
          return true;

        if (base["_type"] == "Name" && base.contains("id"))
        {
          const std::string base_name = base["id"].get<std::string>();
          const std::string base_type = th.get_var_type(base_name);
          if (base_type == "list" || base_type == "List")
            return true;
          if (base_type == "str")
            return false;

          symbol_id base_sid(
            python_file, current_class_name, current_function_name);
          base_sid.set_object(base_name);
          symbolt *base_symbol = converter_.find_symbol(base_sid.to_string());
          return symbol_points_to_list(base_symbol);
        }

        return false;
      };

      // Check if this is a tuple by looking up the variable's type
      if (var_type == "tuple" || var_type.empty())
      {
        if (var_symbol && var_symbol->get_type().id() == "struct")
        {
          const struct_typet &struct_type =
            to_struct_type(var_symbol->get_type());

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
      if (var_type == "dict" || var_symbol_is_dict)
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
      else if (var_symbol_is_list)
      {
        // Prefer symbol type over inferred frontend type when they conflict.
        func_name = kGetObjectSize;
      }
      else if (var_type.empty() && is_list_slice_assignment())
      {
        // list slicing assigned to a variable may have no annotation.
        // Use list size semantics for len(sub) where sub = lst[a:b].
        func_name = kGetObjectSize;
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
  else if (
    is_assume_call(function_id) || is_assert_call(function_id) ||
    is_cover_call(function_id) || is_unreachable_call(function_id))
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
    // An explicit `Base.__init__(self, ...)` call names the receiver class and
    // passes self explicitly. is_constructor_call() flags any `.__init__`, but
    // here the constructor is the named class's own, stored under the class
    // name (@C@Base@F@Base), not the literal "__init__". Resolve both the class
    // and the (renamed) function to the receiver so the symbol is found; a
    // direct `Base()` call keeps func_name as the class name already.
    if (func_name == "__init__" && json_utils::is_class(obj_name, ast))
    {
      class_name = obj_name;
      func_name = obj_name;
    }
    else
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
      {
        // Reaching here means the attribute-chain resolver above could not
        // determine a struct type for the receiver (most commonly because a
        // function parameter is unannotated and python_annotation could not
        // infer it from any call site), and `obj_name` ended up as the
        // trailing attribute name rather than a real variable. Surface a
        // diagnostic that names the unresolved member call and points at
        // the likely missing annotation, instead of the cryptic
        // "Variable <attr> not found" the call previously emitted.
        throw std::runtime_error(
          "Could not resolve type of receiver in member call '." + obj_name +
          "." + func_name +
          "()'. Add a type annotation to the function parameter or local "
          "variable so the attribute chain can be typed.");
      }

      // Extract class name from the type, following symbol references
      typet var_type = var_symbol->get_type().is_pointer()
                         ? var_symbol->get_type().subtype()
                         : var_symbol->get_type();

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
  if (call_["func"]["_type"] == "Attribute")
  {
    exprt handled_string_method =
      converter_.get_string_handler().handle_string_attribute_call(call_);
    if (!handled_string_method.is_nil())
      return handled_string_method;
  }

  symbol_id function_id = build_function_id();

  if (is_len_call(function_id) && !call_["args"].empty())
  {
    exprt arg_expr = converter_.get_expr(call_["args"][0]);

    // If len() argument is a list-typed symbol, force list-size semantics.
    // This avoids misrouting to strlen() when variable annotations are absent
    // (e.g., sub = lst[a:b]; len(sub)).
    if (arg_expr.is_symbol())
    {
      symbolt *arg_symbol = converter_.find_symbol(
        to_symbol_expr(arg_expr).get_identifier().as_string());
      if (arg_symbol)
      {
        auto list_struct =
          try_get_pylist_struct_type(arg_symbol->get_type(), converter_.ns);
        if (list_struct)
        {
          code_typet list_size_type;
          list_size_type.return_type() = size_type();
          code_typet::argumentt arg_type;
          arg_type.type() = pointer_typet(*list_struct);
          list_size_type.arguments().push_back(arg_type);

          symbol_exprt list_size_func("c:@F@__ESBMC_list_size", list_size_type);

          return build_call(
            list_size_func,
            size_type(),
            {arg_expr.type().is_pointer() ? arg_expr
                                          : build_address_of(arg_expr)});
        }
      }
    }

    if (arg_expr.type().is_signedbv() || arg_expr.type().is_unsignedbv())
      return from_integer(1, long_long_int_type());

    // len() of a tuple-typed expression (e.g. an inline str.partition() result
    // that is not bound to a Name, so the __ESBMC_len_tuple routing above never
    // fires) is the number of components. Without this the call falls through to
    // strlen() and reports the wrong length (#5114).
    if (arg_expr.type().id() == "struct")
    {
      const struct_typet &struct_type = to_struct_type(arg_expr.type());
      if (struct_type.tag().as_string().find("tag-tuple") == 0)
        return from_integer(struct_type.components().size(), size_type());
    }

    exprt len_string_fast_path =
      converter_.get_string_handler().try_handle_len_string_fast_path(
        call_, arg_expr);
    if (!len_string_fast_path.is_nil())
      return len_string_fast_path;
  }

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
    // If it's actually a list, call list_size directly
    auto list_struct =
      try_get_pylist_struct_type(obj_expr.type(), converter_.ns);
    if (list_struct)
    {
      // It's a list, not a dict: call list_size on it directly
      code_typet list_size_type;
      list_size_type.return_type() = size_type();
      code_typet::argumentt arg_type;
      arg_type.type() = pointer_typet(*list_struct);
      list_size_type.arguments().push_back(arg_type);

      symbol_exprt list_size_func("c:@F@__ESBMC_list_size", list_size_type);

      return build_call(list_size_func, size_type(), {obj_expr});
    }

    // It's genuinely a dict: get the keys member
    typet keys_type = pointer_typet(struct_typet());
    exprt keys_member = build_member(obj_expr, "keys", keys_type);

    // Create the list_get_size function symbol
    code_typet list_get_size_type;
    list_get_size_type.return_type() = size_type();
    code_typet::argumentt arg_type;
    arg_type.type() = keys_type;
    list_get_size_type.arguments().push_back(arg_type);

    symbol_exprt list_get_size_func(
      "c:@F@__ESBMC_list_size", list_get_size_type);

    // Create the function call
    return build_call(list_get_size_func, size_type(), {keys_member});
  }

  // Special handling for assume calls: convert to code_assume instead of function call
  if (is_assume_call(function_id))
  {
    if (call_["args"].empty())
      throw std::runtime_error("__ESBMC_assume requires one boolean argument");

    exprt condition = converter_.get_expr(call_["args"][0]);
    if (!condition.type().is_bool())
      condition = build_typecast(condition, bool_type());

    // Create code_assume statement
    codet assume_code("assume");
    assume_code.copy_to_operands(condition);
    assume_code.location() = converter_.get_location_from_decl(call_);

    return assume_code;
  }

  // __ESBMC_assert(cond[, msg]) — same message lowering as native
  // `assert cond, msg` in converter_stmt.cpp.
  if (is_assert_call(function_id))
  {
    const auto &args = call_["args"];
    if (args.empty() || args.size() > 2)
      throw std::runtime_error(
        "__ESBMC_assert takes one or two arguments (cond[, msg])");

    exprt condition = converter_.get_expr(args[0]);
    if (!condition.type().is_bool())
      condition = build_typecast(condition, bool_type());

    code_assertt assert_code(condition);
    assert_code.location() = converter_.get_location_from_decl(call_);

    if (args.size() == 2)
    {
      const auto &msg_node = args[1];
      std::string msg;
      if (
        msg_node["_type"] == "Constant" && msg_node.contains("value") &&
        msg_node["value"].is_string())
        msg = msg_node["value"].get<std::string>();
      else if (msg_node["_type"] == "JoinedStr")
        msg = "<formatted string message>";

      if (!msg.empty())
        assert_code.location().comment(msg);
    }

    return assert_code;
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

  // __ESBMC_unreachable: emit a call to the C-style symbol so symex's
  // intrinsic handler in symex_main.cpp fires under
  // --enable-unreachability-intrinsic. Lowering matches the atomic_*
  // pattern below: register a void(void) symbol id and call it.
  if (is_unreachable_call(function_id))
  {
    if (!call_["args"].empty())
      throw std::runtime_error("__ESBMC_unreachable takes no arguments");

    code_typet fn_type;
    fn_type.return_type() = empty_typet();

    locationt location = converter_.get_location_from_decl(call_);
    converter_.ensure_void_void_intrinsic(kEsbmcUnreachable, location);

    code_function_callt call;
    call.function() = symbol_exprt("c:@F@" + kEsbmcUnreachable, fn_type);
    call.type() = empty_typet();
    call.location() = location;
    return call;
  }

  // __ESBMC_atomic_begin / __ESBMC_atomic_end: thin wrappers around the C
  // intrinsics with the same names. Emitting a call to a symbol whose
  // base name matches is enough; goto-symex lowers it to the atomic-section
  // builtins in src/goto-programs/builtin_functions.cpp.
  {
    const std::string &func_name = function_id.get_function();
    const bool is_atomic_begin = func_name == kEsbmcAtomicBegin;
    const bool is_atomic_end = func_name == kEsbmcAtomicEnd;
    if (is_atomic_begin || is_atomic_end)
    {
      if (!call_["args"].empty())
        throw std::runtime_error(func_name + " takes no arguments");

      code_typet atomic_type;
      atomic_type.return_type() = empty_typet();

      locationt location = converter_.get_location_from_decl(call_);
      converter_.ensure_void_void_intrinsic(func_name, location);
      // do_atomic_begin emits a call to __ESBMC_yield to force a context
      // switch before entering the atomic section, dereferencing the symbol
      // unconditionally; the Python frontend does not pull in the C
      // intrinsics header, so register the yield symbol here too. Only
      // atomic_begin needs it; atomic_end does not call yield.
      if (is_atomic_begin)
        converter_.ensure_void_void_intrinsic("__ESBMC_yield", location);

      code_function_callt atomic_call;
      atomic_call.function() = symbol_exprt("c:@F@" + func_name, atomic_type);
      atomic_call.type() = empty_typet();
      atomic_call.location() = location;
      return atomic_call;
    }
  }

  // __ESBMC_spawn_thread: the symex-level thread-spawn intrinsic used
  // by the Python frontend's threading.Thread lowering (see parser.py
  // lower_threading_thread_usage). Symex's intrinsic_spawn_thread
  // (builtin_functions/threads.cpp) requires its operand to be a
  // literal address_of(symbol_expr(trampoline_function)); the Python
  // converter receives a bare Name → symbol_exprt and wraps it here.
  {
    const std::string &func_name = function_id.get_function();
    if (func_name == kEsbmcSpawnThread)
    {
      auto &symbol_table = converter_.symbol_table();
      locationt location = converter_.get_location_from_decl(call_);

      code_typet trampoline_type;
      trampoline_type.return_type() = empty_typet();
      typet param_type = pointer_typet(trampoline_type);

      code_typet fn_type;
      fn_type.return_type() = uint_type();
      fn_type.arguments().push_back(code_typet::argumentt(param_type));

      const std::string symbol_id = "c:@F@" + func_name;
      if (symbol_table.find_symbol(symbol_id.c_str()) == nullptr)
      {
        symbolt symbol = converter_.create_symbol(
          converter_.python_file(), func_name, symbol_id, location, fn_type);
        converter_.add_symbol(symbol);
      }

      if (call_["args"].size() != 1)
        throw std::runtime_error(func_name + " takes exactly one argument");
      exprt arg = converter_.get_expr(call_["args"][0]);
      if (arg.type().is_code())
        arg = build_address_of(arg);
      if (arg.type() != param_type)
        arg = build_typecast(arg, param_type);

      code_function_callt call;
      call.function() = symbol_exprt(symbol_id, fn_type);
      call.type() = fn_type.return_type();
      call.location() = location;
      call.arguments().push_back(arg);
      return call;
    }
  }

  // __pyt_init_tid / __pyt_join / __pyt_terminate / __ESBMC_pylock_block_and_check:
  // C-side bookkeeping helpers for the Python threading lowering (defined
  // in pthread_lib.c, pulled into the GOTO program via the python_c_models
  // whitelist in cprover_library.cpp). Register the matching C-style
  // symbol id so the call resolves to the linked body rather than a
  // freshly-created Python-frontend symbol with no implementation.
  {
    const std::string &func_name = function_id.get_function();
    const bool is_init_tid = func_name == kPytInitTid;
    const bool is_join = func_name == kPytJoin;
    const bool is_terminate = func_name == kPytTerminate;
    const bool is_lock_block = func_name == kPyLockBlockAndCheck;
    if (is_init_tid || is_join || is_terminate || is_lock_block)
    {
      auto &symbol_table = converter_.symbol_table();
      locationt location = converter_.get_location_from_decl(call_);

      const bool takes_uint_arg = is_init_tid || is_join;

      code_typet fn_type;
      fn_type.return_type() = empty_typet();
      if (takes_uint_arg)
        fn_type.arguments().push_back(code_typet::argumentt(uint_type()));

      const std::string symbol_id = "c:@F@" + func_name;
      if (symbol_table.find_symbol(symbol_id.c_str()) == nullptr)
      {
        symbolt symbol = converter_.create_symbol(
          converter_.python_file(), func_name, symbol_id, location, fn_type);
        converter_.add_symbol(symbol);
      }

      code_function_callt call;
      call.function() = symbol_exprt(symbol_id, fn_type);
      call.type() = empty_typet();
      call.location() = location;

      if (takes_uint_arg)
      {
        if (call_["args"].size() != 1)
          throw std::runtime_error(func_name + " takes exactly one argument");
        exprt arg = converter_.get_expr(call_["args"][0]);
        if (arg.type() != uint_type())
          arg = build_typecast(arg, uint_type());
        call.arguments().push_back(arg);
      }
      else if (!call_["args"].empty())
      {
        throw std::runtime_error(func_name + " takes no arguments");
      }

      return call;
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
