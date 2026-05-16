#pragma once

#include <python-frontend/json_utils.h>
#include <python-frontend/global_scope.h>
#include <python-frontend/module_manager.h>
#include <python-frontend/module.h>
#include <python-frontend/type_utils.h>
#include <python-frontend/python_annotation/annotation_intrinsics.h>
#include <python-frontend/python_annotation/annotation_parser.h>
#include <python-frontend/python_annotation/annotation_utils.h>
#include <util/message.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

// `InferResult`, `builtin_functions` and the pure-helper utilities
// (`has_annotation`, `has_return_none`, `get_type_from_json`,
// `get_base_var_name`, `invert_substrings`,
// `infer_type_from_default_arg_shape`) were moved to the
// `python_annotation_intrinsics` and `python_annotation_utils`
// modules. The using declarations below preserve the unqualified
// names so existing call sites inside this header remain byte-
// identical except, for `builtin_functions`, the trailing parens
// (`builtin_functions.find(...)` -> `builtin_functions().find(...)`).
using python_annotation_intrinsics::builtin_functions;
using python_annotation_parser::find_function_recursive;
using python_annotation_parser::find_lambda_in_body;
using python_annotation_utils::get_base_var_name;
using python_annotation_utils::get_type_from_json;
using python_annotation_utils::has_annotation;
using python_annotation_utils::has_return_none;
using python_annotation_utils::infer_type_from_default_arg_shape;
using python_annotation_utils::InferResult;
using python_annotation_utils::invert_substrings;

template <class Json>
class python_annotation
{
public:
  python_annotation(Json &ast, global_scope &gs)
    : ast_(ast),
      gs_(gs),
      current_func(nullptr),
      parent_func(nullptr),
      current_line_(0)
  {
    python_filename_ = ast_["filename"].template get<std::string>();
    if (ast_.contains("ast_output_dir"))
      module_manager_ =
        module_manager::create(ast_["ast_output_dir"], python_filename_);
  }

  void preprocess_constructor_calls(const Json &node)
  {
    if (node.is_object())
    {
      // Check if this is a constructor call
      if (
        node.contains("_type") && node["_type"] == "Call" &&
        node.contains("func") && node["func"]["_type"] == "Name")
      {
        const std::string &func_name = node["func"]["id"];
        // Find the class in ast_["body"] using reference to modify in-place
        for (Json &class_node : ast_["body"])
        {
          if (
            class_node["_type"] == "ClassDef" &&
            class_node["name"] == func_name)
          {
            // Find __init__ method
            for (Json &member : class_node["body"])
            {
              if (
                member["_type"] == "FunctionDef" &&
                member["name"] == "__init__")
              {
                // Annotate parameters based on this call
                if (member.contains("args") && member["args"].contains("args"))
                {
                  Json &params = member["args"]["args"];
                  const Json &call_args = node["args"];
                  // Skip self (index 0), match remaining params with call args
                  for (size_t i = 1;
                       i < params.size() && (i - 1) < call_args.size();
                       ++i)
                  {
                    Json &param = params[i];
                    // Only annotate if the parameter is not yet annotated;
                    // do not overwrite explicit annotations (e.g., Optional["List"])
                    // with a less-specific type inferred from a call site (e.g., NoneType).
                    if (
                      param.contains("annotation") &&
                      !param["annotation"].is_null())
                      continue;
                    const Json &arg = call_args[i - 1];
                    std::string arg_type = get_argument_type(arg);
                    if (!arg_type.empty())
                      add_parameter_annotation(param, arg_type);
                  }
                }
              }
            }
          }
        }
      }
      // Recursively search all object fields
      for (auto &kv : node.items())
        preprocess_constructor_calls(kv.value());
    }
    else if (node.is_array())
    {
      // Search in arrays (such as list literals)
      for (const auto &element : node)
        preprocess_constructor_calls(element);
    }
  }

  // Preprocess method calls on temporary instances to infer parameter types.
  // Handles patterns such as A().method(arg1, arg2) by annotating the method's
  // parameters from the argument types at the call site.
  void preprocess_method_calls(const Json &node)
  {
    if (node.is_object())
    {
      // Detect: A().method(args)
      // func._type == "Attribute", func.value._type == "Call" (the A() part),
      // func.value.func._type == "Name" (the class name A)
      if (
        node.contains("_type") && node["_type"] == "Call" &&
        node.contains("func") && node["func"]["_type"] == "Attribute" &&
        node["func"].contains("value") &&
        node["func"]["value"]["_type"] == "Call" &&
        node["func"]["value"].contains("func") &&
        node["func"]["value"]["func"]["_type"] == "Name" &&
        node["func"]["value"]["func"].contains("id") &&
        node["func"].contains("attr"))
      {
        const std::string &class_name =
          node["func"]["value"]["func"]["id"].template get<std::string>();
        const std::string &method_name =
          node["func"]["attr"].template get<std::string>();
        const Json &call_args =
          node.contains("args") ? node["args"] : Json::array();

        for (Json &class_node : ast_["body"])
        {
          if (
            class_node["_type"] == "ClassDef" &&
            class_node["name"] == class_name)
          {
            for (Json &member : class_node["body"])
            {
              if (
                member["_type"] == "FunctionDef" &&
                member["name"] == method_name && member.contains("args") &&
                member["args"].contains("args"))
              {
                Json &params = member["args"]["args"];
                // Skip self (index 0); match remaining params to call args
                for (size_t i = 1;
                     i < params.size() && (i - 1) < call_args.size();
                     ++i)
                {
                  Json &param = params[i];
                  // Only annotate if the parameter is not yet annotated
                  if (
                    param.contains("annotation") &&
                    !param["annotation"].is_null())
                    continue;
                  const Json &arg = call_args[i - 1];
                  std::string arg_type = get_argument_type(arg);
                  if (!arg_type.empty())
                    add_parameter_annotation(param, arg_type);
                }
              }
            }
          }
        }
      }

      // Recursively search all object fields
      for (auto &kv : node.items())
        preprocess_method_calls(kv.value());
    }
    else if (node.is_array())
    {
      for (const auto &element : node)
        preprocess_method_calls(element);
    }
  }

  void add_type_annotation()
  {
    // First pass: preprocess all constructor calls to infer parameter types
    preprocess_constructor_calls(ast_);
    // Also preprocess method calls on temporary instances: A().method(args)
    preprocess_method_calls(ast_);

    // Second pass: add type annotations to global scope variables
    annotate_global_scope();
    current_line_ = 0;

    // Add type annotation to all functions and class methods
    for (Json &element : ast_["body"])
    {
      // Process top-level functions
      if (element["_type"] == "FunctionDef")
        annotate_function(element);
      // Process classes and their methods
      else if (element["_type"] == "ClassDef")
        annotate_class(element);
    }

    apply_pending_specializations();
  }

  /// Register an additional AST whose multi-axis subscript usages must be
  /// considered when inferring `__getitem__` / `__setitem__` key tuple
  /// types. Used when annotating an imported module to make the importing
  /// module's call sites visible (GitHub #4545). The caller retains
  /// ownership; @p ast must outlive this annotator.
  void add_extra_subscript_inference_source(const Json &ast)
  {
    extra_subscript_inference_asts_.push_back(&ast);
  }

  void add_type_annotation(const std::string &func_name)
  {
    current_line_ = 0;

    for (Json &elem : ast_["body"])
    {
      if (elem["_type"] == "FunctionDef" && elem["name"] == func_name)
      {
        get_global_elements(elem["body"]);
        // Add type annotations to global scope variables
        if (!referenced_global_elements.empty())
          annotate_global_scope();
        filter_global_elements_ = false;

        // Add annotation to a specific function
        annotate_function(elem);
        return;
      }
    }
  }

private:
  Json find_var_node_for_inference(const std::string &var_name)
  {
    Json var_node =
      json_utils::find_var_decl(var_name, get_current_func_name(), ast_);

    if (
      !has_annotation(var_node) &&
      (var_node.empty() ||
       (var_node.contains("_type") && var_node["_type"] == "arg")) &&
      parent_func != nullptr)
    {
      if (
        (*parent_func).contains("args") &&
        (*parent_func)["args"].contains("args"))
      {
        var_node =
          find_annotated_assign(var_name, (*parent_func)["args"]["args"]);
        // Also search keyword-only args
        if (var_node.empty() && (*parent_func)["args"].contains("kwonlyargs"))
          var_node = find_annotated_assign(
            var_name, (*parent_func)["args"]["kwonlyargs"]);
      }

      if (!has_annotation(var_node) && (*parent_func).contains("body"))
        var_node = find_annotated_assign(var_name, (*parent_func)["body"]);
    }

    if (
      !has_annotation(var_node) &&
      (var_node.empty() ||
       (var_node.contains("_type") && var_node["_type"] == "arg")))
    {
      Json global_var_node = json_utils::find_var_decl(var_name, "", ast_);
      if (!global_var_node.empty())
        var_node = global_var_node;
    }

    if (!has_annotation(var_node) && var_node.empty())
      var_node = find_annotated_assign(var_name, ast_["body"]);

    return var_node;
  }

  // Infer return type from non-recursive return statements
  std::string
  infer_from_return_statements(const Json &body, const std::string &func_name)
  {
    for (const Json &stmt : body)
    {
      // Found a return statement
      if (stmt["_type"] == "Return" && !stmt["value"].is_null())
      {
        const Json &return_val = stmt["value"];

        // Skip recursive calls
        if (
          return_val["_type"] == "Call" && return_val.contains("func") &&
          return_val["func"]["_type"] == "Name" &&
          return_val["func"]["id"] == func_name)
        {
          continue; // Skip this recursive call
        }

        // Handle function calls (including nested functions)
        if (
          return_val["_type"] == "Call" && return_val.contains("func") &&
          return_val["func"]["_type"] == "Name")
        {
          const std::string &called_func = return_val["func"]["id"];

          // Try to get the return type of the called function
          try
          {
            std::string called_func_type =
              get_function_return_type(called_func, ast_);
            if (!called_func_type.empty() && called_func_type != "NoneType")
              return called_func_type;
          }
          catch (std::runtime_error &)
          {
            // Function not found, continue with normal inference
          }
        }

        // Reuse get_argument_type to infer the return value type
        std::string inferred_type = get_argument_type(return_val);
        if (!inferred_type.empty())
          return inferred_type;
      }

      // Recursively check nested blocks (if/while/for/try)
      if (stmt.contains("body") && stmt["body"].is_array())
      {
        std::string nested_type =
          infer_from_return_statements(stmt["body"], func_name);
        if (!nested_type.empty())
          return nested_type;
      }

      if (stmt.contains("orelse") && stmt["orelse"].is_array())
      {
        std::string nested_type =
          infer_from_return_statements(stmt["orelse"], func_name);
        if (!nested_type.empty())
          return nested_type;
      }
    }

    return ""; // Couldn't infer
  }

  // Method to infer and annotate unannotated function parameters
  void infer_parameter_types(Json &function_element)
  {
    const std::string &func_name = function_element["name"];

    // Check if function has a meaningful body
    bool has_meaningful_body = false;
    if (function_element.contains("body") && !function_element["body"].empty())
    {
      for (const auto &stmt : function_element["body"])
      {
        if (stmt.contains("_type") && stmt["_type"] != "Pass")
        {
          has_meaningful_body = true;
          break;
        }
      }
    }

    // Determine where to search for function calls:
    // - For nested functions: search in parent function's body
    // - For top-level functions: search in global AST body
    const Json &search_context =
      (parent_func != nullptr)
        ? (*parent_func)["body"] // Search in parent function's body
        : ast_["body"];          // Search in global body

    // Find all calls to this function in the AST
    std::vector<Json> function_calls =
      find_function_calls(func_name, search_context);

    // Determine the call-site context for resolving argument names.
    // For top-level functions, use global scope (empty context).
    // For nested functions, use the parent function context.
    std::string call_site_context;
    if (parent_func != nullptr)
    {
      std::string cur = current_func_name_context_;
      size_t pos = cur.rfind("@F@");
      if (pos != std::string::npos)
        call_site_context = cur.substr(0, pos);
      else if (parent_func->contains("name"))
        call_site_context = (*parent_func)["name"].template get<std::string>();
    }

    // For each parameter, try to infer its type from the function calls
    if (
      function_element.contains("args") &&
      function_element["args"].contains("args"))
    {
      Json &params = function_element["args"]["args"];

      for (size_t i = 0; i < params.size(); ++i)
      {
        Json &param = params[i];

        // Check if parameter needs type inference or refinement
        bool needs_inference = false;
        bool has_partial_annotation = false;
        std::string current_type;

        if (param.contains("annotation") && !param["annotation"].is_null())
        {
          // Extract current annotation
          if (param["annotation"].contains("id"))
            current_type = param["annotation"]["id"];
          // Check if it's a generic container without element type
          if (
            current_type == "list" || current_type == "dict" ||
            current_type == "set")
          {
            has_partial_annotation = true;
            needs_inference = true; // Try to refine with element type
          }
          else if (current_type == "tuple")
          {
            // Bare `tuple` annotation: try to specialise to
            // `tuple[T0, T1, ...]` by inspecting call sites so the unpacker
            // can recover the struct shape (GitHub #4515).
            if (!function_calls.empty())
            {
              std::vector<std::string> elem_types =
                infer_tuple_param_types_from_calls(i, function_calls);
              if (!elem_types.empty())
              {
                int lineno = param.value("lineno", 0);
                int col_offset =
                  param.value("col_offset", 0) +
                  param["arg"].template get<std::string>().size() + 1;
                param["annotation"] = create_tuple_subscript_annotation(
                  elem_types, lineno, col_offset, lineno);
              }
            }
            continue;
          }
          else
            continue; // Fully annotated, skip
        }
        else
          needs_inference = true; // No annotation at all

        std::string inferred_type;

        // Try to infer type from function calls if available
        if (needs_inference && !function_calls.empty())
        {
          std::vector<std::string> call_types =
            collect_parameter_types_from_calls(i, function_calls);

          if (
            parent_func == nullptr && call_types.size() > 1 &&
            are_all_user_classes(call_types))
          {
            bool already_pending = false;
            for (const auto &spec : pending_specializations_)
            {
              if (spec.function_name == func_name && spec.param_index == i)
              {
                already_pending = true;
                break;
              }
            }

            if (!already_pending)
            {
              pending_specializations_.push_back(
                {func_name, i, std::move(call_types)});
            }

            continue;
          }

          std::string saved_ctx = current_func_name_context_;
          current_func_name_context_ = call_site_context;
          inferred_type = infer_parameter_type_from_calls(i, function_calls);
          current_func_name_context_ = saved_ctx;
        }

        // If still no type, try to infer from the parameter's default value.
        // This handles cases like def h(op=g) where g is a function alias.
        if (inferred_type.empty() && !has_partial_annotation)
        {
          const auto &args_node = function_element["args"];
          if (args_node.contains("defaults"))
          {
            const auto &defaults = args_node["defaults"];
            size_t defaults_start = params.size() - defaults.size();
            if (i >= defaults_start)
            {
              const Json &def_node = defaults[i - defaults_start];
              if (
                def_node.contains("_type") && def_node["_type"] == "Name" &&
                def_node.contains("id"))
              {
                const std::string &def_name =
                  def_node["id"].template get<std::string>();
                // Use non-throwing (const) overload to avoid crashing when
                // def_name is a variable (not a FunctionDef).
                const nlohmann::json &const_body =
                  static_cast<const nlohmann::json &>(ast_["body"]);
                // Direct function reference (op=f)?
                if (!json_utils::find_function(const_body, def_name).empty())
                  inferred_type = "Any";
                // Function alias (op=g where g=f)?
                else
                {
                  for (const auto &stmt : ast_["body"])
                  {
                    if (
                      stmt.contains("_type") && stmt["_type"] == "Assign" &&
                      stmt.contains("targets") && !stmt["targets"].empty() &&
                      stmt["targets"][0].contains("id") &&
                      stmt["targets"][0]["id"] == def_name &&
                      stmt.contains("value") &&
                      stmt["value"].contains("_type") &&
                      stmt["value"]["_type"] == "Name" &&
                      stmt["value"].contains("id") &&
                      !json_utils::find_function(
                         const_body,
                         stmt["value"]["id"].template get<std::string>())
                         .empty())
                    {
                      inferred_type = "Any";
                      break;
                    }
                  }
                }
              }
            }
          }
        }

        // For __getitem__/__setitem__ key parameters, infer a
        // `tuple[slice, slice, ...]` annotation from multi-axis subscript
        // call sites on instances of the enclosing class (GitHub #4539).
        // The Subscript syntax t[i:j, k:l] never appears as an explicit
        // __getitem__ call in the AST, so the standard call-site inference
        // never sees it. Without this, `key` stays unannotated and `key[0]`
        // inside __getitem__ falls through to list indexing.
        if (
          inferred_type.empty() && !current_class_name_.empty() &&
          (func_name == "__getitem__" || func_name == "__setitem__") && i == 1)
        {
          std::vector<std::string> elem_types =
            infer_subscript_key_tuple_types(current_class_name_);
          if (!elem_types.empty())
          {
            int lineno = param.value("lineno", 0);
            int col_offset = param.value("col_offset", 0) +
                             param["arg"].template get<std::string>().size() +
                             1;
            param["annotation"] = create_tuple_subscript_annotation(
              elem_types, lineno, col_offset, lineno);
            continue;
          }
        }

        // Apply inference results
        if (has_partial_annotation)
        {
          // Refine partial annotation with inferred element type
          if (!inferred_type.empty() && inferred_type != current_type)
            add_parameter_annotation(param, inferred_type);
        }
        else
        {
          // Handle completely unannotated parameters
          if (inferred_type.empty() && !has_meaningful_body)
            inferred_type = "Any"; // Default fallback for stub functions

          if (!inferred_type.empty())
          {
            // Add annotation to parameter
            add_parameter_annotation(param, inferred_type);
          }
        }
      }
    }
  }

  // For a class @p class_name defining __getitem__/__setitem__, return the
  // synthesised per-element type names for a `tuple[T1, T2, ...]` annotation
  // when a multi-axis subscript usage `instance[a, b, ...]` exists whose
  // `instance` is annotated as @p class_name. Slice and scalar elements are
  // both supported, including all-scalar shapes (GitHub #4542). Both
  // module-scope `AnnAssign` and function-parameter annotations are
  // considered as sources for the instance→class mapping. When called on an
  // imported module, the importing module's AST is also scanned so the
  // cross-module call site is visible (GitHub #4545). Returns empty when no
  // qualifying subscript exists, when any element's type cannot be
  // inferred, or when sites disagree on arity or per-position element
  // types.
  std::vector<std::string>
  infer_subscript_key_tuple_types(const std::string &class_name)
  {
    // Unanimous-wins on per-position element types. The first qualifying
    // site fixes the expected type vector; any later site that disagrees on
    // arity or on a per-position type abandons synthesis so the existing
    // void* default keeps working for those call sites.
    bool any_reject = false;
    std::vector<std::string> expected;
    auto reducer = [&](const Json &elts, const Json *enclosing_func) {
      if (any_reject)
        return;
      std::vector<std::string> types =
        infer_tuple_element_types(elts, enclosing_func);
      if (types.empty())
      {
        any_reject = true;
        return;
      }
      if (expected.empty())
        expected = std::move(types);
      else if (expected != types)
        any_reject = true;
    };

    auto scan = [&](const Json &ast) {
      std::unordered_map<std::string, std::string> var_to_class;
      collect_class_instance_map(ast, var_to_class);
      visit_class_tuple_subscripts(ast, class_name, var_to_class, reducer);
    };

    scan(ast_);
    for (const Json *extra : extra_subscript_inference_asts_)
      scan(*extra);

    if (any_reject)
      return {};
    return expected;
  }

  // Populate @p out with a name→class-name map for every variable in @p ast
  // that has a Name-typed annotation. Two sources are consulted:
  //   * Module-scope `AnnAssign(target=Name, annotation=Name)` — e.g.
  //     `t: Tile = Tile(...)`.
  //   * Function-parameter annotations on any FunctionDef / AsyncFunctionDef
  //     (including methods nested in ClassDef) — e.g.
  //     `def f(t: Tile): ...`.
  // Conflicts across sites with the same name resolve last-write-wins. The
  // visitor filters by class_name at consumption time, so in the common
  // case — a parameter `t: ThisCls` shadowing a module-scope `t: OtherCls` —
  // only sites for the winning class match. A rare shadowing pattern (a
  // module-scope `t: OtherCls` together with a parameter `t: ThisCls` *and*
  // a module-scope multi-axis subscript `t[a, b]` that legitimately belongs
  // to OtherCls) could feed the wrong shape into ThisCls's reducer; if the
  // shapes disagree the unanimous-wins reducer bails out, otherwise an
  // incorrect annotation is synthesised. Not observed in practice; not
  // worth the complexity of lexical scoping until it bites.
  void collect_class_instance_map(
    const Json &ast,
    std::unordered_map<std::string, std::string> &out) const
  {
    auto record = [&](const Json &target_id, const Json &annotation) {
      if (
        !target_id.is_string() || !annotation.is_object() ||
        !annotation.contains("_type") || annotation["_type"] != "Name" ||
        !annotation.contains("id") || !annotation["id"].is_string())
        return;
      out[target_id.template get<std::string>()] =
        annotation["id"].template get<std::string>();
    };

    std::function<void(const Json &)> walk = [&](const Json &node) {
      if (node.is_object())
      {
        if (
          node.contains("_type") && node["_type"] == "AnnAssign" &&
          node.contains("target") && node["target"].is_object() &&
          node["target"].contains("_type") &&
          node["target"]["_type"] == "Name" && node["target"].contains("id") &&
          node.contains("annotation"))
          record(node["target"]["id"], node["annotation"]);
        else if (
          node.contains("_type") && node["_type"] == "arg" &&
          node.contains("arg") && node.contains("annotation"))
          record(node["arg"], node["annotation"]);
        for (auto it = node.begin(); it != node.end(); ++it)
          walk(it.value());
      }
      else if (node.is_array())
      {
        for (const auto &elem : node)
          walk(elem);
      }
    };

    if (ast.is_object() && ast.contains("body"))
      walk(ast["body"]);
  }

  // Walk @p node and invoke @p visit(elts, enclosing_func) for every
  // Subscript whose value is a Name resolving to @p class_name and whose
  // slice is a Tuple. @p visit receives the tuple's `elts` JSON array along
  // with a pointer to the enclosing FunctionDef/AsyncFunctionDef (or nullptr
  // at module scope). The enclosing function lets the reducer resolve
  // `Name` elts against arguments and local annotations of the AST being
  // scanned — essential when scanning a foreign module's AST whose locals
  // are not visible to this annotator's own `ast_`-scoped lookup (GitHub
  // #4558).
  template <typename Visitor>
  void visit_class_tuple_subscripts(
    const Json &node,
    const std::string &class_name,
    const std::unordered_map<std::string, std::string> &var_to_class,
    Visitor &&visit,
    const Json *enclosing_func = nullptr)
  {
    if (node.is_object())
    {
      const Json *next_enclosing = enclosing_func;
      if (
        node.contains("_type") &&
        (node["_type"] == "FunctionDef" || node["_type"] == "AsyncFunctionDef"))
        next_enclosing = &node;

      if (
        node.contains("_type") && node["_type"] == "Subscript" &&
        node.contains("value") && node["value"].is_object() &&
        node["value"].contains("_type") && node["value"]["_type"] == "Name" &&
        node["value"].contains("id") && node.contains("slice") &&
        node["slice"].is_object() && node["slice"].contains("_type") &&
        node["slice"]["_type"] == "Tuple" && node["slice"].contains("elts") &&
        node["slice"]["elts"].is_array() && !node["slice"]["elts"].empty())
      {
        auto it =
          var_to_class.find(node["value"]["id"].template get<std::string>());
        if (it != var_to_class.end() && it->second == class_name)
          visit(node["slice"]["elts"], next_enclosing);
      }
      for (auto it = node.begin(); it != node.end(); ++it)
        visit_class_tuple_subscripts(
          it.value(), class_name, var_to_class, visit, next_enclosing);
    }
    else if (node.is_array())
    {
      for (const auto &elem : node)
        visit_class_tuple_subscripts(
          elem, class_name, var_to_class, visit, enclosing_func);
    }
  }

  // Resolve a `Name` elt against @p enclosing_func's parameters and local
  // annotated assigns. Returns the annotation type name on success, or "" if
  // the name is not declared in that scope. Used as a foreign-scope fallback
  // when @p elts come from an AST other than `ast_` — `get_argument_type`'s
  // own `find_var_node_for_inference` only searches `ast_` (GitHub #4558).
  std::string resolve_name_in_enclosing_func(
    const std::string &name,
    const Json &enclosing_func)
  {
    auto lookup = [&](const Json &scope) -> std::string {
      Json node = find_annotated_assign(name, scope);
      if (has_annotation(node))
        return json_utils::get_annotation_type_name(node["annotation"]);
      return "";
    };

    if (enclosing_func.contains("args"))
    {
      const Json &arg_spec = enclosing_func["args"];
      for (const char *key : {"args", "kwonlyargs"})
        if (arg_spec.contains(key))
          if (std::string t = lookup(arg_spec[key]); !t.empty())
            return t;
    }
    if (enclosing_func.contains("body"))
      return lookup(enclosing_func["body"]);
    return "";
  }

  /// @brief Resolve @p func_name through any `from X import *` in @c ast_.
  ///
  /// Iterates wildcard ImportFrom nodes at module scope and probes each
  /// module's exported functions via @c module_manager_. Returns the
  /// function's declared return type if found; "" otherwise. A NoneType /
  /// missing return type falls back to @p func_name so callers treat it as
  /// a class-constructor-style call, matching the named-import branch's
  /// heuristic. Introduced for GitHub #4564.
  std::string resolve_wildcard_import_func(const std::string &func_name)
  {
    if (!module_manager_ || !ast_.contains("body"))
      return "";
    for (const auto &node : ast_["body"])
    {
      if (
        !node.contains("_type") || node["_type"] != "ImportFrom" ||
        !node.contains("names") || !node.contains("module"))
        continue;
      bool is_star = false;
      for (const auto &name : node["names"])
        if (name.contains("name") && name["name"] == "*")
        {
          is_star = true;
          break;
        }
      if (!is_star)
        continue;
      auto module = module_manager_->get_module(node["module"]);
      if (!module)
        continue;
      auto func_info = module->get_function(func_name);
      if (func_info.name_.empty())
        continue;
      if (
        func_info.return_type_.empty() || func_info.return_type_ == "NoneType")
        return func_name;
      return func_info.return_type_;
    }
    return "";
  }

  /// @brief Resolve a tuple-key elt against @p enclosing_func's scope.
  ///
  /// Generalises @ref resolve_name_in_enclosing_func to expressions whose
  /// operands are names declared outside this annotator's @c ast_ —
  /// constants, unary ops, and binary arithmetic. Numeric promotion follows
  /// Python semantics (any @c float operand → @c float; @c int + @c int →
  /// @c int). Returns "" when any operand is not resolvable in the foreign
  /// scope, signalling the caller to fall through to @c ast_-based
  /// inference. Introduced for GitHub #4564.
  std::string
  resolve_expr_in_enclosing_func(const Json &elt, const Json &enclosing_func)
  {
    if (!elt.contains("_type"))
      return "";
    const std::string &t = elt["_type"];
    if (t == "Constant")
      return get_type_from_constant(elt);
    if (t == "Name" && elt.contains("id"))
      return resolve_name_in_enclosing_func(elt["id"], enclosing_func);
    if (t == "UnaryOp" && elt.contains("operand"))
      return resolve_expr_in_enclosing_func(elt["operand"], enclosing_func);
    if (t == "BinOp" && elt.contains("left") && elt.contains("right"))
    {
      std::string l =
        resolve_expr_in_enclosing_func(elt["left"], enclosing_func);
      std::string r =
        resolve_expr_in_enclosing_func(elt["right"], enclosing_func);
      if (l.empty() || r.empty())
        return "";
      if (l == "float" || r == "float")
        return "float";
      if (l == "int" && r == "int")
        return "int";
      return "";
    }
    return "";
  }

  // Resolve each element of @p elts to a Python type name suitable for a
  // synthesised `tuple[...]` annotation. Slice literals and `slice(...)`
  // calls map to "slice"; scalars and other expressions are routed through
  // get_argument_type(). Non-slice elts are first looked up in @p
  // enclosing_func (when non-null) via @ref resolve_expr_in_enclosing_func
  // so foreign-AST scans can resolve variables defined outside this
  // annotator's `ast_` — including compound arithmetic expressions like
  // `bm * k` whose operand names are foreign-scope parameters
  // (GitHub #4558, #4564). Returns an empty vector when any element resolves
  // to "" or "Any" — bailing keeps the synthesised callee struct consistent
  // with whatever the caller actually builds at the subscript site.
  std::vector<std::string> infer_tuple_element_types(
    const Json &elts,
    const Json *enclosing_func = nullptr)
  {
    std::vector<std::string> types;
    types.reserve(elts.size());
    for (const Json &elt : elts)
    {
      if (!elt.contains("_type"))
        return {};
      const std::string &t = elt["_type"];
      if (t == "Slice")
      {
        types.emplace_back("slice");
        continue;
      }
      if (
        t == "Call" && elt.contains("func") && elt["func"].is_object() &&
        elt["func"].contains("_type") && elt["func"]["_type"] == "Name" &&
        elt["func"].contains("id") && elt["func"]["id"] == "slice")
      {
        types.emplace_back("slice");
        continue;
      }
      std::string resolved;
      if (enclosing_func != nullptr)
        resolved = resolve_expr_in_enclosing_func(elt, *enclosing_func);
      if (resolved.empty())
        resolved = get_argument_type(elt);
      if (resolved.empty() || resolved == "Any")
        return {};
      types.push_back(std::move(resolved));
    }
    return types;
  }

  // Method to find all function calls to a specific function
  std::vector<Json>
  find_function_calls(const std::string &func_name, const Json &body)
  {
    std::vector<Json> calls;
    find_function_calls_recursive(func_name, body, calls);
    return calls;
  }

  // Recursive helper to find function calls
  void find_function_calls_recursive(
    const std::string &func_name,
    const Json &node,
    std::vector<Json> &calls)
  {
    if (node.is_object())
    {
      // Check if this is a function call to our target function
      if (
        node.contains("_type") && node["_type"] == "Call" &&
        node.contains("func") && node["func"]["_type"] == "Name" &&
        node["func"]["id"] == func_name)
      {
        calls.push_back(node);
      }

      // Recursively search all fields
      for (auto it = node.begin(); it != node.end(); ++it)
        find_function_calls_recursive(func_name, it.value(), calls);
    }
    else if (node.is_array())
    {
      for (const auto &element : node)
        find_function_calls_recursive(func_name, element, calls);
    }
  }

  std::vector<std::string> collect_parameter_types_from_calls(
    size_t param_index,
    const std::vector<Json> &function_calls)
  {
    std::vector<std::string> types;
    std::unordered_set<std::string> seen;

    for (const Json &call : function_calls)
    {
      if (!call.contains("args") || param_index >= call["args"].size())
        continue;

      std::string arg_type = get_argument_type(call["args"][param_index]);
      if (arg_type.empty())
        continue;

      if (seen.insert(arg_type).second)
        types.push_back(arg_type);
    }

    return types;
  }

  // For a parameter annotated bare `tuple`, recover the element types by
  // intersecting all call sites that pass a Tuple literal at position
  // @p param_index. Returns empty when arity disagrees, any caller passes a
  // non-Tuple-literal argument, or no usable call sites exist; element types
  // that disagree across calls fall back to "Any". Used to specialise bare
  // `tuple` parameter annotations to `tuple[T0, T1, ...]` so tuple-unpacking
  // sees the struct shape (GitHub #4515).
  std::vector<std::string> infer_tuple_param_types_from_calls(
    size_t param_index,
    const std::vector<Json> &function_calls)
  {
    std::vector<std::string> result;
    bool initialized = false;

    for (const Json &call : function_calls)
    {
      if (!call.contains("args") || param_index >= call["args"].size())
        continue;

      const Json &arg = call["args"][param_index];
      if (
        !arg.contains("_type") || arg["_type"] != "Tuple" ||
        !arg.contains("elts"))
        return {};

      std::vector<std::string> shape;
      shape.reserve(arg["elts"].size());
      for (const Json &elt : arg["elts"])
      {
        std::string t = get_argument_type(elt);
        // Empty result and bare container kinds without parameters all
        // resolve to incomplete/empty types downstream. Clamp them to "Any"
        // so the synthesised struct never carries an empty_typet() component.
        if (t.empty() || t == "tuple" || t == "list" || t == "dict")
          t = "Any";
        shape.push_back(std::move(t));
      }

      if (!initialized)
      {
        result = std::move(shape);
        initialized = true;
      }
      else
      {
        if (shape.size() != result.size())
          return {};
        for (size_t k = 0; k < result.size(); ++k)
          if (result[k] != shape[k])
            result[k] = "Any";
      }
    }

    return result;
  }

  // Declarations only — definitions live in
  // python_annotation/annotation_symbolic.inl, included after the class.
  bool are_all_user_classes(const std::vector<std::string> &types) const;

  std::string build_specialized_function_name(
    const std::string &base_name,
    const std::string &class_name) const;

  std::string
  resolve_name_assigned_class(const std::string &name, const Json &node);

  void rewrite_specialized_calls(
    const std::string &original_name,
    size_t param_index,
    const std::unordered_map<std::string, std::string> &specialized_names,
    Json &node);

  void apply_pending_specializations();

  // Returns the ancestors of a class (including itself) in BFS order via
  // "bases". Handles multiple inheritance and cycle-safe (visited guard).
  // "object" is intentionally excluded — it has no ESBMC IR representation.
  std::vector<std::string> get_class_ancestors(const std::string &class_name);

  // Returns the lowest common ancestor of two user-defined class types, or "".
  std::string
  find_common_ancestor(const std::string &type_a, const std::string &type_b);

  // Method to infer parameter type from function calls
  std::string infer_parameter_type_from_calls(
    size_t param_index,
    const std::vector<Json> &function_calls)
  {
    std::string inferred_type;

    for (const Json &call : function_calls)
    {
      if (call.contains("args") && param_index < call["args"].size())
      {
        const Json &arg = call["args"][param_index];
        std::string arg_type = get_argument_type(arg);

        if (!arg_type.empty())
        {
          if (inferred_type.empty())
            inferred_type = arg_type;
          else if (inferred_type != arg_type)
          {
            std::string common = find_common_ancestor(inferred_type, arg_type);
            if (!common.empty())
              inferred_type = common;
            else
              return "Any";
          }
        }
      }
    }

    return inferred_type;
  }

  // Method to get the type of an argument in a function call
  std::string get_argument_type(const Json &arg);

  // Method to get the full type of a list literal
  std::string get_list_type_from_literal(const Json &list_arg);

  /* Get the global elements referenced by a function */
  void get_global_elements(const Json &node)
  {
    // Checks if the current node is a variable identifier
    if (
      node.contains("_type") && node["_type"] == "Name" && node.contains("id"))
    {
      const std::string &var_name = node["id"];
      Json var_node = json_utils::find_var_decl(var_name, "", ast_);
      if (!var_node.empty())
      {
        gs_.add_variable(var_name);
        referenced_global_elements.push_back(var_node);
      }
    }

    if (
      node.contains("_type") && node["_type"] == "Call" &&
      node.contains("func") && node["func"]["_type"] == "Name")
    {
      const std::string &class_name = node["func"]["id"];
      // Checks if the current node is a constructor call
      Json class_node = json_utils::find_class(ast_["body"], class_name);
      if (!class_node.empty())
      {
        gs_.add_class(class_name);
        referenced_global_elements.push_back(class_node);
      }
      else
      {
        const auto &func_name = node["func"]["id"];
        if (!type_utils::is_builtin_type(func_name))
        {
          try
          {
            auto func_node = json_utils::find_function(ast_["body"], func_name);
            get_global_elements(func_node);
          }
          catch (std::runtime_error &)
          {
          }
        }
      }
    }

    // Recursively iterates through all fields of the node if it is an object
    if (node.is_object())
    {
      for (auto it = node.begin(); it != node.end(); ++it)
        get_global_elements(it.value());
    }

    // Iterates over the elements if the current node is an array
    else if (node.is_array())
    {
      for (const auto &element : node)
        get_global_elements(element);
    }
    filter_global_elements_ = true;
  }

  void annotate_global_scope()
  {
    add_annotation(ast_);
  }

  void annotate_function(Json &function_element)
  {
    std::string saved_func_name_context = current_func_name_context_;
    Json *saved_parent_func = parent_func; // Save previous parent

    const std::string &func_name =
      function_element["name"].template get<std::string>();

    // Build hierarchical path ONLY if we're not inside a class
    if (!current_class_name_.empty())
    {
      // We're inside a class - do NOT accumulate hierarchical context
      current_func_name_context_ = func_name;
    }
    else if (!saved_func_name_context.empty())
    {
      // Nested function outside a class - accumulate context
      current_func_name_context_ = saved_func_name_context + "@F@" + func_name;
    }
    else
    {
      // Top-level function
      current_func_name_context_ = func_name;
    }

    parent_func = current_func; // Current becomes parent
    current_func = &function_element;

    // Infer types for unannotated parameters based on function calls
    infer_parameter_types(function_element);

    // Add type annotations within the function
    add_annotation(function_element);

    // Skip return type inference for __init__ (constructors always return None)
    if (func_name == "__init__")
    {
      // Constructors return None by default - no need to infer
      if (function_element["returns"].is_null())
      {
        function_element["returns"] = {
          {"_type", "Constant"},
          {"value", nullptr}, // None
          {"lineno", function_element["lineno"]},
          {"col_offset", function_element["col_offset"]},
          {"end_lineno", function_element["lineno"]},
          {"end_col_offset",
           function_element["col_offset"].template get<int>() + 4}};
      }

      // Update the end column offset after adding annotations
      update_end_col_offset(function_element);
      current_func = nullptr;
      parent_func = saved_parent_func; // Restore previous parent
      current_func_name_context_ = saved_func_name_context;
      return; // Exit early for __init__
    }

    // Check if we should override the return annotation
    bool should_override =
      config.options.get_bool_option("override-return-annotation");
    bool has_no_annotation = function_element["returns"].is_null();

    if (has_no_annotation || should_override)
    {
      std::string inferred_type =
        infer_from_return_statements(function_element["body"], func_name);

      // Only add annotation if we successfully inferred a type from return statements
      // and the function does not have mixed value+None returns
      if (!inferred_type.empty() && inferred_type != "NoneType")
      {
        // Check if there are also None returns (mixed value+None)
        // If so, leave the annotation as null for the converter to handle
        // via Optional wrapping
        bool has_none_return = has_return_none(function_element["body"]);

        if (!has_none_return)
        {
          // Update the function node to include the return type annotation
          function_element["returns"] = {
            {"_type", "Name"},
            {"id", inferred_type},
            {"ctx", {{"_type", "Load"}}},
            {"lineno", function_element["lineno"]},
            {"col_offset", function_element["col_offset"]},
            {"end_lineno", function_element["lineno"]},
            {"end_col_offset",
             function_element["col_offset"].template get<int>() +
               inferred_type.size()}};
        }
      }
      else if (inferred_type == "NoneType")
      {
        // Function only returns None - annotate as None
        function_element["returns"] = {
          {"_type", "Constant"},
          {"value", nullptr},
          {"lineno", function_element["lineno"]},
          {"col_offset", function_element["col_offset"]},
          {"end_lineno", function_element["lineno"]},
          {"end_col_offset",
           function_element["col_offset"].template get<int>() + 4}};
      }
      // If no return type could be inferred, leave returns as null
      // (function has no explicit return statement)
    }

    // Update the end column offset after adding annotations
    update_end_col_offset(function_element);

    current_func = nullptr;
    parent_func = saved_parent_func; // Restore previous parent
    current_func_name_context_ = saved_func_name_context;
  }

  void annotate_class(Json &class_element)
  {
    std::string saved_class_name = current_class_name_;
    std::string saved_context = current_func_name_context_;
    //    current_func_name_context_ = ""; // Reset for class methods

    current_class_name_ = class_element["name"].template get<std::string>();

    for (Json &class_member : class_element["body"])
    {
      // Process methods in the class
      if (class_member["_type"] == "FunctionDef")
      {
        // Add type annotations within the class member function
        annotate_function(class_member);
      }
      // Process unannotated class attributes (e.g., species = "Homo sapiens")
      else if (class_member["_type"] == "Assign")
      {
        std::string inferred_type;

        // Infer type from the RHS value
        if (
          infer_type(class_member, class_element, inferred_type) ==
          InferResult::OK)
        {
          // Convert Assign to AnnAssign with the inferred type
          update_assignment_node(class_member, inferred_type);
        }
        else
        {
          // If type inference fails, throw error with helpful message
          std::string attr_name =
            class_member["targets"][0]["id"].template get<std::string>();
          throw std::runtime_error(
            "Cannot infer type for class attribute '" + attr_name +
            "' in class '" + class_element["name"].template get<std::string>() +
            "'. Please add explicit type annotation.");
        }
      }
    }

    current_class_name_ = saved_class_name;
    current_func_name_context_ = saved_context;
  }

  // Declarations only — definitions live in
  // python_annotation/annotation_conversion.inl, included after the class.
  std::string get_current_func_name();

  std::string get_type_from_constant(const Json &element);

  std::string get_type_from_binary_expr(const Json &stmt, const Json &body);

  std::string infer_lambda_return_type(const Json &lambda_elem) const;

  std::string
  get_function_return_type(const std::string &func_name, const Json &ast);

  std::string get_type_from_lhs(const std::string &id, const Json &body);

  std::string get_list_subtype(const Json &list);

  bool extract_type_info(
    const Json &annotation,
    std::string &base_type,
    std::string &element_type);

  std::string infer_dict_value_type(const Json &var_node);

  std::string
  resolve_subscript_type(const Json &subscript_node, const Json &body);

  bool
  is_imported_name_in_body(const std::string &name, const Json &stmts) const;

  std::string get_type_from_rhs_variable(const Json &element, const Json &body);

  std::string get_type_from_call(const Json &element);

  std::string get_type_from_method(const Json &call);

  // Method to infer type from conditional expressions (IfExp)
  std::string get_type_from_ifexp(const Json &ifexp_node, const Json &body);

  InferResult
  infer_type(const Json &stmt, const Json &body, std::string &inferred_type);

  std::string match_literal_argument(
    const Json &call_node,
    std::vector<Json> overloads) const;

  // Find the best matching overload
  std::string resolve_overload_return_type(
    const std::string &func_name,
    const Json &call_node) const;


  std::string get_object_name(const Json &call, const std::string &prefix);

  std::string get_string_method_return_type(const std::string &method) const;


  // Method to infer type from conditional expressions (IfExp)


  void infer_loop_variable_types(Json &while_stmt)
  {
    if (!while_stmt.contains("body"))
      return;

    for (auto &stmt : while_stmt["body"])
    {
      // Look for pattern: loop_var: Any = iterable[ESBMC_index_N]
      // A bare annotation like `x: int` has value == null; nlohmann::json's
      // `contains("value")` returns true for present-but-null members, so an
      // explicit is_null() guard is required to avoid a type_error on the
      // subsequent subscript.
      if (
        stmt["_type"] == "AnnAssign" && stmt.contains("value") &&
        !stmt["value"].is_null() && stmt["value"]["_type"] == "Subscript")
      {
        if (!stmt["value"]["value"].contains("id"))
          continue;

        std::string iter_var =
          stmt["value"]["value"]["id"].template get<std::string>();

        // Find the iterable's annotation
        Json iter_node;
        if (current_func != nullptr && (*current_func).contains("body"))
          iter_node = find_annotated_assign(iter_var, (*current_func)["body"]);
        if (
          iter_node.empty() && current_func != nullptr &&
          (*current_func).contains("args"))
          iter_node =
            find_annotated_assign(iter_var, (*current_func)["args"]["args"]);
        if (iter_node.empty())
          iter_node = find_annotated_assign(iter_var, ast_["body"]);

        if (iter_node.empty() || !iter_node.contains("annotation"))
          continue;

        auto &iter_annotation = iter_node["annotation"];

        // Extract element type from container annotation
        if (
          iter_annotation["_type"] == "Subscript" &&
          iter_annotation.contains("value") &&
          iter_annotation["value"].contains("id"))
        {
          std::string container_type =
            iter_annotation["value"]["id"].template get<std::string>();

          if (container_type == "list" || container_type == "List")
          {
            // Extract T from list[T]
            if (iter_annotation.contains("slice"))
              stmt["annotation"] = iter_annotation["slice"];
          }
          else if (container_type == "dict" || container_type == "Dict")
          {
            // For dict iteration, iterate over keys (first type parameter)
            if (
              iter_annotation.contains("slice") &&
              iter_annotation["slice"]["_type"] == "Tuple" &&
              iter_annotation["slice"].contains("elts") &&
              !iter_annotation["slice"]["elts"].empty())
            {
              stmt["annotation"] = iter_annotation["slice"]["elts"][0];
            }
          }
        }
        else if (iter_annotation["_type"] == "Name")
        {
          // For str iteration, element type is also str
          std::string type_name =
            iter_annotation["id"].template get<std::string>();
          if (type_name == "str")
            stmt["annotation"] = iter_annotation;
        }
      }
    }
  }

  void add_annotation(Json &body)
  {
    for (auto &element : body["body"])
    {
      auto itr = std::find(
        referenced_global_elements.begin(),
        referenced_global_elements.end(),
        element);

      if (filter_global_elements_ && itr == referenced_global_elements.end())
        continue;

      if (element.contains("lineno"))
        current_line_ = element["lineno"].template get<int>();

      auto &stmt_type = element["_type"];

      if (stmt_type == "If" || stmt_type == "While" || stmt_type == "Try")
      {
        add_annotation(element);

        // Infer loop variable types for preprocessor-transformed for loops
        if (stmt_type == "While")
          infer_loop_variable_types(element);

        // Process else block if it exists
        if (
          stmt_type == "If" && element.contains("orelse") &&
          !element["orelse"].empty())
        {
          // Create a temporary body structure for the else block
          Json else_body = {{"body", element["orelse"]}};
          add_annotation(else_body);
          // Update the original orelse with annotated version
          element["orelse"] = else_body["body"];
        }

        continue;
      }

      if (stmt_type == "FunctionDef")
      {
        // Only annotate nested functions, not the current function itself
        if (current_func != nullptr && &element != current_func)
          annotate_function(element);

        continue;
      }

      const std::string function_flag = config.options.get_option("function");
      if (!function_flag.empty())
      {
        if (
          stmt_type == "Expr" && element.contains("value") &&
          element["value"]["_type"] == "Call" &&
          element["value"]["func"]["_type"] == "Name")
        {
          auto &func_node = json_utils::find_function(
            ast_["body"], element["value"]["func"]["id"]);
          if (!func_node.empty())
            add_annotation(func_node);
        }
      }

      if (stmt_type != "Assign" || !element["type_comment"].is_null())
        continue;

      // Skip tuple/list unpacking assignments
      // The C++ converter will handle them directly with proper type inference
      if (
        element.contains("targets") && !element["targets"].empty() &&
        element["targets"][0].contains("_type") &&
        (element["targets"][0]["_type"] == "Tuple" ||
         element["targets"][0]["_type"] == "List"))
      {
        continue;
      }

      std::string inferred_type("");

      // Check if RHS is a type identifier
      if (element["value"]["_type"] == "Name")
      {
        const std::string &rhs_name = element["value"]["id"];
        if (type_utils::is_type_identifier(rhs_name))
        {
          // This is a type object assignment: x = int
          inferred_type = "type";
          update_assignment_node(element, inferred_type);
          if (itr != referenced_global_elements.end())
            *itr = element;
          continue;
        }
      }

      // Check if LHS was previously annotated
      if (
        element.contains("targets") && element["targets"][0]["_type"] == "Name")
      {
        inferred_type = get_type_from_lhs(element["targets"][0]["id"], body);
      }

      if (infer_type(element, body, inferred_type) == InferResult::UNKNOWN)
        continue;

      const auto &rhs = element["value"];

      if (
        (rhs["_type"] == "Constant" && rhs["value"].is_null()) ||
        (rhs["_type"] == "NameConstant" && rhs["value"].is_null()) ||
        (rhs["_type"] == "Name" && rhs.contains("id") && rhs["id"] == "None"))
        inferred_type = "NoneType";

      if (inferred_type.empty())
      {
        std::ostringstream oss;
        oss << "Type inference failed for "
            << stmt_type.template get<std::string>() << " at line "
            << current_line_;

        throw std::runtime_error(oss.str());
      }

      update_assignment_node(element, inferred_type);
      if (itr != referenced_global_elements.end())
        *itr = element;
    }
  }

  // Declarations only — definitions live in
  // python_annotation/annotation_expr.inl, included after the class.
  Json create_name_annotation(
    const std::string &type_id,
    int lineno,
    int col_offset,
    int end_lineno,
    int end_col_offset);

  Json create_subscript_annotation(
    const std::string &base_type,
    const std::string &element_type,
    int lineno,
    int col_offset,
    int end_lineno);

  // Build a `tuple[t0, t1, ...]` Subscript annotation node. Used by the
  // parameter-inference pass to specialise bare `tuple` annotations once
  // the element types have been recovered from call sites (GitHub #4515).
  Json create_tuple_subscript_annotation(
    const std::vector<std::string> &elem_types,
    int lineno,
    int col_offset,
    int end_lineno);

  Json create_annotation_from_type(
    const std::string &inferred_type,
    int lineno,
    int col_offset,
    int end_lineno);

  void update_assignment_node(Json &element, const std::string &inferred_type);

  void add_parameter_annotation(Json &param, const std::string &type);

  void update_end_col_offset(Json &ast);

  /**
   * @brief Resolve the user-defined class name of variable @p obj from the
   *        surrounding scope (function parameters, local declarations, or
   *        globals).
   * @return The class name, or an empty string when not resolvable.
   */
  std::string resolve_object_class_name(const std::string &obj);

  /**
   * @brief Infer the type of the element at position @p index in @p rhs, the
   *        right-hand side of a tuple/list unpacking assignment.
   *
   * Handles two RHS shapes: a Tuple/List literal (types element @p index
   * directly), and an `obj.attr` Attribute access (resolves @p obj to a
   * class, then recurses into the matching `self.attr = ...` initialiser).
   *
   * @return The inferred element type, or "Any" when the RHS shape is not
   *         recognised.
   */
  std::string infer_unpacked_element_type(const Json &rhs, size_t index);

  /**
   * @brief Synthesise an AnnAssign-shaped Json for @p node_name when @p elem
   *        is a tuple/list unpacking `Assign` that binds it (GitHub #4532).
   * @return The synthetic node, or an empty Json on no match.
   */
  Json
  match_unpacking_assignment(const Json &elem, const std::string &node_name)
  {
    if (
      !elem.contains("_type") || elem["_type"] != "Assign" ||
      !elem.contains("targets") || !elem["targets"].is_array() ||
      elem["targets"].empty())
      return Json();
    const Json &target = elem["targets"][0];
    if (
      !target.is_object() || !target.contains("_type") ||
      (target["_type"] != "Tuple" && target["_type"] != "List") ||
      !target.contains("elts") || !target["elts"].is_array())
      return Json();

    const Json &elts = target["elts"];
    for (size_t i = 0; i < elts.size(); ++i)
    {
      if (
        !elts[i].is_object() || !elts[i].contains("_type") ||
        elts[i]["_type"] != "Name" || !elts[i].contains("id") ||
        elts[i]["id"] != node_name)
        continue;
      std::string elem_type = elem.contains("value")
                                ? infer_unpacked_element_type(elem["value"], i)
                                : "Any";
      return Json{
        {"_type", "AnnAssign"},
        {"target", Json{{"_type", "Name"}, {"id", node_name}}},
        {"annotation", Json{{"_type", "Name"}, {"id", elem_type}}},
        {"lineno", elem.value("lineno", 0)}};
    }
    return Json();
  }

  const Json
  find_annotated_assign(const std::string &node_name, const Json &body)
  {
    for (const Json &elem : body)
    {
      // Check if this element is the variable we're looking for
      if (
        elem.contains("_type") &&
        ((elem["_type"] == "AnnAssign" && elem.contains("target") &&
          elem["target"].contains("id") &&
          elem["target"]["id"].template get<std::string>() == node_name) ||
         (elem["_type"] == "Assign" && elem.contains("targets") &&
          elem["targets"].is_array() && !elem["targets"].empty() &&
          elem["targets"][0].contains("_type") &&
          elem["targets"][0]["_type"] == "Name" &&
          elem["targets"][0].contains("id") &&
          elem["targets"][0]["id"].template get<std::string>() == node_name) ||
         (elem["_type"] == "arg" && elem["arg"] == node_name)))
      {
        return elem;
      }

      // Tuple/list unpacking assignment (`A, B = rhs`): synthesise an
      // AnnAssign so callers reading node["annotation"]["id"] resolve the
      // destructured element's type (GitHub #4532).
      if (Json synth = match_unpacking_assignment(elem, node_name);
          !synth.empty())
        return synth;

      // Recursively search inside nested blocks
      if (elem.contains("_type"))
      {
        const std::string &elem_type = elem["_type"];

        // Search inside If blocks (both if and else branches)
        if (elem_type == "If")
        {
          if (elem.contains("body") && !elem["body"].empty())
          {
            Json result = find_annotated_assign(node_name, elem["body"]);
            if (!result.empty())
              return result;
          }
          if (elem.contains("orelse") && !elem["orelse"].empty())
          {
            Json result = find_annotated_assign(node_name, elem["orelse"]);
            if (!result.empty())
              return result;
          }
        }
        // Search inside While blocks
        else if (elem_type == "While")
        {
          if (elem.contains("body") && !elem["body"].empty())
          {
            Json result = find_annotated_assign(node_name, elem["body"]);
            if (!result.empty())
              return result;
          }
        }
        // Search inside For blocks
        else if (elem_type == "For")
        {
          if (elem.contains("body") && !elem["body"].empty())
          {
            Json result = find_annotated_assign(node_name, elem["body"]);
            if (!result.empty())
              return result;
          }
        }
        // Search inside Try blocks
        else if (elem_type == "Try")
        {
          if (elem.contains("body") && !elem["body"].empty())
          {
            Json result = find_annotated_assign(node_name, elem["body"]);
            if (!result.empty())
              return result;
          }
          if (elem.contains("handlers") && !elem["handlers"].empty())
          {
            for (const Json &handler : elem["handlers"])
            {
              if (handler.contains("body") && !handler["body"].empty())
              {
                Json result = find_annotated_assign(node_name, handler["body"]);
                if (!result.empty())
                  return result;
              }
            }
          }
        }
      }
    }
    return Json();
  }

  Json &ast_;
  global_scope &gs_;
  std::shared_ptr<module_manager> module_manager_;
  Json *current_func;
  Json
    *parent_func; // Track parent function for nested function scope resolution
  int current_line_;
  std::string python_filename_;
  bool filter_global_elements_ = false;
  std::vector<Json> referenced_global_elements;
  std::set<std::string> functions_in_analysis_;
  std::set<std::string> resolving_rhs_vars_;
  std::string current_func_name_context_;
  std::string current_class_name_;
  struct pending_specializationt
  {
    std::string function_name;
    size_t param_index;
    std::vector<std::string> class_types;
  };
  std::vector<pending_specializationt> pending_specializations_;
  std::vector<const Json *> extra_subscript_inference_asts_;
};

// Out-of-class template member definitions. These files are part of
// this header and must not be included independently — the
// definitions need the `python_annotation<Json>` class layout above.
#include <python-frontend/python_annotation/annotation_conversion.inl>
#include <python-frontend/python_annotation/annotation_expr.inl>
#include <python-frontend/python_annotation/annotation_symbolic.inl>

