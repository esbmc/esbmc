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
// names so existing call sites inside the .inl files remain byte-
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

// Python-frontend annotation pass. Walks the JSON AST produced by
// `parser.py` and decorates every assignment, parameter, and return
// site with an inferred type so the downstream converter sees a
// fully-typed program. Method definitions are split across .inl
// files in python_annotation/ and are `#include`d at the bottom of
// this header. The class shape, member layout, and template
// parameter are unchanged from the pre-split version — this header
// is a thin facade over the .inl files.
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

  // Entry points exercised by callers outside python_annotation.h.
  void preprocess_constructor_calls(const Json &node);
  void preprocess_method_calls(const Json &node);
  void add_type_annotation();
  void add_type_annotation(const std::string &func_name);

  // Register an additional AST whose multi-axis subscript usages must be
  // considered when inferring `__getitem__` / `__setitem__` key tuple
  // types. Used when annotating an imported module to make the importing
  // module's call sites visible (GitHub #4545). The caller retains
  // ownership; @p ast must outlive this annotator.
  void add_extra_subscript_inference_source(const Json &ast);

private:
  // Implementations live in annotation_conversion.inl unless noted.

  // ----- AST lookup and walk helpers -----
  Json find_var_node_for_inference(const std::string &var_name);
  const Json
  find_annotated_assign(const std::string &node_name, const Json &body);
  Json
  match_unpacking_assignment(const Json &elem, const std::string &node_name);
  std::vector<Json>
  find_function_calls(const std::string &func_name, const Json &body);
  void find_function_calls_recursive(
    const std::string &func_name,
    const Json &node,
    std::vector<Json> &calls);
  void get_global_elements(const Json &node);

  // ----- type inspectors -----
  std::string get_current_func_name();
  std::string get_type_from_constant(const Json &element);
  std::string get_type_from_lhs(const std::string &id, const Json &body);
  std::string get_list_subtype(const Json &list);
  std::string get_list_type_from_literal(const Json &list_arg);
  std::string get_object_name(const Json &call, const std::string &prefix);
  std::string get_string_method_return_type(const std::string &method) const;
  bool extract_type_info(
    const Json &annotation,
    std::string &base_type,
    std::string &element_type);
  std::string infer_dict_value_type(const Json &var_node);
  std::string
  resolve_subscript_type(const Json &subscript_node, const Json &body);
  bool
  is_imported_name_in_body(const std::string &name, const Json &stmts) const;
  std::string infer_unpacked_element_type(const Json &rhs, size_t index);
  std::string resolve_object_class_name(const std::string &obj);
  std::string match_literal_argument(
    const Json &call_node,
    std::vector<Json> overloads) const;
  std::string resolve_overload_return_type(
    const std::string &func_name,
    const Json &call_node) const;

  // ----- core inference dispatchers -----
  std::string get_argument_type(const Json &arg);
  std::string get_type_from_binary_expr(const Json &stmt, const Json &body);
  std::string get_type_from_rhs_variable(const Json &element, const Json &body);
  std::string get_type_from_call(const Json &element);
  std::string get_type_from_method(const Json &call);
  std::string get_type_from_ifexp(const Json &ifexp_node, const Json &body);
  InferResult
  infer_type(const Json &stmt, const Json &body, std::string &inferred_type);
  std::string
  get_function_return_type(const std::string &func_name, const Json &ast);
  std::string infer_lambda_return_type(const Json &lambda_elem) const;
  std::string
  infer_from_return_statements(const Json &body, const std::string &func_name);

  // ----- annotation-node constructors (annotation_expr.inl) -----
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

  // ----- class inheritance and specialisation (annotation_symbolic.inl) -----
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
  // Returns the lowest common ancestor of two user-defined class types,
  // or "" when none exists.
  std::string
  find_common_ancestor(const std::string &type_a, const std::string &type_b);

  // ----- per-class subscript-key tuple inference -----
  std::vector<std::string>
  infer_subscript_key_tuple_types(const std::string &class_name);
  void collect_class_instance_map(
    const Json &ast,
    std::unordered_map<std::string, std::string> &out) const;
  template <typename Visitor>
  void visit_class_tuple_subscripts(
    const Json &node,
    const std::string &class_name,
    const std::unordered_map<std::string, std::string> &var_to_class,
    Visitor &&visit,
    const Json *enclosing_func = nullptr);
  std::string resolve_name_in_enclosing_func(
    const std::string &name,
    const Json &enclosing_func);
  std::string resolve_wildcard_import_func(const std::string &func_name);
  std::string
  resolve_expr_in_enclosing_func(const Json &elt, const Json &enclosing_func);
  std::vector<std::string> infer_tuple_element_types(
    const Json &elts,
    const Json *enclosing_func = nullptr);

  // ----- function-call collection and parameter-type inference -----
  std::vector<std::string> collect_parameter_types_from_calls(
    size_t param_index,
    const std::vector<Json> &function_calls);
  std::vector<std::string> infer_tuple_param_types_from_calls(
    size_t param_index,
    const std::vector<Json> &function_calls);
  std::string infer_parameter_type_from_calls(
    size_t param_index,
    const std::vector<Json> &function_calls);
  void infer_parameter_types(Json &function_element);

  // ----- top-level orchestration entry points -----
  void annotate_global_scope();
  void annotate_function(Json &function_element);
  void annotate_class(Json &class_element);
  void infer_loop_variable_types(Json &while_stmt);
  void add_annotation(Json &body);

  // ----- state -----
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
