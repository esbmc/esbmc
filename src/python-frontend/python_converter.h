#pragma once

#include <nlohmann/json.hpp>
#include <python-frontend/complex_handler.h>
#include <python-frontend/function_call/cache.h>
#include <python-frontend/global_scope.h>
#include <python-frontend/python_dict_handler.h>
#include <python-frontend/python_math.h>
#include <python-frontend/string/string_handler.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/type_utils.h>
#include <python-frontend/python_set.h>
#include <util/context.h>
#include <util/namespace.h>
#include <util/std_code.h>
#include <util/symbol_generator.h>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>

class codet;
class struct_typet;
class function_id;
class symbol_id;
class function_call_expr;
class type_handler;
class string_builder;
class module_locator;
class tuple_handler;
class python_typechecking;
class python_class_builder;
class python_lambda;
class python_exception_handler;
class get_expr_depth_guard;

/**
 * @class python_converter
 * @brief Main converter for transforming Python AST into ESBMC's intermediate representation.
 *
 * This class is responsible for converting Python source code (represented as JSON AST)
 * into GOTO programs that can be processed by ESBMC's symbolic execution engine.
 */
class python_converter
{
public:
  python_converter(
    contextt &_context,
    const nlohmann::json *ast,
    const global_scope &gs);

  ~python_converter();

  void convert();

  // Accessors for handlers
  void set_converting_rhs(bool value)
  {
    is_converting_rhs = value;
  }
  void set_current_lhs(exprt *value)
  {
    current_lhs = value;
  }
  void set_current_func_name(const std::string &name)
  {
    current_func_name_ = name;
  }
  void set_current_element_type(const typet &type)
  {
    current_element_type = type;
  }
  symbolt *add_symbol_and_get_ptr(symbolt &symbol)
  {
    return symbol_table_.move_symbol_to_context(symbol);
  }

  exprt create_lhs_expression(
    const nlohmann::json &target,
    symbolt *lhs_symbol,
    const locationt &location);

  const std::string &get_current_func_name() const
  {
    return current_func_name_;
  }
  const nlohmann::json &get_ast_json() const
  {
    return *ast_json;
  }
  exprt get_expr(const nlohmann::json &element);

  /**
   * @brief Handles tuple `+`/`*` operators and lexicographic ordering.
   *
   * Concatenation builds a new tuple struct; repetition with a constant int
   * builds an n-fold repeat; `Lt`/`LtE`/`Gt`/`GtE` lower to element-wise
   * lexicographic comparisons. Returns nil_exprt for non-tuple operands or
   * unsupported variants. Public so the sorted()/reversed() lowering can reuse
   * the comparator for a convert-time sorting network over tuples.
   */
  exprt handle_tuple_operations(
    const std::string &op,
    exprt &lhs,
    exprt &rhs,
    const nlohmann::json &element);

  std::string get_op(const std::string &op, const typet &type) const;
  typet get_type_from_annotation(
    const nlohmann::json &annotation_node,
    const nlohmann::json &element);

  string_builder &get_string_builder();

  python_dict_handler *get_dict_handler()
  {
    return dict_handler_;
  }

  python_exception_handler &get_exception_handler()
  {
    return *exception_handler_;
  }

  const python_exception_handler &get_exception_handler() const
  {
    return *exception_handler_;
  }

  python_math &get_math_handler()
  {
    return math_handler_;
  }

  complex_handler &get_complex_handler()
  {
    return complex_handler_;
  }

  string_handler &get_string_handler()
  {
    return string_handler_;
  }

  tuple_handler &get_tuple_handler() const
  {
    return *tuple_handler_;
  }

  const nlohmann::json &ast() const
  {
    return *ast_json;
  }

  contextt &symbol_table() const
  {
    return symbol_table_;
  }

  type_handler &get_type_handler()
  {
    return type_handler_;
  }

  bool type_assertions_enabled() const;

  const std::string &python_file() const
  {
    return current_python_file;
  }

  const std::string &main_python_filename() const
  {
    return main_python_file;
  }

  const std::string &current_function_name() const
  {
    return current_func_name_;
  }

  const std::string &current_classname() const
  {
    return current_class_name_;
  }

  const namespacet &name_space() const
  {
    return ns;
  }

  function_call_cache &get_function_call_cache()
  {
    return function_call_cache_;
  }

  const function_call_cache &get_function_call_cache() const
  {
    return function_call_cache_;
  }

  void add_symbol(const symbolt s)
  {
    symbol_table_.add(s);
  }

  void add_instruction(const exprt &expr)
  {
    if (current_block)
      current_block->copy_to_operands(expr);
  }

  void update_symbol(const exprt &expr) const;

  symbolt *find_symbol(const std::string &symbol_id) const;

  bool is_imported_module(const std::string &module_name) const;

  const std::string
  get_imported_module_path(const std::string &module_name) const
  {
    if (imported_modules.find(module_name) != imported_modules.end())
      return imported_modules.at(module_name);

    return {};
  }

  symbolt create_symbol(
    const std::string &module,
    const std::string &name,
    const std::string &id,
    const locationt &location,
    const typet &type) const;

  // Register a ``void name(void)`` C intrinsic in the Python symbol
  // table if not already present. Used for ESBMC built-ins whose bodies
  // live in the C library (cprover_library) but whose call sites are
  // synthesised by the Python frontend (atomic_begin/end, yield, the
  // pthread main hooks, __pyt_*); the C-style ``c:@F@<name>`` id is
  // needed so c2goto's linker resolves the call to the library body.
  void ensure_void_void_intrinsic(
    const std::string &name,
    const locationt &location);

  exprt make_char_array_expr(
    const std::vector<unsigned char> &string_literal,
    const typet &t);

  exprt get_literal(const nlohmann::json &element);

  locationt get_location_from_decl(const nlohmann::json &element) const;

  void copy_location_fields_from_decl(
    const nlohmann::json &from,
    nlohmann::json &to) const;

  exprt handle_string_comparison(
    const std::string &op,
    exprt &lhs,
    exprt &rhs,
    const nlohmann::json &element);

  python_typechecking &get_typechecker();
  const python_typechecking &get_typechecker() const;

private:
  friend class complex_handler;
  friend class function_call_expr;
  friend class numpy_call_expr;
  friend class function_call_builder;
  friend class type_handler;
  friend class python_list;
  friend class string_handler;
  friend class tuple_handler;
  friend class python_typechecking;
  friend class python_class_builder;
  friend class python_dict_handler;
  friend class python_set;
  friend class python_exception_handler;
  friend class get_expr_depth_guard;
  friend class python_converter_test_access;

  template <typename Func>
  decltype(auto) with_ast(const nlohmann::json *new_ast, Func &&f)
  {
    const nlohmann::json *old_ast = ast_json;
    ast_json = new_ast;
    auto result = f();
    ast_json = old_ast;
    return result;
  }

  void load_c_intrisics(code_blockt &block);

  void get_var_assign(const nlohmann::json &ast_node, codet &target_block);

  void preregister_global_variables(const nlohmann::json &ast_body);

  /// None/Optional redesign (step A/B): if `annotation` is a nullable reference
  /// to a user class — `Optional[Class]` or `Class | None` — return the user
  /// class name; else "". Callers form `Class*` and build the class on demand
  /// (so its struct symbol is complete, not a null/incomplete stub) per plan
  /// section 10.
  std::string annotated_optional_class(const nlohmann::json &annotation) const;

  typet
  resolve_variable_type(const std::string &var_name, const locationt &loc);

  void get_compound_assign(const nlohmann::json &ast_node, codet &target_block);

  void
  get_return_statements(const nlohmann::json &ast_node, codet &target_block);

  void get_function_definition(const nlohmann::json &function_node);

  void
  get_class_definition(const nlohmann::json &class_node, codet &target_block);

  exprt get_unary_operator_expr(const nlohmann::json &element);

  /// Walrus operator `(target := value)` (PEP 572): bind value to target as a
  /// side effect (emitted into the current block) and evaluate to that value.
  exprt get_named_expr(const nlohmann::json &element);

  /// True if a walrus operator (NamedExpr) appears anywhere in `node`'s tree.
  /// Used to refuse a walrus in conditionally / repeatedly evaluated positions
  /// (while-test, ternary branch, short-circuit operand) where the single
  /// unconditional binding emitted by get_named_expr would be unsound.
  static bool contains_named_expr(const nlohmann::json &node);

  exprt get_binary_operator_expr(const nlohmann::json &element);

  /// Coarse Python-level type category used to decide whether two operands
  /// in an `Eq`/`NotEq` comparison are cross-type (Python's rule: different
  /// types compare unequal without coercion, except within the numeric
  /// tower of `bool`/`int`/`float`/`complex`).
  ///
  /// Returns one of `"numeric"`, `"string"`, `"bytes"`, `"list"`, `"dict"`,
  /// `"tuple"`, or `""` when the category cannot be determined (unannotated
  /// `any_type` void*, or a user-defined class instance whose `__eq__` must
  /// still be dispatched). The caller MUST fall through to the existing
  /// handling on the empty result.
  std::string get_python_type_category(const typet &t) const;

  bool is_bytes_literal(const nlohmann::json &element);

  // V.3: build the `is` identity equality in IREP2 (shared by the `is` and
  // `is not` paths); the public wrappers back-migrate once at the legacy seam.
  expr2tc build_is_equality(const exprt &lhs, const exprt &rhs);

  exprt get_binary_operator_expr_for_is(const exprt &lhs, const exprt &rhs);

  exprt get_negated_is_expr(const exprt &lhs, const exprt &rhs);

  exprt get_resolved_value(const exprt &expr);

  exprt get_function_constant_return(const exprt &func_value);

  exprt resolve_function_call(const exprt &func_expr, const exprt &args_expr);

  exprt get_lambda_expr(const nlohmann::json &element);

  codet convert_expression_to_code(exprt &expr);

  std::string remove_quotes_from_type_string(const std::string &type_string);

  bool function_has_missing_return_paths(const nlohmann::json &function_node);

  exprt materialize_list_function_call(
    const exprt &expr,
    const nlohmann::json &element,
    codet &target_block);

  symbolt create_return_temp_variable(
    const typet &return_type,
    const locationt &location,
    const std::string &func_name);

  // Stage 1 object-model migration (#3067): copy a constructed class *value*
  // onto a fresh non-expiring `__ESBMC_new_object` heap object and return the
  // pointer to it, so a `-> Cls` function can hand back a `Cls*` reference that
  // survives its frame. `current_func_return_type_` must be the migrated
  // pointer type. Used by both return paths in get_return_statements.
  exprt box_value_on_heap(
    const exprt &value,
    const locationt &location,
    codet &target_block);

  void register_instance_attribute(
    const std::string &symbol_id,
    const std::string &attr_name,
    const std::string &var_name,
    const std::string &class_tag);

  bool is_instance_attribute(
    const std::string &symbol_id,
    const std::string &attr_name,
    const std::string &var_name,
    const std::string &class_tag);

  exprt create_member_expression(
    const symbolt &symbol,
    const std::string &attr_name,
    const typet &attr_type);

  typet clean_attribute_type(const typet &attr_type);

  std::string create_normalized_self_key(const std::string &class_tag);

  std::string extract_class_name_from_tag(const std::string &tag_name);

  // True iff `t` denotes a user-defined Python class struct — either the struct
  // itself or a `symbol_typet("tag-<Class>")` reference to it (robust to whether
  // the struct has been built yet). Excludes the list/dict/object model structs.
  bool is_user_class_struct_type(const typet &t);

  // True iff `t` is a pointer to a user-defined class struct (a migrated
  // `Class*` instance). Used to gate the object-model migration's
  // None-keeps-Class* and dunder-dispatch-through-pointer paths to real classes.
  bool is_user_class_pointer(const typet &t);

  exprt resolve_identity_function_call(
    const exprt &func_expr,
    const exprt &args_expr);

  bool is_identity_function(
    const exprt &func_value,
    const std::string &func_identifier);

  exprt handle_none_comparison(
    const std::string &op,
    const exprt &lhs,
    const exprt &rhs);

  /// Rewrites `sl.start/stop/step is/is not None` to a check of the
  /// corresponding `has_start/has_stop/has_step` flag on __ESBMC_PySliceObj.
  /// Returns `nil_exprt()` when the operands are not a slice-member access
  /// paired with a None literal, so the caller continues with default
  /// None-comparison handling.
  exprt try_lower_slice_member_is_none(
    const std::string &op,
    const exprt &lhs,
    const exprt &rhs);

  symbolt &create_tmp_symbol(
    const nlohmann::json &element,
    const std::string var_name,
    const typet &symbol_type,
    const exprt &symbol_value);

  /// Same as above, but for callers that already have a resolved
  /// `locationt` (e.g. string_handler's nondet-fallback materialisation)
  /// rather than an AST node to derive one from.
  symbolt &create_tmp_symbol(
    const locationt &location,
    const std::string var_name,
    const typet &symbol_type,
    const exprt &symbol_value);

  exprt get_logical_operator_expr(const nlohmann::json &element);

  exprt get_conditional_stm(const nlohmann::json &ast_node);

  bool is_coverage_mode() const;

  bool is_pytest_generation_mode() const;

  bool is_model_file(const nlohmann::json &node) const;

  exprt get_function_call(const nlohmann::json &ast_block);

  exprt get_block(
    const nlohmann::json &ast_block,
    bool is_function_body = false,
    bool is_loop_body = false);

  exprt get_static_array(const nlohmann::json &arr, const typet &shape);

  void adjust_statement_types(exprt &lhs, exprt &rhs) const;

  symbol_id create_symbol_id() const;

  symbol_id create_symbol_id(const std::string &filename) const;

  void promote_int_to_float(exprt &op, const typet &target_type) const;

  void handle_float_division(exprt &lhs, exprt &rhs, exprt &bin_expr) const;

  exprt get_tuple_expr(const nlohmann::json &element);

  /**
   * @brief Build a Python slice object from a `Slice` AST node.
   *
   * Lowers `lower`, `upper` and `step` into integer fields of a
   * `PySliceObject` struct constant; absent components leave their integer
   * field zero and clear the corresponding `has_*` flag.
   */
  exprt build_slice_object(const nlohmann::json &slice_node);

  /**
   * @brief Build a Python slice object from a `slice()` builtin call.
   *
   * Supports the one-, two- and three-argument forms; missing trailing
   * arguments and explicit `None` arguments are recorded via the `has_*`
   * flags. Both lowering paths share the same `PySliceObject` shape.
   */
  exprt build_slice_from_args(
    const nlohmann::json &args,
    const nlohmann::json &source_node);

  std::pair<exprt, exprt>
  resolve_comparison_operands_internal(const exprt &lhs, const exprt &rhs);

  bool
  has_unsupported_side_effects_internal(const exprt &lhs, const exprt &rhs);

  TypeFlags infer_types_from_returns(const nlohmann::json &function_body);

  void process_function_arguments(
    const nlohmann::json &function_node,
    code_typet &type,
    const symbol_id &id,
    const locationt &location);

  size_t register_function_argument(
    const nlohmann::json &element,
    code_typet &type,
    const symbol_id &id,
    const locationt &location,
    bool is_keyword_only);

  void validate_return_paths(
    const nlohmann::json &function_node,
    const code_typet &type,
    exprt &function_body);

  exprt compare_constants_internal(
    const std::string &op,
    const exprt &lhs,
    const exprt &rhs);

  exprt handle_indexed_comparison_internal(
    const std::string &op,
    const exprt &lhs,
    const exprt &rhs);

  exprt handle_type_mismatches(
    const std::string &op,
    const exprt &lhs,
    const exprt &rhs);

  void get_attributes_from_self(
    const nlohmann::json &func_node,
    struct_typet &clazz);

  // Infer the static type of `class_name.attr_name` by scanning non-None
  // assignments to that attribute — both `self.attr = <rhs>` within the
  // class's own methods and `<var>.attr = <rhs>` at module scope where
  // `<var>` was assigned `class_name(...)`. Returns nil if no useful
  // information is found (caller falls back to any_type()).
  typet infer_attr_type_from_usage(
    const std::string &class_name,
    const std::string &attr_name);

  /**
   * @brief Build a tuple struct type from an AST value node when the annotation
   *        is bare `tuple`.
   *
   * Walks a `Tuple` literal's elements and synthesises a struct_typet whose
   * components mirror the element types. Constants are typed by their JSON
   * kind; `Name` elements are resolved through @p param_annotations (used to
   * recover types of parameters referenced inside `__init__`-style bodies);
   * everything else falls back to `any_type()`. Returns `empty_typet()` when
   * the node is not a Tuple literal.
   *
   * @param value_node       AST node holding the RHS of `attr: tuple = <rhs>`.
   * @param param_annotations Parameter-name → annotation AST map for the
   *                          enclosing function (may be empty).
   * @return A struct_typet tagged as a tuple, or empty_typet on failure.
   */
  typet infer_tuple_struct_from_value(
    const nlohmann::json &value_node,
    const std::unordered_map<std::string, nlohmann::json> &param_annotations);

  exprt get_return_from_func(const char *func_symbol_id);

  void create_builtin_symbols();

  bool import_module_into_block(
    const nlohmann::json &import_node,
    module_locator &locator,
    code_blockt &code);

  nlohmann::json build_dunder_call(
    const nlohmann::json &object,
    const std::string &dunder_name,
    const nlohmann::json &args,
    const nlohmann::json &source_node) const;

  exprt store_call_result(
    exprt call_expr,
    const locationt &location,
    const std::string &temp_prefix);

  void process_module_imports(
    const nlohmann::json &module_ast,
    module_locator &locator,
    code_blockt &accumulated_code);

  /// Walk the import graph rooted at @p module_ast (without annotating or
  /// generating code) and parse each reachable module's JSON AST into
  /// @ref module_ast_pool_. Mirrors @ref process_module_imports's reach:
  /// top-level imports plus imports directly inside top-level
  /// FunctionDef bodies. Idempotent on names already pooled. Used so that
  /// annotators for any one module can see subscript usages from any
  /// other module in the graph (GitHub #4554).
  void pre_collect_module_asts(
    const nlohmann::json &module_ast,
    module_locator &locator);

  symbolt *find_function_in_base_classes(
    const std::string &class_name,
    const std::string &symbol_id,
    std::string method_name,
    bool is_ctor) const;

  symbolt *find_imported_symbol(const std::string &symbol_id) const;
  symbolt *find_nested_function_symbol(const std::string &name) const;
  symbolt *find_symbol_in_global_scope(const std::string &symbol_id) const;

  void copy_instance_attributes(
    const std::string &src_obj_id,
    const std::string &target_obj_id);

  void update_instance_from_self(
    const std::string &class_name,
    const std::string &func_name,
    const std::string &obj_symbol_id);

  size_t get_type_size(const nlohmann::json &ast_node);

  void append_models_from_directory(
    std::list<std::string> &file_list,
    const std::string &dir_path);

  exprt handle_membership_operator(
    exprt &lhs,
    exprt &rhs,
    const nlohmann::json &element,
    bool invert);

  std::string extract_non_none_type(const nlohmann::json &annotation_node);

  void
  get_delete_statement(const nlohmann::json &ast_node, codet &target_block);

  // =========================================================================
  // Helper methods for get_var_assign
  // =========================================================================

  std::pair<std::string, typet>
  extract_type_info(const nlohmann::json &ast_node);

  void handle_array_unpacking(
    const nlohmann::json &ast_node,
    const nlohmann::json &target,
    exprt &rhs,
    codet &target_block);

  void handle_list_literal_unpacking(
    const nlohmann::json &ast_node,
    const nlohmann::json &target,
    codet &target_block);

  void handle_assignment_type_adjustments(
    symbolt *lhs_symbol,
    exprt &lhs,
    exprt &rhs,
    const std::string &lhs_type,
    const nlohmann::json &ast_node,
    bool is_ctor_call);

  // =========================================================================
  // Dictionary assignment helper methods
  // =========================================================================

  /**
   * @brief Gets RHS expression with dict subscript type resolution.
   *
   * When the RHS is a dict subscript and the target has a primitive type
   * (int, bool, etc.), fetches the dict value with the correct type instead
   * of the default char* pointer.
   *
   * @param ast_node The assignment AST node.
   * @param target_type The expected type from the LHS.
   * @return The RHS expression with proper type resolution.
   */
  exprt get_rhs_with_dict_resolution(
    const nlohmann::json &ast_node,
    const typet &target_type);

  // =========================================================================
  // Type inference helper methods
  // =========================================================================

  /**
   * @brief Infers type from function return when annotation is "Any".
   *
   * When a variable is annotated with "Any" but initialized from a function
   * call, attempts to infer the actual type from the function's return type.
   *
   * @param ast_node The assignment AST node.
   * @param lhs_type The current LHS type string ("Any" or other).
   * @return Updated type string (empty if type was inferred successfully).
   */
  std::string infer_type_from_any_annotation(
    const nlohmann::json &ast_node,
    const std::string &lhs_type);

  // =========================================================================
  // Unpacking helper methods
  // =========================================================================

  /**
   * @brief Handles tuple/list unpacking assignment.
   *
   * Detects and processes assignments where the target is a Tuple or List,
   * performing unpacking of the RHS into multiple variables.
   *
   * @param ast_node The assignment AST node.
   * @param target The assignment target (Tuple or List node).
   * @param target_block The code block to add generated code to.
   * @return true if this was an unpacking assignment and was handled.
   */
  bool handle_unpacking_assignment(
    const nlohmann::json &ast_node,
    const nlohmann::json &target,
    codet &target_block);

  /**
   * Desugar a tuple/list unpacking whose targets include lvalues (a[i],
   * obj.attr) into temp-mediated single assignments routed through the normal
   * single-assignment path. Evaluates every RHS element into a temporary
   * first (Python evaluates the whole RHS before assigning), then stores each
   * target, so the swap a[i], a[j] = a[j], a[i] is handled soundly (#4792).
   * Requires the RHS to be a Tuple/List literal of matching arity.
   */
  void desugar_unpacking_with_lvalue_targets(
    const nlohmann::json &ast_node,
    const nlohmann::json &target,
    codet &target_block);

  // =========================================================================
  // Symbol creation helper methods
  // =========================================================================

  /**
   * @brief Creates symbol for unannotated assignment with inferrable types.
   *
   * For assignments without type annotations where the RHS type can be
   * inferred (Lambda, Call, BoolOp), creates the appropriate symbol.
   *
   * @param ast_node The assignment AST node.
   * @param target The assignment target node.
   * @param sid The symbol ID for the variable.
   * @param is_global Whether this is a global variable.
   * @return Pointer to created symbol, or nullptr if not applicable.
   */
  symbolt *create_symbol_for_unannotated_assign(
    const nlohmann::json &ast_node,
    const nlohmann::json &target,
    const symbol_id &sid,
    bool is_global);

  /**
   * @brief Checks if variable is in global declarations.
   *
   * @param sid The symbol ID to check.
   * @return true if the variable is declared as global.
   */
  bool is_global_variable(const symbol_id &sid) const;

  /**
   * @brief Extracts variable name from assignment target.
   *
   * Handles Name, Attribute, and Subscript target types.
   *
   * @param target The assignment target node.
   * @return The extracted variable name.
   * @throws std::runtime_error if target type is unsupported.
   */
  std::string extract_target_name(const nlohmann::json &target) const;

  // =========================================================================
  // RHS processing helper methods
  // =========================================================================

  /**
   * @brief Handles function call RHS assignment.
   *
   * Processes assignments where the RHS is a function call, handling
   * constructor calls, instance attribute copying, and list return types.
   *
   * @param ast_node The assignment AST node.
   * @param lhs_symbol The LHS symbol.
   * @param lhs The LHS expression.
   * @param rhs The RHS expression (function call).
   * @param location The source location.
   * @param is_ctor_call Whether this is a constructor call.
   * @param target_block The code block to add generated code to.
   */
  void handle_function_call_rhs(
    const nlohmann::json &ast_node,
    symbolt *lhs_symbol,
    exprt &lhs,
    exprt &rhs,
    const locationt &location,
    bool is_ctor_call,
    codet &target_block);

  /**
   * @brief Handles string literal assignment to str-typed variable.
   *
   * When assigning a string literal to a variable with str type annotation,
   * converts the literal to a character array expression.
   *
   * @param ast_node The assignment AST node.
   * @param lhs_type The LHS type string.
   * @param rhs The current RHS expression.
   * @return Updated RHS expression (character array if applicable).
   */
  exprt handle_string_literal_rhs(
    const nlohmann::json &ast_node,
    const std::string &lhs_type,
    const exprt &rhs);

  // =========================================================================
  // Helper methods for binary operator expression handling
  // =========================================================================

  /**
   * @brief Handles type identity checks (value is type_identifier).
   *
   * Handles identity checks involving Python type objects.
   * Type objects are singletons, so identity comparisons between
   * type objects can be resolved by comparing their identifiers.
   *
   * @param op The operator string ("Is" or "IsNot").
   * @param lhs The left operand expression.
   * @param rhs The right operand expression.
   * @param left The left operand JSON AST node.
   * @param right The right operand JSON AST node.
   * @return Boolean result expression, or nil_exprt if not a type identity check.
   */
  exprt handle_type_identity_check(
    const std::string &op,
    const exprt &lhs,
    const exprt &rhs,
    const nlohmann::json &left,
    const nlohmann::json &right);

  /**
   * @brief Converts function calls in binary operands to side effects.
   * @param lhs Left operand expression (may be modified).
   * @param rhs Right operand expression (may be modified).
   */
  void convert_function_calls_to_side_effects(exprt &lhs, exprt &rhs);

  /**
   * @brief Handles chained comparison expressions (e.g., 0 <= x <= 10).
   * @param element The JSON AST node containing the chained comparison.
   * @param bin_expr The base binary expression.
   * @return The combined expression for the chained comparison.
   */
  exprt handle_chained_comparisons_logic(
    const nlohmann::json &element,
    exprt &bin_expr);

  /**
   * @brief Determines if None comparison setup is needed.
   *
   * Checks whether the operation involves None comparisons that shouldn't
   * unwrap optional types.
   *
   * @param op The operator string (e.g., "Eq", "Is").
   * @param lhs The left operand expression.
   * @param rhs The right operand expression.
   * @return true if this is a None comparison, false otherwise.
   */
  bool handle_none_check_setup(
    const std::string &op,
    const exprt &lhs,
    const exprt &rhs);

  /**
   * @brief Handles array-related binary operations.
   *
   * Processes zero-length array comparisons and string concatenation.
   *
   * @param op The operator string.
   * @param lhs The left operand expression (may be modified for concatenation).
   * @param rhs The right operand expression (may be modified for concatenation).
   * @param left The left operand JSON AST node.
   * @param right The right operand JSON AST node.
   * @param element The full binary operation JSON AST node.
   * @return The result expression, or nil_exprt if not an array operation.
   */
  exprt handle_array_operations(
    const std::string &op,
    exprt &lhs,
    exprt &rhs,
    const nlohmann::json &left,
    const nlohmann::json &right,
    const nlohmann::json &element);

  /**
   * @brief Handles list-related binary operations.
   *
   * Processes list comparison, concatenation, and repetition.
   *
   * @param op The operator string.
   * @param lhs The left operand expression (may be modified for repetition).
   * @param rhs The right operand expression (may be modified for repetition).
   * @param left The left operand JSON AST node.
   * @param right The right operand JSON AST node.
   * @param element The full binary operation JSON AST node.
   * @return The result expression, or nil_exprt if not a list operation.
   */
  exprt handle_list_operations(
    const std::string &op,
    exprt &lhs,
    exprt &rhs,
    const nlohmann::json &left,
    const nlohmann::json &right,
    const nlohmann::json &element);

  /**
   * @brief Handles type mismatches in relational operations.
   *
   * Processes single-character comparisons and float-vs-string comparisons.
   *
   * @param op The operator string.
   * @param lhs The left operand expression (may be modified).
   * @param rhs The right operand expression (may be modified).
   * @param element The full binary operation JSON AST node.
   * @return The result expression, or nil_exprt if no special handling needed.
   */
  exprt handle_relational_type_mismatches(
    const std::string &op,
    exprt &lhs,
    exprt &rhs,
    const nlohmann::json &element);

  /**
   * @brief Handles string-related binary operations.
   *
   * Processes string type inference and delegates to string handler.
   *
   * @param op The operator string.
   * @param lhs The left operand expression (may be modified).
   * @param rhs The right operand expression (may be modified).
   * @param left The left operand JSON AST node.
   * @param right The right operand JSON AST node.
   * @param element The full binary operation JSON AST node.
   * @return The result expression, or nil_exprt if not a string operation.
   */
  exprt handle_string_binary_operations(
    const std::string &op,
    exprt &lhs,
    exprt &rhs,
    const nlohmann::json &left,
    const nlohmann::json &right,
    const nlohmann::json &element);

  /**
   * @brief Builds a binary expression with proper type and location.
   *
   * Determines result type, sets location, handles type promotions,
   * and adds operands to the expression.
   *
   * @param op The operator string.
   * @param lhs The left operand expression (may be modified for type casting).
   * @param rhs The right operand expression (may be modified for type casting).
   * @return The constructed binary expression.
   */
  exprt build_binary_expression(const std::string &op, exprt &lhs, exprt &rhs);

  /**
   * @brief Promotes operands for IEEE floating-point operations.
   *
   * @param bin_expr The binary expression (operands may be modified).
   * @param lhs The original left operand.
   * @param rhs The original right operand.
   */
  void
  promote_ieee_operands(exprt &bin_expr, const exprt &lhs, const exprt &rhs);

  /**
   * @brief Infers function return type from return statements in the body.
   *
   * @param body The JSON AST node representing the function body statements.
   * @return The inferred return type (struct_typet for tuples), or empty_typet
   *         if no inferrable return type is found.
   */
  typet infer_return_type_from_body(const nlohmann::json &body);

  // =========================================================================
  // String method helpers
  // =========================================================================

  exprt handle_string_type_mismatch(
    const exprt &lhs,
    const exprt &rhs,
    const std::string &op);

  void process_forward_reference(
    const nlohmann::json &annotation,
    codet &target_block);

  // =========================================================================
  // Optional type helpers
  // =========================================================================

  /// Wrap values in Optional
  exprt wrap_in_optional(const exprt &value, const typet &optional_type);

  // =========================================================================
  // Enum support helpers
  // =========================================================================

  /// Build a struct expression for an enum member (e.g. TrafficLight.GREEN)
  /// so that it carries the enum class type rather than a raw int.
  exprt make_enum_member_struct_expr(
    const symbolt &int_sym,
    const std::string &class_name,
    const std::string &member_name);

  /// Handle Optional value access
  exprt unwrap_optional_if_needed(
    const exprt &expr,
    const nlohmann::json &element = nlohmann::json());

  // =========================================================================
  // Dunder method dispatch for user-defined struct types
  // =========================================================================

  static std::string op_to_dunder(const std::string &op);
  static std::string op_to_rdunder(const std::string &op);
  symbolt *find_dunder_method(
    const std::string &class_name,
    const std::string &dunder_name);
  bool has_dunder_method(
    const nlohmann::json &value_node,
    const std::string &dunder_name);
  exprt dispatch_dunder_operator(
    const std::string &op,
    exprt &lhs,
    exprt &rhs,
    const locationt &loc);
  exprt dispatch_unary_dunder_operator(
    const std::string &op,
    exprt &operand,
    const locationt &loc);

  // =========================================================================
  // Member variables
  // =========================================================================

  contextt &symbol_table_;
  const nlohmann::json *ast_json;
  /// The entry-point module AST, captured at construction. `ast_json` is
  /// temporarily swapped to imported modules during conversion (see with_ast),
  /// so this retains a stable handle to the top-level module whose body holds
  /// the constructor call sites used by cross-module attribute-type inference.
  const nlohmann::json *entry_ast_;
  const global_scope &global_scope_;
  type_handler type_handler_;
  string_builder *string_builder_;
  symbol_generator sym_generator_;

  namespacet ns;
  typet current_element_type;
  typet current_func_return_type_;
  std::string main_python_file;
  std::string current_python_file;
  nlohmann::json imported_module_json;
  std::string current_func_name_;
  std::string current_class_name_;
  std::size_t get_expr_depth_ = 0;
  code_blockt *current_block;
  exprt *current_lhs;
  string_handler string_handler_;
  python_math math_handler_;
  complex_handler complex_handler_;
  tuple_handler *tuple_handler_;
  python_dict_handler *dict_handler_;
  python_typechecking *typechecker_ = nullptr;
  python_lambda *lambda_handler_;
  python_exception_handler *exception_handler_;

  bool is_converting_lhs = false;
  bool is_converting_rhs = false;
  bool is_loading_models = false;
  bool is_importing_module = false;
  bool base_ctor_called = false;
  bool build_static_lists = false;

  /// Map object to list of instance attributes
  std::map<std::string, std::set<std::string>> instance_attr_map;
  /// Map imported modules to their corresponding paths
  std::unordered_map<std::string, std::string> imported_modules;

  /// Pool of every reachable imported module's parsed JSON AST, keyed by
  /// module name. Populated by @ref pre_collect_module_asts before any
  /// annotation runs so that @ref import_module_into_block can expose the
  /// full pool to each module's annotator as extra subscript inference
  /// sources (GitHub #4554). Owns the JSONs to keep their addresses stable
  /// across annotator calls.
  std::map<std::string, nlohmann::json> module_ast_pool_;
  /// Maps any symbol currently known to refer to an input() string
  /// (e.g. $input_str$N or a variable aliasing it) to its $input_len$N symbol ID
  std::unordered_map<std::string, std::string> input_str_to_len_sym_;

  /// Straight-line dynamic-retyping support (#4770, #4774). Maps the original
  /// (first-typed) symbol ID of a variable to the symbol that currently holds
  /// its value after an incompatible numeric<->string reassignment. Loads of
  /// the variable (and the LHS of a later reassignment) are redirected through
  /// this map so they observe the new type. Only populated for retypes at
  /// block_nesting_ == 1 (an unconditional top-level statement), where there is
  /// no control-flow join that could make the runtime type ambiguous.
  std::unordered_map<std::string, std::string> retype_aliases_;

  /// Flow-sensitive class tracking (#4771/#4772). Maps a straight-line lvalue
  /// access path -- "v" for a Name `v`, "v.attr" for an `obj.attr` lvalue -- to
  /// the class (struct tag, without the "tag-" prefix) most recently assigned
  /// to it at the current program point, last-write-wins. Only written for
  /// unconditional top-level (block_nesting_ == 1) assignments and cleared on
  /// entry to any nested/conditional body, so a class is never carried across a
  /// control-flow join. Read in converter_expr to resolve nested attribute
  /// access on a field the usage-site scanner left as any_type().
  std::unordered_map<std::string, std::string> flow_class_map_;

  /// Canonicalise a Name / `Name.attr` AST node into a flow_class_map_ key.
  /// Returns "" for any other shape (e.g. subscript, nested attribute base).
  std::string flow_lvalue_path(const nlohmann::json &node) const;

  /// Class name of an assignment RHS for flow_class_map_: a `Cls(...)` call to a
  /// known class, or a Name already tracked in flow_class_map_. Else "".
  std::string flow_rhs_class(const nlohmann::json &rhs) const;

  /// User-class name returned by a non-constructor call RHS (`y = f(...)` where
  /// `f` is annotated `-> Cls`), so the LHS can be typed as a `Cls*` reference.
  /// Returns "" for constructor calls, unannotated returns, or non-class types.
  /// Scope: only a direct `Name` call to a module-level function with a `Name`
  /// or forward-reference-string return annotation. Method calls
  /// (`obj.method()`), `Attribute` annotations (`-> mod.Cls`), and nested or
  /// imported callees are not resolved here — those reach `Cls*` typing via the
  /// explicit-annotation fallback in get_var_assign instead.
  std::string call_return_class(const nlohmann::json &rhs) const;

  /// Nesting depth of get_block() invocations. The module/imported-module body
  /// is depth 1; every nested body (function, if/while/for, try/except) is
  /// deeper because those bodies are converted through get_block() too.
  unsigned block_nesting_ = 0;

  /// How many of the enclosing get_block() frames are genuine function/method
  /// bodies (bumped only when get_block is called for a function body). The
  /// "unconditional spine" — the module body plus the chain of function bodies
  /// containing the current statement — is exactly the frames where straight-
  /// line retyping (#4770/#4774) is sound: there is no control-flow join that
  /// could leave the runtime type ambiguous. That spine is precisely
  /// block_nesting_ == function_body_depth_ + 1 (the +1 is the module body);
  /// any if/while/for/try body adds a block_nesting_ frame WITHOUT a
  /// function_body_depth_ frame, so the equality fails and retyping is refused.
  /// Fail-safe: an unrecognised block kind is treated as conditional.
  unsigned function_body_depth_ = 0;

  /// How many enclosing get_block() frames are while/for loop bodies. A loop
  /// target variable (and any rebinding inside the body) leaks past the loop in
  /// Python, so reverting its retype at the body's join would be wrong (it would
  /// hide the leaked value). Dynamic retyping (#4770/#4774) is therefore refused
  /// while loop_body_depth_ > 0 and left to the existing fallback — the
  /// pre-#5716 behaviour — whereas if/else/try bodies do retype-with-revert.
  unsigned loop_body_depth_ = 0;

  function_call_cache function_call_cache_;

  std::vector<std::string> global_declarations;
  std::vector<std::string> local_loads;
  bool is_right = false;
  std::vector<std::string> scope_stack_;

  exprt extract_type_from_boolean_op(const exprt &bool_op);
};
