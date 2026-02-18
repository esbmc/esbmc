#pragma once

#include <nlohmann/json.hpp>
#include <python-frontend/global_scope.h>
#include <python-frontend/python_dict_handler.h>
#include <python-frontend/python_math.h>
#include <python-frontend/string_handler.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/type_utils.h>
#include <python-frontend/python_set.h>
#include <util/context.h>
#include <util/namespace.h>
#include <util/std_code.h>
#include <util/symbol_generator.h>
#include <map>
#include <set>
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
  std::string get_op(const std::string &op, const typet &type) const;
  typet get_type_from_annotation(
    const nlohmann::json &annotation_node,
    const nlohmann::json &element);

  string_builder &get_string_builder();

  python_dict_handler *get_dict_handler()
  {
    return dict_handler_;
  }

  python_math &get_math_handler()
  {
    return math_handler_;
  }

  string_handler &get_string_handler()
  {
    return string_handler_;
  }

  tuple_handler &get_tuple_handler()
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

  typet
  resolve_variable_type(const std::string &var_name, const locationt &loc);

  void get_compound_assign(const nlohmann::json &ast_node, codet &target_block);

  void
  get_return_statements(const nlohmann::json &ast_node, codet &target_block);

  void get_function_definition(const nlohmann::json &function_node);

  void
  get_class_definition(const nlohmann::json &class_node, codet &target_block);

  exprt get_unary_operator_expr(const nlohmann::json &element);

  exprt get_binary_operator_expr(const nlohmann::json &element);

  bool is_bytes_literal(const nlohmann::json &element);

  exprt get_binary_operator_expr_for_is(const exprt &lhs, const exprt &rhs);

  exprt get_negated_is_expr(const exprt &lhs, const exprt &rhs);

  exprt get_resolved_value(const exprt &expr);

  exprt get_function_constant_return(const exprt &func_value);

  exprt resolve_function_call(const exprt &func_expr, const exprt &args_expr);

  symbolt create_assert_temp_variable(const locationt &location);

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

  symbolt &create_tmp_symbol(
    const nlohmann::json &element,
    const std::string var_name,
    const typet &symbol_type,
    const exprt &symbol_value);

  exprt get_logical_operator_expr(const nlohmann::json &element);

  exprt get_conditional_stm(const nlohmann::json &ast_node);

  exprt get_function_call(const nlohmann::json &ast_block);

  exprt get_block(const nlohmann::json &ast_block);

  exprt get_static_array(const nlohmann::json &arr, const typet &shape);

  void adjust_statement_types(exprt &lhs, exprt &rhs) const;

  symbol_id create_symbol_id() const;

  symbol_id create_symbol_id(const std::string &filename) const;

  void promote_int_to_float(exprt &op, const typet &target_type) const;

  void handle_float_division(exprt &lhs, exprt &rhs, exprt &bin_expr) const;

  exprt get_tuple_expr(const nlohmann::json &element);

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
    const nlohmann::json &method_body,
    struct_typet &clazz);

  exprt get_return_from_func(const char *func_symbol_id);

  void create_builtin_symbols();

  void process_module_imports(
    const nlohmann::json &module_ast,
    module_locator &locator,
    code_blockt &accumulated_code);

  symbolt *find_function_in_base_classes(
    const std::string &class_name,
    const std::string &symbol_id,
    std::string method_name,
    bool is_ctor) const;

  symbolt *find_imported_symbol(const std::string &symbol_id) const;
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
  // Assertion helper methods
  // =========================================================================

  /**
   * @brief Handles assertions on list expressions.
   *
   * In Python, empty lists are falsy, so `assert []` should fail.
   * This method converts `assert list_var` to `assert len(list_var) > 0`
   * by calling __ESBMC_list_size and checking the result.
   *
   * @param element The assertion AST node.
   * @param test The test expression (a list or list-returning function call).
   * @param block The code block to add generated statements to.
   * @param attach_assert_message Lambda to attach user assertion messages.
   */
  void handle_list_assertion(
    const nlohmann::json &element,
    const exprt &test,
    code_blockt &block,
    const std::function<void(code_assertt &)> &attach_assert_message);

  /**
   * @brief Handles assertions on function call expressions.
   *
   * Materializes function calls in assertions.
   * For None-returning functions, executes the call and asserts False.
   * For other functions, stores result in temp var and asserts on that.
   *
   * @param element The assertion AST node.
   * @param func_call_expr The function call expression to assert on.
   * @param is_negated Whether the assertion is negated (assert not func()).
   * @param block The code block to add generated statements to.
   * @param attach_assert_message Lambda to attach user assertion messages.
   */
  void handle_function_call_assertion(
    const nlohmann::json &element,
    const exprt &func_call_expr,
    bool is_negated,
    code_blockt &block,
    const std::function<void(code_assertt &)> &attach_assert_message);

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

  /// Handle Optional value access
  exprt unwrap_optional_if_needed(const exprt &expr);

  // =========================================================================
  // Member variables
  // =========================================================================

  contextt &symbol_table_;
  const nlohmann::json *ast_json;
  const global_scope &global_scope_;
  type_handler type_handler_;
  string_builder *string_builder_;
  symbol_generator sym_generator_;

  namespacet ns;
  typet current_element_type;
  std::string main_python_file;
  std::string current_python_file;
  nlohmann::json imported_module_json;
  std::string current_func_name_;
  std::string current_class_name_;
  code_blockt *current_block;
  exprt *current_lhs;
  string_handler string_handler_;
  python_math math_handler_;
  tuple_handler *tuple_handler_;
  python_dict_handler *dict_handler_;
  python_typechecking *typechecker_ = nullptr;
  python_lambda *lambda_handler_;

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

  std::vector<std::string> global_declarations;
  std::vector<std::string> local_loads;
  bool is_right = false;
  std::vector<std::string> scope_stack_;

  exprt extract_type_from_boolean_op(const exprt &bool_op);
};
