#pragma once

#include <python-frontend/global_scope.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/type_utils.h>
#include <python-frontend/string_handler.h>
#include <util/context.h>
#include <util/namespace.h>
#include <util/std_code.h>
#include <util/symbol_generator.h>
#include <nlohmann/json.hpp>
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

class python_converter
{
public:
  python_converter(
    contextt &_context,
    const nlohmann::json *ast,
    const global_scope &gs);

  ~python_converter();

  void convert();

  string_builder &get_string_builder();

  string_handler &get_string_handler()
  {
    return string_handler_;
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

  const std::string &python_file() const
  {
    return current_python_file;
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

  exprt get_expr(const nlohmann::json &element);

  locationt get_location_from_decl(const nlohmann::json &element);

  exprt handle_string_comparison(
    const std::string &op,
    exprt &lhs,
    exprt &rhs,
    const nlohmann::json &element);

private:
  friend class function_call_expr;
  friend class numpy_call_expr;
  friend class function_call_builder;
  friend class type_handler;
  friend class python_list;
  bool processing_list_elements = false;

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

  exprt handle_power_operator(exprt base, exprt exp);

  exprt build_power_expression(const exprt &base, const BigInt &exp);

  bool is_bytes_literal(const nlohmann::json &element);

  exprt get_binary_operator_expr_for_is(const exprt &lhs, const exprt &rhs);

  exprt get_negated_is_expr(const exprt &lhs, const exprt &rhs);

  exprt get_resolved_value(const exprt &expr);

  exprt get_function_constant_return(const exprt &func_value);

  exprt resolve_function_call(const exprt &func_expr, const exprt &args_expr);

  exprt handle_power_operator_sym(exprt base, exprt exp);

  symbolt create_assert_temp_variable(const locationt &location);

  exprt get_lambda_expr(const nlohmann::json &element);

  codet convert_expression_to_code(exprt &expr);

  std::string remove_quotes_from_type_string(const std::string &type_string);

  bool function_has_missing_return_paths(const nlohmann::json &function_node);

  typet get_type_from_annotation(
    const nlohmann::json &annotation_node,
    const nlohmann::json &element);

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

  exprt compute_math_expr(const exprt &expr) const;

  void promote_int_to_float(exprt &op, const typet &target_type) const;

  void handle_float_division(exprt &lhs, exprt &rhs, exprt &bin_expr) const;

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

  // helper methods for get_var_assign
  std::pair<std::string, typet>
  extract_type_info(const nlohmann::json &ast_node);

  exprt create_lhs_expression(
    const nlohmann::json &target,
    symbolt *lhs_symbol,
    const locationt &location);

  void handle_assignment_type_adjustments(
    symbolt *lhs_symbol,
    exprt &lhs,
    exprt &rhs,
    const std::string &lhs_type,
    const nlohmann::json &ast_node,
    bool is_ctor_call);

  // Helper methods for binary operator expression handling
  void convert_function_calls_to_side_effects(exprt &lhs, exprt &rhs);

  exprt handle_chained_comparisons_logic(
    const nlohmann::json &element,
    exprt &bin_expr);

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

  bool is_converting_lhs = false;
  bool is_converting_rhs = false;
  bool is_loading_models = false;
  bool is_importing_module = false;
  bool base_ctor_called = false;
  bool build_static_lists = false;

  // Map object to list of instance attributes
  std::map<std::string, std::set<std::string>> instance_attr_map;
  // Map imported modules to their corresponding paths
  std::unordered_map<std::string, std::string> imported_modules;

  std::vector<std::string> global_declarations;
  std::vector<std::string> local_loads;
  bool is_right = false;
};
