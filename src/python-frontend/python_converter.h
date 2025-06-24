#pragma once

#include <python-frontend/global_scope.h>
#include <python-frontend/type_handler.h>
#include <util/context.h>
#include <util/namespace.h>
#include <util/std_code.h>
#include <util/symbol_generator.h>
#include <nlohmann/json.hpp>
#include <map>
#include <set>

class codet;
class struct_typet;
class function_id;
class symbol_id;
class function_call_expr;

class python_converter
{
public:
  python_converter(
    contextt &_context,
    const nlohmann::json *ast,
    const global_scope &gs);

  void convert();

  const nlohmann::json &ast() const
  {
    return *ast_json;
  }

  contextt &symbol_table() const
  {
    return symbol_table_;
  }

  const type_handler &get_type_handler() const
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

private:
  friend class function_call_expr;
  friend class numpy_call_expr;
  friend class function_call_builder;

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> a710fa95b (Improved memcpy function)
  template <typename Func>
  decltype(auto) with_ast(const nlohmann::json *new_ast, Func &&f)
  {
    const nlohmann::json *old_ast = ast_json;
    ast_json = new_ast;
    auto result = f();
    ast_json = old_ast;
    return result;
  }

<<<<<<< HEAD
=======
>>>>>>> e7c955101 (Update stats-300s.txt)
=======
>>>>>>> a710fa95b (Improved memcpy function)
  void load_c_intrisics();

  void get_var_assign(const nlohmann::json &ast_node, codet &target_block);

  typet
  resolve_variable_type(const std::string &var_name, const locationt &loc);

  void get_compound_assign(const nlohmann::json &ast_node, codet &target_block);

  void
  get_return_statements(const nlohmann::json &ast_node, codet &target_block);

  void get_function_definition(const nlohmann::json &function_node);

  void
  get_class_definition(const nlohmann::json &class_node, codet &target_block);

  locationt get_location_from_decl(const nlohmann::json &ast_node);

  exprt get_expr(const nlohmann::json &element);

  exprt get_unary_operator_expr(const nlohmann::json &element);

  exprt get_binary_operator_expr(const nlohmann::json &element);

<<<<<<< HEAD
<<<<<<< HEAD
  exprt handle_power_operator(exprt base, exprt exp);

  exprt build_power_expression(const exprt &base, const BigInt &exp);

  bool is_zero_length_array(const exprt &expr);

  void ensure_string_array(exprt &expr);

  BigInt get_string_size(const exprt &expr);

  bool is_bytes_literal(const nlohmann::json &element);

  exprt get_binary_operator_expr_for_is(const exprt &lhs, const exprt &rhs);

  exprt get_negated_is_expr(const exprt &lhs, const exprt &rhs);

  exprt get_array_base_address(const exprt &arr);

  exprt get_resolved_value(const exprt &expr);

  exprt get_function_constant_return(const exprt &func_value);

  exprt resolve_function_call(const exprt &func_expr, const exprt &args_expr);

  exprt handle_power_operator_sym(exprt base, exprt exp);

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 39b012f9f ([python] handles assertions involving function calls and refactor attribute handling (#2510))
  symbolt create_assert_temp_variable(const locationt &location);

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

<<<<<<< HEAD
  exprt resolve_identity_function_call(
    const exprt &func_expr,
    const exprt &args_expr);

  bool is_identity_function(
    const exprt &func_value,
    const std::string &func_identifier);

  exprt handle_string_concatenation(
    const exprt &lhs,
    const exprt &rhs,
    const nlohmann::json &left,
    const nlohmann::json &right);

  exprt handle_string_comparison(
    const std::string &op,
    exprt &lhs,
    exprt &rhs,
    const nlohmann::json &element);

  exprt handle_string_operations(
    const std::string &op,
    exprt &lhs,
    exprt &rhs,
    const nlohmann::json &left,
    const nlohmann::json &right,
    const nlohmann::json &element);
=======
  exprt handle_power_operator(exprt lhs, exprt rhs);
>>>>>>> d76a37c5d ([python] Support python true division semantics (/) with float type coercion (#2425))
=======
  exprt handle_power_operator(exprt base, exprt exp);
>>>>>>> 59fb1bf8f ([python] enhanced handling of true division (#2505))

  exprt build_power_expression(const exprt &base, const BigInt &exp);

  bool is_zero_length_array(const exprt &expr);

  void ensure_string_array(exprt &expr);

  BigInt get_string_size(const exprt &expr);

  bool is_bytes_literal(const nlohmann::json &element);

  exprt get_binary_operator_expr_for_is(const exprt &lhs, const exprt &rhs);

  exprt get_negated_is_expr(const exprt &lhs, const exprt &rhs);

  exprt get_array_base_address(const exprt &arr);

  exprt get_resolved_value(const exprt &expr);

  exprt get_function_constant_return(const exprt &func_value);

  exprt resolve_function_call(const exprt &func_expr, const exprt &args_expr);

=======
>>>>>>> 8c98a1162 ([python] implement pow symbolically (#2506))
=======
>>>>>>> 39b012f9f ([python] handles assertions involving function calls and refactor attribute handling (#2510))
  exprt resolve_identity_function_call(
    const exprt &func_expr,
    const exprt &args_expr);

  bool is_identity_function(
    const exprt &func_value,
    const std::string &func_identifier);

  exprt handle_string_concatenation(
    const exprt &lhs,
    const exprt &rhs,
    const nlohmann::json &left,
    const nlohmann::json &right);

  exprt handle_string_comparison(
    const std::string &op,
    exprt &lhs,
    exprt &rhs,
    const nlohmann::json &element);

  exprt handle_string_operations(
    const std::string &op,
    exprt &lhs,
    exprt &rhs,
    const nlohmann::json &left,
    const nlohmann::json &right,
    const nlohmann::json &element);

  exprt get_logical_operator_expr(const nlohmann::json &element);

  exprt get_conditional_stm(const nlohmann::json &ast_node);

  exprt get_function_call(const nlohmann::json &ast_block);

  exprt make_char_array_expr(
    const std::vector<unsigned char> &string_literal,
    const typet &t);

  exprt get_literal(const nlohmann::json &element);

  exprt get_block(const nlohmann::json &ast_block);

  void adjust_statement_types(exprt &lhs, exprt &rhs) const;

  symbol_id create_symbol_id() const;

  symbol_id create_symbol_id(const std::string &filename) const;

  exprt compute_math_expr(const exprt &expr) const;

  void promote_int_to_float(exprt &op, const typet &target_type) const;

<<<<<<< HEAD
<<<<<<< HEAD
  void handle_float_division(exprt &lhs, exprt &rhs, exprt &bin_expr) const;

=======
>>>>>>> d76a37c5d ([python] Support python true division semantics (/) with float type coercion (#2425))
=======
  void handle_float_division(exprt &lhs, exprt &rhs, exprt &bin_expr) const;

>>>>>>> 117989e74 ([python] Improved inference to handle true division (/) operator (#2426))
  void get_attributes_from_self(
    const nlohmann::json &method_body,
    struct_typet &clazz);

  symbolt *find_function_in_base_classes(
    const std::string &class_name,
    const std::string &symbol_id,
    std::string method_name,
    bool is_ctor) const;

  symbolt *find_imported_symbol(const std::string &symbol_id) const;
  symbolt *find_symbol_in_global_scope(const std::string &symbol_id) const;

  void update_instance_from_self(
    const std::string &class_name,
    const std::string &func_name,
    const std::string &obj_symbol_id);

  void append_models_from_directory(
    std::list<std::string> &file_list,
    const std::string &dir_path);

  contextt &symbol_table_;
  const nlohmann::json *ast_json;
  const global_scope &global_scope_;
  type_handler type_handler_;
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

  bool is_converting_lhs = false;
  bool is_converting_rhs = false;
  bool is_loading_models = false;
  bool is_importing_module = false;
  bool base_ctor_called = false;

  // Map object to list of instance attributes
  std::map<std::string, std::set<std::string>> instance_attr_map;
  // Map imported modules to their corresponding paths
  std::unordered_map<std::string, std::string> imported_modules;
};
