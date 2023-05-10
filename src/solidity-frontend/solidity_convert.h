#ifndef SOLIDITY_FRONTEND_SOLIDITY_CONVERT_H_
#define SOLIDITY_FRONTEND_SOLIDITY_CONVERT_H_

#include <memory>
#include <stack>
#include <map>
#include <util/context.h>
#include <util/namespace.h>
#include <util/std_types.h>
#include <nlohmann/json.hpp>
#include <solidity-frontend/solidity_grammar.h>
#include <solidity-frontend/pattern_check.h>

class solidity_convertert
{
public:
  solidity_convertert(
    contextt &_context,
    nlohmann::json &_ast_json,
    const std::string &_sol_func,
    const std::string &_contract_path);
  virtual ~solidity_convertert() = default;

  bool convert();

protected:
  contextt &context;
  namespacet ns;
  nlohmann::json
    &ast_json; // json for Solidity AST. Use vector for multiple contracts
  const std::string &sol_func;      // Solidity function to be verified
  const std::string &contract_path; //smart contract source file

  std::string absolute_path;
  std::string contract_contents = "";
  int global_scope_id; // scope id of "ContractDefinition"

  unsigned int current_scope_var_num;
  const nlohmann::json *current_functionDecl;
  const nlohmann::json *current_forStmt;
  // Use current level of BinOp type as the "anchor" type for numerical literal conversion:
  // In order to remove the unnecessary implicit IntegralCast. We need type of current level of BinaryOperator.
  // All numeric literals will be implicitly converted to this type. Pop it when finishing the current level of BinaryOperator.
  // TODO: find a better way to deal with implicit type casting if it's not able to cope with compelx rules
  std::stack<const nlohmann::json *> current_BinOp_type;
  std::string current_functionName;

  std::string current_contractName;
  std::string current_fileName;

  bool convert_ast_nodes(const nlohmann::json &contract_def);

  // conversion functions
  // get decl in rule contract-body-element
  bool get_decl(const nlohmann::json &ast_node, exprt &new_expr);
  // get decl in rule variable-declaration-statement, e.g. function local declaration
  bool get_var_decl_stmt(const nlohmann::json &ast_node, exprt &new_expr);
  bool get_var_decl(const nlohmann::json &ast_node, exprt &new_expr);
  bool get_function_definition(const nlohmann::json &ast_node, exprt &new_expr);
  bool get_function_params(const nlohmann::json &pd, exprt &param);
  bool get_struct_class(const nlohmann::json &ast_node);
  bool add_implicit_constructor();
  bool get_implicit_ctor_call(const int ref_decl_id, exprt &new_expr);
  bool
  get_struct_class_fields(const nlohmann::json &ast_node, struct_typet &type);
  bool
  get_struct_class_method(const nlohmann::json &ast_node, struct_typet &type);
  bool get_access_from_decl(
    const nlohmann::json &ast_node,
    struct_typet::componentt &comp);
  bool get_block(
    const nlohmann::json &expr,
    exprt &
      new_expr); // For Solidity's mutually inclusive: rule block and rule statement
  bool get_statement(const nlohmann::json &block, exprt &new_expr);
  bool get_expr(const nlohmann::json &expr, exprt &new_expr);
  bool get_expr(
    const nlohmann::json &expr,
    const nlohmann::json &expr_common_type,
    exprt &new_expr);
  bool get_binary_operator_expr(const nlohmann::json &expr, exprt &new_expr);
  bool get_compound_assign_expr(const nlohmann::json &expr, exprt &new_expr);
  bool get_unary_operator_expr(
    const nlohmann::json &expr,
    const nlohmann::json &int_literal_type,
    exprt &new_expr);
  bool
  get_conditional_operator_expr(const nlohmann::json &expr, exprt &new_expr);
  bool get_cast_expr(
    const nlohmann::json &cast_expr,
    exprt &new_expr,
    const nlohmann::json int_literal_type = nullptr);
  bool get_var_decl_ref(const nlohmann::json &decl, exprt &new_expr);
  bool get_func_decl_ref(const nlohmann::json &decl, exprt &new_expr);
  bool get_decl_ref_builtin(const nlohmann::json &decl, exprt &new_expr);
  bool get_type_description(const nlohmann::json &type_name, typet &new_type);
  bool get_func_decl_ref_type(const nlohmann::json &decl, typet &new_type);
  bool get_array_to_pointer_type(const nlohmann::json &decl, typet &new_type);
  bool
  get_elementary_type_name(const nlohmann::json &type_name, typet &new_type);
  bool get_parameter_list(const nlohmann::json &type_name, typet &new_type);
  void get_state_var_decl_name(
    const nlohmann::json &ast_node,
    std::string &name,
    std::string &id);
  void get_var_decl_name(
    const nlohmann::json &ast_node,
    std::string &name,
    std::string &id);
  void get_function_definition_name(
    const nlohmann::json &ast_node,
    std::string &name,
    std::string &id);
  bool get_constructor_call(const nlohmann::json &ast_node, exprt &new_expr);
  bool get_contract_name(const int ref_decl_id, std::string &contract_name);
  // line number and locations
  void
  get_location_from_decl(const nlohmann::json &ast_node, locationt &location);
  void get_start_location_from_stmt(
    const nlohmann::json &ast_node,
    locationt &location);
  void get_final_location_from_stmt(
    const nlohmann::json &ast_node,
    locationt &location);
  unsigned int
  get_line_number(const nlohmann::json &ast_node, bool final_position = false);
  unsigned int add_offset(const std::string &src, unsigned int start_position);
  std::string get_src_from_json(const nlohmann::json &ast_node);

  symbolt *move_symbol_to_context(symbolt &symbol);

  // auxiliary functions
  std::string get_modulename_from_path(std::string path);
  std::string get_filename_from_path(std::string path);
  const nlohmann::json &find_decl_ref(int ref_decl_id);
  const nlohmann::json &find_constructor_ref(nlohmann::json &contract_def);
  void convert_expression_to_code(exprt &expr);
  bool check_intrinsic_function(const nlohmann::json &ast_node);
  nlohmann::json make_implicit_cast_expr(
    const nlohmann::json &sub_expr,
    std::string cast_type);
  nlohmann::json make_pointee_type(const nlohmann::json &sub_expr);
  nlohmann::json make_callexpr_return_type(const nlohmann::json &type_descrpt);
  nlohmann::json make_array_elementary_type(const nlohmann::json &type_descrpt);
  nlohmann::json make_array_to_pointer_type(const nlohmann::json &type_descrpt);
  std::string get_array_size(const nlohmann::json &type_descrpt);
  bool is_dyn_array(const nlohmann::json &json_in);
  nlohmann::json add_dyn_array_size_expr(
    const nlohmann::json &type_descriptor,
    const nlohmann::json &dyn_array_node);

  void get_default_symbol(
    symbolt &symbol,
    std::string module_name,
    typet type,
    std::string name,
    std::string id,
    locationt location);

  // literal conversion functions
  bool convert_integer_literal(
    const nlohmann::json &integer_literal,
    std::string the_value,
    exprt &dest);
  bool convert_bool_literal(
    const nlohmann::json &bool_literal,
    std::string the_value,
    exprt &dest);
  bool convert_string_literal(std::string the_value, exprt &dest);

  static constexpr const char *mode = "C++";

  // The prefix for the id of each class
  std::string prefix = "tag-";

  // json nodes that always empty
  // used as the return value for find_constructor_ref when
  // dealing with the implicit constructor call
  // this is to avoid reference to stack memory associated with local variable returned
  const nlohmann::json empty_json;

private:
  bool get_elementary_type_name_uint(
    SolidityGrammar::ElementaryTypeNameT &type,
    typet &out);
  bool get_elementary_type_name_int(
    SolidityGrammar::ElementaryTypeNameT &type,
    typet &out);
  
  /*
   * Methods for virtual tables and virtual pointers
   *  TODO: add link to wiki page
   */
  std::string vtable_type_prefix = "virtual_table::";
  std::string vtable_ptr_suffix = "@vtable_pointer";
  std::string thunk_prefix = "thunk::";
  using function_switch = std::map<irep_idt, exprt>;
  using switch_table = std::map<irep_idt, function_switch>;
  using overriden_map = std::map<std::string, const nlohmann::json *>;
  /*
   * traverse methods to:
   *  1. convert virtual methods and add them to class' type
   *  2. If there is no existing vtable type symbol,
   *    add virtual table type symbol in the context
   *    and the virtual pointer.
   *    Then add a new entry in the vtable.
   *  3. If vtable type symbol exists, do not add it or the vptr,
   *    just add a new entry in the existing vtable.
   *  4. If method X overrides a base method,
   *    add a thunk function that does late casting of the `this` parameter
   *    and redirects the call to the overriding function (i.e. method X itself)
   *
   *  Last but not least, set up the vtable varaible symbols (i.e. these are the struct variables
   *  instantiated from the vtable type symbols)
   */
  bool get_struct_class_virtual_methods(
    const nlohmann::json *contract_def,
    struct_typet &type);
  /*
   * additional annotations for virtual or overriding methods
   *
   * Params:
   *  - md: the virtual method AST we are currently dealing with
   *  - comp: the `component` as in `components` list in class type IR
   *    this `component` represents the type of the virtual method
   */
  bool annotate_virtual_overriding_methods(
    const nlohmann::json *method_def,
    struct_typet::componentt &comp);
  /*
   * Check the existence of virtual table type symbol.
   * If it exists, return its pointer. Otherwise, return nullptr.
   */
  symbolt *check_vtable_type_symbol_existence(const struct_typet &type);
  /*
   * Add virtual table(vtable) type symbol.
   * This is added as a type symbol in the symbol table.
   *
   * This is done the first time when we encounter a virtual method in a class
   */
  symbolt *add_vtable_type_symbol(
    const struct_typet::componentt &comp,
    struct_typet &type);
  /*
   * Add virtual pointer(vptr).
   * Vptr is NOT a symbol but rather simply added as a component to the class' type.
   *
   * This is done the first time we encounter a virtual method in a class
   */
  void add_vptr(struct_typet &type);
  /*
   * Add an entry to the virtual table type
   *
   * This is done when NOT the first time we encounter a virtual method in a class
   * in which case we just want to add a new entry to the virtual table type
   */
  void add_vtable_type_entry(
    struct_typet &type,
    struct_typet::componentt &comp,
    symbolt *vtable_type_symbol);

  /*
   * Get the overriden methods to which we need to create thunks.
   *
   * Params:
   *  - md: clang AST of the overriding method
   *  - map: key: a map that takes method id as key and pointer to the overriden method AST
   */
  void
  get_overriden_methods(const nlohmann::json *method, overriden_map &map);
  /*
   * add a thunk function for each overriding method
   *
   * Params:
   *  - md: clang AST of the overriden method in base class
   *  - component: ESBMC IR representing the the overriding method in derived class' type
   *  - type: ESBMC IR representing the derived class' type
   */
  void add_thunk_method(
    const nlohmann::json *method,
    const struct_typet::componentt &component,
    struct_typet &type);
  /*
   * change the type of 'this' pointer from derived class type to base class type
   */
  void update_thunk_this_type(
    typet &thunk_symbol_type,
    const std::string &base_class_id);
  /*
   * Add symbol for arguments of the thunk method
   * Params:
   *  - thunk_func_symb: function symbol for the thunk method
   */
  void add_thunk_method_arguments(symbolt &thunk_func_symb);
  /*
   * Add thunk function body
   * Params:
   *  - thunk_func_symb: function symbol for the thunk method
   *  - component: ESBMC IR representing the the overriding method in derived class' type
   */
  void add_thunk_method_body(
    symbolt &thunk_func_symb,
    const struct_typet::componentt &component);
  /*
   * Add thunk body that contains return value
   * Params:
   *  - thunk_func_symb: function symbol for the thunk method
   *  - component: ESBMC IR representing the the overriding method in derived class' type
   *  - late_cast_this: late casting of `this`
   */
  void add_thunk_method_body_return(
    symbolt &thunk_func_symb,
    const struct_typet::componentt &component,
    const typecast_exprt &late_cast_this);
  /*
   * Add thunk body that does NOT contain return value
   * Params:
   *  - thunk_func_symb: function symbol for the thunk method
   *  - component: ESBMC IR representing the the overriding method in derived class' type
   *  - late_cast_this: late casting of `this`
   */
  void add_thunk_method_body_no_return(
    symbolt &thunk_func_symb,
    const struct_typet::componentt &component,
    const typecast_exprt &late_cast_this);
  /*
   * Add thunk function as a `method` in the derived class' type
   * Params:
   *  - thunk_func_symbol: thunk function symbol
   *  - type: derived class' type
   *  - comp: `component` representing the overriding function
   */
  void add_thunk_component_to_type(
    const symbolt &thunk_func_symb,
    struct_typet &type,
    const struct_typet::componentt &comp);
  /*
   * Set an intuitive name to thunk function
   */
  void
  set_thunk_name(symbolt &thunk_func_symb, const std::string &base_class_id);
  /*
   * Recall that we mode the virtual function table as struct of function pointers.
   * This function adds the symbols for these struct variables.
   *
   * Params:
   *  - cxxrd: clang AST node representing the class/struct we are currently dealing with
   *  - type: ESBMC IR representing the type the class/struct we are currently dealing with
   *  - vft_value_map: representing the vtable value maps for this class/struct we are currently dealing with
   */
  void setup_vtable_struct_variables(
    const nlohmann::json *contract_def,
    const struct_typet &type);
  /*
   * This function builds the vtable value map -
   * a map representing the function switch table
   * with each key-value pair in the form of:
   *  Class X : {VirtualName Y : FunctionID}
   *
   * where X represents the name of a virtual/thunk/overriding function and function ID represents the
   * actual function we are calling when calling the virtual/thunk/overriding function
   * via a Class X* pointer, something like:
   *   xptr->Y()
   *
   * Params:
   *  - struct_type: ESBMC IR representing the type of the class/struct we are currently dealing with
   *  - vtable_value_map: representing the vtable value maps for this class/struct we are currently dealing with
   */
  void build_vtable_map(
    const struct_typet &struct_type,
    switch_table &vtable_value_map);
  /*
   * Create the vtable variable symbols and add them to the symbol table.
   * Each vtable variable represents the actual function switch table, which
   * is modelled as a struct of function pointers, e.g.:
   *  Vtable tag.Base@Base =
   *    {
   *      .do_something = &TagBase::do_someting();
   *    };
   *
   * Params:
   *  - cxxrd: clang AST node representing the class/struct we are currently dealing with
   *  - struct_type: ESBMC IR representing the type the class/struct we are currently dealing with
   *  - vtable_value_map: representing the vtable value maps for this class/struct we are currently dealing with
   */
  void add_vtable_variable_symbols(
    const nlohmann::json *contraact_def,
    const struct_typet &struct_type,
    const switch_table &vtable_value_map);

  /*
   * Methods for resolving a clang::MemberExpr to virtual/overriding method
   */
  bool perform_virtual_dispatch(const nlohmann::json &member);
  bool is_md_virtual_or_overriding(const nlohmann::json &contrract_def);
  bool is_fd_virtual_or_overriding(const nlohmann::json &func_decl);
  bool get_vft_binding_expr(const nlohmann::json &member, exprt &new_expr);
  /*
   * Get the base dereference for VFT bound MemberExpr
   *
   * Params:
   *  - member: the method to which this MemberExpr refers
   *  - new_expr: ESBMC IR to represent x dereferencing as in `x->X@vtable_ptr->F`
   */
  bool
  get_vft_binding_expr_base(const nlohmann::json &member, exprt &new_expr);
  /*
   * Get the vtable poitner dereferencing for VFT bound MemberExpr
   *
   * Params:
   *  - member: the method to which this MemberExpr refers
   *  - new_expr: ESBMC IR to represent X@Vtable_ptr dereferencing as in `x->X@vtable_ptr->F`
   *  - base_deref: the base dereferencing expression incorporated in vtable pointer dereferencing expression
   */
  void get_vft_binding_expr_vtable_ptr(
    const nlohmann::json &member,
    exprt &new_expr,
    const exprt &base_deref);
  /*
   * Get the Function for VFT bound MemberExpr
   *
   * Params:
   *  - member: the method to which this MemberExpr refers
   *  - new_expr: ESBMC IR to represent F in `x->X@vtable_ptr->F`
   *  - vtable_ptr_deref: the vtable pointer dereferencing expression
   */
  bool get_vft_binding_expr_function(
    const nlohmann::json &member,
    exprt &new_expr,
    const exprt &vtable_ptr_deref);
};

#endif /* SOLIDITY_FRONTEND_SOLIDITY_CONVERT_H_ */
