#ifndef SOLIDITY_FRONTEND_SOLIDITY_CONVERT_H_
#define SOLIDITY_FRONTEND_SOLIDITY_CONVERT_H_

#include <memory>
#include <stack>
#include <vector>
#include <map>
#include <queue>
#include <util/context.h>
#include <util/namespace.h>
#include <util/std_types.h>
#include <util/std_code.h>
#include <util/string_constant.h>
#include <nlohmann/json.hpp>
#include <solidity-frontend/solidity_grammar.h>
#include <solidity-frontend/pattern_check.h>
#include <clang-c-frontend/symbolic_types.h>

class solidity_convertert
{
public:
  solidity_convertert(
    contextt &_context,
    nlohmann::json &_ast_json,
    const std::string &_sol_func,
    const std::string &_contract_path,
    const bool _is_bound);
  virtual ~solidity_convertert() = default;

  bool convert();
  static bool is_low_level_call(const std::string &name);

protected:
  void merge_multi_files();
  std::vector<nlohmann::json> topological_sort(
    std::unordered_map<std::string, std::unordered_set<std::string>> &graph,
    std::unordered_map<std::string, nlohmann::json> &path_to_json);
  void contract_precheck();
  void populate_auxilary_vars();
  bool convert_ast_nodes(const nlohmann::json &contract_def);

  // conversion functions
  // get decl in rule contract-body-element
  bool get_contract_definition(const std::string &c_name);
  bool get_non_function_decl(const nlohmann::json &ast_node, exprt &new_expr);
  bool get_function_decl(const nlohmann::json &ast_node);
  // get decl in rule variable-declaration-statement, e.g. function local declaration
  bool get_var_decl_stmt(const nlohmann::json &ast_node, exprt &new_expr);
  bool get_var_decl(const nlohmann::json &ast_node, exprt &new_expr);
  bool get_function_definition(const nlohmann::json &ast_node);
  bool get_function_params(const nlohmann::json &pd, exprt &param);
  void get_function_this_pointer_param(
    const std::string &contract_name,
    const std::string &ctor_id,
    const std::string &debug_modulename,
    const locationt &location_begin,
    code_typet &type);
  bool get_default_function(
    const std::string name,
    const std::string id,
    symbolt &added_symbol);
  bool get_unbound_funccall(
    const std::string contractName,
    code_function_callt &call);
  void get_static_contract_instance_name(
    const std::string c_name,
    std::string &name,
    std::string &id);
  void get_static_contract_instance(const std::string c_name, symbolt &sym);

  // handle the non-contract definition, including struct/enum/error/event/abstract/...
  bool get_noncontract_defition(nlohmann::json &ast_node);
  bool
  get_noncontract_decl_ref(const nlohmann::json &ast_node, exprt &new_expr);
  bool get_struct_class(const nlohmann::json &ast_node);
  void add_enum_member_val(nlohmann::json &ast_node);
  bool get_error_definition(const nlohmann::json &ast_node);
  void add_empty_body_node(nlohmann::json &ast_node);

  // handle inheritance
  void merge_inheritance_ast(
    nlohmann::json &c_node,
    const std::string &c_name,
    std::set<std::string> &merged_list);

  // handle constructor
  bool get_constructor(
    const nlohmann::json &ast_node,
    const std::string &contract_name);
  bool add_implicit_constructor(const std::string &contract_name);
  bool get_implicit_ctor_ref(exprt &new_expr, const std::string &contract_name);
  bool get_instantiation_ctor_call(
    const std::string &contract_name,
    exprt &new_expr);
  void move_to_initializer(const exprt &expr);
  bool move_initializer_to_ctor(
    const std::string contract_name,
    std::string ctor_id = "");
  bool move_inheritance_to_ctor(
    const std::string contract_name,
    std::string ctor_id,
    symbolt &sym);
  void move_to_front_block(const exprt &expr);
  void move_to_back_block(const exprt &expr);

  // handle contract variables and functions
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
  bool get_init_expr(
    const nlohmann::json &ast_node,
    const typet &dest_type,
    exprt &new_expr);
  bool get_binary_operator_expr(const nlohmann::json &expr, exprt &new_expr);
  bool get_compound_assign_expr(const nlohmann::json &expr, exprt &new_expr);
  bool get_unary_operator_expr(
    const nlohmann::json &expr,
    const nlohmann::json &literal_type,
    exprt &new_expr);
  bool
  get_conditional_operator_expr(const nlohmann::json &expr, exprt &new_expr);
  bool get_cast_expr(
    const nlohmann::json &cast_expr,
    exprt &new_expr,
    const nlohmann::json literal_type = nullptr);
  bool get_var_decl_ref(
    const nlohmann::json &decl,
    bool is_this_ptr,
    exprt &new_expr);
  void get_symbol_decl_ref(
    const std::string &sym_name,
    const std::string &sym_id,
    const typet &t,
    exprt &new_expr);
  bool get_func_decl_ref(const nlohmann::json &decl, exprt &new_expr);
  bool
  get_func_decl_id_ref(const std::string &func_id, nlohmann::json &decl_ref);
  bool get_func_decl_this_ref(const nlohmann::json &decl, exprt &new_expr);
  bool get_func_decl_this_ref(
    const std::string contract_name,
    const std::string &func_id,
    exprt &new_expr);
  bool get_enum_member_ref(const nlohmann::json &decl, exprt &new_expr);
  bool get_esbmc_builtin_ref(const nlohmann::json &decl, exprt &new_expr);
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
  void get_local_var_decl_name(
    const nlohmann::json &ast_node,
    std::string &name,
    std::string &id);
  void get_function_definition_name(
    const nlohmann::json &ast_node,
    std::string &name,
    std::string &id);
  bool get_var_decl_name(
    const nlohmann::json &decl,
    std::string &name,
    std::string &id);
  bool get_non_library_function_call(
    const exprt &func,
    const typet &t,
    const nlohmann::json &decl_ref,
    const nlohmann::json &epxr,
    side_effect_expr_function_callt &call);
  bool get_ctor_call(
    const exprt &func,
    const typet &t,
    const nlohmann::json &decl_ref,
    const nlohmann::json &epxr,
    side_effect_expr_function_callt &call);
  bool
  get_new_object_ctor_call(const nlohmann::json &ast_node, exprt &new_expr);
  bool get_new_object_ctor_call(
    const std::string &contract_name,
    const std::string &ctor_id,
    const nlohmann::json param_list,
    exprt &new_expr);
  bool get_current_contract_name(
    const nlohmann::json &ast_node,
    std::string &contract_name);
  bool get_library_function_call(
    const exprt &func,
    const typet &t,
    const nlohmann::json &caller,
    side_effect_expr_function_callt &call);
  void get_library_function_call_no_args(
    const std::string &func_name,
    const std::string &func_id,
    const typet &t,
    const locationt &l,
    exprt &new_expr);
  void get_malloc_function_call(
    const locationt &loc,
    side_effect_expr_function_callt &_call);
  void get_calloc_function_call(
    const locationt &loc,
    side_effect_expr_function_callt &_call);
  void get_arrcpy_function_call(
    const locationt &loc,
    side_effect_expr_function_callt &calc_call);
  void get_str_assign_function_call(
    const locationt &loc,
    side_effect_expr_function_callt &_call);
  void get_memcpy_function_call(
    const locationt &loc,
    side_effect_expr_function_callt &_call);
  bool is_library_function(const std::string &id);
  bool get_empty_array_ref(const nlohmann::json &ast_node, exprt &new_expr);
  void get_aux_array_name(std::string &aux_name, std::string &aux_id);
  void get_aux_array(const exprt &src_expr, exprt &new_expr);
  void get_aux_var(std::string &aux_name, std::string &aux_id);
  void get_aux_function(std::string &aux_name, std::string &aux_id);
  void get_size_expr(const exprt &rhs, exprt &size_expr);
  void store_update_dyn_array(
    const exprt &dyn_arr,
    const exprt &size_expr,
    exprt &store_call);

  // tuple
  bool get_tuple_definition(const nlohmann::json &ast_node);
  bool get_tuple_instance(const nlohmann::json &ast_node, exprt &new_expr);
  void get_tuple_name(
    const nlohmann::json &ast_node,
    std::string &name,
    std::string &id);
  bool get_tuple_instance_name(
    const nlohmann::json &ast_node,
    std::string &name,
    std::string &id);
  bool get_tuple_function_ref(const nlohmann::json &ast_node, exprt &new_expr);
  bool get_tuple_member_call(
    const irep_idt instance_id,
    const exprt &comp,
    exprt &new_expr);
  void get_tuple_assignment(code_blockt &_block, const exprt &lop, exprt rop);
  void get_tuple_function_call(code_blockt &_block, const exprt &op);
  void get_llc_ret_tuple(exprt &new_expr);

  // string
  void
  get_string_assignment(const exprt &lhs, const exprt &rhs, exprt &new_expr);

  // mapping
  bool get_mapping_type(const nlohmann::json &ast_node, typet &t);
  bool get_mapping_key_expr(
    const symbolt &sym,
    const std::string &postfix,
    exprt &new_expr);
  void get_mapping_key_name(
    const std::string &m_name,
    const std::string &m_id,
    std::string &k_name,
    std::string &k_id);

  // line number and locations
  void
  get_location_from_node(const nlohmann::json &ast_node, locationt &location);
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
  bool multi_transaction_verification(const std::string &contractName);
  bool multi_contract_verification_bound();
  bool multi_contract_verification_unbound();
  void reset_auxiliary_vars();

  // auxiliary functions
  std::string get_modulename_from_path(std::string path);
  std::string get_filename_from_path(std::string path);
  const nlohmann::json &
  find_parent(const nlohmann::json &json, const nlohmann::json &target);
  const nlohmann::json &find_decl_ref(int ref_decl_id);
  const nlohmann::json &
  find_decl_ref(int ref_decl_id, std::string &contract_name);
  const nlohmann::json &find_constructor_ref(int ref_decl_id);
  const nlohmann::json &find_constructor_ref(const std::string &contract_name);
  void convert_expression_to_code(exprt &expr);
  bool check_intrinsic_function(const nlohmann::json &ast_node);
  nlohmann::json make_implicit_cast_expr(
    const nlohmann::json &sub_expr,
    std::string cast_type);
  nlohmann::json make_return_type_from_typet(typet type);
  nlohmann::json make_pointee_type(const nlohmann::json &sub_expr);
  nlohmann::json make_array_elementary_type(const nlohmann::json &type_descrpt);
  nlohmann::json make_array_to_pointer_type(const nlohmann::json &type_descrpt);
  std::string get_array_size(const nlohmann::json &type_descrpt);
  void get_size_of_expr(const typet &elem_type, exprt &size_of_expr);
  bool is_dyn_array(const nlohmann::json &json_in);
  nlohmann::json add_dyn_array_size_expr(
    const nlohmann::json &type_descriptor,
    const nlohmann::json &dyn_array_node);
  bool is_mapping(const nlohmann::json &ast_node);
  void change_balance(const std::string cname, const exprt &value);

  void get_default_symbol(
    symbolt &symbol,
    std::string module_name,
    typet type,
    std::string name,
    std::string id,
    locationt location);

  bool get_ctor_call_id(const std::string &contract_name, std::string &ctor_id);
  std::string get_explicit_ctor_call_id(const std::string &contract_name);
  std::string get_implict_ctor_call_id(const std::string &contract_name);
  bool get_sol_builtin_ref(const nlohmann::json expr, exprt &new_expr);
  void get_temporary_object(exprt &call, exprt &new_expr);
  bool get_unbound_function(const std::string &c_name, symbolt &sym);
  bool get_unbound_expr(const nlohmann::json expr, exprt &new_expr);
  void convert_unboundcall_nondet(
    exprt &new_expr,
    const typet common_type,
    const locationt &l);
  bool add_auxiliary_members(const std::string contract_name);
  void get_builtin_symbol(
    const std::string name,
    const std::string id,
    const typet t,
    const locationt &l,
    const exprt &val,
    const std::string c_name);
  void get_low_level_call(
    const nlohmann::json &json,
    const nlohmann::json &args,
    exprt &new_expr);
  void get_low_level_call(
    const nlohmann::json &json,
    const nlohmann::json &args,
    const exprt &base,
    exprt &new_expr,
    const std::string bs_contract_name);
  void call_modelling(
    const bool has_arguments,
    const exprt &base,
    const std::string &bs_contract_name,
    const std::string &tgt_f_name,
    exprt &trusted_expr);
  void staticcall_modelling(
    const exprt &base,
    const std::string &bs_contract_name,
    const std::string &tgt_f_name,
    exprt &trusted_expr);
  void delegatecall_modelling(
    const exprt &base,
    const std::string &bs_contract_name,
    const std::string &tgt_f_name,
    exprt &trusted_expr);
  void transfer_modelling(
    const exprt &base,
    const std::string &bs_contract_name,
    exprt &trusted_expr);
  void send_modelling(
    const exprt &base,
    const std::string &bs_contract_name,
    exprt &trusted_expr);
  void get_low_level_memcall(
    const std::string new_bs_contract_name,
    const exprt &new_base,
    const nlohmann::json &func_json,
    side_effect_expr_function_callt &_call);
  const nlohmann::json &
  get_func_decl_ref(const std::string &c_name, const std::string &f_name);
  void extend_extcall_modelling(
    const std::string &c_contract_name,
    const locationt &sol_loc);
  bool member_extcall_harness(
    const nlohmann::json &json,
    const nlohmann::json &args,
    const exprt &base,
    exprt &new_expr);
  bool member_builtin_harness(
    const nlohmann::json &json,
    const nlohmann::json &args,
    const exprt &base,
    exprt &new_expr);
  bool get_new_temporary_obj(
    const std::string &c_name,
    const std::string &ctor_ins_name,
    const std::string &ctor_ins_id,
    const locationt &ctor_ins_loc,
    symbolt &added,
    codet &decl);
  void get_low_level_harness(
    const typet &struct_type,
    const exprt &base,
    const std::string &bs_contract_name,
    symbolt &harness);
  void
  get_addr_expr(const std::string &cname, const exprt &base, exprt &new_expr);
  bool set_addr_cname_mapping(
    const std::string &cname,
    const exprt &base,
    exprt &new_expr);
  bool arbitrary_modelling2(
    const std::string &contract_name,
    const struct_typet::componentst &methods,
    const exprt &base,
    codet &body);
  bool populate_nil_this_arguments(
    const exprt &ctor,
    const exprt &this_object,
    side_effect_expr_function_callt &call);
  bool get_this_object(const exprt &func, exprt &this_object);
  bool get_high_level_member_access(
    const nlohmann::json &expr,
    const exprt &base,
    const exprt &member,
    const bool is_func_call,
    exprt &new_expr);
  bool get_bind_cname(const nlohmann::json &json, exprt &bind_cname_expr);
  void get_nondet_contract_name(
    const exprt src_expr,
    const typet dest_type,
    exprt &new_expr);
  bool assign_nondet_contract_name(const std::string &_cname, exprt &new_expr);
  bool assign_param_nondet(
    const nlohmann::json &decl_ref,
    side_effect_expr_function_callt &call);
  bool get_base_contract_name(const exprt &base, std::string &cname);

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
  void convert_type_expr(const namespacet &ns, exprt &dest, const typet &type);
  bool
  convert_hex_literal(std::string the_value, exprt &dest, const int n = 256);
  // check if it's a bytes type
  bool is_bytes_type(const typet &t);

  contextt &context;
  namespacet ns;
  // json for Solidity AST. Use vector for multiple contracts
  nlohmann::json src_ast_json_array = nlohmann::json::array();
  // json for Solidity AST. Use object for single contract
  nlohmann::json src_ast_json;
  // Solidity function to be verified
  const std::string &sol_func;
  //smart contract source file
  const std::string &contract_path;

  std::string absolute_path;
  std::string contract_contents = "";
  // scope id of "ContractDefinition"
  int global_scope_id;

  unsigned int current_scope_var_num;
  const nlohmann::json *current_functionDecl;
  const nlohmann::json *current_forStmt;
  const nlohmann::json *current_typeName;
  // store multiple exprt and flatten the block
  code_blockt expr_frontBlockDecl;
  code_blockt expr_backBlockDecl;
  code_blockt ctor_frontBlockDecl;
  code_blockt ctor_backBlockDecl;
  // for tuple
  bool current_lhsDecl;
  bool current_rhsDecl;
  // Use current level of BinOp type as the "anchor" type for numerical literal conversion:
  // In order to remove the unnecessary implicit IntegralCast. We need type of current level of BinaryOperator.
  // All numeric literals will be implicitly converted to this type. Pop it when finishing the current level of BinaryOperator.
  // TODO: find a better way to deal with implicit type casting if it's not able to cope with compelx rules
  std::stack<const nlohmann::json *> current_BinOp_type;
  std::string current_functionName;

  //! Be careful of using 'current_contractName'. This might lead to trouble in inheritance.
  //! If you are not sure, use 'get_current_contract_name' instead.
  std::string current_contractName;
  std::string current_baseContractName;
  std::string current_fileName;

  // Auxiliary data structures:
  // Mapping from the node 'id' to the exported symbol (i.e. contract, error, constant var ....)
  std::unordered_map<int, std::string> exportedSymbolsList;
  // Inheritance Order Record <contract_name, Contract_id>
  std::unordered_map<std::string, std::vector<int>> linearizedBaseList;
  // Who inherits from me?
  std::unordered_map<std::string, std::unordered_set<std::string>>
    inheritanceMap;
  //std::unordered_map<std::string, std::unordered_set<std::string>> functionSignature;
  // contract name list
  std::unordered_map<int, std::string> contractNamesMap;
  std::set<std::string> contractNamesList;
  // Store the ast_node["id"] of contract/struct/function/...
  std::unordered_map<int, std::string> scope_map;
  // Store state variables
  code_blockt initializers;
  // For inheritance
  const nlohmann::json *ctor_modifier;
  const nlohmann::json *based_contracts;

  static constexpr const char *mode = "C++";

  // The prefix for the id of each class
  std::string prefix = "tag-";

  // json nodes that always empty
  // used as the return value for find_constructor_ref when
  // dealing with the implicit constructor call
  // this is to avoid reference to stack memory associated with local variable returned
  const nlohmann::json empty_json;

  // --function
  std::string tgt_func;
  // --contract
  std::string tgt_cnt;

  // for auxiliary var name
  int aux_counter;

  // bound setting
  bool is_bound;

  // NONDET
  side_effect_expr_function_callt nondet_bool_expr;
  side_effect_expr_function_callt nondet_uint_expr;

private:
  bool get_elementary_type_name_uint(
    SolidityGrammar::ElementaryTypeNameT &type,
    typet &out);
  bool get_elementary_type_name_int(
    SolidityGrammar::ElementaryTypeNameT &type,
    typet &out);
  bool get_elementary_type_name_bytesn(
    SolidityGrammar::ElementaryTypeNameT &type,
    typet &out);
};

#endif /* SOLIDITY_FRONTEND_SOLIDITY_CONVERT_H_ */
