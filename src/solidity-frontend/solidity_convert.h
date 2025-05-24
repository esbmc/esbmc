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
    const std::string &_sol_cnts,
    const std::string &_sol_func,
    const std::string &_contract_path);
  virtual ~solidity_convertert() = default;

  bool convert();
  static bool is_low_level_call(const std::string &name);
  static bool is_low_level_property(const std::string &name);

  // find reference
  static const nlohmann::json &
  find_last_parent(const nlohmann::json &json, const nlohmann::json &target);
  static const nlohmann::json &find_parent_contract(
    const nlohmann::json &json,
    const nlohmann::json &target);
  static const nlohmann::json &
  find_decl_ref_in_contract(const nlohmann::json &j, int ref_id);
  static const nlohmann::json &
  find_decl_ref_global(const nlohmann::json &j, int ref_id);
  static const nlohmann::json &
  find_decl_ref_unique_id(const nlohmann::json &json, int ref_id);
  static const nlohmann::json &
  find_decl_ref(const nlohmann::json &json, int ref_id);
  static const nlohmann::json &find_constructor_ref(int ref_decl_id);
  static const nlohmann::json &
  find_constructor_ref(const std::string &contract_name);

  // json nodes that always empty
  // used as the return value for find_constructor_ref when
  // dealing with the implicit constructor call
  // this is to avoid reference to stack memory associated with local variable returned
  static const nlohmann::json empty_json;
  //! Be careful of using 'current_contractName'. This might lead to trouble in inheritance.
  //! If you are not sure, use 'get_current_contract_name' instead.
  static std::string current_baseContractName;

  // json for Solidity AST. Use object for contract
  static nlohmann::json src_ast_json;

protected:
  typedef struct func_sig
  {
    std::string name;
    std::string id;
    std::string visibility;
    code_typet type;
    bool is_payable;
    bool is_inherit;
    bool is_library;

    func_sig(
      const std::string &name,
      const std::string &id,
      const std::string &visibility,
      const code_typet &type,
      bool is_payable,
      bool is_inherit,
      bool is_library)
      : name(name),
        id(id),
        visibility(visibility),
        type(type),
        is_payable(is_payable),
        is_inherit(is_inherit),
        is_library(is_library)
    {
    }

    bool operator==(const func_sig &other) const
    {
      return id == other.id;
    }
  } func_sig;

  void merge_multi_files();
  void topological_sort(
    std::unordered_map<std::string, std::unordered_set<std::string>> &graph,
    std::unordered_map<std::string, nlohmann::json> &path_to_json,
    nlohmann::json &sorted_files);
  void contract_precheck();
  bool populate_auxilary_vars();
  bool
  populate_function_signature(nlohmann::json &json, const std::string &cname);
  bool populate_low_level_functions(const std::string &cname);
  bool convert_ast_nodes(
    const nlohmann::json &contract_def,
    const std::string &cname);

  // conversion functions
  // get decl in rule contract-body-element
  bool get_contract_definition(const std::string &c_name);
  bool get_non_function_decl(const nlohmann::json &ast_node, exprt &new_expr);
  bool get_function_decl(const nlohmann::json &ast_node);
  bool get_var_decl(
    const nlohmann::json &ast_node,
    const nlohmann::json &initialValue,
    exprt &new_expr);
  bool get_var_decl(const nlohmann::json &ast_node, exprt &new_expr);
  bool get_function_definition(const nlohmann::json &ast_node);
  bool get_function_params(
    const nlohmann::json &pd,
    const std::string &cname,
    exprt &param);
  void get_function_this_pointer_param(
    const std::string &contract_name,
    const std::string &ctor_id,
    const std::string &debug_modulename,
    const locationt &location_begin,
    code_typet &type);
  bool get_unbound_funccall(
    const std::string contractName,
    code_function_callt &call);
  void get_static_contract_instance_name(
    const std::string c_name,
    std::string &name,
    std::string &id);
  void add_static_contract_instance(const std::string c_name);
  void get_static_contract_instance_ref(const std::string &c_name, exprt &new_expr);
  void get_inherit_static_contract_instance_name(
    const std::string bs_c_name,
    const std::string c_name,
    std::string &name,
    std::string &id);
  void get_inherit_static_contract_instance(
    const std::string bs_c_name,
    const std::string c_name,
    const nlohmann::json &args_list,
    symbolt &sym);
  void get_inherit_ctor_definition(const std::string c_name, exprt &new_expr);
  void get_inherit_ctor_definition_name(
    const std::string c_name,
    std::string &name,
    std::string &id);
  void get_contract_mutex_name(
    const std::string c_name,
    std::string &name,
    std::string &id);
  void get_contract_mutex_expr(
    const std::string c_name,
    const exprt &this_expr,
    exprt &expr);
  bool get_high_level_call_wrapper(
    const std::string c_name,
    const exprt &this_expr,
    exprt &front_block,
    exprt &back_block);
  bool is_sol_builin_symbol(const std::string &cname, const std::string &name);

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
    const std::string &c_name,
    nlohmann::json &c_node,
    std::set<std::string> &merged_list);
  void add_inherit_label(nlohmann::json &node, const std::string &cname);

  // handle constructor
  bool get_constructor(
    const nlohmann::json &ast_node,
    const std::string &contract_name);
  bool add_implicit_constructor(const std::string &contract_name);
  bool get_implicit_ctor_ref(const std::string &contract_name, exprt &new_expr);
  bool get_instantiation_ctor_call(
    const std::string &contract_name,
    exprt &new_expr);
  void move_to_initializer(const exprt &expr);
  bool move_initializer_to_ctor(
    const nlohmann::json *based_contracts,
    const std::string contract_name);
  bool move_initializer_to_ctor(
    const nlohmann::json *based_contracts,
    const std::string contract_name,
    bool is_aux_ctor);
  bool move_inheritance_to_ctor(
    const nlohmann::json *based_contracts,
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
    const nlohmann::json &init_value,
    const nlohmann::json &literal_type,
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
    const bool is_this_ptr,
    exprt &new_expr);
  void get_symbol_decl_ref(
    const std::string &sym_name,
    const std::string &sym_id,
    const typet &t,
    exprt &new_expr);
  bool get_func_decl_ref(const nlohmann::json &decl, exprt &new_expr);
  bool get_func_decl_this_ref(const nlohmann::json &decl, exprt &new_expr);
  bool get_func_decl_this_ref(
    const std::string contract_name,
    const std::string &func_id,
    exprt &new_expr);
  bool get_ctor_decl_this_ref(const nlohmann::json &caller, exprt &this_object);
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
    const std::string &cname,
    std::string &name,
    std::string &id);
  void get_local_var_decl_name(
    const nlohmann::json &ast_node,
    const std::string &cname,
    std::string &name,
    std::string &id);
  void get_function_definition_name(
    const nlohmann::json &ast_node,
    std::string &name,
    std::string &id);
  void get_error_definition_name(
    const nlohmann::json &ast_node,
    std::string &name,
    std::string &id);
  bool get_var_decl_name(
    const nlohmann::json &decl,
    std::string &name,
    std::string &id);
  bool get_non_library_function_call(
    const nlohmann::json &decl_ref,
    const nlohmann::json &caller,
    side_effect_expr_function_callt &call);
  bool get_ctor_call(
    const nlohmann::json &decl_ref,
    const nlohmann::json &epxr,
    side_effect_expr_function_callt &call);
  bool
  get_new_object_ctor_call(const nlohmann::json &ast_node, exprt &new_expr);
  bool get_new_object_ctor_call(
    const std::string &contract_name,
    const nlohmann::json param_list,
    exprt &new_expr);
  void get_current_contract_name(
    const nlohmann::json &ast_node,
    std::string &contract_name);
  bool get_library_function_call(
    const nlohmann::json &decl_ref,
    const nlohmann::json &caller,
    side_effect_expr_function_callt &call);
  bool get_library_function_call(
    const exprt &func,
    const typet &t,
    const nlohmann::json &decl_ref,
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
  bool is_esbmc_library_function(const std::string &id);
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
  bool construct_tuple_assigments(
    const nlohmann::json &ast_node,
    const exprt &lhs,
    const exprt &rhs);
  void get_tuple_assignment(const exprt &lop, exprt rop);
  void get_tuple_function_call(const exprt &op);
  void get_llc_ret_tuple(symbolt &sym);

  // string
  void
  get_string_assignment(const exprt &lhs, const exprt &rhs, exprt &new_expr);

  // mapping
  void get_mapping_inf_arr_name(
    const std::string &cname,
    const std::string &name,
    std::string &arr_name,
    std::string &arr_id);
  bool is_mapping_set_lvalue(const nlohmann::json &json);

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
  bool multi_transaction_verification(
    const std::string &contractName,
    bool is_final_main);
  bool multi_contract_verification_bound(std::set<std::string> &tgt_set);
  bool multi_contract_verification_unbound(std::set<std::string> &tgt_set);
  void reset_auxiliary_vars();

  // auxiliary functions
  std::string get_modulename_from_path(std::string path);
  std::string get_filename_from_path(std::string path);
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
  bool is_func_sig_cover(const std::string &derived, const std::string &base);
  bool is_var_getter_matched(
    const std::string &cname,
    const std::string &tname,
    const typet &ttype);

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
  bool get_unbound_expr(
    const nlohmann::json expr,
    const std::string &cname,
    exprt &new_expr);
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
  void move_builtin_to_contract(
    const std::string cname,
    const exprt &sym,
    bool is_method);
  void move_builtin_to_contract(
    const std::string cname,
    const exprt &sym,
    const std::string &access,
    bool is_method);
  const nlohmann::json &
  get_func_decl_ref(const std::string &c_name, const std::string &f_name);
  void get_builtin_property_expr(
    const std::string &name,
    const exprt &base,
    const locationt &loc,
    exprt &new_expr);
  void get_aux_property_function(
    const exprt &addr,
    const typet &return_t,
    const locationt &loc,
    const std::string &property_name,
    exprt &new_expr);
  bool get_new_temporary_obj(
    const std::string &c_name,
    const std::string &ctor_ins_name,
    const std::string &ctor_ins_id,
    const locationt &ctor_ins_loc,
    symbolt &added,
    codet &decl);
  void
  get_addr_expr(const std::string &cname, const exprt &base, exprt &new_expr);
  void get_new_object(const typet &t, exprt &this_object);
  bool get_high_level_member_access(
    const nlohmann::json &expr,
    const nlohmann::json &literal_type,
    const exprt &base,
    const exprt &member,
    const exprt &_mem_call,
    const bool is_func_call,
    exprt &new_expr);
  bool get_high_level_member_access(
    const nlohmann::json &expr,
    const exprt &base,
    const exprt &member,
    const exprt &_mem_call,
    const bool is_func_call,
    exprt &new_expr);
  bool get_low_level_member_accsss(
    const nlohmann::json &expr,
    const nlohmann::json &options,
    const std::string mem_name,
    const exprt &base,
    const exprt &arg,
    exprt &new_expr);
  bool has_callable_func(const std::string &cname);
  bool
  has_target_function(const std::string &cname, const std::string func_name);
  func_sig
  get_target_function(const std::string &cname, const std::string &func_name);
  bool get_call_definition(const std::string &cname, exprt &new_expr);
  bool get_call_value_definition(const std::string &cname, exprt &new_expr);
  bool get_transfer_definition(const std::string &cname, exprt &new_expr);
  bool get_send_definition(const std::string &cname, exprt &new_expr);
  bool model_transaction(
    const nlohmann::json &expr,
    const exprt &base,
    const exprt &value,
    const locationt &loc,
    exprt &front_block,
    exprt &back_block);

  bool get_bind_cname_expr(const nlohmann::json &json, exprt &bind_cname_expr);
  void get_nondet_expr(const typet &t, exprt &new_expr);
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
  // Solidity contracts/ function to be verified
  const std::string &tgt_cnts;
  const std::string &tgt_func;
  //smart contract source file
  const std::string &contract_path;

  std::string absolute_path;
  std::string contract_contents = "";

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

  // Auxiliary data structures:
  // Mapping from the node 'id' to the exported symbol (i.e. contract, error, constant var ....)
  std::unordered_map<int, std::string> exportedSymbolsList;
  // Inheritance Order Record <contract_name, Contract_id>
  std::unordered_map<std::string, std::vector<int>> linearizedBaseList;
  // Who inherits from me?
  std::unordered_map<std::string, std::unordered_set<std::string>>
    inheritanceMap;
  // structural typing indentical
  std::unordered_map<std::string, std::unordered_set<std::string>>
    structureTypingMap;

  // contract name list
  std::unordered_map<int, std::string> contractNamesMap;
  std::set<std::string> contractNamesList;
  // Store the ast_node["id"] of struct/error
  // where entity contains "members": [{}, {}...]
  std::unordered_map<int, std::string> member_entity_scope;
  // Store state variables
  code_blockt initializers;
  // For inheritance
  const nlohmann::json *ctor_modifier;

  static constexpr const char *mode = "C++";

  std::unordered_map<std::string, std::vector<func_sig>> funcSignatures;

  // The prefix for the id of each class
  std::string prefix = "tag-";

  // for auxiliary var name
  int aux_counter;

  // bound setting
  bool is_bound;

  // reentry-check setting
  bool is_reentry_check;

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
