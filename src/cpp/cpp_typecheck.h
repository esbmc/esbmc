/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_CPP_TYPECHECK_H
#define CPROVER_CPP_TYPECHECK_H

#include <ansi-c/c_typecheck_base.h>
#include <cassert>
#include <cpp/cpp_member_spec.h>
#include <cpp/cpp_parse_tree.h>
#include <cpp/cpp_scopes.h>
#include <cpp/cpp_template_type.h>
#include <cpp/cpp_typecheck_resolve.h>
#include <cpp/cpp_util.h>
#include <cpp/template_map.h>
#include <list>
#include <map>
#include <set>
#include <util/std_code.h>
#include <util/std_types.h>

bool cpp_typecheck(
  cpp_parse_treet &cpp_parse_tree,
  contextt &context,
  const std::string &module,
  message_handlert &message_handler);

bool cpp_typecheck(
  exprt &expr,
  message_handlert &message_handler,
  const namespacet &ns);

class cpp_typecast_rank
{
public:
  // Total score of what conversions occurred to make this conversion happen,
  // summed from:
  //   0) None?
  //   1) An exact-match conversion
  //   2) Promotion
  //   3) Conversion (including floating point and pointer casting).
  //   4) User defined conversion.
  unsigned int rank;

  // How many template arguments exist in this templated function.
  unsigned int templ_distance;

  bool has_ptr_to_bool;    // Self explanatory.
  bool has_ptr_to_base;    // Derived ptr casted down to base ptr
  bool has_ptr_to_voidptr; // Any pointer converted to void ptr type.

  // There are more conditions to detect; These are the most relevant at the
  // moment.

  cpp_typecast_rank()
  {
    rank = 0;
    templ_distance = 0;
    has_ptr_to_bool = false;
    has_ptr_to_base = false;
    has_ptr_to_voidptr = false;
  }

  cpp_typecast_rank &operator+=(const cpp_typecast_rank &ref)
  {
    rank += ref.rank;
    templ_distance += ref.templ_distance;
    has_ptr_to_bool |= ref.has_ptr_to_bool;
    has_ptr_to_base |= ref.has_ptr_to_base;
    has_ptr_to_voidptr |= ref.has_ptr_to_voidptr;
    return *this;
  }

  bool operator<(const cpp_typecast_rank &ref) const
  {
    if(rank < ref.rank)
      return true;
    if(rank > ref.rank)
      return false;

    // Put funkier rules here. Note that less-than means a better match.

    // Prefer functions with the fewest template parameters.
    if(templ_distance < ref.templ_distance)
      return true;
    if(templ_distance > ref.templ_distance)
      return false;

    if(!has_ptr_to_bool && ref.has_ptr_to_bool)
      return true;

    // Pointer-to-base-ptr is better than casting it to void.
    if(has_ptr_to_base && ref.has_ptr_to_voidptr)
      return true;

    // Insert here: other rules.
    return false;
  }
};

class cpp_typecheckt : public c_typecheck_baset
{
public:
  cpp_typecheckt(
    cpp_parse_treet &_cpp_parse_tree,
    contextt &_context,
    const std::string &_module,
    message_handlert &message_handler)
    : c_typecheck_baset(_context, _module, message_handler),
      cpp_parse_tree(_cpp_parse_tree),
      template_counter(0),
      anon_counter(0),
      disable_access_control(false)
  {
  }

  cpp_typecheckt(
    cpp_parse_treet &_cpp_parse_tree,
    contextt &_context1,
    const contextt &_context2,
    const std::string &_module,
    message_handlert &message_handler)
    : c_typecheck_baset(_context1, _context2, _module, message_handler),
      cpp_parse_tree(_cpp_parse_tree),
      template_counter(0),
      anon_counter(0),
      disable_access_control(false)
  {
  }

  ~cpp_typecheckt() override = default;

  void typecheck() override;

  // overload to use C++ syntax

  std::string to_string(const typet &type) override;
  std::string to_string(const exprt &expr) override;

  friend class cpp_typecheck_resolvet;
  friend class cpp_declarator_convertert;

  exprt resolve(
    const cpp_namet &cpp_name,
    const cpp_typecheck_resolvet::wantt want,
    const cpp_typecheck_fargst &fargs,
    bool fail_with_exception = true)
  {
    cpp_typecheck_resolvet cpp_typecheck_resolve(*this);
    return cpp_typecheck_resolve.resolve(
      cpp_name, want, fargs, fail_with_exception);
  }

  void typecheck_expr(exprt &expr) override;

  bool cpp_is_pod(const typet &type) const;

  codet cpp_constructor(
    const locationt &location,
    const exprt &object,
    const exprt::operandst &operands);

protected:
  cpp_scopest cpp_scopes;

  cpp_parse_treet &cpp_parse_tree;
  irep_idt current_mode;

  void convert(cpp_linkage_spect &linkage_spec);
  void convert(cpp_namespace_spect &namespace_spec);
  void convert(cpp_usingt &cpp_using);
  void convert(cpp_itemt &item);
  void convert(cpp_declarationt &declaration);
  void convert(cpp_declaratort &declarator);

  void convert_initializer(symbolt &symbol);
  void convert_function(symbolt &symbol);

  void convert_pmop(exprt &expr);

  void convert_anonymous_union(cpp_declarationt &declaration, codet &new_code);

  void convert_compound_ano_union(
    const cpp_declarationt &declaration,
    const irep_idt &access,
    struct_typet::componentst &components);

  //
  // Templates
  //
  void salvage_default_parameters(
    const template_typet &old_type,
    template_typet &new_type);

  void
  check_template_restrictions(const irept &cpp_name, const typet &final_type);

  void convert_template_declaration(cpp_declarationt &declaration);

  void convert_non_template_declaration(cpp_declarationt &declaration);

  void convert_template_function_or_member_specialization(
    cpp_declarationt &declaration);

  void convert_class_template_specialization(cpp_declarationt &declaration);

  void typecheck_class_template(cpp_declarationt &declaration);

  void typecheck_function_template(cpp_declarationt &declaration);

  void typecheck_class_template_member(cpp_declarationt &declaration);

  std::string class_template_identifier(
    const irep_idt &base_name,
    const template_typet &template_type,
    const cpp_template_args_non_tct &partial_specialization_args);

  std::string function_template_identifier(
    const irep_idt &base_name,
    const template_typet &template_type,
    const typet &function_type);

  cpp_template_args_tct typecheck_template_args(
    const locationt &location,
    const symbolt &template_symbol,
    const cpp_template_args_non_tct &template_args);

  class instantiationt
  {
  public:
    locationt location;
    irep_idt identifier;
    cpp_template_args_tct full_template_args;
  };

  typedef std::list<instantiationt> instantiation_stackt;
  instantiation_stackt instantiation_stack;

  void show_instantiation_stack(std::ostream &);

  class instantiation_levelt
  {
  public:
    instantiation_levelt(instantiation_stackt &_instantiation_stack)
      : instantiation_stack(_instantiation_stack)
    {
      instantiation_stack.emplace_back();
    }

    ~instantiation_levelt()
    {
      instantiation_stack.pop_back();
    }

  private:
    instantiation_stackt &instantiation_stack;
  };

  const symbolt *handle_recursive_template_instance(
    const symbolt &template_symbol,
    const cpp_template_args_tct &full_template_args,
    const exprt &new_decl);

  const symbolt &instantiate_template(
    const locationt &location,
    const symbolt &template_symbol,
    const cpp_template_args_tct &specialization_template_args,
    const cpp_template_args_tct &full_template_args,
    const typet &specialization = typet("nil"));

  void put_template_args_in_scope(
    const template_typet &template_type,
    const cpp_template_args_tct &template_args);

  void put_template_arg_into_scope(
    const template_parametert &template_param,
    const exprt &argument);

  const symbolt *is_template_instantiated(
    const irep_idt &template_symbol_name,
    const irep_idt &template_pattern_name) const;

  void mark_template_instantiated(
    const irep_idt &template_symbol_name,
    const irep_idt &template_pattern_name,
    const irep_idt &instantiated_symbol_name);

  unsigned template_counter;
  unsigned anon_counter;

  template_mapt template_map;

  std::string template_suffix(const cpp_template_args_tct &template_args);

  void convert_arguments(const irep_idt &mode, code_typet &function_type);

  void convert_argument(const irep_idt &mode, code_typet::argumentt &argument);

  bool has_incomplete_args(cpp_template_args_tct template_args_tc);

  //
  // Misc
  //

  void find_constructor(const typet &dest_type, exprt &symbol_expr);

  void default_ctor(
    const locationt &location,
    const irep_idt &base_name,
    cpp_declarationt &ctor) const;

  void default_cpctor(const symbolt &, cpp_declarationt &cpctor) const;

  void default_assignop(const symbolt &symbol, cpp_declarationt &cpctor);

  void
  default_assignop_value(const symbolt &symbol, cpp_declaratort &declarator);

  void default_dtor(const symbolt &symb, cpp_declarationt &dtor);

  void dtor(const symbolt &symb, code_blockt &vtables, code_blockt &dtors);

  void check_member_initializers(
    const irept &bases,
    const struct_typet::componentst &components,
    const irept &initializers);

  bool check_component_access(
    const irept &component,
    const struct_typet &struct_type);

  void full_member_initialization(
    const struct_typet &struct_type,
    irept &initializers);

  bool find_cpctor(const symbolt &symbol) const;
  bool find_assignop(const symbolt &symbol) const;
  bool find_dtor(const symbolt &symbol) const;

  bool find_parent(
    const symbolt &symb,
    const irep_idt &base_name,
    irep_idt &identifier);

  bool get_component(
    const locationt &location,
    const exprt &object,
    const irep_idt &component_name,
    exprt &member);

  void new_temporary(
    const locationt &location,
    const typet &,
    const exprt::operandst &ops,
    exprt &temporary);

  void new_temporary(
    const locationt &location,
    const typet &,
    const exprt &op,
    exprt &temporary);

  void static_initialization();
  void do_not_typechecked();
  void clean_up();

  void add_base_components(
    const struct_typet &from,
    const irep_idt &access,
    struct_typet &to,
    std::set<irep_idt> &bases,
    std::set<irep_idt> &vbases,
    bool is_virtual);

  bool cast_away_constness(const typet &t1, const typet &t2) const;

  void do_virtual_table(const symbolt &symbol);

  // we need to be able to delay the typechecking
  // of function bodies to handle methods with
  // bodies in the class definition
  struct function_bodyt
  {
  public:
    function_bodyt(
      symbolt *_function_symbol,
      template_mapt _template_map,
      instantiation_stackt _instantiation_stack)
      : function_symbol(_function_symbol),
        template_map(std::move(_template_map)),
        instantiation_stack(std::move(_instantiation_stack))
    {
    }

    symbolt *function_symbol;
    template_mapt template_map;
    instantiation_stackt instantiation_stack;
  };

  typedef std::list<function_bodyt> function_bodiest;
  function_bodiest function_bodies;

  void add_function_body(symbolt *_function_symbol)
  {
    function_bodies.emplace_back(
      _function_symbol, template_map, instantiation_stack);
  }

  // types

  bool convert_typedef(typet &type);
  void typecheck_type(typet &type) override;

  cpp_scopet &typecheck_template_parameters(template_typet &type);

  std::string fetch_compound_name(const typet &type);
  void typecheck_compound_type(typet &type) override;
  void check_array_types(typet &type);
  void typecheck_enum_type(typet &type);

  // determine the scope into which a tag goes
  // (enums, structs, union, classes)
  cpp_scopet &tag_scope(
    const irep_idt &_base_name,
    bool has_body,
    bool tag_only_declaration);

  void typecheck_compound_declarator(
    const symbolt &symbol,
    const cpp_declarationt &declaration,
    cpp_declaratort &declarator,
    struct_typet::componentst &components,
    const irep_idt &access,
    bool is_static,
    bool is_typedef,
    bool is_mutable);

  void typecheck_friend_declaration(
    symbolt &symbol,
    cpp_declarationt &cpp_declaration);

  void put_compound_into_scope(const irept &compound);
  void typecheck_compound_body(symbolt &symbol);
  void typecheck_enum_body(symbolt &symbol);
  void typecheck_function_bodies();
  void typecheck_compound_bases(struct_typet &type);
  void add_anonymous_members_to_scope(const symbolt &struct_union_symbol);

  void move_member_initializers(
    irept &initializers,
    const typet &type,
    exprt &value);

  static bool has_const(const typet &type);
  static bool has_volatile(const typet &type);

  void typecheck_member_function(
    const irep_idt &compound_symbol,
    struct_typet::componentt &component,
    irept &initializers,
    const typet &method_qualifier,
    exprt &value);

  void adjust_method_type(
    const irep_idt &compound_symbol,
    typet &method_type,
    const typet &method_qualifier);

  // for function overloading
  irep_idt function_identifier(const typet &type);

  using c_typecheck_baset::zero_initializer;
  void zero_initializer(
    const exprt &object,
    const typet &type,
    const locationt &location,
    exprt::operandst &ops);

  // code conversion
  void typecheck_code(codet &code) override;
  virtual void typecheck_catch(codet &code);
  virtual void typecheck_throw_decl(codet &code);
  virtual void typecheck_member_initializer(codet &code);
  void typecheck_decl(codet &code) override;
  void typecheck_block(codet &code) override;
  void typecheck_ifthenelse(codet &code) override;
  void typecheck_while(codet &code) override;
  void typecheck_switch(codet &code) override;

  const struct_typet &this_struct_type();

  codet cpp_destructor(
    const locationt &location,
    const typet &type,
    const exprt &object);

  // expressions
  void typecheck_expr_main(exprt &expr) override;
  void typecheck_expr_member(exprt &expr) override;
  void typecheck_expr_ptrmember(exprt &expr) override;
  void typecheck_expr_throw(exprt &expr);
  void typecheck_function_expr(exprt &expr, const cpp_typecheck_fargst &fargs);
  void typecheck_expr_cpp_name(exprt &expr, const cpp_typecheck_fargst &fargs);
  void typecheck_expr_member(exprt &expr, const cpp_typecheck_fargst &fargs);
  void typecheck_expr_ptrmember(exprt &expr, const cpp_typecheck_fargst &fargs);
  void typecheck_cast_expr(exprt &expr);
  void typecheck_expr_trinary(exprt &expr) override;
  void typecheck_expr_binary_arithmetic(exprt &expr) override;
  void typecheck_expr_explicit_typecast(exprt &expr);
  void typecheck_expr_explicit_constructor_call(exprt &expr);
  void typecheck_expr_address_of(exprt &expr) override;
  void typecheck_expr_dereference(exprt &expr) override;
  void typecheck_expr_typeid(exprt &expr);
  void typecheck_expr_function_identifier(exprt &expr) override;
  void typecheck_expr_reference_to(exprt &expr);
  void typecheck_expr_this(exprt &expr);
  void typecheck_expr_new(exprt &expr);
  void typecheck_expr_sizeof(exprt &expr) override;
  void typecheck_expr_delete(exprt &expr);
  void typecheck_expr_side_effect(side_effect_exprt &expr) override;
  void typecheck_side_effect_assignment(exprt &expr) override;
  void typecheck_side_effect_increment(side_effect_exprt &expr);
  void typecheck_expr_typecast(exprt &expr) override;
  void typecheck_expr_index(exprt &expr) override;
  void typecheck_expr_rel(exprt &expr) override;
  void typecheck_expr_comma(exprt &expr) override;

  void typecheck_function_call_arguments(
    side_effect_expr_function_callt &expr) override;

  bool operator_is_overloaded(exprt &expr);
  bool overloadable(const exprt &expr);

  void add_implicit_dereference(exprt &expr);

  void typecheck_side_effect_function_call(
    side_effect_expr_function_callt &expr) override;

  void typecheck_method_application(side_effect_expr_function_callt &expr);

  void typecheck_assign(codet &code) override;

public:
  //
  // Type Conversions
  //

  bool standard_conversion_lvalue_to_rvalue(const exprt &expr, exprt &new_expr)
    const;

  bool standard_conversion_array_to_pointer(const exprt &expr, exprt &new_expr)
    const;

  bool standard_conversion_function_to_pointer(
    const exprt &expr,
    exprt &new_expr) const;

  bool standard_conversion_qualification(
    const exprt &expr,
    const typet &,
    exprt &new_expr) const;

  bool standard_conversion_integral_promotion(
    const exprt &expr,
    exprt &new_expr) const;

  bool standard_conversion_floating_point_promotion(
    const exprt &expr,
    exprt &new_expr) const;

  bool standard_conversion_integral_conversion(
    const exprt &expr,
    const typet &type,
    exprt &new_expr) const;

  bool standard_conversion_floating_integral_conversion(
    const exprt &expr,
    const typet &type,
    exprt &new_expr) const;

  bool standard_conversion_floating_point_conversion(
    const exprt &expr,
    const typet &type,
    exprt &new_expr) const;

  bool standard_conversion_pointer(
    const exprt &expr,
    const typet &type,
    exprt &new_expr);

  bool standard_conversion_pointer_to_member(
    const exprt &expr,
    const typet &type,
    exprt &new_expr);

  bool standard_conversion_boolean(const exprt &expr, exprt &new_expr) const;

  bool standard_conversion_sequence(
    const exprt &expr,
    const typet &type,
    exprt &new_expr,
    class cpp_typecast_rank &rank);

  bool user_defined_conversion_sequence(
    const exprt &expr,
    const typet &type,
    exprt &new_expr,
    class cpp_typecast_rank &rank);

  bool reference_related(const exprt &expr, const typet &type) const;

  bool reference_compatible(
    const exprt &expr,
    const typet &type,
    class cpp_typecast_rank &rank) const;

  bool reference_binding(
    exprt expr,
    const typet &type,
    exprt &new_expr,
    class cpp_typecast_rank &rank);

  bool implicit_conversion_sequence(
    const exprt &expr,
    const typet &type,
    exprt &new_expr,
    class cpp_typecast_rank &rank);

  bool implicit_conversion_sequence(
    const exprt &expr,
    const typet &type,
    class cpp_typecast_rank &rank);

  bool implicit_conversion_sequence(
    const exprt &expr,
    const typet &type,
    exprt &new_expr);

  void reference_initializer(exprt &expr, const typet &type);

  void implicit_typecast(exprt &expr, const typet &type) override;

  void get_bases(const struct_typet &type, std::set<irep_idt> &set_bases) const;

  void get_virtual_bases(const struct_typet &type, std::list<irep_idt> &vbases)
    const;

  bool subtype_typecast(const struct_typet &from, const struct_typet &to) const;

  void make_ptr_typecast(exprt &expr, const typet &dest_type);

  // the C++ typecasts

  bool const_typecast(const exprt &expr, const typet &type, exprt &new_expr);

  bool dynamic_typecast(const exprt &expr, const typet &type, exprt &new_expr);

  bool reinterpret_typecast(
    const exprt &expr,
    const typet &type,
    exprt &new_expr,
    bool check_constantness = true);

  bool static_typecast(
    const exprt &expr,
    const typet &type,
    exprt &new_expr,
    bool check_constantness = true);

private:
  std::list<irep_idt> dinis;   // Static Default-Initialization List
  bool disable_access_control; // Disable protect and private
};

#endif
