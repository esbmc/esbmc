/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_IREP_H
#define CPROVER_IREP_H

#include <vector>
#include <list>
#include <map>
#include <string>

#include <cassert>
#include "global.h"

#define USE_DSTRING
#define SHARING

#include "dstring.h"

typedef dstring irep_idt;
typedef dstring irep_namet;
typedef dstring_hash irep_id_hash;

#define forall_irep(it, irep) \
  for(irept::subt::const_iterator it=(irep).begin(); \
      it!=(irep).end(); it++)

#define Forall_irep(it, irep) \
  for(irept::subt::iterator it=(irep).begin(); \
      it!=(irep).end(); it++)

#define forall_named_irep(it, irep) \
  for(irept::named_subt::const_iterator it=(irep).begin(); \
      it!=(irep).end(); it++)

#define Forall_named_irep(it, irep) \
  for(irept::named_subt::iterator it=(irep).begin(); \
      it!=(irep).end(); it++)

#include <iostream>

class typet;

class irept
{
public:
  typedef std::vector<irept> subt;
  //typedef std::list<irept> subt;

  typedef std::map<irep_namet, irept> named_subt;

  // Dump contents of irep to stdout. Debugging only.
  void dump() const;

  bool is_nil() const { return id()=="nil"; }
  bool is_not_nil() const { return id()!="nil"; }

  explicit irept(const irep_idt &_id);

  #ifdef SHARING
  inline irept():data(NULL)
  {
  }

  inline irept(const irept &irep):data(irep.data)
  {
    if(data!=NULL)
    {
      assert(data->ref_count!=0);
      data->ref_count++;
      #ifdef IREP_DEBUG
      std::cout << "COPY " << data << " " << data->ref_count << std::endl;
      #endif
    }
  }

  inline irept &operator=(const irept &irep)
  {
    dt *tmp;
    assert(&irep!=this); // check if we assign to ourselves

    #ifdef IREP_DEBUG
    std::cout << "ASSIGN\n";
    #endif

    tmp = data;
    data=irep.data;
    if(data!=NULL) data->ref_count++;
    remove_ref(tmp);
    return *this;
  }

  ~irept()
  {
    remove_ref(data);
    data=NULL;
  }
  #else
  irept()
  {
  }
  #endif

  inline const irep_idt &id() const
  { return read().data; }

  inline const std::string &id_string() const
  { return read().data.as_string(); }

  inline void id(const irep_idt &_data)
  { write().data=_data; }

  // These methods should be protected; however to make things play nice with
  // the C++ frontend right now, they're made public.
//protected:
  // This class has to be able to fiddle with ireps directly.
  friend class irep_serializationt;

  const irept &find(const irep_namet &name) const;
  irept &add(const irep_namet &name);

  const std::string &get_string(const irep_namet &name) const
  {
    return get(name).as_string();
  }

  const irep_idt &get(const irep_namet &name) const;
  bool get_bool(const irep_namet &name) const;

  inline void set(const irep_namet &name, const irep_idt &value)
  { add(name).id(value); }

  void set(const irep_namet &name, const long value);
  void set(const irep_namet &name, const irept &irep);
//public:
  void remove(const irep_namet &name);
  void move_to_sub(irept &irep);
  void move_to_named_sub(const irep_namet &name, irept &irep);

  inline typet &type() { return (typet &)(add(s_type)); }
  inline const typet &type() const { return (typet &)(find(s_type)); }

  inline bool is_identifier_set(void) const {
    return (get(a_identifier) != "");
  }

  inline const irep_idt &identifier(void) const {
    return get(a_identifier);
  }

  inline const irep_idt &width(void) const {
    return get(a_width);
  }

  inline const irep_idt &statement(void) const {
    return get(a_statement);
  }

  inline const irep_idt &name(void) const {
    return get(a_name);
  }

  inline const irep_idt &component_name(void) const {
    return get(a_comp_name);
  }

  inline const irep_idt &tag(void) const {
    return get(a_tag);
  }

  inline const irep_idt &from(void) const {
    return get(a_from);
  }

  inline const irep_idt &file(void) const {
    return get(a_file);
  }

  inline const irep_idt &line(void) const {
    return get(a_line);
  }

  inline const irep_idt &function(void) const {
    return get(a_function);
  }

  inline const irept &function_irep(void) const {
    return find(a_function);
  }

  inline const irep_idt &column(void) const {
    return get(a_column);
  }

  inline const irep_idt &destination(void) const {
    return get(a_destination);
  }

  inline const irep_idt &access(void) const {
    return get(a_access);
  }

  inline const irep_idt &base_name(void) const {
    return get(a_base_name);
  }

  inline const irep_idt &comment(void) const {
    return get(a_comment);
  }

  inline const irep_idt &event(void) const {
    return get(a_event);
  }

  inline const irept &event_irep(void) const {
    return find(a_event);
  }

  inline const irep_idt &literal(void) const {
    return get(a_literal);
  }

  inline const irep_idt &loopid(void) const {
    return get(a_loopid);
  }

  inline const irep_idt &mode(void) const {
    return get(a_mode);
  }

  inline const irep_idt &module(void) const {
    return get(a_module);
  }

  inline const irep_idt &pretty_name(void) const {
    return get(a_pretty_name);
  }

  inline const irep_idt &property(void) const {
    return get(a_property);
  }

  inline const irep_idt &size(void) const {
    return get(a_size);
  }

  inline const irept &size_irep(void) const {
    return find(a_size);
  }

  inline const irep_idt &integer_bits(void) const {
    return get(a_integer_bits);
  }

  inline const irep_idt &to(void) const {
    return get(a_to);
  }

  inline const irep_idt &failed_symbol(void) const {
    return get(a_failed_symbol);
  }

  inline const irep_idt &type_id(void) const {
    return get(a_type_id);
  }

  inline const irept &arguments(void) const {
    return find(s_arguments);
  }

  inline const irept &components(void) const {
    return find(s_components);
  }

  inline const irept &cmt_type(void) const {
    return find(a_cmt_type);
  }

  inline const irept &return_type(void) const {
    return find(s_return_type);
  }

  inline const irept &body(void) const {
    return find(s_body);
  }

  inline const irept &member_irep(void) const {
    return find(s_member);
  }

  inline const irept &labels_irep(void) const {
    return find(s_labels);
  }

  inline const irept &c_sizeof_type(void) const {
    return find(a_c_sizeof_type);
  }

  inline const irept &bv(void) const {
    return find(s_bv);
  }

  inline const irept &targets(void) const {
    return find(s_targets);
  }

  inline const irept &variables(void) const {
    return find(s_variables);
  }

  inline const irept &object_type(void) const {
    return find(a_object_type);
  }

  inline const irept &initializer(void) const {
    return find(s_initializer);
  }

  inline const irept &cmt_size(void) const {
    return find(a_cmt_size);
  }

  inline const irept &code(void) const {
    return find(a_code);
  }

  inline const irept &guard(void) const {
    return find(a_guard);
  }

  inline const irept &location(void) const {
    return find(a_location);
  }

  inline const irept &declaration_type(void) const {
    return find(s_declaration_type);
  }

  inline const irept &decl_value(void) const {
    return find(s_decl_value);
  }

  inline const irept &end_location(void) const {
    return find(a_end_location);
  }

  inline const irept &symvalue(void) const {
    return find(s_symvalue);
  }

  inline const irept &cmt_location(void) const {
    return find(s_cmt_location);
  }

  inline const irept &decl_ident(void) const {
    return find(s_decl_ident);
  }

  inline bool is_decl_ident_set(void) const {
    return (get(s_decl_ident) != "");
  }

  inline const irept &elements(void) const {
    return find(s_elements);
  }

  inline bool is_dynamic_set(void) const {
    const irep_idt &c = get(a_dynamic);
    return (c != "");
  }

  inline bool dynamic(void) const {
    return get_bool(a_dynamic);
  }

  inline const irep_idt &cmt_base_name(void) const {
    return get(a_cmt_base_name);
  }

  inline const irep_idt &id_class(void) const {
    return get(a_id_class);
  }

  inline const irep_idt &cmt_identifier(void) const {
    return get(a_cmt_identifier);
  }

  inline const irep_idt &cformat(void) const {
    return get(a_cformat);
  }

  inline const irep_idt &cmt_width(void) const {
    return get(a_cmt_width);
  }

  inline const irept &offsetof_type(void) const {
    return find(s_offsetof_type);
  }

  inline bool axiom(void) const {
    return get_bool(a_axiom);
  }

  inline bool cmt_constant(void) const {
    return get_bool(a_cmt_constant);
  }

  inline bool dfault(void) const {
    return get_bool(a_default);
  }

  inline bool ellipsis(void) const {
    return get_bool(a_ellipsis);
  }

  inline bool explict(void) const {
    return get_bool(a_explicit);
  }

  inline bool file_local(void) const {
    return get_bool(a_file_local);
  }

  inline bool hex_or_oct(void) const {
    return get_bool(a_hex_or_oct);
  }

  inline bool hide(void) const {
    return get_bool(a_hide);
  }

  inline bool implicit(void) const {
    return get_bool(a_implicit);
  }

  inline bool incomplete(void) const {
    return get_bool(a_incomplete);
  }

  inline bool initialization(void) const {
    return get_bool(a_initialization);
  }

  inline bool inlined(void) const {
    return get_bool(a_inlined);
  }

  inline bool invalid_object(void) const {
    return get_bool(a_invalid_object);
  }

  inline bool is_parameter(void) const {
    return get_bool(a_is_parameter);
  }

  inline bool is_expression(void) const {
    return get_bool(a_is_expression);
  }

  inline bool is_extern(void) const {
    return get_bool(a_is_extern);
  }

  inline bool is_macro(void) const {
    return get_bool(a_is_macro);
  }

  inline bool is_type(void) const {
    return get_bool(a_is_type);
  }

  inline bool cmt_lvalue(void) const {
    return get_bool(a_cmt_lvalue);
  }

  inline bool lvalue(void) const {
    return get_bool(a_lvalue);
  }

  inline bool reference(void) const {
    return get_bool(a_reference);
  }

  inline bool restricted(void) const {
    return get_bool(a_restricted);
  }

  inline bool static_lifetime(void) const {
    return get_bool(a_static_lifetime);
  }

  inline bool theorem(void) const {
    return get_bool(a_theorem);
  }

  inline bool cmt_unsigned(void) const {
    return get_bool(a_cmt_unsigned);
  }

  inline bool user_provided(void) const {
    return get_bool(a_user_provided);
  }

  inline bool cmt_volatile(void) const {
    return get_bool(a_cmt_volatile);
  }

  inline bool zero_initializer(void) const {
    return get_bool(a_zero_initializer);
  }

  inline void arguments(const irept &val) {
    set(s_arguments, val);
  }

  inline void decl_ident(const irept &val) {
    set(s_decl_ident, val);
  }

  inline void identifier(const irep_idt ident) {
    set(a_identifier, ident);
  }

  inline void statement(const irep_idt statm) {
    set(a_statement, statm);
  }

  inline void comment(const irep_idt cmt) {
    set(a_comment, cmt);
  }

  inline void component_name(const irep_idt name) {
    set(a_comp_name, name);
  }

  inline void hex_or_oct(const irep_idt which) {
    set(a_hex_or_oct, which);
  }

  inline void hex_or_oct(bool which) {
    set(a_hex_or_oct, which);
  }

  inline void cmt_width(const irep_idt width) {
    set(a_cmt_width, width);
  }

  inline void cmt_width(unsigned int width) {
    set(a_cmt_width, width);
  }

  inline void width(const irep_idt width) {
    set(a_width, width);
  }

  inline void width(unsigned int width) {
    set(a_width, width);
  }

  inline void cmt_unsigned(const irep_idt val) {
    set(a_cmt_unsigned, val);
  }

  inline void cmt_unsigned(bool val) {
    set(a_cmt_unsigned, val);
  }

  inline void property(const irep_idt prop) {
    set(a_property, prop);
  }

  inline void cmt_base_name(const irep_idt name) {
    set(a_cmt_base_name, name);
  }

  inline void cmt_lvalue(const irep_idt val) {
    set(a_cmt_lvalue, val);
  }

  inline void cmt_lvalue(bool val) {
    set(a_cmt_lvalue, val);
  }

  inline void name(const irep_idt val) {
    set(a_name, val);
  }

  inline void cformat(const irep_idt val) {
    set(a_cformat, val);
  }

  inline void flavor(const irep_idt val) {
    set(a_flavor, val);
  }

  inline void function(const irep_idt val) {
    set(a_function, val);
  }

  inline void size(const irep_idt val) {
    set(a_size, val);
  }

  inline void size(const irept &val) {
    set(a_size, val);
  }

  inline void user_provided(const irep_idt val) {
    set(a_user_provided, val);
  }

  inline void user_provided(bool val) {
    set(a_user_provided, val);
  }

  inline void destination(const irep_idt val) {
    set(a_destination, val);
  }

  inline void cmt_constant(const irep_idt val) {
    set(a_cmt_constant, val);
  }

  inline void cmt_constant(bool val) {
    set(a_cmt_constant, val);
  }

  inline void cmt_active(const irep_idt val) {
    set(a_cmt_active, val);
  }

  inline void base_name(const irep_idt val) {
    set(a_base_name, val);
  }

  inline void code(const irept &val) {
    set(a_code, val);
  }

  inline void component(const irep_idt val) {
    set(a_component, val);
  }

  inline void component(unsigned int val) {
    set(a_component, val);
  }

  inline void c_sizeof_type(const irept &val) {
    set(a_c_sizeof_type, val);
  }

  inline void dfault(bool val) {
    set(a_default, val);
  }

  inline void dynamic(bool val) {
    set(a_dynamic, val);
  }

  inline void end_location(const irept &val) {
    set(a_end_location, val);
  }

  inline void ellipsis(bool val) {
    set(a_ellipsis, val);
  }

  inline void axiom(bool val) {
    set(a_axiom, val);
  }

  inline void event(const irep_idt &val) {
    set(a_event, val);
  }

  inline void failed_symbol(const irep_idt val) {
    set(a_failed_symbol, val);
  }

  inline void file(const irep_idt val) {
    set(a_file, val);
  }

  inline void file_local(bool val) {
    set(a_file_local, val);
  }

  inline void guard(const irept &val) {
    set(a_guard, val);
  }

  inline void hide(bool val) {
    set(a_hide, val);
  }

  inline void id_class(unsigned int val) {
    set(a_id_class, val);
  }

  inline void cmt_identifier(const irep_idt val) {
    set(a_cmt_identifier, val);
  }

  inline void implicit(bool val) {
    set(a_implicit, val);
  }

  inline void incomplete(bool val) {
    set(a_incomplete, val);
  }

  inline void inlined(bool val) {
    set(a_inlined, val);
  }

  inline void invalid_object(bool val) {
    set(a_invalid_object, val);
  }

  inline void is_parameter(bool val) {
    set(a_is_parameter, val);
  }

  inline void is_expression(bool val) {
    set(a_is_expression, val);
  }

  inline void is_extern(bool val) {
    set(a_is_extern, val);
  }

  inline void is_macro(bool val) {
    set(a_is_macro, val);
  }

  inline void is_type(bool val) {
    set(a_is_type, val);
  }

  inline void label(const irep_idt &val) {
    set(a_label, val);
  }

  inline void lhs(bool val) {
    set(a_lhs, val);
  }

  inline void line(const irep_idt &val) {
    set(a_line, val);
  }

  inline void line(unsigned int val) {
    set(a_line, val);
  }

  inline void location(const irept &val) {
    set(a_location, val);
  }

  inline void lvalue(bool val) {
    set(a_lvalue, val);
  }

  inline void mode(const irep_idt &val) {
    set(a_mode, val);
  }

  inline void module(const irep_idt &val) {
    set(a_module, val);
  }

  inline void object_type(const irept &val) {
    set(a_object_type, val);
  }

  inline void pretty_name(const irep_idt &val) {
    set(a_pretty_name, val);
  }

  inline void restricted(bool val) {
    set(a_restricted, val);
  }

  inline void cmt_size(const irept &val) {
    set(a_cmt_size, val);
  }

  inline void static_lifetime(bool val) {
    set(a_static_lifetime, val);
  }

  inline void tag(const irep_idt &val) {
    set(a_tag, val);
  }

  inline void theorem(bool val) {
    set(a_theorem, val);
  }

  inline void cmt_type(const irept &val) {
    set(a_cmt_type, val);
  }

  inline void type_id(unsigned int val) {
    set(a_type_id, val);
  }

  inline void cmt_volatile(bool val) {
    set(a_cmt_volatile, val);
  }

  inline void zero_initializer(bool val) {
    set(a_zero_initializer, val);
  }

  inline void components(const irept &val) {
    set(s_components, val);
  }

  inline void offsetof_type(const irept &val) {
    set(s_offsetof_type, val);
  }

  inline void declaration_type(const irept &val) {
    set(s_declaration_type, val);
  }

  inline void bv(const irept &val) {
    set(s_bv, val);
  }

  inline void initializer(const irept &val) {
    set(s_initializer, val);
  }

  inline void labels(const irept &val) {
    set(s_labels, val);
  }

  inline void symvalue(const irept &val) {
    set(s_symvalue, val);
  }

  inline void member_irep(const irept &val) {
    set(s_member, val);
  }

  inline void return_type(const irept &val) {
    set(s_return_type, val);
  }

  inline void variables(const irept &val) {
    set(s_variables, val);
  }

  inline void targets(const irept &val) {
    set(s_targets, val);
  }

  inline bool is_address_of() const { return id() == id_address_of; }
  inline bool is_and() const { return id() == id_and; }
  inline bool is_or() const { return id() == id_or; }
  inline bool is_array() const { return id() == id_array; }
  inline bool is_bool() const { return id() == id_bool; }
  inline bool is_code() const { return id() == id_code; }
  inline bool is_constant() const { return id() == id_constant; }
  inline bool is_dereference() const { return id() == id_dereference; }
  inline bool is_empty() const { return id() == id_empty; }
  inline bool is_fixedbv() const { return id() == id_fixedbv; }
  inline bool is_floatbv() const { return id() == id_floatbv; }
  inline bool is_incomplete_array() const { return id() == id_incomplete_array;}
  inline bool is_index() const { return id() == id_index; }
  inline bool is_member() const { return id() == id_member; }
  inline bool is_not() const { return id() == id_not; }
  inline bool is_notequal() const { return id() == id_notequal; }
  inline bool is_pointer() const { return id() == id_pointer; }
  inline bool is_signedbv() const { return id() == id_signedbv; }
  inline bool is_struct() const { return id() == id_struct; }
  inline bool is_symbol() const { return id() == id_symbol; }
  inline bool is_typecast() const { return id() == id_typecast; }
  inline bool is_union() const { return id() == id_union; }
  inline bool is_unsignedbv() const { return id() == id_unsignedbv; }

  friend bool operator==(const irept &i1, const irept &i2);

  friend inline bool operator!=(const irept &i1, const irept &i2)
  { return !(i1==i2); }

  friend std::ostream& operator<< (std::ostream& out, const irept &irep);

  std::string to_string() const;

  void swap(irept &irep)
  {
    std::swap(irep.data, data);
  }

  friend bool operator<(const irept &i1, const irept &i2);
  friend bool ordering(const irept &i1, const irept &i2);

  int compare(const irept &i) const;

  void clear();

  void make_nil() { clear(); id("nil"); }

  subt &get_sub() { return write().sub; } // DANGEROUS
  const subt &get_sub() const { return read().sub; }
  named_subt &get_named_sub() { return write().named_sub; } // DANGEROUS
  const named_subt &get_named_sub() const { return read().named_sub; }
  named_subt &get_comments() { return write().comments; } // DANGEROUS
  const named_subt &get_comments() const { return read().comments; }

  size_t hash() const;
  size_t full_hash() const;

  friend bool full_eq(const irept &a, const irept &b);

  std::string pretty(unsigned indent=0) const;

protected:
  static bool is_comment(const irep_namet &name)
  { return !name.empty() && name[0]=='#'; }

public:
  static const irep_idt s_type, s_arguments, s_components;
  static const irep_idt s_return_type, s_body, s_member, s_labels;
  static const irep_idt s_bv, s_targets, s_variables;
  static const irep_idt s_initializer, s_declaration_type, s_decl_value;
  static const irep_idt s_symvalue, s_cmt_location, s_decl_ident;
  static const irep_idt s_elements, s_offsetof_type;
  static const irep_idt a_width, a_name, a_statement, a_identifier, a_comp_name;
  static const irep_idt a_tag, a_from, a_file, a_line, a_function, a_column;
  static const irep_idt a_access, a_destination, a_base_name, a_comment,a_event;
  static const irep_idt a_literal, a_loopid, a_mode, a_module;
  static const irep_idt a_pretty_name, a_property, a_size, a_integer_bits, a_to;
  static const irep_idt a_failed_symbol, a_dynamic, a_cmt_base_name, a_id_class;
  static const irep_idt a_cmt_identifier, a_cformat, a_cmt_width, a_axiom;
  static const irep_idt a_cmt_constant, a_default;
  static const irep_idt a_ellipsis, a_explicit, a_file_local;
  static const irep_idt a_hex_or_oct, a_hide, a_implicit, a_incomplete;
  static const irep_idt a_initialization, a_inlined, a_invalid_object;
  static const irep_idt a_is_parameter, a_is_expression;
  static const irep_idt a_is_extern, a_is_macro;
  static const irep_idt a_is_type, a_cmt_lvalue;
  static const irep_idt a_lvalue, a_reference, a_static_lifetime, a_theorem;
  static const irep_idt a_cmt_unsigned, a_user_provided, a_cmt_volatile;
  static const irep_idt a_zero_initializer, a_restricted, a_flavor;
  static const irep_idt a_cmt_active, a_code, a_component, a_c_sizeof_type;
  static const irep_idt a_end_location, a_guard, a_label, a_lhs, a_location;
  static const irep_idt a_object_type, a_cmt_size, a_cmt, a_type_id;
  static const irep_idt a_cmt_type;

  static const irep_idt id_address_of, id_and, id_or, id_array, id_bool, id_code;
  static const irep_idt id_constant, id_dereference, id_empty, id_fixedbv;
  static const irep_idt id_floatbv, id_incomplete_array, id_index, id_member;
  static const irep_idt id_not, id_notequal, id_pointer, id_signedbv;
  static const irep_idt id_struct, id_symbol, id_typecast, id_union;
  static const irep_idt id_unsignedbv;

  class dt
  {
  public:
    #ifdef SHARING
    unsigned ref_count;
    #endif

    dstring data;

    named_subt named_sub;
    named_subt comments;
    subt sub;

    void clear()
    {
      data.clear();
      sub.clear();
      named_sub.clear();
      comments.clear();
    }

    void swap(dt &d)
    {
      d.data.swap(data);
      d.sub.swap(sub);
      d.named_sub.swap(named_sub);
      d.comments.swap(comments);
    }

    #ifdef SHARING
    dt():ref_count(1)
    {
    }
    #else
    dt()
    {
    }
    #endif
  };

protected:
  #ifdef SHARING
  dt *data;

  void remove_ref(dt *old_data);

  const dt &read() const;

  inline dt &write()
  {
    detatch();
    return *data;
  }

  void detatch();
  #else
  dt data;

  inline const dt &read() const
  {
    return data;
  }

  inline dt &write()
  {
    return data;
  }
  #endif
};

extern inline const std::string &id2string(const irep_idt &d)
{
  return d.as_string();
}

extern inline const std::string &name2string(const irep_namet &n)
{
  return n.as_string();
}

struct irep_hash hash_map_hasher_superclass(irept)
{
  size_t operator()(const irept &irep) const { return irep.hash(); }
  bool operator()(const irept &i1, const irept &i2) const {
    return i1.hash() < i2.hash();
  }
};

struct irep_full_hash
{
  size_t operator()(const irept &irep) const { return irep.full_hash(); }
};

struct irep_full_eq
{
  bool operator()(const irept &i1, const irept &i2) const
  {
    return full_eq(i1, i2);
  }
};

const irept &get_nil_irep();

#endif
