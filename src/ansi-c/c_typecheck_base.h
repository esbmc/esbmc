/*******************************************************************\

Module: ANSI-C Language Type Checking

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_C_TYPECHECK_BASE_H
#define CPROVER_C_TYPECHECK_BASE_H

#include <context.h>
#include <typecheck.h>
#include <namespace.h>
#include <std_code.h>
#include <std_types.h>

class c_typecheck_baset:
  public typecheckt,
  public namespacet
{
public:
  c_typecheck_baset(
    contextt &_context,
    const std::string &_module,
    message_handlert &_message_handler):
    typecheckt(_message_handler),
    namespacet(_context),
    context(_context),
    module(_module),
    mode("C")
  {
  }

  c_typecheck_baset(
    contextt &_context1,
    const contextt &_context2,
    const std::string &_module,
    message_handlert &_message_handler):
    typecheckt(_message_handler),
    namespacet(_context1, _context2),
    context(_context1),
    module(_module),
    mode("C")
  {
  }

  virtual ~c_typecheck_baset() { }

  virtual void typecheck()=0;
  virtual void typecheck_expr(exprt &expr);

protected:
  contextt &context;
  const std::string &module;
  std::string mode;

  typedef hash_map_cont<irep_idt, std::string, irep_id_hash> id_replace_mapt;
  id_replace_mapt id_replace_map;

  // apply id_replace_map
  void replace_symbol(irept &symbol);

  // overload to use language specific syntax
  virtual std::string to_string(const exprt &expr);
  virtual std::string to_string(const typet &type);

  //
  // service functions
  //

  // initializers
  struct init_statet
  {
  protected:
    const exprt array;
    unsigned pos;

  public:
    explicit init_statet(const exprt &_array):array(_array), pos(0)
    {
    }

    unsigned remaining() const
    {
      return array.operands().size()-pos;
    }

    bool has_next() const
    {
      return pos<array.operands().size();
    }

    init_statet &operator ++(int x __attribute__((unused)))
    {
      pos++;
      return *this;
    }

    const exprt &operator *() const
    {
      return array.operands()[pos];
    }

    const exprt *operator ->() const
    {
      return &(array.operands()[pos]);
    }
  };

  virtual bool zero_initializer(exprt &value, const typet &type) const;

  virtual void do_initializer(
    exprt &value,
    typet &type,
    bool force_constant);

  virtual exprt do_initializer_rec(
    init_statet &state,
    const typet &type,
    bool force_constant,
    bool go_down);

  virtual exprt do_initializer_rec(
    const exprt &value,
    const typet &type,
    bool force_constant);

  virtual exprt do_initializer_union(
    init_statet &state,
    const union_typet &type,
    bool force_constant);

  virtual exprt do_initializer_struct(
    init_statet &state,
    const struct_typet &type,
    bool force_constant);

  virtual exprt do_initializer_array(
    init_statet &state,
    const array_typet &type,
    bool force_constant);

  virtual exprt do_initializer_incomplete_array(
    init_statet &state,
    const typet &type,
    bool force_constant);

  virtual exprt do_designated_initializer(
    const exprt &value,
    const struct_typet &type,
    bool force_constant);

  virtual exprt do_designated_union_initializer(
    const exprt &value,
    const union_typet &type,
    bool force_constant);

  // typecasts

  virtual void implicit_typecast(exprt &expr, const typet &type);
  virtual void implicit_typecast_arithmetic(exprt &expr);
  virtual void implicit_typecast_arithmetic(exprt &expr1, exprt &expr2);

  virtual void implicit_typecast_bool(exprt &expr)
  {
    implicit_typecast(expr, typet("bool"));
  }

  // code
  virtual void start_typecheck_code();
  virtual void typecheck_code(codet &code);

  virtual void typecheck_assign(codet &expr);
  virtual void typecheck_asm(codet &code);
  virtual void typecheck_block(codet &code);
  virtual void typecheck_break(codet &code);
  virtual void typecheck_continue(codet &code);
  virtual void typecheck_decl(codet &code);
  virtual void typecheck_expression(codet &code);
  virtual void typecheck_for(codet &code);
  virtual void typecheck_goto(codet &code);
  virtual void typecheck_ifthenelse(codet &code);
  virtual void typecheck_label(code_labelt &code);
  virtual void typecheck_switch_case(code_switch_caset &code);
  virtual void typecheck_return(codet &code);
  virtual void typecheck_switch(codet &code);
  virtual void typecheck_while(codet &code);
  virtual void typecheck_start_thread(codet &code);

  bool break_is_allowed;
  bool continue_is_allowed;
  bool case_is_allowed;
  typet switch_op_type;
  typet return_type;

  // expressions
  virtual void typecheck_expr_builtin_va_arg(exprt &expr);
  virtual void typecheck_expr_builtin_offsetof(exprt &expr);
  virtual void typecheck_expr_main(exprt &expr);
  virtual void typecheck_expr_operands(exprt &expr);
  virtual void typecheck_expr_comma(exprt &expr);
  virtual void typecheck_expr_constant(exprt &expr);
  virtual void typecheck_expr_side_effect(side_effect_exprt &expr);
  virtual void typecheck_expr_unary_arithmetic(exprt &expr);
  virtual void typecheck_expr_unary_boolean(exprt &expr);
  virtual void typecheck_expr_binary_arithmetic(exprt &expr);
  virtual void typecheck_expr_pointer_arithmetic(exprt &expr);
  virtual void typecheck_expr_binary_boolean(exprt &expr);
  virtual void typecheck_expr_trinary(exprt &expr);
  virtual void typecheck_expr_address_of(exprt &expr);
  virtual void typecheck_expr_dereference(exprt &expr);
  virtual void typecheck_expr_member(exprt &expr);
  virtual void typecheck_expr_ptrmember(exprt &expr);
  virtual void typecheck_expr_rel(exprt &expr);
  virtual void adjust_float_arith(exprt &);
  virtual void typecheck_expr_index(exprt &expr);
  virtual void typecheck_expr_typecast(exprt &expr);
  virtual void typecheck_expr_symbol(exprt &expr);
  virtual void typecheck_expr_sizeof(exprt &expr);
  virtual void typecheck_expr_function_identifier(exprt &expr);
  virtual void typecheck_side_effect_gcc_conditional_expression(side_effect_exprt &expr);
  virtual void typecheck_side_effect_function_call(side_effect_expr_function_callt &expr);
  virtual void typecheck_side_effect_assignment(exprt &expr);
  virtual void typecheck_side_effect_statement_expression(side_effect_exprt &expr);
  virtual void typecheck_function_call_arguments(side_effect_expr_function_callt &expr);
  virtual void do_special_functions(side_effect_expr_function_callt &expr);

  virtual void make_constant(exprt &expr);
  virtual void make_constant_index(exprt &expr);
  virtual void make_constant_rec(exprt &expr);

  // types
  virtual void typecheck_type(typet &type);
  virtual void typecheck_compound_type(typet &type);
  virtual void typecheck_code_type(code_typet &type);
  virtual void typecheck_symbol_type(typet &type);
  virtual void typecheck_c_bit_field_type(typet &type);
  virtual void typecheck_typeof_type(typet &type);
  virtual void typecheck_array_type(array_typet &type);
  virtual void adjust_function_argument(typet &type) const;

  void make_index_type(exprt &expr);

  // environment
  void add_argc_argv(const symbolt &main_symbol);

  // context management
  bool move_symbol(symbolt &symbol, symbolt *&new_symbol);
  bool move_symbol(symbolt &symbol)
  { symbolt *new_symbol; return move_symbol(symbol, new_symbol); }

  // top level stuff
  void typecheck_symbol(symbolt &symbol);
  void typecheck_new_symbol(symbolt &symbol);
  void typecheck_symbol_redefinition(symbolt &old_symbol, symbolt &new_symbol);
  void typecheck_function_body(symbolt &symbol);

  virtual void do_initializer(symbolt &symbol);
};

#endif
