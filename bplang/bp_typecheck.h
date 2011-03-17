/*******************************************************************\

Module: Boolean Program Typechecking

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_BP_TYPECHECK_H
#define CPROVER_BP_TYPECHECK_H

#include <list>

#include <context.h>
#include <std_code.h>
#include <message.h>

#include "bp_parse_tree.h"

bool bp_typecheck(
  bp_parse_treet &bp_parse_tree,
  contextt &context,
  const std::string &module,
  message_handlert &message_handler);

#include <typecheck.h>

class bp_typecheckt:public typecheckt
{
 public:
  bp_typecheckt(
    bp_parse_treet &_bp_parse_tree,
    contextt &_context,
    const std::string &_module,
    message_handlert &_message_handler):
    typecheckt(_message_handler),
    bp_parse_tree(_bp_parse_tree),
    context(_context),
    module(_module)
  {
    mode="bp";
  }

  virtual ~bp_typecheckt() { }

  void convert(bp_parse_treet::declarationst &declarations);

  virtual void typecheck();

  // overload to use BP syntax
  
  virtual std::string to_string(const typet &type);
  virtual std::string to_string(const exprt &expr);

protected:
  bp_parse_treet &bp_parse_tree;
  contextt &context;
  const std::string &module;

  void typecheck_expr(exprt &expr);
  void typecheck_code(codet &code);
  void typecheck_code_goto(codet &code);
  void typecheck_code_block(codet &code);
  void typecheck_code_constrain(codet &code);
  void typecheck_code_non_deterministic_goto(codet &code);
  void typecheck_code_enforce(codet &code);
  void typecheck_code_abortif(codet &code);
  void typecheck_code_dead(codet &code);
  void typecheck_code_function_call(code_function_callt &code);
  void typecheck_code_assign(codet &code);
  void typecheck_code_ifthenelse(codet &code);
  void typecheck_code_return(codet &code);
  void typecheck_code_decl(codet &code);
  void convert_declaration(exprt &declaration);
  void convert_function(exprt &declaration);
  void convert_function_arguments(symbolt &symbol);
  void convert_variable(exprt &declaration);
  void typecheck_boolean_operands(exprt &exprt);
  
  irep_idt function_name, mode;
  unsigned number_of_returned_variables;
  std::list<irep_idt> function_identifiers;
};

#endif
