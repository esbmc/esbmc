/*******************************************************************\

Module: ANSI-C Conversion

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_C_TYPECHECK_H
#define CPROVER_C_TYPECHECK_H

#include "c_typecheck_base.h"

bool c_typecheck(
  std::list<symbolt> &symbols,
  contextt &context,
  message_handlert &message_handler,
  const std::string &module);

bool c_typecheck(
  exprt &expr,
  const contextt &context,
  message_handlert &message_handler,
  const std::string &module);

class c_typecheckt:public c_typecheck_baset
{
public:
  c_typecheckt(
    std::list<symbolt> &_symbols,
    contextt &_context,
    const std::string &_module,
    message_handlert &_message_handler):
    c_typecheck_baset(_context, _module, _message_handler),
    symbols(_symbols)
  {
  }
      
  c_typecheckt(
    std::list<symbolt> &_symbols,
    contextt &_context1,
    const contextt &_context2,
    const std::string &_module,
    message_handlert &_message_handler):
    c_typecheck_baset(_context1, _context2, _module, _message_handler),
    symbols(_symbols)
  {
  }
      
  virtual void typecheck();

  virtual void typecheck_expr(exprt &expr)
  {
    c_typecheck_baset::typecheck_expr(expr);
  }
 
protected:
  std::list<symbolt> &symbols;
};

#endif
