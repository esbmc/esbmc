/*******************************************************************\

Module: SMT-LIB Frontend, Linking

Author: CM Wintersteiger

\*******************************************************************/

#ifndef SMT_LINK_
#define SMT_LINK_

#include <message.h>
#include <context.h>
#include <typecheck.h>
#include <namespace.h>

bool smt_link(
  contextt &context,
  contextt &new_context,
  message_handlert &message_handler,
  const std::string &module);
  
class smt_linkt: public typecheckt
{
public:
  smt_linkt(
    contextt &_context,
    contextt &_new_context,
    const std::string &_module,
    message_handlert &_message_handler):
    typecheckt(_message_handler),
    context(_context),
    new_context(_new_context),
    module(_module),
    ns(_context, _new_context)
  { }
   
  virtual void typecheck();
 
protected:
  void duplicate(symbolt &in_context, symbolt &new_symbol);
  void duplicate_type(symbolt &in_context, symbolt &new_symbol);
  void duplicate_symbol(symbolt &in_context, symbolt &new_symbol);
  void move(symbolt &new_symbol);

  // overload to use language specific syntax
  //virtual std::string to_string(const exprt &expr);
  //virtual std::string to_string(const typet &type);

  contextt &context;
  contextt &new_context;
  std::string module;
  namespacet ns;
};

#endif /*SMT_LINK_*/
