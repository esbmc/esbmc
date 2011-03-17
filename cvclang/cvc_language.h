/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_CVC_LANGUAGE_H
#define CPROVER_CVC_LANGUAGE_H

#include "language.h"

class cvc_languaget:public languaget
{
public:
  virtual bool parse(
    std::istream &instream,
    const std::string &path,
    message_handlert &message_handler);
             
  virtual bool typecheck(
    contextt &context,
    const std::string &module,
    message_handlert &message_handler);
  
  virtual void show_parse(std::ostream &out);
  
  virtual ~cvc_languaget() { }
  
  virtual bool from_expr(
    const exprt &expr,
    std::string &code,
    const namespacet &ns);

  virtual bool from_type(
    const typet &type,
    std::string &code,
    const namespacet &ns);

  virtual bool to_expr(
    const std::string &code,
    const std::string &module,
    exprt &expr,
    message_handlert &message_handler,
    const namespacet &ns);

  virtual languaget *new_language()
  { return new cvc_languaget; }
};

languaget *new_cvc_language();
 
#endif
