/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_IREP_LANGUAGE_H
#define CPROVER_IREP_LANGUAGE_H

#include "language.h"

class irep_languaget:public languaget
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
  
  virtual ~irep_languaget() { }
  
  virtual bool to_expr(
    const std::string &code,
    const std::string &module,
    exprt &expr,
    message_handlert &message_handler,
    const namespacet &ns);
                       
  virtual languaget *new_language()
  { return new irep_languaget; }
  
protected:
  void *root;
};
 
languaget *new_intrep_language();

#endif
