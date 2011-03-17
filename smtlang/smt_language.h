/*******************************************************************\

Module: SMT-LIB Frontend

Author: CM Wintersteiger

\*******************************************************************/

#ifndef CPROVER_SMT_LANGUAGE_H
#define CPROVER_SMT_LANGUAGE_H

#include <language.h>
#include <message_stream.h>

#include "smt_parse_tree.h"

class smt_languaget:public languaget
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
    
  virtual bool final(
    contextt &context,
    message_handlert &message_handler);
  
  virtual void show_parse(std::ostream &out);
  
  virtual ~smt_languaget() { }
  
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
  { return new smt_languaget; }
  
protected:
  smt_parse_treet parse_tree;
  
private:
  void check_double_sorts(const exprt&, const symbolst&, message_streamt&);
  void check_double_functions(const exprt&, const symbolst&, message_streamt&);
  void check_double_predicates(const exprt&, const symbolst&, message_streamt&);
  void check_double_function_signatures(
            const exprt&, 
            const typet&, 
            const typet&,
            message_streamt&);
};

languaget *new_smt_language();
 
#endif
