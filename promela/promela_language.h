/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_PROMELA_LANGUAGE_H
#define CPROVER_PROMELA_LANGUAGE_H

#include <language.h>

#include "promela_parse_tree.h"

class promela_languaget:public languaget
{
 public:
  virtual bool preprocess(std::istream &instream,
                          const std::string &path,
                          std::ostream &outstream,
                          std::ostream &err);

  virtual bool parse(std::istream &instream,
                     const std::string &path,
                     std::ostream &err);
             
  virtual bool typecheck(contextt &context,
                         const std::string &module,
                         std::ostream &err);

  virtual bool final(contextt &context, std::ostream &err);

  virtual void make_sequent(const contextt &context,
                            const symbolt &symbol,
                            h_sequentt &sequent);
  
  virtual void show_parse(std::ostream &out);
  
  virtual ~promela_languaget();
  promela_languaget() { /* c_parsetree=NULL;*/ }
  
  virtual bool from_expr(const exprt &expr, std::string &code,
                         const namespacet &ns);

  virtual bool from_type(const typet &type, std::string &code,
                         const namespacet &ns);

  virtual bool to_expr(const std::string &code,
                       const std::string &module,
                       exprt &expr,
                       std::ostream &err,
                       const namespacet &ns);
                       
  virtual languaget *new_language()
  { return new promela_languaget; }
   
  virtual std::string id() { return "promela"; }
  virtual std::string description() { return "PROMELA"; }

  virtual void modules_provided(std::set<std::string> &modules);
  
 protected:
  promela_parse_treet promela_parse_tree;
  std::string parse_path;
  bool get_expr(class TransUnit &unit, std::ostream &err,
                const namespacet &ns, exprt &expr);
};
 
languaget *new_promela_language();
 
#endif
