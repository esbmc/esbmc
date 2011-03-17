/*******************************************************************\

Module: SpecC Language Type Checking

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef PROMELA_TYPECHECK_H
#define PROMELA_TYPECHECK_H

#include <set>

#include <context.h>
#include <namespace.h>
#include <typecheck.h>

#include "promela_parser.h"

bool promela_typecheck(promela_parse_treet &promela_parse_tree,
                       contextt &context,
                       const std::string &module,
                       std::ostream &err);

bool promela_typecheck(exprt &expr,
                       std::ostream &err,
                       const namespacet &ns);

class promela_typecheckt:public typecheckt
{
 public:
  promela_typecheckt(promela_parse_treet &_promela_parse_tree,
                     contextt &_context,
                     const std::string &_module,
                     std::ostream &_err):
                     typecheckt(_err),
                     promela_parse_tree(_promela_parse_tree)
  {
  }

  promela_typecheckt(promela_parse_treet &_promela_parse_tree,
                     contextt &_context1,
                     const contextt &_context2,
                     const std::string &_module,
                     std::ostream &_err):
                     typecheckt(_err),
                     promela_parse_tree(_promela_parse_tree)
  {
  }

  virtual ~promela_typecheckt() { }

  virtual void typecheck();

  // overload to use Promela syntax
  
  virtual std::string to_string(const typet &type);
  virtual std::string to_string(const exprt &expr);

  // expressions
  void typecheck_expr(exprt &expr);
  
 protected:
  promela_parse_treet &promela_parse_tree;

  #if 0
  const symbolt &convert_declaration(exprt &declaration);
  void convert_parameters(const exprt &declaration, symbolt &symbol);
  
  // code
  void typecheck_code(exprt &code);
  std::set<std::string> local_identifiers;
  
  // expressions
  void typecheck_expr_constant(exprt &expr);
  void typecheck_expr_extractbits(exprt &expr);
  void typecheck_expr_concatentation(exprt &expr);
  void typecheck_expr_sideeffect(exprt &expr);
  void typecheck_expr_symbol(exprt &expr);
  void typecheck_expr_unary_arithmetic(exprt &expr);
  void typecheck_expr_unary_boolean(exprt &expr);
  void typecheck_expr_binary_arithmetic(exprt &expr);
  bool typecheck_expr_pointer_arithmetic(exprt &expr);
  void typecheck_expr_binary_boolean(exprt &expr);
  void typecheck_expr_trinary(exprt &expr);
  void typecheck_expr_address_of(exprt &expr);
  void typecheck_expr_dereference(exprt &expr);
  void typecheck_expr_member(exprt &expr);
  void typecheck_expr_ptrmember(exprt &expr);
  void typecheck_expr_rel(exprt &expr);
  void typecheck_expr_index(exprt &expr);
  void typecheck_expr_typecast(exprt &expr);
  void typecheck_expr_sizeof(exprt &expr);
  void typecheck_deref_ptr(const exprt &ptr,
                           exprt &dest) const;
  void typecheck_sideeffect_functioncall(exprt &expr);
  void typecheck_sideeffect_assignment(exprt &expr);
  
  // types
  void typecheck_type(typet &type);
  
  virtual bool zero_initializer(exprt &value, const typet &type) const;
  
  virtual bool zero_initializer(exprt &value) const
  {
    return c_typecheck_baset::zero_initializer(value);
  }
  #endif
};

#endif
