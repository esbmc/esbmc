/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_CPP_DECLARATOR_H
#define CPROVER_CPP_DECLARATOR_H

#include <cpp/cpp_name.h>
#include <util/expr.h>

class cpp_declaratort : public exprt
{
public:
  cpp_declaratort() : exprt("cpp-declarator")
  {
    value().make_nil();
    name().make_nil();
    location().make_nil();
  }

  cpp_declaratort(const typet &type) : exprt("cpp-declarator", type)
  {
    value().make_nil();
    name().make_nil();
    location().make_nil();
  }

  cpp_namet &name()
  {
    return static_cast<cpp_namet &>(add("name"));
  }
  const cpp_namet &name() const
  {
    return static_cast<const cpp_namet &>(find("name"));
  }

  exprt &value()
  {
    return static_cast<exprt &>(add("value"));
  }
  const exprt &value() const
  {
    return static_cast<const exprt &>(find("value"));
  }

  // initializers for function arguments
  exprt &init_args()
  {
    return static_cast<exprt &>(add("init_args"));
  }
  const exprt &init_args() const
  {
    return static_cast<const exprt &>(find("init_args"));
  }

  irept &method_qualifier()
  {
    return add("method_qualifier");
  }
  const irept &method_qualifier() const
  {
    return find("method_qualifier");
  }

  irept &member_initializers()
  {
    return add("member_initializers");
  }
  const irept &member_initializers() const
  {
    return find("member_initializers");
  }

  exprt &throw_decl()
  {
    return static_cast<exprt &>(add("throw_decl"));
  }
  const exprt &throw_decl() const
  {
    return static_cast<const exprt &>(find("throw_decl"));
  }

  void output(std::ostream &out) const;

  typet merge_type(const typet &declaration_type) const;
};

#define forall_cpp_declarators(it, expr)                                       \
  for (cpp_declarationt::declaratorst::const_iterator it =                     \
         (expr).declarators().begin();                                         \
       it != (expr).declarators().end();                                       \
       it++)

#define Forall_cpp_declarators(it, expr)                                       \
  if ((expr).has_operands())                                                   \
    for (cpp_declarationt::declaratorst::iterator it =                         \
           (expr).declarators().begin();                                       \
         it != (expr).declarators().end();                                     \
         it++)

#endif
