/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_SYMBOL_H
#define CPROVER_SYMBOL_H

#include <algorithm>
#include <iostream>
#include <list>
#include <util/expr.h>
#include <util/location.h>

class symbolt
{
public:
  typet type;
  exprt value;
  locationt location;
  irep_idt name;
  irep_idt module;
  irep_idt base_name;
  irep_idt mode;
  irep_idt pretty_name;

  const irep_idt &display_name() const;

  // global use
  bool is_type, is_macro, is_parameter, is_used;

  // ANSI-C
  bool lvalue, static_lifetime, file_local, is_extern;

  symbolt();

  void clear();

  void swap(symbolt &b);

  void show(std::ostream &out = std::cout) const;
  void dump() const;

  void to_irep(irept &dest) const;
  void from_irep(const irept &src);
};

std::ostream &operator<<(std::ostream &out, const symbolt &symbol);

typedef std::list<symbolt *> symbol_listt;

#define forall_symbol_list(it, expr)                                           \
  for(symbol_listt::const_iterator it = (expr).begin(); it != (expr).end();    \
      it++)

#define Forall_symbol_list(it, expr)                                           \
  for(symbol_listt::iterator it = (expr).begin(); it != (expr).end(); it++)

typedef std::list<const symbolt *> symbolptr_listt;

#define forall_symbolptr_list(it, list)                                        \
  for(symbolptr_listt::const_iterator it = (list).begin(); it != (list).end(); \
      it++)

#define Forall_symbolptr_list(it, list)                                        \
  for(symbolptr_listt::iterator it = (list).begin(); it != (list).end(); it++)

#endif
