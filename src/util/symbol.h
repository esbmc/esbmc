/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_SYMBOL_H
#define CPROVER_SYMBOL_H

#include <iostream>
#include <algorithm>
#include <list>

#include "expr.h"
#include "location.h"

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

  const irep_idt &display_name() const
  {
    return pretty_name==""?name:pretty_name;
  }

  // global use
  bool is_type, is_macro, is_parameter;

  // ANSI-C
  bool lvalue, static_lifetime, file_local, is_extern;

  symbolt()
  {
    clear();
  }

  void clear()
  {
    value.make_nil();
    location.make_nil();
    lvalue=static_lifetime=file_local=is_extern=
    is_type=is_parameter=is_macro=false;
    name=module=base_name=mode=pretty_name="";
  }

  void swap(symbolt &b)
  {
    #define SYM_SWAP1(x) x.swap(b.x)

    SYM_SWAP1(type);
    SYM_SWAP1(value);
    SYM_SWAP1(name);
    SYM_SWAP1(pretty_name);
    SYM_SWAP1(module);
    SYM_SWAP1(base_name);
    SYM_SWAP1(mode);
    SYM_SWAP1(location);

    #define SYM_SWAP2(x) std::swap(x, b.x)

    SYM_SWAP2(is_type);
    SYM_SWAP2(is_macro);
    SYM_SWAP2(is_parameter);
    SYM_SWAP2(lvalue);
    SYM_SWAP2(static_lifetime);
    SYM_SWAP2(file_local);
    SYM_SWAP2(is_extern);
  }

  void show(std::ostream &out = std::cout) const;
  void dump() const;

  void to_irep(irept &dest) const;
  void from_irep(const irept &src);
};

std::ostream &operator<<(std::ostream &out,
                         const symbolt &symbol);

typedef std::list<symbolt*> symbol_listt;

#define forall_symbol_list(it, expr) \
  for(symbol_listt::const_iterator it=(expr).begin(); \
      it!=(expr).end(); it++)

#define Forall_symbol_list(it, expr) \
  for(symbol_listt::iterator it=(expr).begin(); \
      it!=(expr).end(); it++)

typedef std::list<const symbolt *> symbolptr_listt;

#define forall_symbolptr_list(it, list) \
  for(symbolptr_listt::const_iterator it=(list).begin(); \
      it!=(list).end(); it++)

#define Forall_symbolptr_list(it, list) \
  for(symbolptr_listt::iterator it=(list).begin(); \
      it!=(list).end(); it++)

#endif
