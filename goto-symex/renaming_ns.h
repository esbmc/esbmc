/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_SYMEX_RENAMING_NS_H
#define CPROVER_GOTO_SYMEX_RENAMING_NS_H

#include <namespace.h>

class renaming_nst:public namespacet
{
public:
  renaming_nst(
    const namespacet &_ns,
    class goto_symex_statet &_state):
    namespacet(_ns),
    state(_state)
  {
  }
   
  virtual bool lookup(const irep_idt &name, const symbolt *&symbol) const
  {
    expr2tc tmp = expr2tc(new symbol2t(type_pool.get_empty(), name));
    state.get_original_name(tmp);
    return namespacet::lookup(to_symbol2t(tmp).get_symbol_name(), symbol);
  }
  
  virtual bool lookup(const expr2tc &name, const symbolt *&symbol) const
  {
    expr2tc tmp = name;
    state.get_original_name(tmp);
    return namespacet::lookup(to_symbol2t(tmp).get_symbol_name(), symbol);
  }

  const symbolt &lookup(const irep_idt &name) const
  {
    return namespacet::lookup(state.get_original_name(name));
  }
  
protected:
  class goto_symex_statet &state;
};
 
#endif
