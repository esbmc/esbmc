/*******************************************************************\

Module: Slicer for symex traces

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_SYMEX_SLICE_H
#define CPROVER_GOTO_SYMEX_SLICE_H

#include <hash_cont.h>

#include "renaming.h"
#include "symex_target_equation.h"

void slice(symex_target_equationt &equation);
void simple_slice(symex_target_equationt &equation);

class symex_slicet
{
public:
  symex_slicet();
  void slice(symex_target_equationt &equation);
  void slice_for_symbols(symex_target_equationt &equation, const expr2tc &expr);

protected:
  typedef hash_set_cont<std::string, string_hash> symbol_sett;

  symbol_sett depends;
  bool single_slice;

  void get_symbols(const expr2tc &expr);

  void slice(symex_target_equationt::SSA_stept &SSA_step);
  void slice_assignment(symex_target_equationt::SSA_stept &SSA_step);
  void slice_renumber(symex_target_equationt::SSA_stept &SSA_step);
};

#endif
