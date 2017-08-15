/*******************************************************************\

Module: Slicer for symex traces

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_SYMEX_SLICE_H
#define CPROVER_GOTO_SYMEX_SLICE_H

#include <goto-symex/renaming.h>
#include <goto-symex/symex_target_equation.h>
#include <util/hash_cont.h>

u_int64_t slice(boost::shared_ptr<symex_target_equationt> &eq);
u_int64_t simple_slice(boost::shared_ptr<symex_target_equationt> &eq);

class symex_slicet
{
public:
  symex_slicet();
  void slice(boost::shared_ptr<symex_target_equationt> &eq);
  void slice_for_symbols(
    boost::shared_ptr<symex_target_equationt> &eq,
    const expr2tc &expr);

  u_int64_t ignored;

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
