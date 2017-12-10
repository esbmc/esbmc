/*******************************************************************\

Module: Slicer for symex traces

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_SYMEX_SLICE_H
#define CPROVER_GOTO_SYMEX_SLICE_H

#include <goto-symex/renaming.h>
#include <goto-symex/symex_target_equation.h>
#include <util/hash_cont.h>

BigInt slice(boost::shared_ptr<symex_target_equationt> &eq, bool slice_assume);
BigInt simple_slice(boost::shared_ptr<symex_target_equationt> &eq);

class symex_slicet
{
public:
  symex_slicet(bool assume);
  void slice(boost::shared_ptr<symex_target_equationt> &eq);

  typedef hash_set_cont<std::string, string_hash> symbol_sett;
  symbol_sett depends;
  BigInt ignored;

protected:
  bool slice_assumes;
  std::function<bool(const symbol2t &)> add_to_deps;

  bool
  get_symbols(const expr2tc &expr, std::function<bool(const symbol2t &)> fn);

  void slice(symex_target_equationt::SSA_stept &SSA_step);
  void slice_assume(symex_target_equationt::SSA_stept &SSA_step);
  void slice_assignment(symex_target_equationt::SSA_stept &SSA_step);
  void slice_renumber(symex_target_equationt::SSA_stept &SSA_step);
};

#endif
