#pragma once

#include <goto-programs/goto_functions.h>
#include <util/irep/irep_idt.h>

#include <memory>
#include <set>

/// Proves assertions of the final GOTO program unfailable without symex, for
/// the early answer of #10252, failing closed.
///
/// The uncaught-exception checks test only `__ESBMC_exc_thrown` and
/// `__ESBMC_exc_typeid`. When every reachable write to them is a constant of a
/// recognised shape, the values they can hold are known, and a check is proved
/// when it holds for all of them.
class python_claim_dischargert
{
public:
  python_claim_dischargert(
    const goto_functionst &goto_functions,
    const std::set<irep_idt> &reachable);
  ~python_claim_dischargert();

  bool proves(goto_programt::const_targett assertion) const;

private:
  struct implt;
  std::unique_ptr<implt> impl;
};
