#pragma once

#include <goto-programs/goto_functions.h>
#include <util/irep/irep_idt.h>

#include <map>
#include <memory>
#include <set>

class contextt;

/// Proves assertions of the final GOTO program unfailable without symex, for
/// the early answer of #10252. Two arguments, each failing closed:
///
/// - The uncaught-exception checks test only `__ESBMC_exc_thrown` and
///   `__ESBMC_exc_typeid`. When every reachable write to them is a constant of
///   a recognised shape, the values they can hold are known, and a check is
///   proved when it holds for all of them.
/// - Other assertions are proved from what holds on every path to them in
///   their own function: the sign of an integer local, or whether a pointer
///   local is null, as branch conditions, assumptions, earlier assertions and
///   constant assignments fix it. Only locals whose address is never taken
///   count. A pointer is also known non-null when it was assigned an address,
///   the result of a fixed-size `__ESBMC_alloca`, the result of a function
///   whose every return is non-null, or a parameter that every call site
///   passes non-null, for a function whose address is never taken.
///
/// Using an earlier assertion as a fact is sound because the early answer
/// needs every reachable assertion proved: on any execution, the first
/// assertion to fail would have to fail with all earlier ones holding, and the
/// facts at it then hold too.
class python_claim_dischargert
{
public:
  python_claim_dischargert(
    const goto_functionst &goto_functions,
    const contextt &context,
    const std::set<irep_idt> &reachable);
  ~python_claim_dischargert();

  bool proves(goto_programt::const_targett assertion) const;

private:
  struct implt;
  std::unique_ptr<implt> impl;
};
