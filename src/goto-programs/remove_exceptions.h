#pragma once

class goto_functionst;
class contextt;
class namespacet;

/// Lower C++/Python exception dispatch (THROW/CATCH) into ordinary guarded
/// control flow over the global exception state (issue #5075).
///
/// A `throw` is rewritten to arm the $esbmc_exc_{thrown,typeid,value} globals
/// (see exception_globals.h) and branch, in catch-declaration order, to the
/// first matching handler — the match expressed as a guard
/// `typeid in concrete_subtype_ids(C)` (see exception_typeidt) over a symbolic
/// type-id, with `catch (...)` as an unconditional branch.
///
/// Scope (whole-program, all-or-nothing — a program containing anything outside
/// it is unsupported: the imperative symex path that used to handle the residual
/// has been removed, so the unlowered THROW/CATCH reach symex and are rejected):
///   - C++ class and primitive throws; reference, value (with single-inheritance
///     slicing), pointer and void* catches, and catch-all;
///   - nested try regions with innermost-out propagation;
///   - inter-procedural propagation through direct and indirect calls, and
///     rethrow;
///   - uncaught detection at main and __ESBMC_main (incl. static-init throws);
///   - std::bad_cast from a failed dynamic_cast<T&>, noexcept / throw(T...)
///     exception specifications, and std::terminate / std::unexpected routing;
///   - Python try/except/raise via the same machinery.
/// Not yet lowered (makes the program unsupported): destructor unwinding.
///
/// Always runs (before goto_partial_inline); the lowering is the only exception
/// path.
void remove_exceptions(
  goto_functionst &goto_functions,
  contextt &context,
  const namespacet &ns);
