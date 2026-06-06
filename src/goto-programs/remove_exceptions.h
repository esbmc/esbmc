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
/// Scope (whole-program, all-or-nothing — anything outside it leaves the entire
/// program to the imperative symex path):
///   - C++ class and primitive throws; reference, value (with single-inheritance
///     slicing), pointer and void* catches, and catch-all;
///   - nested try regions with innermost-out propagation;
///   - inter-procedural propagation through direct and indirect calls, and
///     rethrow;
///   - uncaught detection at main and __ESBMC_main (incl. static-init throws);
///   - Python try/except/raise via the same machinery.
/// A small residual still forces fallback (std::bad_exception by value, some
/// unusual try-block layouts); the pass warns when it declines (report_fallback).
///
/// Run before goto_partial_inline; on by default, with --no-lower-exceptions
/// selecting the legacy imperative dispatch as a fallback (#5075).
void remove_exceptions(
  goto_functionst &goto_functions,
  contextt &context,
  const namespacet &ns);
