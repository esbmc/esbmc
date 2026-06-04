#pragma once

class goto_functionst;
class contextt;
class namespacet;

/// Lower C++/Python exception dispatch (THROW/CATCH) into ordinary guarded
/// control flow over the global exception state (issue #5075).
///
/// PoC scope: intra-procedural, single-level try regions, reference catches,
/// no destructor unwinding. A `throw` is rewritten to arm the
/// __ESBMC_exc_{thrown,typeid,value} globals (see exception_globals.h) and
/// branch, in catch-declaration order, to the first handler whose static type
/// is a supertype of the thrown dynamic type — the match expressed as a guard
/// `typeid in concrete_subtype_ids(C)` (see exception_typeidt) over a symbolic
/// type-id, with `catch (...)` as an unconditional branch. Regions outside the
/// PoC subset (nested, value-catch, throws crossing calls) are left untouched
/// for the existing symex path to handle.
///
/// Run before goto_partial_inline, gated on --lower-exceptions (default off)
/// until the lowered path reaches parity with the imperative dispatch.
void remove_exceptions(
  goto_functionst &goto_functions,
  contextt &context,
  const namespacet &ns);
