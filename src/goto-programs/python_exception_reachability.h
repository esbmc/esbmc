#pragma once

#include <util/irep/irep_idt.h>

class goto_functionst;
class contextt;

/// Whether no assertion that can fail is reachable from the entry in the final
/// GOTO program, so that nothing is left for symex to report (#10252). Run
/// after every transformation that adds or filters checks. Every assertion
/// that python_claim_dischargert does not prove blocks the answer, whatever
/// its comment, so a check this pass does not know about cannot be missed. With
/// @p allow_loops false, a loop or call cycle also blocks it, for a run in
/// which symex reports a failed unwinding assertion that this pass cannot see.
/// Otherwise it does not: the pass reads the program, not a bounded unrolling
/// of it, so no execution of any length reaches an assertion that can fail.
/// Any instruction where symex may add a claim of its own blocks the answer
/// too: a dereference, a call through a pointer or to a function with no body
/// when @p pointer_checks is set, and free, delete, va_arg, realloc or an
/// oversized allocation always.
bool no_reachable_property(
  const goto_functionst &goto_functions,
  const contextt &context,
  bool allow_loops,
  bool pointer_checks);
