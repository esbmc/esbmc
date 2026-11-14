#pragma once

#include <util/irep/irep_idt.h>

class goto_functionst;
class contextt;

/// Whether no assertion that can fail is reachable from the entry in the final
/// GOTO program, so that nothing is left for symex to report (#10252). Run
/// after every transformation that adds or filters checks. Every assertion
/// that is not provably true blocks the answer, whatever its comment, so a
/// check this pass does not know about cannot be missed. With
/// @p allow_loops false, a loop also blocks it, because symex adds an
/// unwinding assertion per loop that this pass cannot see.
bool no_reachable_property(
  const goto_functionst &goto_functions,
  const contextt &context,
  bool allow_loops);
