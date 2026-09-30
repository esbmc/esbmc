#pragma once

#include <goto-programs/goto_functions.h>
#include <transition-system/transition_system.h>
#include <util/config/options.h>
#include <util/symtab/context.h>

#include <string>

/** Recognise a single unbounded loop in main (after the k-induction
 * transformation) and extract it into `ts`, running symbolic execution over
 * one iteration. Returns whether the operation succeded.
 *
 * For
 *
 *     unsigned char x = 0;
 *     for (;;) { __ESBMC_assert(x < 4, "range"); x = (x + 1) & 3; }
 *
 * symex of one iteration gives (SSA names shortened):
 *
 *     x@1 == 0               prefix
 *     x@2 == nondet          the havoc: state at the loop head
 *     guard => x@2 < 4       the claim
 *     x@3 == (x@2 + 1) & 3   the body's assignment
 *
 * which is carved into
 *
 *     states[0]   name "x", init 0, pre x@2, post x@3
 *     body_defs   x@3 == (x@2 + 1) & 3
 *     bad[0]      x@2 >= 4
 *     back_guard  true
 */
bool extract_transition_system(
  const goto_functionst &goto_functions,
  contextt &context,
  const optionst &options,
  transition_systemt &ts);
