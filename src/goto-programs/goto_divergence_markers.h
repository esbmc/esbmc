#ifndef GOTO_PROGRAMS_GOTO_DIVERGENCE_MARKERS_H_
#define GOTO_PROGRAMS_GOTO_DIVERGENCE_MARKERS_H_

#include <goto-programs/goto_functions.h>
#include <memory>

/// Reduce non-termination of loops such as `while (nondet()) { ... }` to
/// reachability.
///
/// A loop head is divergent when, from every state there, choosing the values
/// of fresh nondet branch guards (`t = nondet(); if (t) ...`) keeps the
/// program running forever, whatever the other branches do: no path the
/// chooser cannot avoid leads to an ASSUME, to an abort-like call or out of
/// the function. Reaching a divergent loop head is then a non-terminating
/// execution, so a base case that reaches it refutes termination.
///
/// Returns a copy of @p goto_functions with an ASSERT(false) before every
/// divergent user loop head, or nullptr when there is none or the program
/// creates threads. In the copy the markers are the only properties: a user
/// assertion becomes an ASSUME of its condition and an abort-like call an
/// ASSUME(false), so a path ends where the real program would. @p
/// goto_functions is left unchanged.
std::unique_ptr<goto_functionst>
make_divergence_program(const goto_functionst &goto_functions);

/// The comment carried by the markers make_divergence_program inserts.
extern const char *const divergence_marker_comment;

#endif /* GOTO_PROGRAMS_GOTO_DIVERGENCE_MARKERS_H_ */
