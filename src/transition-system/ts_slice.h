#pragma once

#include <transition-system/transition_system.h>

/** Drop the state variables, definitions and inputs that no property,
 * assumption, invariant or back-edge guard depends on.
 *
 * A backward slice over one iteration is not enough. Nothing in a single step
 * reads a state's value at the back edge, so an ordinary slicer deletes it --
 * and that value *is* the transition relation, read by the next step. Reaching
 * a state through its head symbol therefore also pulls in its `post` and
 * `init`.
 */
void slice_transition_system(transition_systemt &ts);
