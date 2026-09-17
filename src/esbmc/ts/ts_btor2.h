#pragma once

#include <esbmc/ts/transition_system.h>

#include <iosfwd>

/// Write the transition system as a BTOR2 safety problem with a single bad
/// property, for external hardware model checkers. Throws std::runtime_error
/// on anything outside scalar bit-vector and boolean terms.
void write_btor2(const transition_systemt &ts, std::ostream &out);
