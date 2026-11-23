#pragma once

#include <transition-system/transition_system.h>

#include <iosfwd>

/**
 * @brief Write the transition system as a BTOR2 safety problem with a single
 * @c bad property, for external hardware model checkers such as rIC3.
 *
 * @param ts The extracted transition system.
 * @param out Stream that receives the BTOR2 text.
 * @throws std::runtime_error If @p ts uses anything outside scalar bit-vector
 *         and boolean terms (floats, pointers, arrays).
 */
void write_btor2(const transition_systemt &ts, std::ostream &out);
