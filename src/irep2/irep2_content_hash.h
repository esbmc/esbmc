#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <irep2/irep2.h>

/// Maps a name to the form that should be hashed, letting a caller normalise
/// SSA names without rewriting the expression.
using irep2_name_mappert = std::function<std::string(const irep_idt &)>;

/**
 * @brief Structural hash of \p e that depends only on its content.
 *
 * Every irep_idt contributes its characters (through \p rename), so the result
 * is stable across processes -- unlike crc(), whose irep_idt hash is the
 * string-pool index. Nothing is copied or rendered to text: the walk reads the
 * expression in place and feeds the two rolling lanes, which chain across
 * calls so a sequence of expressions accumulates into one 128-bit value.
 */
void irep2_content_hash(
  const expr2tc &e,
  const irep2_name_mappert &rename,
  uint64_t &lo,
  uint64_t &hi);
