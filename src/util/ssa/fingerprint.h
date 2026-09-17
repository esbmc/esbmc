#pragma once

#include <cstdint>
#include <string>
#include <util/ssa/algorithms.h>

/**
 * @brief Normalisation applied to SSA symbols before a claim's cone is
 *        digested.
 *
 * ESBMC bakes a source character offset into every local's name, so a literal
 * digest only ever matches a byte-identical re-run. The normalising modes
 * quantify how much of that instability is recoverable.
 */
enum class fingerprint_modet
{
  /// No normalisation. Baseline for the measurement.
  raw,
  /// Strip the `<file>@<character-offset>@` segment ESBMC bakes into every
  /// local's name, re-disambiguating names that collide once it is gone.
  srcloc,
  /// Alpha-rename every symbol: the digest sees structure and types only.
  full,
};

/**
 * @brief Digest the surviving steps of a per-claim sliced equation.
 *
 * Each non-ignored step contributes its type and every expression
 * convert_internal_step hands the solver for it: `guard` and `cond` for an
 * assume, assert, branching or assignment, plus `lhs`/`rhs` for a renumber and
 * the arguments for an output, neither of which is reachable through `cond`.
 *
 * Symbols reachable only through a type (a symbolic array size) are not
 * normalised, which can only cost hits, never manufacture them.
 */
uint64_t ssa_cone_digest(
  const symex_target_equationt::SSA_stepst &steps,
  fingerprint_modet mode);

/// A cone's 128-bit key. Wide enough to stand alone, so a cache need not store
/// and re-compare the text it summarises.
struct ssa_cone_keyt
{
  uint64_t lo;
  uint64_t hi;
};

ssa_cone_keyt ssa_cone_key(
  const symex_target_equationt::SSA_stepst &steps,
  fingerprint_modet mode);

/// #ssa_cone_key rendered as 32 hex digits.
std::string ssa_cone_key_string(
  const symex_target_equationt::SSA_stepst &steps,
  fingerprint_modet mode);

/// FNV-1a over \p s. Stable across processes, unlike irep2's crc().
uint64_t fingerprint_hash(const std::string &s);

/// Number of steps the digest covered.
size_t ssa_cone_size(const symex_target_equationt::SSA_stepst &steps);
