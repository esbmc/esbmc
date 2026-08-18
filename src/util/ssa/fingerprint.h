#pragma once

#include <cstdint>
#include <string>
#include <util/ssa/algorithms.h>

/**
 * @brief Normalisation applied to SSA symbols before a claim's cone is
 *        digested.
 *
 * SSA names carry path-dependent counters (see renaming.h): an edit anywhere
 * upstream renumbers every downstream version, so a literal digest only ever
 * matches a byte-identical re-run. The two normalising modes quantify how much
 * of that instability is recoverable.
 */
enum class fingerprint_modet
{
  /// No normalisation. Baseline for the measurement.
  raw,
  /// Keep base names; canonicalise the L1/L2/thread/node counters per name.
  counters,
  /// As `counters`, but also strip the `<file>@<character-offset>@` segment
  /// ESBMC bakes into every local's name, re-disambiguating names that
  /// collide once it is gone.
  srcloc,
  /// Alpha-rename every symbol: the digest sees structure and types only.
  full,
};

/**
 * @brief Digest the surviving steps of a per-claim sliced equation.
 *
 * Only the (type, guard, cond) triple of each non-ignored step is fed in;
 * for an assignment `cond` is the `lhs == rhs` equality, so that triple is a
 * complete account of what the step contributes to the formula.
 *
 * Symbols reachable only through a type (a symbolic array size) are not
 * normalised, which can only cost hits, never manufacture them.
 */
uint64_t ssa_cone_digest(
  const symex_target_equationt::SSA_stepst &steps,
  fingerprint_modet mode);

/**
 * @brief The canonical text a cone's digest is taken over.
 *
 * A persistent cache stores this alongside the digest and compares it on a
 * hit, so a digest collision cannot turn into a wrong verdict.
 */
std::string ssa_cone_text(
  const symex_target_equationt::SSA_stepst &steps,
  fingerprint_modet mode);

/// FNV-1a over \p s. Stable across processes, unlike irep2's crc().
uint64_t fingerprint_hash(const std::string &s);

/// Number of steps the digest covered.
size_t ssa_cone_size(const symex_target_equationt::SSA_stepst &steps);
