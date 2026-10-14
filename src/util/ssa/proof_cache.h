#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <string>
#include <util/config/options.h>

/**
 * @brief Content-addressed store of claims already proved unsatisfiable.
 *
 * A claim's key is the 128-bit digest of its sliced SSA cone (see
 * fingerprint.h) combined with a fingerprint of everything else the verdict was
 * contingent on: the ESBMC build, every option in effect, and the data model.
 * An entry is the file named by that key; its presence is the proof.
 *
 * Only UNSAT is stored. A cached SAT could not reproduce its counterexample,
 * and a report must never show a trace that was not just produced.
 */
class proof_cachet
{
public:
  /// \param dir cache directory, created if absent
  /// \param options every option in effect, folded into each key
  /// \param build_identity the ESBMC that is running, from
  ///        #proof_cache_build_identity
  proof_cachet(
    const std::string &dir,
    const optionst &options,
    const std::string &build_identity);

  /// True iff a proof is stored for \p cone_key.
  bool proved(const std::string &cone_key) const;

  /// Record \p cone_key as proved. Idempotent.
  void record(const std::string &cone_key) const;

  size_t hits() const
  {
    return hit_count;
  }
  size_t misses() const
  {
    return miss_count;
  }

private:
  std::filesystem::path entry_path(const std::string &cone_key) const;

  std::filesystem::path dir;
  /// Digest of the ESBMC build, the option set, and the data model. Held
  /// hashed: it is the same for every claim, and each key needs it.
  uint64_t context_hash;
  // Solving runs on several threads under --parallel-solving.
  mutable std::atomic<size_t> hit_count{0};
  mutable std::atomic<size_t> miss_count{0};
};

/// Digest of everything outside the cone that the verdict depends on. Every
/// option is folded in unabridged bar a short allowlist of ones that reach
/// nothing but the report: a curated list of "options that matter" is the
/// classic way a cache like this goes unsound.
std::string
proof_cache_context(const optionst &options, const std::string &build_identity);

/// Why a proof cache would not be consulted under \p options, or empty when
/// it would be. Covers what the option set alone decides; a run adds the
/// thread-interleaving exclusion on top.
std::string proof_cache_inactive_reason(const optionst &options);

/// Say, once per run, that a named cache directory will not be consulted. A
/// run that reuses nothing is otherwise indistinguishable from one whose cache
/// is working.
void report_proof_cache_inactive(const std::string &why);

/// True when the solver has just contradicted a proof the cache stood in for.
/// Only a counterexample does: a vacuous discharge, a solver error and an
/// SMT-LIB-only emission all mean the claim was not re-checked, not that the
/// stored proof was wrong.
bool proof_cache_contradicted(bool was_hit, bool counterexample_found);

/// The ESBMC that is running, as far as it can be established: \p build_id
/// alone when that names one build, otherwise it plus a digest of the
/// executable. Empty when neither is available -- a key that cannot name the
/// verifier must not be used to skip it.
std::string proof_cache_build_identity(const std::string &build_id);
