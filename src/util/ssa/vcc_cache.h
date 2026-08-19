#pragma once

#include <atomic>
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
class vcc_cachet
{
public:
  /// \param dir cache directory, created if absent
  /// \param options every option in effect, folded into each key
  vcc_cachet(const std::string &dir, const optionst &options);

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
  std::string entry_path(const std::string &cone_key) const;

  std::string dir;
  /// Digest of the ESBMC build, the option set, and the data model.
  std::string context;
  // Solving runs on several threads under --parallel-solving.
  mutable std::atomic<size_t> hit_count{0};
  mutable std::atomic<size_t> miss_count{0};
};

/// Digest of everything outside the cone that the verdict depends on. Every
/// option is folded in unabridged: a curated list of "options that matter"
/// is the classic way a cache like this goes unsound.
std::string vcc_cache_context(const optionst &options);
