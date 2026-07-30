#ifndef CPROVER_GOTO_PROGRAMS_PROPERTY_VERDICT_H
#define CPROVER_GOTO_PROGRAMS_PROPERTY_VERDICT_H

#include <atomic>
#include <cstddef>
#include <map>
#include <mutex>
#include <string>

/// Outcome of checking one property. Ordered by dominance: when a property is
/// checked more than once in a run, the numerically greater verdict survives.
enum class property_verdictt
{
  Passed = 0,
  Unknown = 1,
  Failed = 2
};

/// A verdict together with a note explaining how it was reached -- that a
/// discharge came from interval analysis rather than the solver, or that it was
/// vacuous. Empty when the verdict needs no qualification.
struct property_resultt
{
  property_verdictt verdict;
  std::string note;
};

/// One verdict per property for a whole verification run.
///
/// `--multi-property` re-checks every property in every thread interleaving,
/// and a property can be discharged under one schedule while being violated
/// under another. Reporting each check as it happens then prints contradictory
/// verdicts for a single source assertion (esbmc/esbmc discussion #6391).
/// Recording verdicts here instead lets the run report each property once,
/// with the dominant verdict.
class property_verdict_tablet
{
public:
  /// Records \p verdict for \p property, annotated with \p note. Keeps the
  /// dominant verdict when the property has already been checked. Safe to call
  /// from parallel solver threads.
  void record(
    const std::string &property,
    property_verdictt verdict,
    const std::string &note = "");

  /// How many distinct properties have been checked.
  std::size_t size() const;

  /// Whether any property has been found violated. Lock-free, so it stays
  /// usable from the SIGALRM timeout handler, which must report an
  /// already-established violation before _exit() discards it.
  bool has_violation() const
  {
    return violation;
  }

  /// A copy of the results recorded so far, keyed by property.
  std::map<std::string, property_resultt> snapshot() const;

  /// Discards all verdicts.
  void clear();

private:
  mutable std::mutex mutex;
  std::map<std::string, property_resultt> results;
  std::atomic<bool> violation{false};
};

#endif
