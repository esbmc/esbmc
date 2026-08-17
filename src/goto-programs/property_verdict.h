#ifndef CPROVER_GOTO_PROGRAMS_PROPERTY_VERDICT_H
#define CPROVER_GOTO_PROGRAMS_PROPERTY_VERDICT_H

#include <util/irep/location.h>

#include <atomic>
#include <cstddef>
#include <map>
#include <mutex>
#include <string>

/// Outcome of checking one property. Ordered by dominance: when a property is
/// checked more than once in a run, the numerically greater verdict survives.
///
/// NotChecked is the weakest so that seeding the table with the program's whole
/// property set never masks a real verdict recorded later. It is not a failure
/// to report it: without --multi-property ESBMC solves one monolithic formula
/// and stops at the first violation, so the properties it never separated out
/// have genuinely not been decided, and saying so beats claiming they passed.
enum class property_verdictt
{
  NotChecked = 0,
  Passed = 1,
  Unknown = 2,
  Failed = 3
};

/// Where a property lives, kept apart from the table key so the report can sort
/// by source position and print a path once per file rather than once per line.
struct property_locationt
{
  std::string file;
  std::string function;
  std::string description;
  unsigned line = 0;
  unsigned column = 0;
};

property_locationt
property_location(const locationt &, const std::string &description);

/// A verdict together with where it applies and a note explaining how it was
/// reached -- that a discharge came from interval analysis rather than the
/// solver, or that it was vacuous. The note is empty when the verdict needs no
/// qualification.
struct property_resultt
{
  property_verdictt verdict;
  std::string note;
  property_locationt loc;
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
  /// Records \p verdict for \p property at \p loc, annotated with \p note.
  /// Keeps the dominant verdict when the property has already been checked.
  /// Safe to call from parallel solver threads.
  ///
  /// \p loc is required rather than defaulted: the report sorts and groups on
  /// it, so a caller that does not know where its property is would silently
  /// sort to the top of the table with no file or line.
  void record(
    const std::string &property,
    property_verdictt verdict,
    const property_locationt &loc,
    const std::string &note = "");

  /// Raises every NotChecked entry to Passed. Call only once the run has
  /// established that *all* properties hold -- a monolithic UNSAT refutes the
  /// disjunction of every claim violation, so each claim holds -- and never
  /// after a merely bounded round such as a k-induction base case.
  void promote_unchecked_to_passed();

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
