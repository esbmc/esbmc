#ifndef GOTO_PROGRAMS_WRITE_FOOTPRINT_H_
#define GOTO_PROGRAMS_WRITE_FOOTPRINT_H_

#include <pointer-analysis/value_sets.h>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class goto_functionst;

/// Conservatively bounds writes across reachable loops and their callees.
/// Only fields preceding every possible write are retained; unknown effects
/// keep whole-object havoc. Bounds are shared across loops, not loop-specific.
class write_footprintt
{
public:
  /// Complete pointer definitions, or empty if they cannot be safely traced.
  using sourcest = std::function<std::vector<expr2tc>(const expr2tc &)>;
  /// Includes callees of loops, even when their own bodies contain no loop.
  using in_loopt = std::function<bool(const irep_idt &, unsigned)>;

  write_footprintt(
    const goto_functionst &functions,
    value_setst &points_to,
    const std::unordered_set<irep_idt, irep_id_hash> &reachable,
    const in_loopt &in_loop,
    const sourcest &sources);

  /// Fields to havoc, or nullptr when no prefix can safely be preserved.
  const std::vector<expr2tc> *partial_writes(const expr2tc &object) const;

private:
  std::unordered_map<irep_idt, BigInt, irep_id_hash> lower_bounds;
  mutable std::unordered_map<expr2tc, std::vector<expr2tc>, irep2_hash>
    member_targets;
};

#endif
