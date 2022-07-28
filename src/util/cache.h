#pragma once

#include <set>
#include <util/algorithms.h>

// Helper definitions
using crc_hash = size_t; // from irep2_meta_templates.h
/**
 * @brief This class stores all asserts conditions and guards
 *        for a given SSA, if some pair was already added
 *        then it will convert it to be a trivial case,
 *        e.g. GUARD => COND ----> GUARD => 1
 */
class crc_assert_cache : public ssa_algorithm
{
public:
  using crc_pair = std::pair<crc_hash, crc_hash>;
  crc_assert_cache(
    symex_target_equationt::SSA_stepst &steps,
    std::set<crc_pair> &crc_set,
    bool trivial_value)
    : ssa_algorithm(steps, true), crc_set(crc_set), trivial_value(trivial_value)
  {
  }

  void run_on_assert(symex_target_equationt::SSA_stept &) override;

  static std::set<crc_pair> create_empty_set()
  {
    return std::set<crc_pair>();
  }

  unsigned get_hits()
  {
    return hits;
  }

protected:
  std::set<crc_pair> &crc_set;
  /// value to be set for  the COND
  bool trivial_value;

private:
  unsigned hits = 0;
};
