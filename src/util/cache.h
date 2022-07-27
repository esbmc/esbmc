#pragma once

#include <set>
#include <util/algorithms.h>

using crc_hash = unsigned long long;
using crc_pair = std::pair<crc_hash, crc_hash>;
class crc_assert_cache : public ssa_algorithm
{
public:
  explicit crc_assert_cache(
    symex_target_equationt::SSA_stepst &steps,
    std::set<crc_pair> &crc_set,
    bool is_forward_condition)
    : ssa_algorithm(steps, true),
      crc_set(crc_set),
      is_forward_condition(is_forward_condition)
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
  bool is_forward_condition;

private:
  unsigned hits = 0;
};
