// Negative counterpart of pair_tuple_interface: get really reads the pair's
// members, so a claim contradicting them is refuted rather than vacuously
// held (issue #5868).
#include <utility>

int main()
{
  std::pair<int, int> p(7, 8);
  __ESBMC_assert(std::get<0>(p) == 8, "must not hold");
  return 0;
}
