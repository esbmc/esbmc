// upper_bound was scanning BACKWARDS from last-1 and returning the last
// position satisfying `val <= *it`, instead of [upper.bound]'s first position
// satisfying `val < *it`. On a sorted {1,2,3,4,5,6} it answered
// upper_bound(3) == &6 rather than &4 -- deterministically wrong, not
// nondeterministic, which is why it looked plausible and survived.
//
// lower_bound's comparator overload additionally tested
// `comp(val, *it) || val == *it`, requiring Ty to have operator== which
// [lower.bound] does not; a comparator-only type would not compile.
//
// Verified against clang++ -std=c++17 -fsanitize=address,undefined: exits 0.
#include <algorithm>
#include <utility>
#include <cassert>
static bool gt(int a, int b)
{
  return a > b;
}
int main()
{
  int v[7] = {1, 2, 3, 3, 3, 5, 6};

  // [lower.bound] / [upper.bound] around a run of equal values
  assert(std::lower_bound(v, v + 7, 3) == v + 2);
  assert(std::upper_bound(v, v + 7, 3) == v + 5);

  // value absent: both point at the first greater element
  assert(std::lower_bound(v, v + 7, 4) == v + 5);
  assert(std::upper_bound(v, v + 7, 4) == v + 5);

  // below the range / above the range
  assert(std::lower_bound(v, v + 7, 0) == v);
  assert(std::upper_bound(v, v + 7, 0) == v);
  assert(std::lower_bound(v, v + 7, 9) == v + 7);
  assert(std::upper_bound(v, v + 7, 9) == v + 7);

  std::pair<int *, int *> r = std::equal_range(v, v + 7, 3);
  assert(r.first == v + 2 && r.second == v + 5);

  // comparator overloads on a descending range
  int d[5] = {9, 7, 5, 3, 1};
  assert(std::lower_bound(d, d + 5, 5, gt) == d + 2);
  assert(std::upper_bound(d, d + 5, 5, gt) == d + 3);
  return 0;
}
