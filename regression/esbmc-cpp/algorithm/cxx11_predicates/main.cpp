// The C++11 additions to <algorithm> were absent: all_of, any_of, none_of,
// find_if_not, copy_if, is_partitioned, partition_point, minmax,
// minmax_element, is_heap and is_heap_until. Naming any of them was a parse
// error.
//
// The empty-range cases are the ones easy to get wrong: [alg.all.of] makes
// all_of and none_of true on an empty range and any_of false. minmax_element
// returns the FIRST smallest but the LAST largest.
//
// Verified against clang++ -std=c++17 -fsanitize=address,undefined: exits 0.
#include <algorithm>
#include <utility>
#include <cassert>
static bool odd(int x)
{
  return x % 2 != 0;
}
static bool pos(int x)
{
  return x > 0;
}
int main()
{
  int v[6] = {1, 2, 3, 4, 5, 6};
  assert(std::all_of(v, v + 6, pos));
  assert(!std::all_of(v, v + 6, odd));
  assert(std::any_of(v, v + 6, odd));
  assert(!std::any_of(v, v + 6, [](int x) { return x > 100; }));
  assert(std::none_of(v, v + 6, [](int x) { return x > 100; }));
  assert(!std::none_of(v, v + 6, odd));

  // empty range: all_of/none_of are true, any_of is false
  assert(std::all_of(v, v, pos));
  assert(std::none_of(v, v, pos));
  assert(!std::any_of(v, v, pos));

  assert(std::find_if_not(v, v + 6, odd) == v + 1);

  int out[6];
  int *e = std::copy_if(v, v + 6, out, odd);
  assert(e == out + 3 && out[0] == 1 && out[1] == 3 && out[2] == 5);

  int p[6] = {1, 3, 5, 2, 4, 6};
  assert(std::is_partitioned(p, p + 6, odd));
  assert(!std::is_partitioned(v, v + 6, odd));
  assert(std::partition_point(p, p + 6, odd) == p + 3);

  // minmax returns references, so the arguments must outlive the pair --
  // binding it to literals would leave it dangling.
  int lo = 4, hi = 2;
  std::pair<const int &, const int &> mm = std::minmax(lo, hi);
  assert(mm.first == 2 && mm.second == 4);

  int d[5] = {3, 1, 4, 1, 5};
  std::pair<int *, int *> me = std::minmax_element(d, d + 5);
  assert(*me.first == 1 && *me.second == 5);
  assert(me.first == d + 1); // the FIRST smallest

  int h[5] = {5, 4, 3, 2, 1};
  assert(std::is_heap(h, h + 5));
  int nh[3] = {1, 2, 3};
  assert(!std::is_heap(nh, nh + 3));
  assert(std::is_heap_until(nh, nh + 3) == nh + 1);
  return 0;
}
