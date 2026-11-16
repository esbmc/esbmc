// #9364: partial_sort_copy places the min(N, M) smallest inputs in sorted
// order and returns result_first + min(N, M).
#include <algorithm>
#include <cassert>
#include <functional>

int nondet_int();

int main()
{
  int a[3] = {3, 1, 2};
  int o[3];
  int *r = std::partial_sort_copy(a, a + 3, o, o + 3);
  assert(r == o + 3 && o[0] == 1 && o[1] == 2 && o[2] == 3);

  int b[4] = {4, 1, 3, 2};
  int s[2];
  r = std::partial_sort_copy(b, b + 4, s, s + 2);
  assert(r == s + 2 && s[0] == 1 && s[1] == 2);

  int l[3] = {0, 0, 7};
  r = std::partial_sort_copy(a, a + 2, l, l + 3);
  assert(r == l + 2 && l[0] == 1 && l[1] == 3 && l[2] == 7);

  r = std::partial_sort_copy(b, b + 4, s, s + 2, std::greater<int>());
  assert(r == s + 2 && s[0] == 4 && s[1] == 3);

  r = std::partial_sort_copy(b, b, s, s + 2);
  assert(r == s);

  int x[3] = {nondet_int(), nondet_int(), nondet_int()};
  int m[1];
  r = std::partial_sort_copy(x, x + 3, m, m + 1);
  assert(r == m + 1 && m[0] <= x[0] && m[0] <= x[1] && m[0] <= x[2]);
  return 0;
}
