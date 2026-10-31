// #9396: merging {1, 5} with {2} taken from a larger array writes three
// elements and leaves o[3] alone.
#include <algorithm>
#include <cassert>

int main()
{
  int b[3] = {2, 3, 9};
  int a[2] = {1, 5};
  int o[4] = {0, 0, 0, 0};
  int *r = std::merge(a, a + 2, b, b + 1, o);
  assert(r == o + 4 && o[2] == 3);
  return 0;
}
