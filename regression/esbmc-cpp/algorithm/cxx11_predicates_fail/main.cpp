// Non-vacuity guard for cxx11_predicates: all_of really inspects every element,
// so a predicate that fails partway must make it false.
#include <algorithm>
#include <cassert>

static bool odd(int x)
{
  return x % 2 != 0;
}

int main()
{
  int v[6] = {1, 2, 3, 4, 5, 6};
  assert(std::all_of(v, v + 6, odd));
  return 0;
}
