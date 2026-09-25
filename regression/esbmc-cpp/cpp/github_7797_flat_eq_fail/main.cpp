// #7797: two flat_sets holding comparator-equivalent but unequal keys are not
// equal -- [tab:container.req] gives == on the elements themselves.
#include <flat_set>
#include <cassert>

struct mod10
{
  bool operator()(int a, int b) const
  {
    return a % 10 < b % 10;
  }
};

int main()
{
  std::flat_set<int, mod10> a, b;
  a.insert(3);
  b.insert(13);
  assert(a == b);
  return 0;
}
