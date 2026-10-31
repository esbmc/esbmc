// #9402: binary_search must use the comparator (or operator<), not operator==
#include <algorithm>
#include <cassert>

struct ByTens
{
  bool operator()(int a, int b) const
  {
    return a / 10 < b / 10;
  }
};

struct Tens
{
  int v;
  bool operator<(const Tens &o) const
  {
    return v / 10 < o.v / 10;
  }
  bool operator==(const Tens &o) const
  {
    return v == o.v;
  }
};

int main()
{
  int a[3] = {10, 20, 30};
  assert(std::binary_search(a, a + 3, 12, ByTens()));
  assert(!std::binary_search(a, a + 3, 45, ByTens()));

  Tens t[3] = {{10}, {20}, {30}};
  Tens k = {12};
  Tens m = {45};
  assert(std::binary_search(t, t + 3, k));
  assert(!std::binary_search(t, t + 3, m));
  return 0;
}
