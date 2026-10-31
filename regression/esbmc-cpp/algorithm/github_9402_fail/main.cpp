// #9402: binary_search must use the comparator, not operator==
#include <algorithm>
#include <cassert>

struct ByTens
{
  bool operator()(int a, int b) const
  {
    return a / 10 < b / 10;
  }
};

int main()
{
  int a[3] = {10, 20, 30};
  assert(!std::binary_search(a, a + 3, 12, ByTens()));
  return 0;
}
