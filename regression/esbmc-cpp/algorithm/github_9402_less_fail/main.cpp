// #9402: binary_search without a comparator must use operator<, not operator==
#include <algorithm>
#include <cassert>

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
  Tens t[3] = {{10}, {20}, {30}};
  Tens k = {12};
  assert(!std::binary_search(t, t + 3, k));
  return 0;
}
