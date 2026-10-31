// #9393: std::max returns the first argument on a tie
#include <algorithm>
#include <cassert>

struct S
{
  int k;
  int v;
  bool operator<(const S &o) const
  {
    return k < o.k;
  }
};

int main()
{
  S a{1, 10}, b{1, 20};
  assert(std::max(a, b).v == 20);
  return 0;
}
