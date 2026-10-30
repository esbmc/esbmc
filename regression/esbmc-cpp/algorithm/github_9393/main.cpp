// #9393: std::min and std::max return the first argument on a tie
#include <algorithm>
#include <cassert>
#include <cmath>

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
  S a{1, 10}, b{1, 20}, c{2, 30};
  assert(std::min(a, b).v == 10);
  assert(std::max(a, b).v == 10);
  assert(std::min(a, c).v == 10);
  assert(std::max(a, c).v == 30);
  assert(!std::signbit(std::max(0.0, -0.0)));
  assert(std::signbit(std::min(-0.0, 0.0)));
  assert(std::max(1.0, std::nan("")) == 1.0);
  assert(std::min(1.0, std::nan("")) == 1.0);
  return 0;
}
