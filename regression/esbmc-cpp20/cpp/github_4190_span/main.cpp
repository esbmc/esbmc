// github.com/esbmc/esbmc/issues/4190 — std::span shim.
// Exercises ctors, element access, iteration, and subviews.

#include <span>
#include <array>
#include <cassert>

int sum(std::span<int> s)
{
  int total = 0;
  for (int x : s)
    total += x;
  return total;
}

int main()
{
  int buf[4] = {1, 2, 3, 4};

  std::span<int> sv(buf, 4);
  assert(sv.size() == 4);
  assert(sv.data() == buf);
  assert(sv[0] == 1 && sv[3] == 4);
  assert(!sv.empty());
  assert(sum(sv) == 10);

  // std::array adapter — aggregate brace-init through the shim is
  // unreliable here, so default-construct then assign.
  std::array<int, 3> a;
  a[0] = 10; a[1] = 20; a[2] = 30;
  std::span<int> sv3(a);
  assert(sv3.size() == 3);
  assert(sv3.front() == 10 && sv3.back() == 30);

  auto mid = sv.subspan(1, 2);
  assert(mid.size() == 2 && mid[0] == 2 && mid[1] == 3);

  auto pre = sv.first(2);
  auto suf = sv.last(2);
  assert(pre[0] == 1 && pre[1] == 2);
  assert(suf[0] == 3 && suf[1] == 4);

  return 0;
}
