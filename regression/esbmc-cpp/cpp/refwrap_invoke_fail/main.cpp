#include <functional>
#include <cassert>

int main()
{
  auto lam = [](int x) { return x + 1; };
  std::reference_wrapper<decltype(lam)> rl(lam);
  assert(rl(1) == 3);
  return 0;
}
