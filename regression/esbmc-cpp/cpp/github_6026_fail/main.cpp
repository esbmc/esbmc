#include <tuple>
#include <utility>

int main()
{
  auto t = std::make_tuple(1, 2, 3);
  int s = std::apply([](auto... xs) { return (xs + ...); }, t);
  __ESBMC_assert(s == 7, "deliberately wrong"); // s is actually 6
  return 0;
}
