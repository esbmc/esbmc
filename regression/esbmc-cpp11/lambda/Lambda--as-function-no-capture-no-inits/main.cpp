#include <cassert>
#include <functional>

int main()
{
  std::function<int(int, int)> sum = [](int a, int b) -> int { return a + b; };

  assert(sum(5, 7) == 12);

  return 0;
}