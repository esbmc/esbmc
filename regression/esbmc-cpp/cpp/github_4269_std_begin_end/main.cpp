#include <array>
#include <cassert>

int main()
{
  std::array<int, 3> a = {1, 2, 3};
  int sum = 0;
  for (auto it = std::begin(a); it != std::end(a); ++it)
    sum += *it;
  assert(sum == 6);

  int total = 0;
  for (auto it = std::cbegin(a); it != std::cend(a); ++it)
    total += *it;
  assert(total == 6);
  return 0;
}
