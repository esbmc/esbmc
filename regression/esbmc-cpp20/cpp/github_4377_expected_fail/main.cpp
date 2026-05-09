#include <cassert>
#include <expected>

int main()
{
  std::expected<int, int> e = 5;
  // *e is 5; the assertion below must fail.
  assert(*e == 99);
  return 0;
}
