#include <any>
#include <cassert>

int main()
{
  std::any a = 7;
  // a holds 7; the assertion below must fail.
  assert(std::any_cast<int>(a) == 99);
  return 0;
}
