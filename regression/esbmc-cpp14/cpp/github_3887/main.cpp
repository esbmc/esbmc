#include <cassert>
#include <iostream>
#include <memory>

int main()
{
  std::unique_ptr<int> x = std::make_unique<int>(5);
  assert(x != nullptr);
  assert(*x == 5);
  return 0;
}
