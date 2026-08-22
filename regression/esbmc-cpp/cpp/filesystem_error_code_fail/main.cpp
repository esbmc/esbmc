#include <filesystem>
#include <cassert>

int main()
{
  std::error_code ec;
  assert(ec.value() == 1);
  return 0;
}
