#include <cassert>
#include <filesystem>

int main()
{
  std::filesystem::path p("/x");
  // p is not empty; the assertion below must fail.
  assert(p.empty());
  return 0;
}
