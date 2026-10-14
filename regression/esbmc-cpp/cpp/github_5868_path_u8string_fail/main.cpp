#include <filesystem>
#include <cassert>

int main()
{
  std::filesystem::path p("a/b");
  // The path holds three characters, so u8string() is not empty.
  assert(p.u8string().empty());
  return 0;
}
