#include <filesystem>
#include <cassert>

namespace fs = std::filesystem;

int main()
{
  unsigned n = 0;
  for (const auto &entry : fs::directory_iterator(fs::path("/d")))
  {
    (void)entry;
    ++n;
  }
  // The entry count is non-deterministic, so an empty directory is reachable.
  assert(n > 0 && "directory always yields at least one entry");
  return 0;
}
