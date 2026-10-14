#include <filesystem>
#include <cassert>

namespace fs = std::filesystem;

int main()
{
  // Explicit ctor against the default-constructed end sentinel.
  unsigned n = 0;
  for (fs::directory_iterator it(fs::path("/d")), e; it != e; ++it)
  {
    assert(it->path().parent_path() == fs::path("/d"));
    ++n;
  }
  assert(n <= 3);

  // Two default-constructed iterators are both past-the-end, so equal.
  assert(fs::directory_iterator() == fs::directory_iterator());
  return 0;
}
