#include <filesystem>
#include <cassert>

namespace fs = std::filesystem;

// The error_code ctor plus increment(ec), as json_utils.h drives it.
int main()
{
  std::error_code ec;
  fs::path full("/d/f0");
  unsigned n = 0;
  for (fs::directory_iterator it(full.parent_path(), ec), e; !ec && it != e;
       it.increment(ec))
    ++n;
  assert(n <= 3);
  return 0;
}
