#include <filesystem>
#include <cassert>

namespace fs = std::filesystem;

// operator const path& lets a directory_entry bind to the free functions,
// which is how converter_util.cpp calls is_regular_file(*it).
int main()
{
  for (fs::directory_iterator it(fs::path("/d")), e; it != e; it++)
  {
    if (fs::is_regular_file(*it))
      assert(!it->path().filename().empty());
  }
  return 0;
}
