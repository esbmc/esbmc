#include <filesystem>
#include <string>
#include <cassert>

int main()
{
  std::filesystem::path p("a/b");

  // [fs.path.native.obs]: this model stores a narrow string, so u8string and
  // generic_string yield the same sequence as string().
  assert(p.u8string().size() == 3);
  assert(p.generic_string().size() == 3);
  assert(!p.u8string().empty());
  return 0;
}
