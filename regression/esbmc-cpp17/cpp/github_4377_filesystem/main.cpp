#include <cassert>
#include <filesystem>

int main()
{
  // Construct path from string literal.
  std::filesystem::path p("/tmp/foo.txt");
  assert(!p.empty());

  // Append via operator/.
  std::filesystem::path q = std::filesystem::path("/tmp") / "foo.txt";
  assert(!q.empty());

  // Equality.
  std::filesystem::path a("/x");
  std::filesystem::path b("/x");
  std::filesystem::path c("/y");
  assert(a == b);
  assert(a != c);

  return 0;
}
