// Negative counterpart of github_5868_string_conversions: `pos` really is
// written, so a wrong claim about it is refuted rather than vacuously true.
#include <string>
#include <cassert>

int main()
{
  size_t p = 0;
  assert(std::stoi("42abc", &p) == 42);
  assert(p == 5);
  return 0;
}
