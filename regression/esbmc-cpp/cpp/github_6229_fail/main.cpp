#include <cassert>
#include <string>

int main()
{
  std::string a = "AAAA";

  // s > "lit" was an overload ambiguity (a hard compile error) before the fix.
  // It now resolves to the free operator, and this genuinely-false assertion is
  // correctly reported as a violation.
  assert(a > "AAAA"); // "AAAA" > "AAAA" is false

  return 0;
}
