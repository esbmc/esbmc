#include <cassert>
#include <string>

int main()
{
  std::string a = "AA";

  // Before the fix, "s > literal" did not even compile (ambiguous between the
  // member operator>(const char*) and the free operator>(basic_string&, const
  // char*)). With the fix only the free template survives, and since "AA" is
  // not greater than "AAAA" this genuinely-false assertion is correctly FAILED.
  assert(a > "AAAA");

  return 0;
}
