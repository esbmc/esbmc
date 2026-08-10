#include <cassert>
#include <string>

int main()
{
  std::string a = "a", b = "b";

  // compare() is not symmetric: "a" is less than "b", so this must be
  // reported as violated.
  assert(a.compare(b) > 0);

  return 0;
}
