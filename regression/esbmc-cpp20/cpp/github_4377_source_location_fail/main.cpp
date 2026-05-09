#include <cassert>
#include <source_location>

int main()
{
  std::source_location sl = std::source_location::current();
  // current() must produce a non-zero line; the assertion below must fail.
  assert(sl.line() == 0);
  return 0;
}
