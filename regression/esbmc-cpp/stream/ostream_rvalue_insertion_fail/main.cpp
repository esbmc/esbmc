// Anti-vacuity twin of ostream_rvalue_insertion: the insertion has to reach the
// stream, so the result cannot be empty.
#include <sstream>
#include <string>
#include <cassert>

int main()
{
  std::string s = (std::ostringstream{} << 42).str();
  assert(s.empty());
  return 0;
}
