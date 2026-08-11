#include <cassert>
#include <sstream>
#include <string>

int main()
{
  std::ostringstream o;
  o << 42;

  // The stream holds "42", not the empty string it starts as.
  assert(o.str().size() == 0);

  return 0;
}
