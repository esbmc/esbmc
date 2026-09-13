// Constructing a stream stores a nondeterministic file size. Holding it in a
// signed 32-bit streamsize, or in a signed 64-bit offset, made
// --ir --overflow-check report an overflow inside the constructors (#7540).
#include <sstream>

int main()
{
  std::ostringstream os;
  return 0;
}
