#include <sstream>
#include <cstring>
#include <cassert>
int main() {
  // [ostream.inserters.arithmetic]: a negative value in a non-decimal base
  // goes through the same-width unsigned type.
  std::ostringstream s;
  s << std::hex << (short)-1;
  assert(strcmp(s.str().c_str(), "ffff") == 0);

  std::ostringstream l;
  l << std::hex << (long long)-1;
  assert(strcmp(l.str().c_str(), "ffffffffffffffff") == 0);
  return 0;
}
