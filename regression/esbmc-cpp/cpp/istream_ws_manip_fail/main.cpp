// `is >> std::ws` returns the stream it was applied to.
#include <cassert>
#include <sstream>
int main() {
  std::istringstream s("  5");
  std::istream &r = s >> std::ws;
  assert(&r != &s);
  return 0;
}
