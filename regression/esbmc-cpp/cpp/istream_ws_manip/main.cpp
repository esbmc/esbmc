// `is >> std::ws` needs the istream manipulator overload of operator>>.
#include <cassert>
#include <sstream>
int main() {
  std::istringstream s("  5");
  std::istream &r = s >> std::ws;
  assert(&r == &s);
  return 0;
}
