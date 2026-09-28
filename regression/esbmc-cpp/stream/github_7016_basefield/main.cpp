#include <sstream>
#include <iomanip>
#include <cstring>
#include <cassert>

int main()
{
  std::ostringstream h;
  h << std::hex << 255;
  assert(strcmp(h.str().c_str(), "ff") == 0);

  std::ostringstream o;
  o << std::oct << 8;
  assert(strcmp(o.str().c_str(), "10") == 0);

  std::ostringstream n;
  n << std::hex << -1;
  assert(strcmp(n.str().c_str(), "ffffffff") == 0);

  std::ostringstream b;
  b << std::setbase(16) << std::uppercase << 255;
  assert(strcmp(b.str().c_str(), "FF") == 0);
  return 0;
}
