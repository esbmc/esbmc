#include <sstream>
#include <ostream>
#include <cstring>
#include <cassert>

static void emit(std::ostream &o)
{
  o << "id=" << 12;
}

int main()
{
  std::ostringstream hs;
  emit(hs);
  assert(strcmp(hs.str().c_str(), "id=12") == 0);
  return 0;
}
