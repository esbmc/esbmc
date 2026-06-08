// Issue #4281 (variant): the same false-positive also fired without
// [[gnu::packed]]; covers the bitfield-only path through the ctor
// member-initialiser-list lowering.
#include <cstdint>

struct S
{
  uint32_t flag : 1;
  uint32_t rest : 31;
  S() : flag{0}, rest{0}
  {
  }
};

int main()
{
  S s;
  (void)s.flag;
  return 0;
}
