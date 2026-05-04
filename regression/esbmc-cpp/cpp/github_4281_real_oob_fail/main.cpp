// Issue #4281: confirm the bitfield ctor init-list fix does not silently
// suppress real failures. Writing 0 in the init-list and then asserting
// the bitfield is non-zero must still fail — i.e. ESBMC must reach and
// evaluate the assertion through the same code path the bounds-check fix
// touched.
#include <cassert>
#include <cstdint>

struct [[gnu::packed]] S
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
  assert(s.flag == 1);
  return 0;
}
