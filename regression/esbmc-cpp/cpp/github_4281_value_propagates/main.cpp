// Issue #4281: confirm the bitfield ctor init-list rewrite preserves
// value semantics. The init-list writes 0 to `flag`, so asserting
// `flag == 1` must fail — proving the assignment reaches the bitfield
// and the wrong value is observable. This is a value-propagation
// sanity check; dereferencet bounds-check soundness on bitfields is
// covered by regression/esbmc/struct_bitfields_*.
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
