// Reproducer for https://github.com/esbmc/esbmc/issues/4234
// switch(static_cast<enum class : uint8_t>(...)) with fall-through cases.
// The second case constant was not being type-normalized, causing a width
// mismatch (8-bit vs 32-bit) in mk_eq.  Expected: VERIFICATION FAILED.
#include <cstdint>

enum class Op : uint8_t
{
  A = 0,
  B = 1
};

int main()
{
  uint8_t x = 0;
  int result = 0;
  switch(static_cast<Op>(x))
  {
  case Op::A:
  case Op::B:
    result = 1;
    break;
  default:
    break;
  }
  __ESBMC_assert(result == 0, "should fail: x=0 matches Op::A, falls through to Op::B body");
  return 0;
}
