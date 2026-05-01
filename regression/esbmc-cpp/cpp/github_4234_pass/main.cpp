// Reproducer for https://github.com/esbmc/esbmc/issues/4234 — passing variant
// switch with fall-through cases; assertion is always true.
// Expected: VERIFICATION SUCCESSFUL.
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
  __ESBMC_assert(result == 1, "x=0 → Op::A falls through to Op::B body, result is 1");
  return 0;
}
