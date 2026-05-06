// Reproducer for https://github.com/esbmc/esbmc/issues/4234 — default: case A: variant
// When default: falls through into a case with an enum class : uint8_t constant,
// the case constant inside the default body was not being type-normalized,
// causing the same mk_eq width mismatch.  Expected: VERIFICATION SUCCESSFUL.
#include <cstdint>

enum class Op : uint8_t
{
  A = 0,
  B = 1
};

int main()
{
  uint8_t x = 2; // no named case matches → takes default
  int result = 0;
  switch(static_cast<Op>(x))
  {
  default:
  case Op::A:
    result = 1;
    break;
  case Op::B:
    result = 2;
    break;
  }
  __ESBMC_assert(result == 1, "x=2 → default falls through to Op::A body, result is 1");
  return 0;
}
