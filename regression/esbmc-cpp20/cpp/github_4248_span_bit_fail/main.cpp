// github.com/esbmc/esbmc/issues/4248 — failing variant: bit_cast through
// the bundled <span> shim still produces a real, asserting value.

#include <span>
#include <cstdint>
#include <cassert>

int main()
{
  uint8_t buf[2] = {7, 8};
  std::span<const uint8_t> s(buf, 2);
  // bit_cast preserves the pointer value, so it cannot be null here.
  assert(std::bit_cast<uintptr_t>(s.data()) == 0);
  return 0;
}
