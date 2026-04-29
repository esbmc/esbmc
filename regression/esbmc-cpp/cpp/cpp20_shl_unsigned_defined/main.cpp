// Unsigned left-shift wrap is defined under every C++ standard.
// Under --overflow-check + --std c++20, we must not flag it.
#include <cstdint>

extern "C" unsigned int nondet_uint();

int main()
{
  uint32_t a = nondet_uint();
  uint32_t r = a << 1;
  return r == 0 ? 0 : 1;
}
