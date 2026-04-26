#include <bit>
#include <cstdint>

extern "C" uint32_t __VERIFIER_nondet_uint32_t();

int main()
{
  uint32_t bits = __VERIFIER_nondet_uint32_t();
  float f = std::bit_cast<float>(bits);
  uint32_t round_trip = std::bit_cast<uint32_t>(f);
  __ESBMC_assert(round_trip == bits, "non-pointer bit_cast round-trip");
  return 0;
}
