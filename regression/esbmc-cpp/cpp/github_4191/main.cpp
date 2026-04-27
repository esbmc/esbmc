#include <bit>
#include <cstdint>

extern "C" uint16_t __VERIFIER_nondet_uint16_t();

struct __attribute__((packed)) Header
{
  uint16_t length;
  uint8_t pad[6];
};

int main()
{
  Header h{};
  h.length = __VERIFIER_nondet_uint16_t();

  uint8_t *raw = (uint8_t *)&h;
  Header &mirror = *std::bit_cast<Header *>(raw);

  __ESBMC_assert(mirror.length == h.length, "round-trip via bit_cast");
  return 0;
}
