#include <bit>
#include <cstdint>

struct __attribute__((packed)) Header
{
  uint16_t length;
  uint8_t pad[6];
};

int main()
{
  Header h{};
  h.length = 1;

  uint8_t *raw = (uint8_t *)&h;
  Header &mirror = *std::bit_cast<Header *>(raw);

  mirror.length = 7;
  // h aliases mirror by construction, so h.length is now 7 and the
  // assertion below must fail.
  __ESBMC_assert(h.length == 1, "aliased write must propagate");
  return 0;
}
