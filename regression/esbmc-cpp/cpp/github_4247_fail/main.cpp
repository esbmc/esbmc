#include <bit>
#include <cstdint>

struct Packet
{
  uint8_t data[8];

  // bit_cast<uint8_t*>(this) inside a const method must produce a
  // pointer that aliases data[]; symex must see the actual byte values
  // — assertion below is intentionally false to confirm the bytes
  // really are routed through to the cast result (esbmc#4247).
  uint8_t *bytes() const
  {
    return std::bit_cast<uint8_t *>(this);
  }
};

int main()
{
  Packet pkt{};
  pkt.data[0] = 0xAB;

  uint8_t *p = pkt.bytes();
  __ESBMC_assert(p[0] == 0x00, "intentionally wrong: byte 0 is 0xAB");
  return 0;
}
