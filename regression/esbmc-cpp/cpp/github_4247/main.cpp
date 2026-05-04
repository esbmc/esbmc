#include <bit>
#include <cstdint>

struct Packet
{
  uint8_t data[8];

  // bit_cast<uint8_t*>(this) inside a const method: this is `const
  // Packet*`, target is non-const `uint8_t*`. Standard bit_cast accepts
  // this; the bundled overload must too (esbmc#4247).
  const uint8_t *first_byte_const() const
  {
    return std::bit_cast<const uint8_t *>(this);
  }

  uint8_t *first_byte_mutable() const
  {
    return std::bit_cast<uint8_t *>(this);
  }
};

int main()
{
  Packet pkt{};
  pkt.data[0] = 0xAB;
  pkt.data[7] = 0xCD;

  const uint8_t *p_const = pkt.first_byte_const();
  uint8_t *p_mut = pkt.first_byte_mutable();

  __ESBMC_assert(p_const[0] == 0xAB, "const-qualified view of byte 0");
  __ESBMC_assert(p_mut[7] == 0xCD, "cv-stripped view of byte 7");
  return 0;
}
