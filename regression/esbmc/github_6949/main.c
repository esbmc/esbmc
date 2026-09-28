#include <stdint.h>
#include <stddef.h>

#pragma pack(push, 1)
struct pack1_s
{
  char a;
  uint32_t b;
};
#pragma pack(pop)

#pragma pack(push, 2)
struct pack2_s
{
  char a;
  uint32_t b;
};
union pack2_u
{
  char c;
  uint32_t u;
};
#pragma pack(pop)

struct plain_s
{
  char a;
  uint32_t b;
};

struct __attribute__((packed)) attr_packed_s
{
  char a;
  uint32_t b;
};

/* an attribute that says nothing about layout */
struct __attribute__((deprecated)) unrelated_attr_s
{
  char a;
  uint32_t b;
};

union plain_u
{
  char c;
  uint32_t u;
};

/* Bitfield layout is ABI-specific (MSVC starts a new allocation unit where the
 * Itanium ABI packs), so only the pragma's *effect* is portable: a zero-width
 * bitfield's alignment demand is not capped, so it places the next member
 * exactly where it would without the pragma. */
#pragma pack(push, 2)
struct zero_bitfield_s
{
  char a;
  unsigned : 0;
  char b;
};
#pragma pack(pop)

struct plain_zero_bitfield_s
{
  char a;
  unsigned : 0;
  char b;
};

#pragma pack(push, 1)
struct __attribute__((aligned(8))) pack1_aligned_s
{
  char a;
  uint32_t b;
};
#pragma pack(pop)

int main(void)
{
  struct pack2_s obj;
  struct pack2_s *p = &obj;
  uint32_t direct = p->b;
  union plain_u pu;
  pu.u = 1;
  uint32_t via_union = pu.u;
  (void)direct;
  (void)via_union;

  __ESBMC_assert(sizeof(struct pack1_s) == 5, "pack(1) sizeof is 5");
  __ESBMC_assert(offsetof(struct pack1_s, b) == 1, "pack(1) offsetof(b) is 1");
  __ESBMC_assert(sizeof(struct pack2_s) == 6, "pack(2) sizeof is 6");
  __ESBMC_assert(offsetof(struct pack2_s, b) == 2, "pack(2) offsetof(b) is 2");
  __ESBMC_assert(sizeof(union pack2_u) == 4, "pack(2) union sizeof is 4");
  __ESBMC_assert(sizeof(struct plain_s) == 8, "unpacked sizeof is 8");
  __ESBMC_assert(offsetof(struct plain_s, b) == 4, "unpacked offsetof(b) is 4");
  __ESBMC_assert(
    offsetof(struct zero_bitfield_s, b) ==
      offsetof(struct plain_zero_bitfield_s, b),
    "pack does not cap a zero-width bitfield");
  __ESBMC_assert(offsetof(struct pack1_aligned_s, b) == 1, "pack(1) + aligned(8) offsetof(b) is 1");
  __ESBMC_assert(sizeof(struct pack1_aligned_s) == 8, "pack(1) + aligned(8) sizeof is 8");
  __ESBMC_assert(offsetof(struct attr_packed_s, b) == 1, "packed offsetof(b) is 1");
  __ESBMC_assert(
    offsetof(struct unrelated_attr_s, b) == 4,
    "unrelated attribute leaves layout alone");
  return 0;
}
