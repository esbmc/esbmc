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

int main(void)
{
  __ESBMC_assert(sizeof(struct pack1_s) == 5, "pack(1) sizeof is 5");
  __ESBMC_assert(offsetof(struct pack1_s, b) == 1, "pack(1) offsetof(b) is 1");
  __ESBMC_assert(sizeof(struct pack2_s) == 6, "pack(2) sizeof is 6");
  __ESBMC_assert(offsetof(struct pack2_s, b) == 2, "pack(2) offsetof(b) is 2");
  __ESBMC_assert(sizeof(union pack2_u) == 4, "pack(2) union sizeof is 4");
  __ESBMC_assert(sizeof(struct plain_s) == 8, "unpacked sizeof is 8");
  __ESBMC_assert(offsetof(struct plain_s, b) == 4, "unpacked offsetof(b) is 4");
  return 0;
}
