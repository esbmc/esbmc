#include <stdbool.h>
#include <stdint.h>

typedef struct
{
  uint64_t tmp1 : 64;
} struct_1;

typedef union
{
  unsigned int raw : 2;

  struct
  {
    unsigned int x1 : 1;
    unsigned int x2 : 1;
  };
} union_1;

void func(struct_1 *tmp_p);

void func(struct_1 *tmp_p)
{
  union_1 tmp1 = (union_1)(uint32_t)(tmp_p->tmp1);
}
