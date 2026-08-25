#include <stdint.h>
#include <stddef.h>

/* The index-operand fold added for #7311 must not substitute a
 * quantifier-bound symbol with the like-named program variable's SSA value:
 * here the forall binder i shadows the outer i (2); substituting it would
 * check only p->buf[2] and mask the violation at i == 3. */

typedef struct
{
  uint8_t buf[64];
  size_t buflen;
  uint64_t len;
} ctx;

int main(void)
{
  ctx c = {0};
  ctx *p = &c;
  p->buf[3] = 1;
  int i = 2;
  __ESBMC_assert(
    __ESBMC_forall(&i, !(0 <= i && i < 4) || p->buf[i] == 0), "all zero");
  return 0;
}
