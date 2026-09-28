#include <stdint.h>
#include <stddef.h>

/* The exists side of github_7311_forall: the binder i shadows the outer i
 * (2), and only buf[3] is 1. Substituting the outer value would check just
 * p->buf[2] and turn the satisfiable witness into a failed assertion. */

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
    __ESBMC_exists(&i, 0 <= i && i < 4 && p->buf[i] == 1), "some one");
  return 0;
}
