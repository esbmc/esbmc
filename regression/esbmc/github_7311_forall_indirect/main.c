#include <stdint.h>
#include <stddef.h>

/* Same concern as github_7311_forall, with the binder reached through a
 * pointer: `q` names the pointer, not the variable the quantifier binds, so
 * there is no name for the #7311 index fold to exclude. Folding anything
 * under such a binder would substitute the outer i (2) and check only
 * p->buf[2], masking the violation at i == 3.
 *
 * The verdict also needs rename_quantified to recover the bound name, which
 * it does here because q is constant-propagated to &i before renaming. A
 * binder whose value is not known (`q = nondet ? &i : &j`) is a separate,
 * pre-existing gap in that layer, not something this test pins. */

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
  void *q = &i;
  __ESBMC_assert(
    __ESBMC_forall(q, !(0 <= i && i < 4) || p->buf[i] == 0), "all zero");
  return 0;
}
