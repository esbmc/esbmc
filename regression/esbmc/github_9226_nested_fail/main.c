/* github #9226: byte offsets of elements and members reached through a
 * symbolic index. */
#include <stddef.h>

int nondet_int(void);
struct inner
{
  char c;
  short h[3];
};
struct outer
{
  int x;
  struct inner in[4];
};
struct outer o;

int main(void)
{
  int i = nondet_int();
  int j = nondet_int();
  __ESBMC_assume(i >= 0 && i < 4);
  __ESBMC_assume(j >= 0 && j < 3);
  char *base = (char *)&o;
  char *p = (char *)&o.in[i].h[j];
  __ESBMC_assert(
    p - base == offsetof(struct outer, in) + i * sizeof(struct inner) +
                  offsetof(struct inner, h) + j * sizeof(int),
    "byte offset of o.in[i].h[j]");
  __ESBMC_assert(&o.in[i].h[j] == o.in[i].h + j, "element address");
  return 0;
}
