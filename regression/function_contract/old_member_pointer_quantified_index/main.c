#include <stdlib.h>
struct V { int *buf; unsigned long n; };
static unsigned long q;
void drop_front(struct V *v)
{
  __ESBMC_requires(v->n > 0 && v->buf != NULL);
  __ESBMC_assigns(v->n, __ESBMC_object(v->buf));
  __ESBMC_ensures(v->n == __ESBMC_old(v->n) - 1);
  __ESBMC_ensures(__ESBMC_forall(&q, !(q < v->n) || v->buf[q] == __ESBMC_old(v->buf[q + 1])));
  for (unsigned long i = 0; i + 1 < v->n; i++)
    v->buf[i] = v->buf[i + 1];
  v->n--;
}
int main()
{
  unsigned long n = nondet_ulong();
  __ESBMC_assume(n >= 3 && n <= 8);
  struct V v = {malloc(n * sizeof(int)), n};
  v.buf[0] = 5; v.buf[1] = 7; v.buf[n - 1] = 9;
  drop_front(&v);
  __ESBMC_assert(v.n == n - 1 && v.buf[0] == 7 && v.buf[n - 2] == 9, "shifted");
#ifdef BAD
  __ESBMC_assert(v.buf[0] == 5, "bad");
#endif
  free(v.buf);
}
