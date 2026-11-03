#include <stddef.h>
struct desc
{
  void (*init)(int *);
  int n;
};
static void init_fn(int *v) { *v = 1; }
const struct desc the_desc = {init_fn, 4};

int big(const struct desc *d)
{
  int sum = 0;
  for (int i = 0; i < 1000; i++)
    sum += d->n;
  return sum;
}

int unpack(const struct desc *d)
{
  __ESBMC_requires(d == &the_desc);
  __ESBMC_ensures(__ESBMC_return_value == 1);
  int v = 0;
  if (d->init != NULL)
    d->init(&v);
  else
    v = big(d);
  return v;
}

int main(void)
{
  return 0;
}
