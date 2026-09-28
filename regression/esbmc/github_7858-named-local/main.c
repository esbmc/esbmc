// Pairs with github_7858: the assumption for the named local IS emitted on the
// same run. A fix for #7858 must add d->devnum without losing this.
#include <stdlib.h>

extern int __VERIFIER_nondet_int(void);

struct dev
{
  int devnum;
  int other;
};

static int create_pipe(struct dev *d)
{
  return d->devnum << 8;
}

int main(void)
{
  struct dev *d = malloc(sizeof(struct dev));
  if (!d)
    return 0;

  int local = __VERIFIER_nondet_int();
  if (local == 7)
    return create_pipe(d);
  return 0;
}
