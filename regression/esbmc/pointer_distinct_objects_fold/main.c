/* A pointer is an (object, offset) pair, so addresses rooted in two
 * different objects never compare equal, whatever their offsets. Without
 * the fold the comparison stays symbolic, the impossible arm survives, and
 * each loop below unrolls to its 300 bound. With it every comparison is
 * a constant and the loops are bounded by the true arm. */
typedef int (*handler_t)(int);

static int inc(int x)
{
  return x + 1;
}
static int dec(int x)
{
  return x - 1;
}

int a[4], b[4];

int main(void)
{
  /* Function addresses: a hook checked against one implementation. */
  handler_t h = inc;
  int n = 0;
  for (unsigned short i = 0; i < (h == dec ? 300 : 2); i++)
    n++;
  __ESBMC_assert(n == 2, "distinct functions fold unequal");

  /* Object addresses through constant offsets and casts. */
  int *p = &a[1];
  int m = 0;
  if (p != b + 1 && (char *)p != (char *)&b[0] + 4 && p - 1 != &b[3])
    for (unsigned short i = 0; i < 2; i++)
      m++;
  else
    for (unsigned short i = 0; i < 300; i++)
      m++;
  __ESBMC_assert(m == 2, "distinct arrays fold unequal");
  return 0;
}
