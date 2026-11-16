/* #9786: the inductive step may run past a call through a pointer with no
 * target, but the call's result stays unknown. */
extern int (*ext_cb)(int);

int main(void)
{
  int r = 0;
  r = ext_cb(1);
  __ESBMC_assert(r == 0, "result unchanged");
  unsigned i = 0;
  while (1)
  {
    i = (i + 1) % 4;
    __ESBMC_assert(i < 4, "i stays below 4");
  }
  return 0;
}
