/* #9786: a call through a pointer with no target is skipped in every step, so
 * it must not disable the inductive step. */
extern int (*ext_cb)(int);

int main(void)
{
  ext_cb(1);
  unsigned i = 0;
  while (1)
  {
    i = (i + 1) % 4;
    __ESBMC_assert(i < 4, "i stays below 4");
  }
  return 0;
}
