/* github.com/esbmc/esbmc/issues/6443
 *
 * P is 2-inductive but not 1-inductive: from an arbitrary state one loop
 * body execution establishes it, so assuming P on the earlier unwindings of
 * the inductive step discharges it. Before #6443 the conversion never took
 * effect and this was reported UNKNOWN at every k. */
int main(void)
{
  _Bool x = 0;
  while (1)
  {
    __ESBMC_assert(!x, "P");
    x = 0;
  }
  return 0;
}
