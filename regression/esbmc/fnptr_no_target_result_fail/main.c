/* A call through a pointer with no target is skipped in the open world, so its
 * result is unknown, not the value the left-hand side held before. */
extern int (*ext_cb)(int);

int main(void)
{
  int r = 0;
  r = ext_cb(1);
  __ESBMC_assert(r == 0, "result unchanged");
  return 0;
}
