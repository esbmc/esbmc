/* Under --closed-world-fnptr a call through a pointer with no target cannot
 * happen, so the path past it is dead. */
extern int (*ext_cb)(int);

int main(void)
{
  int r = 0;
  r = ext_cb(1);
  __ESBMC_assert(r == 0, "result unchanged");
  return 0;
}
