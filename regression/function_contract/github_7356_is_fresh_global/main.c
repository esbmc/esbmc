/* __ESBMC_is_fresh on a global pointer. The #7356 havoc runs before the
 * is_fresh allocation so it cannot overwrite the pointer that allocation
 * produces. Measured: this reports an invalid dereference both before that
 * change and with the havoc placed after the allocation.
 */
int *gp;

void f(void)
{
  __ESBMC_requires(__ESBMC_is_fresh(gp, sizeof(int)));
  __ESBMC_assigns(*gp);
  *gp = 1;
}

int main(void)
{
  return 0;
}
