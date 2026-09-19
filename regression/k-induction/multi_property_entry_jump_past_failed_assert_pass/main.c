/* A false assertion on a dead branch must not cost the inductive-step proof
   (esbmc/esbmc#7900). */
int main()
{
  int i = 0;
  goto test;
body:
  if (i > 100)
    __ESBMC_assert(0, "dead");
  __ESBMC_assert(i < 10, "bound");
  i = i + 1;
test:
  if (i < 10)
    goto body;
  return 0;
}
