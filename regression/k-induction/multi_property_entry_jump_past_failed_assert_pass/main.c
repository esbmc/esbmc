/* The dead entry jump now gets a havoc; the inductive-step proof must survive
   (esbmc/esbmc#7900). */
int nondet_int();
int main()
{
  int i = 0, x = 0, y = 0;
  if (x == 1)
  {
    y = 1;
    goto L;
  }
  while (i < 10)
  {
    __ESBMC_assert(i < 10, "bound");
    i = i + 1;
    continue;
  L:
    __ESBMC_assert(0, "v");
    i = i + 1;
  }
  return y;
}
