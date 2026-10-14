/* The native arm folds `(c) || (assert(0), 0)` with an empty else, provided the
   discarded statement is a no-op (esbmc/esbmc#7900). */
extern int nd(void);
int main(void)
{
  int c = nd();
  if (c == 2)
  {
    __ESBMC_assert(0, "j");
    (void)0;
  }
  else
  {
  }
  return 0;
}
