_Bool nondet_bool();

unsigned n = 0;
int x = 0;

int main()
{
  /* Never entered: the inductive step must not assume its entry condition
     on the only path that reaches the second loop. */
  for (unsigned i = 0; i < n; i++)
  {
  }
  while (nondet_bool())
    x++;
  __ESBMC_assert(x != 5, "x reaches 5 after five iterations");
}
