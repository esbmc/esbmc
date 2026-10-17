_Bool nondet_bool();

unsigned n = 0;
int x = 0;

int main()
{
  /* Never entered: the second loop is still proved by induction. */
  for (unsigned i = 0; i < n; i++)
  {
  }
  while (nondet_bool() && x < 10)
    x++;
  __ESBMC_assert(x <= 10, "x stays bounded");
}
