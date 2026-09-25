int g;

/* Assigns-only contract, no requires/ensures: replacing the call must still
   honour the empty frame condition and leave g alone (GitHub #6555). */
int pure(int x)
{
  __ESBMC_assigns();
  return x + 1;
}

int main(void)
{
  g = 3;
  pure(1);
  __ESBMC_assert(g == 3, "replaced call writes nothing");
  return 0;
}
