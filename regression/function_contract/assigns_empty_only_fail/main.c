int g;

/* Declares an empty frame condition but writes a global: the assigns
   compliance check must catch it (GitHub #6555). */
int impure(int x)
{
  __ESBMC_assigns();
  g = x;
  return x + 1;
}

int main(void)
{
  return impure(1) - 2;
}
