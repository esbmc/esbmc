/* Naming an element of one global must not excuse a different global. */
int global[10];
int other;

void f(int i, int v)
{
  __ESBMC_requires(i >= 0 && i < 10);
  __ESBMC_assigns(global[i]);
  __ESBMC_ensures(1);
  global[i] = v;
  other = 7; /* not in __ESBMC_assigns */
}

int main()
{
  return 0;
}
