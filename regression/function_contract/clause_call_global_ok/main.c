int g;
int probe(void);

int f(int x)
{
  g = probe();
  __ESBMC_requires(g > 100);
  __ESBMC_ensures(g > 100);
  return x;
}

int main(void)
{
  return f(5);
}
