// The loop runs 4 times: the call's result moves i past the bound.
int g(int x)
{
  return x == 3 ? 100 : x;
}

int main()
{
  int i, n = 0;
  for (i = 0; i < 10; i++)
  {
    n++;
    i = g(i);
  }
  __ESBMC_assert(n != 4, "the loop ran 4 times");
  return 0;
}
