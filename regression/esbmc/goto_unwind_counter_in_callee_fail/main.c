// The loop runs 4 times: f moves the global counter past the bound.
int i;

void f()
{
  if (i == 3)
    i = 100;
}

int main()
{
  int n = 0;
  for (i = 0; i < 10; i++)
  {
    n++;
    f();
  }
  __ESBMC_assert(n != 4, "the loop ran 4 times");
  return 0;
}
