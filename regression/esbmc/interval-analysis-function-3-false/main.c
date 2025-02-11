int sum(int n)
{
  if (n == 0)
    return 0;
  int tmp = n + sum(n - 1);
  return tmp;
}

int main()
{
  int A = 5;
  int B = sum(A);
  __ESBMC_assert(B <= 10, "Should be able to verify!");  
}
