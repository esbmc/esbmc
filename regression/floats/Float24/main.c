int main()
{
  float a, b, _a=a, _b=b;
  __ESBMC_assume(a==1 && b==2);
  assert(a!=b);
}
