int main()
{
  float a, b;
  __ESBMC_assert((a>b)==(a-b>0), "theorem");
}
