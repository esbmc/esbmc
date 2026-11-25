unsigned long nondet_ulong(void);
int main(void)
{
  unsigned long a = nondet_ulong();
  __ESBMC_assume(a == 6098646942UL);
  unsigned long r = a << 32;
  return (int)(r >> 63);
}
