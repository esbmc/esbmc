// #11013: a union over a byte array is read through its widest member
union u { unsigned char t; unsigned char b[4]; };
int main(void)
{
  unsigned char buf[4] __attribute__((aligned(4))) = {1, 2, 3, 4};
  union u v = *(union u *)buf;
  __ESBMC_assert(v.b[3] != 4, "last byte");
  return 0;
}
