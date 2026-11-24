// #11013: a union over a byte array is written through its widest member
union u { unsigned char t; unsigned char b[4]; };
int main(void)
{
  unsigned char buf[4] __attribute__((aligned(4))) = {0, 0, 0, 0};
  union u n = {.b = {1, 2, 3, 4}};
  *(union u *)buf = n;
  __ESBMC_assert(buf[3] == 4, "last byte");
  return 0;
}
