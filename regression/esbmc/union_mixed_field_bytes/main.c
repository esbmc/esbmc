union u
{
  int a;
  short b;
};

int main(void)
{
  union u u;
  u.b = 1;
  u.a = 0x12345678;

  __ESBMC_assert(u.b == 0x5678, "the narrow field reads the wide write's bytes");
  return 0;
}
