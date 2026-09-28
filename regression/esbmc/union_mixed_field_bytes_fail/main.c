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

  __ESBMC_assert(u.b == 1, "the narrow field must not survive the wide write");
  return 0;
}
