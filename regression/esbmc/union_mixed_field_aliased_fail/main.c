union u
{
  int a;
  short b;
};

int main(void)
{
  union u u;
  u.a = 4;
  u.b = 1;

  __ESBMC_assert(u.a == 4, "a read aliased by a later sibling write must not fold");
  return 0;
}
