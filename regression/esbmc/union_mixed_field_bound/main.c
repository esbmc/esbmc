union u
{
  int a;
  short b;
};

int main(void)
{
  union u u;
  u.b = 1;
  u.a = 4;

  int s = 0;
  for (int i = 0; i < u.a; i++)
    s++;

  __ESBMC_assert(s == 4, "the loop reads the last write, across a sibling field");
  return 0;
}
