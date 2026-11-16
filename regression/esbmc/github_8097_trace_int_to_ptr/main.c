/* #8097: u.s.a[0] reads back the pointer stored through u.p. */
int main()
{
  union
  {
    int *p;
    struct
    {
      int *a[1];
      char c[2];
    } s;
  } u;
  int x;
  u.p = &x;
  u.s.c[1] = 4;
  int *r = u.s.a[0];
  __ESBMC_assert(r == &x, "r aliases u.p");
}
