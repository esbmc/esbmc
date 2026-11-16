/* #8097: printing u.s.c[1] = 4 read c back from u's bytes, which hold &x;
 * decoding them as a pointer asserted side constraints and discarded the
 * model being read. */
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
  __ESBMC_assert(r != &x, "r aliases u.p");
}
