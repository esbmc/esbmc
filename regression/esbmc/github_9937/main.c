/* #9937: writing a member of a union whose other member holds an
 * out-of-bounds pointer must not make the path vacuous. */
int main()
{
  union
  {
    int *p;
    struct
    {
      int *a;
      long c;
    } s;
  } u;
  int arr[2];
  u.p = arr + 100;
  u.s.c = 4;
  __ESBMC_assert(u.s.a == arr + 100 && u.s.c == 4, "kept");

  union
  {
    int *p;
    char b;
  } v;
  v.p = arr + 100;
  v.b = 1;
  __ESBMC_assert(v.p != arr + 100, "byte changed");
}
