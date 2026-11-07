/* #9937: the out-of-bounds pointer reaches the union write through a
 * merge of two branches. */
int nondet_int();

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
  int x;
  int c = nondet_int();
  if (c)
    u.p = arr + 100;
  else
    u.p = &x;
  u.s.c = 4;
  __ESBMC_assert(u.s.a == &x, "a is x");
}
