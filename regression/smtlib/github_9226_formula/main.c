/* github #9226: the address of an array element or member is a byte offset,
 * not a bit offset divided by 8. */
unsigned nondet_uint(void);
struct s
{
  char c;
  int a[10];
};
int a[10];
struct s g;

int main(void)
{
  unsigned i = nondet_uint();
  __ESBMC_assume(i < 10);
  int *p = &a[i];
  int *q = &g.a[i];
  __ESBMC_assert(p != q, "");
  return 0;
}
