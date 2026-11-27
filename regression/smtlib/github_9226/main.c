/* github #9226: the address of an element or member is offset in bytes, with
 * no bit offset divided by 8. */
unsigned nondet_uint(void);
int a[10];
struct s
{
  char c;
  int m[3];
} st[4];

int main(void)
{
  unsigned i = nondet_uint();
  __ESBMC_assume(i < 3);
  int *p = &a[i];
  int *q = &st[i].m[i];
  __ESBMC_assert(p != q, "");
  return 0;
}
