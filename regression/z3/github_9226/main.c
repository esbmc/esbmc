/* #9226: the address of an element or a member is offset in bytes at the
 * width of size_t, with no bit offset divided by 8. */
unsigned nondet_uint(void);
int nondet_int(void);
struct s
{
  int x;
  int arr[4];
} st[3];
int a[10];

int main(void)
{
  unsigned i = nondet_uint(), j = nondet_uint();
  __ESBMC_assume(i < 3 && j < 4);
  __ESBMC_assert(&a[i] - a == i, "element");
  __ESBMC_assert(
    (char *)&st[i].arr[j] - (char *)st == 20 * i + 4 + 4 * j, "member");

  int k = nondet_int();
  __ESBMC_assume(k > -4 && k < 0);
  __ESBMC_assert((char *)&a[k] - (char *)a == 4 * k, "negative index");
  __ESBMC_assert((char *)&a[-1] - (char *)a == -4, "negative constant");
  return 0;
}
