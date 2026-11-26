/* #9226: the byte offset of &st[i].arr[j] still tells elements apart. */
unsigned nondet_uint(void);
struct s
{
  int x;
  int arr[4];
} st[3];

int main(void)
{
  unsigned i = nondet_uint(), j = nondet_uint();
  __ESBMC_assume(i < 3 && j < 4);
  int *q = &st[i].arr[j];
  __ESBMC_assert((char *)q - (char *)st != 44, "st[2].arr[2]");
  return 0;
}
