/* github #9226: the address of a member of an element reached through a
 * symbolic index is a byte offset, not a bit offset divided by 8. */
unsigned nondet_uint(void);
struct s
{
  char c;
  int x;
} st[4];

int main(void)
{
  unsigned i = nondet_uint();
  __ESBMC_assume(i < 4);
  int *p = &st[i].x;
  __ESBMC_assert(p != 0, "");
  return 0;
}
