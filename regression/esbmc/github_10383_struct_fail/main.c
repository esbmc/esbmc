// #10383: VLAs of structs and of pointers with an empty initializer.
int nondet_int(void);
struct s
{
  int x;
  char *p;
};
int main(void)
{
  int n = nondet_int();
  __ESBMC_assume(n > 0 && n < 4);
  struct s a[n] = {};
  int *b[n] = {};
  __ESBMC_assert(a[n - 1].p != 0, "member set");
  __ESBMC_assert(b[0] != 0, "element set");
  return 0;
}
