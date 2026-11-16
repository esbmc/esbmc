extern int __VERIFIER_nondet_int(void);
struct inner { unsigned int x, y, z; };
struct outer { unsigned int a; struct inner in; unsigned int arr[4]; unsigned int b; };
static struct outer o;
int main(void)
{
  o.in.y = 9; o.arr[2] = 9; o.b = 9;
  int i = __VERIFIER_nondet_int();
  if (i > 0)
    return 0;
  unsigned long base = (unsigned long)&o + i * sizeof(struct outer);
  *((unsigned int *)(base + 8)) = 1U;
  *((unsigned int *)(base + 24)) = 2U;
  __ESBMC_assume(i == 0);
  __ESBMC_assert(o.in.y == 1U, "store reaches the nested member");
  __ESBMC_assert(o.arr[2] == 2U, "store reaches the array member");
  __ESBMC_assert(o.b == 9U && o.a == 0U && o.in.x == 0U, "others untouched");
  return 0;
}
