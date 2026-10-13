/* The addresses of two differently named objects, declared or allocated,
 * are never the same object, so each claim below folds in the simplifier
 * and none reaches the solver. */
int a, b;
int arr[4];
struct pair
{
  int x, y;
} s;

int main()
{
  long *p = __ESBMC_alloca(sizeof(long));
  long *q = __ESBMC_alloca(sizeof(long));
  __ESBMC_assert(!__ESBMC_same_object(&a, &b), "two globals");
  __ESBMC_assert(!__ESBMC_same_object(&arr[1], &b), "element and global");
  __ESBMC_assert(!__ESBMC_same_object(&s.y, &arr[2]), "member and element");
  __ESBMC_assert(!__ESBMC_same_object(p, q), "two allocations");
  __ESBMC_assert(__ESBMC_same_object(&arr[0], &arr[3]), "one array");
  return 0;
}
