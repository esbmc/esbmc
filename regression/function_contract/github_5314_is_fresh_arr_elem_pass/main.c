/* github #5314: under --enforce-contract, naming an array element a[k] as an
 * assigns target tripped a spurious "array bounds violated" because the Phase-2B
 * witness index was bounded by ARRAY_ALLOC_ELEMS (100) instead of the real
 * __ESBMC_is_fresh allocation (10 elements). a[0] / a[2] / *(a+2) all in bounds. */
__ESBMC_contract
void f(int *a)
{
  __ESBMC_requires(__ESBMC_is_fresh(a, 10 * sizeof(int)));
  __ESBMC_assigns(a[2]);
  a[2] = 0;
}
