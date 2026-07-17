/* GitHub discussion #6123 (negative): the quantifier behind the helper's
 * return must stay falsifiable. vec contains a duplicate, so uniqueness
 * fails; a change that makes the quantifier vacuously true (the other wrong
 * verdict PR #6105 fixed) would turn this into a spurious SUCCESSFUL. */
#include <stdbool.h>

#define SIZE 4

bool in_range(int lo, int i, int hi)
{
  return lo <= i && i < SIZE && i < hi;
}

bool all_unique(int vec[SIZE], int hi)
{
  int i, j;
  return __ESBMC_forall(
    &i,
    !in_range(0, i, hi) ||
      __ESBMC_forall(&j, !in_range(0, j, hi) || !(i != j) || vec[i] != vec[j]));
}

int main()
{
  int vec[SIZE] = {0, 1, 2, 2};

  __ESBMC_assert(all_unique(vec, SIZE), "all elements unique");
}
