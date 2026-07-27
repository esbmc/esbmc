/* GitHub discussion #6123: quantifiers reached through a helper function's
 * return statement, with pure-function calls in the quantifier bodies
 * (including a nested quantifier whose body also contains a call). */
#include <stdbool.h>

#define SIZE 4

bool in_range(int lo, int i, int hi)
{
  return lo <= i && i < SIZE && i < hi;
}

bool all_lt(int vec[SIZE], int hi)
{
  int i;
  return __ESBMC_forall(&i, !in_range(0, i, SIZE) || vec[i] < hi);
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
  int vec[SIZE] = {0, 1, 2, 3};

  __ESBMC_assert(all_lt(vec, SIZE), "all elements below SIZE");
  __ESBMC_assert(all_unique(vec, SIZE), "all elements unique");
}
