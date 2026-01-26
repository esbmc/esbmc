/* Basic11: Complex logical expressions with && and ||
 * Tests nested logical operators in ensures clauses
 */
#include <assert.h>

int in_range(int x, int min, int max)
{


  __ESBMC_requires(min <= max);
  __ESBMC_ensures(__ESBMC_return_value >= min && __ESBMC_return_value <= max);

  if (x < min)
    return min;
  if (x > max)
    return max;
  return x;
}

int is_valid_score(int score)
{
  __ESBMC_ensures(__ESBMC_return_value == 0 || __ESBMC_return_value == 1);
  __ESBMC_ensures(__ESBMC_return_value == 1 || (score < 0 || score > 100));
  if (score >= 0 && score <= 100)
    return 1;
  return 0;
}

int main()
{



  int result1 = in_range(15, 0, 10);
  assert(result1 >= 0 && result1 <= 10);
  
  int result2 = is_valid_score(50);
  assert(result2 == 1);
  
  return 0;
}

