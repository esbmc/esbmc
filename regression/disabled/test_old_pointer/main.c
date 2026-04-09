/* Test __ESBMC_old with pointer/array operations - should PASS
 * Similar to basic26 but tests __ESBMC_old with array modification
 */

void increment_array_element(int *arr, int index, int increment)
{
  __ESBMC_requires(arr != 0);
  __ESBMC_requires(index >= 0 && index < 10);
  __ESBMC_requires(increment > 0);
  __ESBMC_ensures(arr[index] == __ESBMC_old(arr[index]) + increment);
  
  arr[index] += increment;  // Correct implementation
}

int main()
{
  int array[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  
  // Should increment array[5] by 10 (from 5 to 15)
  increment_array_element(array, 5, 10);
  
  return 0;
}

