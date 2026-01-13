/* Basic26: Pointer return type with __ESBMC_return_value
 * Tests that __ESBMC_return_value works with pointer return types
 */
#include <assert.h>

int* get_pointer(int* arr, int index)
{
  __ESBMC_requires(arr != 0);
  __ESBMC_requires(index >= 0);
  __ESBMC_requires(index < 10);
  // Test __ESBMC_return_value with pointer return type
  __ESBMC_ensures(__ESBMC_return_value != 0);
  __ESBMC_ensures(__ESBMC_return_value == arr + index);
  return arr + index;
}

int* find_value(int* arr, int size, int value)
{
  __ESBMC_requires(arr != 0);
  __ESBMC_requires(size > 0);
  __ESBMC_requires(size <= 10);
  // Return pointer to found element, or NULL if not found
  // Simplified ensures: just check that return value is not null or is null
  __ESBMC_ensures(__ESBMC_return_value == 0 || __ESBMC_return_value != 0);
  for (int i = 0; i < size; i++)
  {
    if (arr[i] == value)
      return arr + i;
  }
  return 0;
}

int main()
{
  int array[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  
  // Test 1: Pointer return type
  int* ptr = get_pointer(array, 5);
  assert(ptr == array + 5);
  assert(ptr != 0);
  assert(*ptr == 5);
  
  // Test 2: Pointer return with search
  int* found = find_value(array, 10, 7);
  assert(found != 0);
  assert(*found == 7);
  
  int* not_found = find_value(array, 10, 99);
  assert(not_found == 0);
  
  return 0;
}

