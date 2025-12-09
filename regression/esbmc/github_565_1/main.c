#include <assert.h>  // Include the assert library

int a;

int main() {
  // Assertion to ensure that 'a' has a valid size for the array
  assert(a >= 0);  // Check that 'a' is positive, since the array size must be positive

  char c[a];
  
  // Optional: Check the size of the array
  assert(sizeof(c) == a * sizeof(char));  // Assert that the size of the array is 'a' times the size of a char

  (void)sizeof(c);  // The (void) sizeof expression is just to silence unused variable warnings
}

