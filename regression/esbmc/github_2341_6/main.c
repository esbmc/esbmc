#include <stdlib.h>
#include <assert.h>

int main() {
  int *ptr = (int*)malloc(5 * sizeof(int));
  if (ptr == NULL)
    return -1;
  assert(ptr != NULL);

  ptr[0] = 10;
  ptr[4] = 50;

  // Expanding memory
  int *temp = (int*)realloc(ptr, 10 * sizeof(int));
  assert(temp != NULL); // Ensure successful reallocation

  ptr = temp;
  assert(ptr[0] == 10); // Data should be preserved
  assert(ptr[4] == 50);

  free(ptr);
  return 0;
}

