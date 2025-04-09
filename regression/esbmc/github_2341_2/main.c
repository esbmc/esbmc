#include <stdlib.h>
#include <assert.h>

int main() {
  int *ptr = (int*)malloc(10 * sizeof(int));
  if (ptr == NULL)
    return -1;
  assert(ptr != NULL);

  ptr[0] = 100;
  ptr[9] = 200;

  int *temp = (int*)realloc(ptr, 5 * sizeof(int)); // Reduce size
  if (temp == NULL)
    return -1;
  assert(temp != NULL); // Ensure successful reallocation

  ptr = temp;
  assert(ptr[0] == 100); // Data should be preserved

  free(ptr);
  return 0;
}

