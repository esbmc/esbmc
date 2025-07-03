#include <stdlib.h>
#include <assert.h>

int main() {
  int *ptr = (int*)malloc(10 * sizeof(int));
  if (ptr == NULL)
    return -1;
  assert(ptr != NULL);

  // Request an excessively large allocation to force failure
  int *temp = (int*)realloc(ptr, (size_t)-1);
  if (temp != NULL)
    return -1;
  assert(temp == NULL); // Ensure reallocation failed
  assert(ptr != NULL);  // Ensure original pointer is still valid

  free(ptr);
  return 0;
}

