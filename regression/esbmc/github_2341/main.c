#include <stdlib.h>
#include <assert.h>

int main() {
  int *ptr = NULL;
  int *temp = (int*)realloc(ptr, 10 * sizeof(int)); 

  if (temp != NULL) { 
      ptr = temp;
      ptr[0] = 42;
      ptr[9] = 99;
  }

  free(ptr);
  return 0;
} 
