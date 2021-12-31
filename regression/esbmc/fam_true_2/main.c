#include <assert.h>
#include <stdlib.h>
typedef struct {
  int v; // FAM needs at least one variable
  int arr[]; // Array of size 0
} FAM;


main() {
  FAM *ptr = (FAM*) malloc(sizeof(FAM));
  assert(ptr->arr != NULL);
  free(ptr);
}
