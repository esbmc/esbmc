#include <assert.h>
typedef struct {
  int v; // FAM needs at least one variable
  int arr[]; // Array of size 0
} FAM;

#include <stdlib.h>
main() {
  int value = 42;
  FAM *ptr = (FAM*) malloc(sizeof(FAM));
  ptr->v = value;
  FAM deref = *ptr;
  assert(deref.v == value); // out-of-bounds (0 sized array)
  free(ptr);
}
