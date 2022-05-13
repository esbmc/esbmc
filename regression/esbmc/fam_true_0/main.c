#include <stdlib.h>
#include <assert.h>
typedef struct {
  int v; // FAM needs at least one variable
  int arr[]; // Array of size 0
} FAM;

int main() {
  FAM F = {1, {}};
  assert(F.arr != NULL);
}