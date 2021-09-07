#include <stdlib.h>
typedef struct {
  int v; // FAM needs at least one variable
  int arr[]; // Array of size 0
} FAM;

int main() {
  FAM F = {1, {}};
  __ESBMC_assert(F.arr != NULL, "FAM should be initialized properly");
}