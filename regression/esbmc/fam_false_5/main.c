
typedef struct {
  int v; // FAM needs at least one variable
  int arr[]; // Array of size 0
} FAM;

#include <stdlib.h>
main() {
  // 10 positions with arr of size 5
  FAM **matrix = (FAM**) malloc(sizeof(FAM*) * 10);
  for(int i = 0; i < 10; i++) {
    matrix[i] = (FAM*) malloc(sizeof(FAM) + sizeof(int)*5);
  }
  // Only address sanitizer detects this
  matrix[9]->arr[5] = 42;
  free(matrix);
}
