
typedef struct {
  int v; // FAM needs at least one variable
  int arr[]; // Array of size 0
} FAM;


main() {
  // 10 positions with arr of size 5
  FAM **matrix = (FAM**) malloc(sizeof(FAM*) * 10);
  for(int i = 0; i < 10; i++) {
    matrix[i] = (FAM*) malloc(sizeof(FAM) + sizeof(int)*5);
  }
  matrix[9]->arr[4] = 42;
  for(int i = 0; i < 10; i++) {
    free(matrix[i]);
  }
  free(matrix);
}
