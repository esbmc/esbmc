
typedef struct {
  int v; // FAM needs at least one variable
  int arr[]; // Array of size 0
} FAM;

main() {
  FAM *ptr = (FAM*) malloc(sizeof(FAM) + sizeof(int)*3);
  ptr->arr[2] = 42; // out-of-bounds
  free(ptr);
}
