
typedef struct {
  int v; // FAM needs at least one variable
  int q;
  double arr[]; // Array of size 0
} FAM;


main() {
  FAM *ptr = (FAM*) malloc(sizeof(FAM));
  ptr->arr[2] = 42; // out-of-bounds (0 sized array)
  free(ptr);
}
