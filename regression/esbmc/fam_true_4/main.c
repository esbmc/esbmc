
typedef struct {
  int v; // FAM needs at least one variable
  int arr[]; // Array of size 0
} FAM;


main() {
  FAM *ptr = (FAM*) malloc(sizeof(FAM) + sizeof(int)*3);
  FAM deref = *ptr;
  deref.v = ptr->arr[2]; // a copy holds no flexible array elements, the heap object does
  free(ptr);
}
