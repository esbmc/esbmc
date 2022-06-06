typedef struct {
  int v; // FAM needs at least one variable
  int arr[]; // Array of size 0
} FAM;

int main() {
  FAM F = {1, {}};
  F.arr[2000] = 7; // out-of-bounds
}