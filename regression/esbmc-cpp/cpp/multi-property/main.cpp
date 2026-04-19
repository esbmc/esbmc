#include <cstdlib>

int* make_array(int n) {
  if (n <= 0) return NULL;
  int *arr = (int *) malloc(n * sizeof(int));
  for (int i = 0; i <= n; i++) {
    arr[i] = i;
  }
  return arr;
}

int main()
{
  make_array(nondet_int());
  return 0;
}
