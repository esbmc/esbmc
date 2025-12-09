#include <cstdlib>

int floor_sqrt(int n) {
  int res = 1;
  while (res * res <= n) {
    res++;
  }
  return res - 1;
}

int* int_roots(int n) {
  if (n <= 0) return NULL;
  int *arr = (int *) malloc(n * sizeof(int));
  if (arr == NULL) return NULL;
  for (int i = 0; i < n; i++) {
    arr[i] = -1;
  }
  int frn = floor_sqrt(n);
  for (int i = 0; i <= frn; i++) {
    arr[i*i] = i;
  }
  return arr;
}

int main()
{
  int *x, number = nondet_int();
  x = int_roots(number);
  return 0;
}
