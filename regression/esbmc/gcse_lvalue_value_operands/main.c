int nondet_int();

// The pointer under a dereference and the index of an array store are values,
// so GCSE may still replace them inside an assignment target; storing to
// `a[k + 1]` does not invalidate `k + 1`.
int main()
{
  int arr[3] = {0, 0, 0};
  int k = nondet_int();
  int *p = arr;
  int x = *(p + k);
  *(p + k) = 5;

  int a[4] = {0, 0, 0, 0};
  int y = a[k + 1];
  a[k + 1] = 7;
  a[k + 1] = 8;
}
