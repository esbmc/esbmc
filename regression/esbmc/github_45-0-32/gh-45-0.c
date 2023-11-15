int linear_search(int *x, int n, int q)
{
  unsigned int j = 0;
  while (j < n && x[j] != q)
    j++;

  if (j < n)
    return 1;
  else
    return 0;
}

int main()
{
  unsigned int SIZE = nondet_uint() / 2 + 1;
  __VERIFIER_assume(SIZE < 2147483648);
  int a[SIZE];
  a[SIZE / 2] = 3;

#if 0
  assert(linear_search(a, SIZE, 3));
#else
  unsigned int j = 0;
  while (j < SIZE && a[j] != 3)
    j++;

  if (j < SIZE)
    return 1;

  assert(0);
#endif
}
