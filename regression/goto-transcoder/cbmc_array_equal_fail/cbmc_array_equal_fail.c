int main(void)
{
  int a[3] = {1, 2, 3};
  int b[3] = {1, 2, 9};
  __CPROVER_assert(__CPROVER_array_equal(a, b), "equal");
  return 0;
}
