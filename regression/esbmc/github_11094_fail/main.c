// Issue #11094: a pointer to a VLA with a zero bound is undefined behaviour.
int main(void)
{
  int arr[2];
  int n = 0;
  int (*p)[n] = &arr;
  return 0;
}
