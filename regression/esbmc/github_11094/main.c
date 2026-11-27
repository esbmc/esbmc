// Issue #11094: a positive bound for a pointer to a VLA is accepted.
int main(void)
{
  int arr[2];
  int n = 2;
  int (*p)[n] = &arr;
  return 0;
}
