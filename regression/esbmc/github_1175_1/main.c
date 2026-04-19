int main()
{
  const int addr = 100;
  int *ptr = (int*)addr;  // Cast const int to pointer
  int v = *ptr;  // Should fail: dereference failure
  return 0;
}
