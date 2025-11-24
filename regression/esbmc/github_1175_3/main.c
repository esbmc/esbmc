int main()
{
  int * const ptr = (int*)200;  // const pointer to invalid address
  int v = *ptr;  // Should fail: dereference failure
  return 0;
}
