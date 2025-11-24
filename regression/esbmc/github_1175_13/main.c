int main()
{
  const int a = 10;
  const int b = 20;
  const int result = a * b * 100;  // = 20000
  int *ptr = (int*)result;  // Use const calculation as address
  int v = *ptr;  // Should fail: dereference failure
  return 0;
}
