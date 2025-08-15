int main()
{
  const char c = 'A';  // ASCII value 65
  int *ptr = (int*)(unsigned long)c;  // Use const char as address
  int v = *ptr;  // Should fail: dereference failure
  return 0;
}
