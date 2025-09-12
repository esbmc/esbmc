int main()
{
  const int base = 50;
  const int offset = 40;
  int *ptr = (int*)(base + offset);  // const expression = 90
  int v = *ptr;  // Should fail: dereference failure
  return 0;
}
