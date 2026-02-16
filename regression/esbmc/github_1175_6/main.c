const int getAddress()
{
  return 0xDEADBEEF;
}

int main()
{
  int *ptr = (int*)getAddress();  // Use const function return as address
  int v = *ptr;  // Should fail: dereference failure
  return 0;
}
