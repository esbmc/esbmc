enum MemoryAddresses {
  INVALID_ADDR = 0x12345678
};

int main()
{
  const enum MemoryAddresses addr = INVALID_ADDR;
  int *ptr = (int*)addr;  // Use const enum as address
  int v = *ptr;  // Should fail: dereference failure
  return 0;
}
