const int GLOBAL_CONST = 0x404040;

int main()
{
  int *ptr = (int*)GLOBAL_CONST;  // Use global const as address
  int v = *ptr;  // Should fail: dereference failure
  return 0;
}
