#define STACK_LIM 96
int main() {
  int y = 5;  // 4 * 8 = 32
  int *yPtr;  // 4 * 8 = 32
  // Expected total here: 32+32= 64.
 
  yPtr = &y;

  // This assert is throwing error because of the latter
  // pointer variable xPtr
  __ESBMC_assert(__ESBMC_stack_size() <= STACK_LIM, "ERROR");
 
  int *xPtr; // 4 * 8 = 32
  // The total now is 64 + 32 = 96.
  return 0;
}
