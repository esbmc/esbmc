#define STACK_LIM 160
int main() {
  int y = 5;  // 4 * 8 = 32
  int *yPtr;  // 8 * 8 = 64
  // Expected total here: 32+64= 96.
 
  yPtr = &y;

  // This assert is throwing error because of the latter
  // pointer variable xPtr
  __ESBMC_assert(__ESBMC_stack_size() <= STACK_LIM, "ERROR");
 
  int *xPtr; // 8 * 8 = 64
  // The total now is 96 + 64 = 160.
  return 0;
}
