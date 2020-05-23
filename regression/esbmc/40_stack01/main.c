#define STACK_LIM 70
int main() {
  int stack_size; // 4 * 8 = 32
  char ptr; // 1 * 8 = 8
  // Expected total here: 8+32= 40.
 
  // This assert is throwing error and it should. Because of the latter
  // variable breaks_stack_lim
  __ESBMC_assert(__ESBMC_stack_size() < STACK_LIM, "ERROR");
 
  int breaks_stack_lim; // 4 * 8 = 32
  // The total now is 40 + 32 = 72.
  return 0;
}
