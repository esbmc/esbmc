#define STACK_LIM 64
int main() {
  char arr[5]; // alloca 5 * i8 = 40
  __ESBMC_assert(__ESBMC_stack_size() <= STACK_LIM, "ERROR");
  return 0;
}
