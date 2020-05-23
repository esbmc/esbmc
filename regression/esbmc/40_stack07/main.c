#define STACK_LIM 64
int main() {
  char arr[10]; // alloca 10 * i8 = 80
  __ESBMC_assert(__ESBMC_stack_size() <= STACK_LIM, "ERROR");
  return 0;
}
