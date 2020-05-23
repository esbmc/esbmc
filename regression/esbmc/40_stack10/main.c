#define STACK_LIM 64

struct MixedData
{
  char var1; // i8
  short var2; // i16
  int var3; // i32
};

// total: 56

struct MixedData a;

int main() {
  __ESBMC_assert(__ESBMC_stack_size() <= STACK_LIM, "ERROR");
  return 0;
}