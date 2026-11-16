typedef struct
{
  int value;
} NvBool;

NvBool flag;

int main(void)
{
  volatile NvBool *pointer = &flag;
  int first = pointer->value;
  int second = pointer->value;
  __ESBMC_assert(first == second, "volatile pointer member reads may differ");
}
