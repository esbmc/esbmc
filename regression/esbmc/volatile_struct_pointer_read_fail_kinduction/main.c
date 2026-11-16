typedef struct
{
  unsigned int value;
} NvBool;

volatile NvBool flag;

int main(void)
{
  volatile NvBool *pointer = &flag;
  unsigned int first = pointer->value;
  unsigned int second = pointer->value;
  __ESBMC_assert(first == second, "volatile struct pointer reads may differ");
}
