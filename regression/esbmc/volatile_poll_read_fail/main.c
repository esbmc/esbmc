typedef struct
{
  int value;
} NvBool;

NvBool flag;

int main(void)
{
  volatile NvBool *pointer = &flag;
  if (pointer->value != 0)
    __ESBMC_assert(0, "a volatile poll can observe a change");
}
