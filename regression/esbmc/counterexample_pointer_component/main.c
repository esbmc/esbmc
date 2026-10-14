// The holding counterpart of counterexample_pointer_component_fail.
int x = 5;
int *arr[2];

int main(void)
{
  arr[0] = &x;
  arr[1] = 0;
  __ESBMC_assert(arr[0] == &x && arr[1] == 0, "elements are not as written");
  return 0;
}
