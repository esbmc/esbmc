/* A const global read by a `requires`. The havoc added for #7356 skips const
 * globals, since a caller cannot have changed one. Measured: this reports an
 * array-bounds violation both before that change and with the havoc placed
 * after the is_fresh step, and verifies with the shipped ordering.
 */
const int N = 4;
int arr[4];

void f(int i)
{
  __ESBMC_requires(i >= 0 && i < N);
  __ESBMC_assigns(arr[i]);
  arr[i] = 1;
}

int main(void)
{
  return 0;
}
