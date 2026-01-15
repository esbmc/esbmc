/* Test that #pragma unroll N overrides --unwindset
 * --unwindset 2:2 would limit loop 2 to 2 iterations
 * But pragma unroll 8 should override it to 8 iterations
 */

int main() {
  int arr[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  int sum = 0;

  #pragma unroll 8
  for (int i = 0; i < 8; i++) {
    sum += arr[i];
  }

  __ESBMC_assert(sum == 28, "Sum should be 28");
  return 0;
}
