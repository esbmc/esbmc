/* Test #pragma unroll (without N) for unlimited unrolling
 * The pragma should override --unwind and unroll fully
 * Loop has 10 iterations, --unwind 3 should be ignored
 */

int main() {
  int arr[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int sum = 0;

  #pragma unroll
  for (int i = 0; i < 10; i++) {
    sum += arr[i];
  }

  __ESBMC_assert(sum == 45, "Sum should be 45");
  return 0;
}
