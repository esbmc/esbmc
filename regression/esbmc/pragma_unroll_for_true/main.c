/* Test #pragma unroll N with for loop
 * The pragma should limit unrolling to 5 iterations
 * The loop naturally terminates at 5, so verification should succeed
 */

int main() {
  int arr[5] = {0, 1, 2, 3, 4};
  int sum = 0;

  #pragma unroll 5
  for (int i = 0; i < 5; i++) {
    sum += arr[i];
  }

  __ESBMC_assert(sum == 10, "Sum should be 10");
  return 0;
}
