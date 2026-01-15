/* Test #pragma unroll N with while loop
 * The pragma should limit unrolling to 5 iterations
 */

int main() {
  int arr[5] = {0, 1, 2, 3, 4};
  int sum = 0;
  int i = 0;

  #pragma unroll 5
  while (i < 5) {
    sum += arr[i];
    i++;
  }

  __ESBMC_assert(sum == 10, "Sum should be 10");
  return 0;
}
