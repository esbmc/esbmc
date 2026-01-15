/* Test #pragma unroll N with for loop
 * Loop bound is 10, but pragma limits unrolling to 5
 * Demonstrates pragma actually constrains the unrolling
 */

int main() {
  int sum = 0;

  #pragma unroll 5
  for (int i = 0; i < 10; i++) {
    sum += i;
  }

  // Only iterations 0-4 are explored due to pragma
  // No assertions to fail, verification succeeds
  return 0;
}
