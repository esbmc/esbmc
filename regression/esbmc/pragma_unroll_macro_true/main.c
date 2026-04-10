/* Test #pragma unroll with macro-defined N
 * Loop bound is 20, macro UNROLL_COUNT is 6
 * Demonstrates pragma works with macro constants
 */

#define UNROLL_COUNT 6

int main() {
  int sum = 0;

  #pragma unroll UNROLL_COUNT
  for (int i = 0; i < 20; i++) {
    sum += i;
  }

  // Only iterations 0-5 are explored due to pragma unroll 6
  return 0;
}
