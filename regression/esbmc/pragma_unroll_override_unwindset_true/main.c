/* Test that #pragma unroll N overrides --unwindset
 * Loop bound is 20, pragma limits to 8, --unwindset tries to limit to 2
 * Pragma should win: unroll 8 times (not 2, not 20)
 */

int main() {
  int sum = 0;

  #pragma unroll 8
  for (int i = 0; i < 20; i++) {
    sum += i;
  }

  // Pragma unroll 8 overrides --unwindset 2:2
  // Only iterations 0-7 are explored
  return 0;
}
