/* The loop stores a 1 in row 0, so the sum is positive. */
long n, sum;
int row, col, done;
short idx, fixed_row;
void abort();
void assume(int cond) {
  if (!cond)
    abort();
}
void check(int cond) {
  if (!cond)
  ERROR: abort();
}
int nondet_int();
short nondet_short();
int main() {
  n = nondet_short();
  assume(n > 0);
  int array[n][n];
  for (; row < n; row++) {
    col = 0;
    for (; col < n; col++)
      array[row][col] = 0;
  }
  while (1) {
    idx = nondet_short();
    assume(idx >= 0 && idx < n);
    array[fixed_row][idx] = 1;
    done = nondet_int();
    if (done)
      break;
  }
  row = 0;
  for (; row < n; row++) {
    col = 0;
    for (; col < n; col++)
      sum = sum + array[row][col];
  }
  check(sum == 0);
}
