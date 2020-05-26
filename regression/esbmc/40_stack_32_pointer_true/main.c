int main() {
  int y = 5;  // 4 * 8 = 32
  int *yPtr;  // 32
  // Expected total here: 32+32= 64.

  yPtr = &y;

  int *xPtr; // 32
  // The total now is 64 + 32 = 96.
  return 0;
}
