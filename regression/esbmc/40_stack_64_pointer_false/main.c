int main() {
  int y = 5;  // 4 * 8 = 32
  int *yPtr;  // 8 * 8 = 64
  // Expected total here: 32+64= 96.
 
  yPtr = &y;
 
  int *xPtr; // 8 * 8 = 64
  // The total now is 96 + 64 = 160.
  return 0;
}
