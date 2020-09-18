// Check if equality between heap and heap is correct
int main(void) {
  int* p = (int*) malloc(sizeof(int)); // Null or unique heap addr
  int* q = (int*) malloc(sizeof(int)); // Null or unique heap addr

  // Force-malloc-success
  if(p == q) __ESBMC_assert(0,"p shouldn't be equal to q with malloc success");
  return 0;
}