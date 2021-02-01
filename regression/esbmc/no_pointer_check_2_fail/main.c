// Check if equality between heap and stack is correct
int main(void) {
  int* p = (int*) malloc(sizeof(int)); // Null or unique heap addr
  int q; // unique stack addr

  if(p != &q) __ESBMC_assert(0,"p should be different to &q");
  return 0;
}