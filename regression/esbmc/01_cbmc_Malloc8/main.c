void *malloc(unsigned size);
void free(void *p);

int main() {
  int *p;
  unsigned int n;

  p=malloc(sizeof(int)*10);
  __ESBMC_assume(p);

  free(p);

  // bad!
  free(p);  
}
