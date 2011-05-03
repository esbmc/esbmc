void *malloc(unsigned size);
void free(void *p);

int main() {
  int *p;
  unsigned int n;

  p=malloc(sizeof(int)*2);

  free(p);
  
  free(p);
}
