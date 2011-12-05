void *malloc(unsigned size);
void free(void *p);

int main() {
  int *p;

  free(p);  
}
