# 1 "main.c"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "main.c"

int main() {
  int *p;
  unsigned int n;

  p=malloc(sizeof(int)*10);

  free(p);


  p[1]=1;
}
