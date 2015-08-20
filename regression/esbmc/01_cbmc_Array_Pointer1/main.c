int a[4];
int *p;

int main() {
  int j;

  a[1] = 1;

  p = a;
  p++;
  *p = 1;

  j = a[1];

  assert(j == 1);
  
  // assignment with cast
  p=(int *)a;
  assert(*p==0);
}
