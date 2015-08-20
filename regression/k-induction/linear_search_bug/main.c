//#define SIZE 3
unsigned int nondet_uint();
unsigned int  SIZE=nondet_uint()/2+1;
int linear_search(int *a, int n, int q) {
  unsigned int j=0;
  while (j<n && a[j]!=q) {
  j++;
  if (j==20) j=-1;
  }
  if (j<SIZE) return 1;
  else return 0;
}
int main() { 
  int a[SIZE];
  a[SIZE/2]=3;
  assert(linear_search(a,SIZE,3));
}
