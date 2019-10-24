unsigned char nondet_uchar();
unsigned short nondet_ushort();

int main(void) {
  unsigned short size = nondet_ushort();
  __ESBMC_assume(size > 0);
  unsigned char a[size], k;
  unsigned short i=0, j=size-1, m, n;
  _Bool found = 0;

  for(i = 0; i < size; ++i)
    a[i] = nondet_uchar();
  k = nondet_uchar();

  while(i <= j) {
    if(a[i]*a[j] == k) {
      found = 1;
      m=i;
      n=j;
      break;
    }
    else if(a[i]*a[j] < k) {
      ++i;
      assert(j>i);
    }
    else {
      --j;
      assert(i>j);
    }
  }

  if(found)
    assert(k == a[m] * a[n]);

  return 0;
}
