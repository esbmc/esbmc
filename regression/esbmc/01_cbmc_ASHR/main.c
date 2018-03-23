int main() {
  int x=-4,y;
  // should succeed
  y=x>>1;
  x>>=1;
  assert(x==-2);
  assert(y==-2);
  
  // should also work with mixed types
  assert(((-2)>>1u)==-1);
}
