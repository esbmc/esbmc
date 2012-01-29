int main() {
  _Bool b1, b2, b3;
  
  b1=0;
  b1++;
  assert(b1);
  
  b2=1;
  b2+=10;
  assert(b2);
  
  b3=b1+b2;
  assert(b3==1);
}
