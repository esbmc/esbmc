int nondet_int();
int main() {
  int i, j=nondet_int();
  i = 2147483640 ;
  i = i+j;
  return 0;
}
