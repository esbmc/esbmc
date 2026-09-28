int main() {
  int a;
  int b = (int) &a;
  int *c = (int*) b;
  *c = 42;
  return 0;
}
