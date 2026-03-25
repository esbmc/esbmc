int add(int a, int b) {
  return a + b;
}

int multiply(int x, int y) {
  return x * y;
}

int main() {
  int p = 1, q = 2, r = 3;
  int sum = add(p, q);
  int prod = multiply(sum, r);
  int result = add(prod, multiply(p, r));
  return result;
}
