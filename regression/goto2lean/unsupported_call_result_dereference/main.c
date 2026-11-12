unsigned f(void) { return 1; }
int main(void) {
  *(unsigned *)0 = f();
  return 0;
}
