char nondet_char();

void init_loop(void *base, unsigned int size) {
  for (int i = 0; i < size; i++) {
    *((char *)base + i) = nondet_char();
  }
}

typedef struct my_obj {
  int foo;
  void *bar;
} obj;

int some_var;

int main() {
  obj bug;
  init_loop(&bug, sizeof(bug));
  __ESBMC_assume(bug.bar == &some_var);
  __ESBMC_assert(0, "this should be reachable");
}
