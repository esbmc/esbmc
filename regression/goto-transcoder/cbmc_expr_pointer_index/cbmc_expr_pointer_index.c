int main() {
  int a[4] = { 10, 20, 30, 40 };
  int *p = &a[2];
  __CPROVER_assert(*p == 30, "dereference through address_of + index");
  __CPROVER_assert(p[-1] == 20, "negative index off an interior pointer");
  return 0;
}
