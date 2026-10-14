int a, b;
int main() {
  __CPROVER_assert(__CPROVER_POINTER_OBJECT(&a) == __CPROVER_POINTER_OBJECT(&a), "same object");
  __CPROVER_assert(__CPROVER_POINTER_OBJECT(&a) != __CPROVER_POINTER_OBJECT(&b), "distinct objects");
  return 0;
}
