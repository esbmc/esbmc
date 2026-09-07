int a, b;
int main() {
  __CPROVER_assert(__CPROVER_POINTER_OBJECT(&a) == __CPROVER_POINTER_OBJECT(&a), "same object");
  __CPROVER_assert(__CPROVER_POINTER_OBJECT(&a) == __CPROVER_POINTER_OBJECT(&b), "wrongly same object");
  return 0;
}
