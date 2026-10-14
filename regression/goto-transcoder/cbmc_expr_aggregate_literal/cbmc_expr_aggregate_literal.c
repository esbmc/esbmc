struct S { int a; int b; };
int main() {
  struct S s = { 1, 2 };
  int arr[4] = { 7, 7, 7, 7 };
  __CPROVER_assert(s.a == 1 && s.b == 2, "struct literal");
  __CPROVER_assert(arr[0] == 7 && arr[3] == 7, "array_of literal");
  return 0;
}
