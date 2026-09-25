/* A variable-length array as a struct member is a GCC extension that clang
   declines to implement -- its own diagnostic says the extension "will never
   be supported" -- so ESBMC, whose frontend is clang, cannot parse it either.

   That is a limitation rather than a defect, but the failure mode still
   matters: it must stay a clean parse diagnostic and never become a crash.
   This pins that (issue #4086). */
void f(int n)
{
  struct S
  {
    int arr[n];
  };
  struct S s;
  s.arr[0] = 42;
}

int main(void)
{
  f(3);
  return 0;
}
