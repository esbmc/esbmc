// VLAs are documented out of scope. The pass must not allocate a
// `bool[<runtime expr>]` shadow for `a` (which would mis-encode), nor
// emit indexed asserts against it. Reads of `a[0]` after the write must
// verify SUCCESSFUL with --uninitialised-vars-check enabled.
int main(int argc, char **argv)
{
  int n = argc;
  int a[n]; // VLA: size known only at runtime
  a[0] = 42;
  return a[0];
}
