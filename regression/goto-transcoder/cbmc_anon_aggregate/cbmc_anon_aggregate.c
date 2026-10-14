/* A nested anonymous union whose layout CBMC encodes in the type-name
   (tag-#anon#...) rather than a separate type symbol. ESBMC cannot yet parse
   that encoding (roadmap §4.3); it must report a clean error and exit, never
   abort() the process. */
struct S {
  union {
    struct { int a, b; };
    int arr[2];
  };
};

int main(void)
{
  struct S s;
  s.a = 1;
  __CPROVER_assert(s.arr[0] == 1, "anon overlay");
  return 0;
}
