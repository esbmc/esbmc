/* Anonymous struct/union members. CBMC serialises an anonymous aggregate's
   layout as a type symbol emitted *after* the struct that contains it, so the
   adapter must resolve the container's anonymous-tag member against that later
   symbol (via the re-check pass) rather than aborting on it. Resolving from
   CBMC's own symbol keeps the type byte-identical to the instruction side. */
struct Reg {
  union {
    struct { int lo, hi; };
    long full;
    int words[2];
  };
};

struct Nested {
  struct {
    union {
      struct { int a, b; };
      long packed;
    };
    int tag;
  };
};

struct WithPtr {
  union {
    int *p;
    long v;
  };
};

int main(void)
{
  struct Reg r;
  r.lo = 1;
  r.hi = 2;
  __CPROVER_assert(r.words[0] == 1 && r.words[1] == 2, "union/array overlay");

  struct Nested n;
  n.a = 5;
  n.b = 6;
  n.tag = 7;
  __CPROVER_assert(n.a + n.b == 11 && n.tag == 7, "double-nested anon");

  int y = 42;
  struct WithPtr w;
  w.p = &y;
  __CPROVER_assert(*w.p == 42, "anon union pointer member");
  return 0;
}
