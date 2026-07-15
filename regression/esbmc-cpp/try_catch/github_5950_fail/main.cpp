#include <cstdlib>

// Negative guard for the #5950 fix.  The fix scopes temporary destructors to the
// substatement they are created in; it must NOT suppress destructors that are
// supposed to run.  Here ~Holder legitimately frees ptr at end of scope, so
// reading through the dangling copy afterwards is a genuine use-after-free that
// must still be reported.
struct Holder
{
  char *ptr;
  Holder() { ptr = (char *)malloc(1); }
  ~Holder() { free(ptr); }
};

int main()
{
  char *dangling;
  {
    Holder h;
    dangling = h.ptr;
  } // h destroyed here; ptr freed
  return *dangling; // use-after-free
}
