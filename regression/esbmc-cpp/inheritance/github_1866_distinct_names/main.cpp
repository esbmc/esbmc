// esbmc/esbmc#1866: non-first base subobject under multiple inheritance.
// KNOWNBUG until the base-subobject rework (#3894) lands; correct result is
// VERIFICATION SUCCESSFUL (verified against clang++/g++). See
// docs/design/cpp-multiple-inheritance-subobjects.md
#include <cassert>
struct B { int e = 22; };
struct A { int b = 111; };
struct f : B, A {};
int main(){ f i=f(); assert(i.e==22); assert(i.b==111); return 0; }
