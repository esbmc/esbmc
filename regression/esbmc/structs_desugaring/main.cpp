#include <assert.h>

struct Counter {
 // int &ref;
  int *ref_ptr;	
  Counter(int &r) : ref_ptr(&r) {}
  void increment() {(*ref_ptr)++; }
};

int main() {
   int x = 0;
   Counter c(x);
   c.increment();
  // int &y = c.ref;
  /// y++;
   //c.ref++;
   (*c.ref_ptr)++;
   assert(x == 2);
   return 0;
} 
