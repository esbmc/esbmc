#include <assert.h>
#include <stdbool.h>

typedef struct {
    bool is_float;
    struct {
       int a;
       short b;
       union {
         int c;
         short d;
         struct {
           int e;
           short f;
         };
       };
       struct {
         float g;
       };
    };
} mychoice_t;

int as_float(mychoice_t* ch) 
{ 
   if (ch->is_float) return ch->a;
   else return ch->b;
} 

int main()
{
  mychoice_t t = {true, 2, 0, 2};
  assert(t.a == 2);
  assert(t.b == 0);
  assert(t.c == 2);
  assert(t.d == 2);
  assert(t.e == 2);
  assert(t.g == 0);

  return 0;
}

