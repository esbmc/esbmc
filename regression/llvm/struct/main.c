//#include <assert.h>

typedef struct {
    int is_float;
    struct {
       int a;
       short b;
    };
} mychoice_t;

int as_blah(mychoice_t* ch) 
{ 
   if (ch->is_float) return ch->a;
   else return ch->b;
} 

int main()
{
  mychoice_t t = {1, 2, 0, 2};
//  assert(t.a == 2);
//  assert(t.b == 0);

  int a = as_blah(&t);

  return 0;
}

