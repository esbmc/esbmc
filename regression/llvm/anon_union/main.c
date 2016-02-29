//#include <stdbool.h>

typedef struct {
    _Bool is_float;
    union {
       float f;
       double s;
    };
} mychoice_t;

double as_float(mychoice_t* ch) 
{ 
   if (ch->is_float) return ch->f;
   else return ch->s;
} 

int main()
{
  mychoice_t t = {1, 2.0f};
  mychoice_t t1 = {0, 2.0f};

  double d = as_float(&t);
  double d1 = as_float(&t1);

  return 0;
}

