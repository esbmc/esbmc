struct RANGE {
  int value;
  int lbl;
};

typedef struct {
  struct RANGE *a[2];
  struct RANGE *b[2];
  struct RANGE *free[]; // int *free[1]
} FAM;

struct RANGE array1[2];
struct RANGE array2[2];
struct RANGE end = {-1, -1};
struct RANGE other = {0, 0};

FAM fam = {
  {
    &array1[0]
  }
  ,
  {
    &array2[0]
  }
  ,
  {
    &other,
    &end
  }
};


void FamInit()
{
  unsigned i;
  struct RANGE **p = fam.a;

  p[0] = &array1[0];
  p[1] = &array1[1];

  p = &fam.a[2]; // array2
  p[0] = &array2[0];
  p[1] = &array2[1];
}


#include <assert.h>
int main() {
  FamInit();
  struct RANGE **pra = fam.a;
  for(int i = 0; i < 100; i++)
  {
    pra[i+1]->lbl;
  }
}
