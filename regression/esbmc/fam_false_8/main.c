typedef struct {
  int *a[2];
  int *b[2];
  int *free[];
} FAM;

int array1[2];
int array2[2];
int end;
int other;

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


#include <assert.h>
int main() {
  int **pra = fam.a;
  pra[7]; // out-of-bounds
}
