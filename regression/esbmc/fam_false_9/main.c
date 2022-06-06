typedef struct {
  int *a[2];
  int *b[2];
  int *free[]; // int *free[1]
} FAM;

int array1[2];
int array2[2];
int pos0;
int pos1;
int pos2;


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
		&pos0,
		&pos1,
		&pos2
	}
};

void FamInit()
{
	unsigned i;
	int **p = fam.a;

	p[0] = &array1[0];
        p[1] = &array1[1];

	p = &fam.a[2]; // array2
	p[0] = &array2[0];
	p[1] = &array2[1];
}


#include <assert.h>
int main() {

  assert(fam.a[0] == &array1[0]);
  assert(fam.a[1] == 0);
  assert(fam.b[0] == &array2[0]);
  assert(fam.b[1] == 0);
  assert(fam.free[0] == &pos0);
  assert(fam.free[1] == &pos1);
  assert(fam.free[2] == &pos2);


  FamInit();
  FAM *dst = (FAM*) malloc(sizeof(int*) * 7);
  memcpy(dst->free, fam.free, 0);
  /*
  int **pra = dst->a;
  //assert(pra[0] == &array1[0]);
  //assert(pra[1] == &array1[1]);
  //assert(pra[2] == &array2[0]);
  //assert(pra[3] == &array2[1]);
  assert(pra[4] == &pos0);
  assert(pra[5] == &pos2);
  */

  free(dst);
}
