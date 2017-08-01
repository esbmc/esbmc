#define balh 1

struct record
{
  _Bool blah;
};

int main()
{
  unsigned int SIZE = 10;
  int a[SIZE][SIZE][SIZE]; 
  a[SIZE / 2][SIZE-1][SIZE-1] = 1;
  assert(a[SIZE / 2][SIZE-1][SIZE-1] == 1);

  struct record a1[SIZE][SIZE][SIZE];
  a1[SIZE / 2][SIZE-1][SIZE-1].blah = 1;
  assert(a1[SIZE / 2][SIZE-1][SIZE-1].blah == 1);
}

