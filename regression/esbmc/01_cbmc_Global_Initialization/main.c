int x = 123;
int y;

// should also work through a typedef
typedef unsigned char uchar;
uchar b[] = "abc";

// addresses are constants
int *p=&y;

int main()
{
  assert(x == 123);
  assert(y == 0);
  assert(b[0]=='a');
  
  assert(p==&y);
}
