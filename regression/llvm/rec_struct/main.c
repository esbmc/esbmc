//#include <assert.h>
//#include <stdio.h>

struct node {
  int blah;
  struct node *next;
  int bleh;
};

struct node b = {
  .blah = 2,
  .next = 0,
  .bleh = 3
};

// just using designated initialization
struct node a = {
  .blah = 1,
  .next = &b,
  .bleh = 2
};

struct node c = {
  .blah = 3,
  .next = 0,
  .bleh = 4
};

int main()
{
  c.next = &b;
  return(0);
}
