#include <iostream>
#include <new>
using namespace std;

struct myclass {
  myclass() {cout <<"myclass constructed\n";}
  ~myclass() {cout <<"myclass destroyed\n";}
};

void my_delete(myclass *q)
{
  delete[] q;
}

int main () {
  myclass * pt;

  pt = new myclass[3];
  delete[] pt;

  my_delete(pt);

  return 0;
}


