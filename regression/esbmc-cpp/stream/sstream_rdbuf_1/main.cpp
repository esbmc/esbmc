// stringstream::rdbuf
#include <iostream>
#include <sstream>
using namespace std;

int main () {
  stringbuf *pbuf;
  stringstream ss;

  pbuf=ss.rdbuf();
  pbuf->sputn ("Sample string",13);
  cout << pbuf->str();

  return 0;
}
