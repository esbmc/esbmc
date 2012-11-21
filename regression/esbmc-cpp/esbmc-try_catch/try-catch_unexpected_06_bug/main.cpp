#include<exception>
#include<cassert>
using namespace std;

void myunexpected()
{
  assert(0);
}

int main() throw(char)
{
  set_unexpected(myunexpected);
  throw 5;
  return 0;
}
