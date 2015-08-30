#include <exception>
using std::exception;

#include <cassert>

void throwException()
{
  try {
    throw exception();
  }
  catch ( exception &caughtException ) {
    int a = 2;
    assert(a != 2);
    throw;
  }
}

int main()
{
  try {
    throwException();
  }
  catch ( exception &caughtException ) {
  }
  return 0;
}
