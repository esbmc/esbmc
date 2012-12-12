#include <exception>
using std::exception;

#include <cassert>

void throwException()
{
  try {
    throw exception();
  }
  catch ( exception() ) {
    try {
      throw;
    }
    catch ( exception() ) {
      try {
        throw;
      }
      catch ( exception() ) {
        try {
          throw;
        }
        catch ( exception() ) {
          try {
            throw;
          }
          catch ( exception() ) {
            throw;
          }
        }
      }
    }
  }
}

int main()
{
  try {
    throwException();
  }
  catch ( exception &caughtException ) {
    assert(0);
  }
  return 0;
}
