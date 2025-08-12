#include <sstream>
#include <cassert>
#include <cmath> // for NAN, INFINITY
using namespace std;

int main () {

  {
    stringstream oss;
    float val = 70.23f;
    oss << val;
    assert(oss.str() == "70.23");
  }

  {
    stringstream oss;
    float val = -3.5f;
    oss << val;
    assert(oss.str() == "-3.5");
  }

  {
    stringstream oss;
    float val = NAN;
    oss << val;
    assert(oss.str() == "nan");
  }

  {
    stringstream oss;
    double val = -0.000123;
    oss << val;
    assert(oss.str().find('-') == 0);
  }

  return 0;
}

