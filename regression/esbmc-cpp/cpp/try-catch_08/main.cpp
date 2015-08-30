#include <iostream>
using std::cout;

int main()
{
  double *ptr[ 50 ];

  for ( int i = 0; i < 50; i++ ) {
    ptr[ i ] = new double[ 50000000000 ];

    if ( ptr[ i ] == 0 ) {
      cout << "Memory allocation failed for ptr[ "
          << i << " ]\n";
      break;
    }
    else
      cout << "Allocated 50000000000 doubles in ptr[ "
      << i << " ]\n";
  }
  return 0;
}
