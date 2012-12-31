// Fig. 13.5: fig13_05.cpp
// Demonstrating standard new throwing bad_alloc when memory
// cannot be allocated.
#include <iostream>
using std::cout;
using std::endl;

#include <new> // standard operator new
using std::bad_alloc;

int main()
{
  double *ptr[ 50 ];

  // attempt to allocate memory
  try {
    // allocate memory for ptr[ i ]; new throws bad_alloc
    // on failure
    for ( int i = 0; i < 50; i++ ) {
      ptr[ i ] = new double[ 5000 ];
      cout << "Allocated 5000 doubles in ptr[ "
          << i << " ]\n";
    }
  } // end try
  // handle bad_alloc exception
  catch ( bad_alloc &memoryAllocationException ) {
    cout << "Exception occurred: "
        << memoryAllocationException.what() << endl;
  } // end catch

  return 0;
}  // end main

/**************************************************************************
 * (C) Copyright 1992-2003 by Deitel & Associates, Inc. and Prentice      *
 * Hall. All Rights Reserved.                                             *
 *                                                                        *
 * DISCLAIMER: The authors and publisher of this book have used their     *
 * best efforts in preparing the book. These efforts include the          *
 * development, research, and testing of the theories and programs        *
 * to determine their effectiveness. The authors and publisher make       *
 * no warranty of any kind, expressed or implied, with regard to these    *
 * programs or to the documentation contained in these books. The authors *
 * and publisher shall not be liable in any event for incidental or       *
 * consequential damages in connection with, or arising out of, the       *
 * furnishing, performance, or use of these programs.                     *
 *************************************************************************/
