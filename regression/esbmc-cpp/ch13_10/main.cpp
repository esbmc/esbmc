// Fig. 13.4: fig13_04.cpp
// Demonstrating pre-standard new returning 0 when memory 
// is not allocated.
#include <iostream>
#include <cassert>

using std::cout;

int main()
{
   double *ptr[ 50 ];

   // allocate memory for ptr
   for ( int i = 0; i < 100; i++ ) {
      assert(i<50);
      ptr[ i ] = new double[ 5000000 ];

      // new returns 0 on failure to allocate memory
      if ( ptr[ i ] == 0 ) {
         cout << "Memory allocation failed for ptr[ " 
              << i << " ]\n";
         break;

      } // end if

      // successful memory allocation
      else
         cout << "Allocated 5000000 doubles in ptr[ " 
              << i << " ]\n";

   } // end for

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
