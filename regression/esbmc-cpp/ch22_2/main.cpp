// Fig 22.8: fig22_08.cpp
// Driver for simple class Array.
#include <iostream>

using std::cout;

#include "array.h"

void outputArray( const Array & );

int main()
{
   Array integers1( 7 );

   outputArray( integers1 );   // output Array integers1

   outputArray( 15 );  // convert 15 to an Array and output  

   return 0;

}  // end main

// print array contents
void outputArray( const Array &arrayToOutput )
{
   cout << "The array received contains:\n" 
        << arrayToOutput << "\n\n";

}  // end outputArray

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

