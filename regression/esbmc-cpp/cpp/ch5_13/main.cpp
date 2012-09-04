// Fig. 5.16: fig05_16.cpp
// Sizeof operator when used on an array name
// returns the number of bytes in the array.
#include <iostream>

using std::cout;
using std::endl;

size_t getSize( double * );  // prototype

int main()
{
   double array[ 20 ];

   cout << "The number of bytes in the array is "
        << sizeof( array );

   cout << "\nThe number of bytes returned by getSize is "
        << getSize( array ) << endl;

   return 0;  // indicates successful termination

} // end main

// return size of ptr
size_t getSize( double *ptr )
{
   return sizeof( ptr );

} // end function getSize

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
