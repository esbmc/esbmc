// Fig. 18.11: fig18_11.cpp
// Using the bitwise shift operators.
#include <iostream>

using std::cout;
using std::cin;
using std::endl;

#include <iomanip>

using std::setw;

void displayBits( unsigned );  // prototype

int main()
{
   unsigned number1 = 960;

   // demonstrate bitwise left shift
   cout << "The result of left shifting\n";
   displayBits( number1 );
   cout << "8 bit positions using the left "
        << "shift operator is\n";
   displayBits( number1 << 8 );

   // demonstrate bitwise right shift
   cout << "\nThe result of right shifting\n";
   displayBits( number1 );
   cout << "8 bit positions using the right "
        << "shift operator is\n";
   displayBits( number1 >> 8 );

   return 0;

} // end main

// display bits of an unsigned integer value
void displayBits( unsigned value )
{
   const int SHIFT = 8 * sizeof( unsigned ) - 1;
   const unsigned MASK = 1 << SHIFT;

   cout << setw( 10 ) << value << " = ";

   for ( unsigned i = 1; i <= SHIFT + 1; i++ ) {
      cout << ( value & MASK ? '1' : '0' );
      value <<= 1;  // shift value left by 1

      if ( i % 8 == 0 )  // output a space after 8 bits
         cout << ' ';

   } // end for

   cout << endl;

} // end function displayBits

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