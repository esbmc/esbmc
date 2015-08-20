// Fig. 18.7: fig18_07.cpp
// Using the bitwise AND, bitwise inclusive OR, bitwise
// exclusive OR and bitwise complement operators.
#include <iostream>

using std::cout;
using std::cin;

#include <iomanip>

using std::endl;
using std::setw;

void displayBits( unsigned );  // prototype

int main()
{
   unsigned number1;
   unsigned number2;
   unsigned mask;
   unsigned setBits;

   // demonstrate bitwise &
   number1 = 2179876355;
   mask = 1;
   cout << "The result of combining the following\n";
   displayBits( number1 );
   displayBits( mask );
   cout << "using the bitwise AND operator & is\n";
   displayBits( number1 & mask );

   // demonstrate bitwise |
   number1 = 15;
   setBits = 241;
   cout << "\nThe result of combining the following\n";
   displayBits( number1 );
   displayBits( setBits );
   cout << "using the bitwise inclusive OR operator | is\n";
   displayBits( number1 | setBits );

   // demonstrate bitwise exclusive OR
   number1 = 139;
   number2 = 199;
   cout << "\nThe result of combining the following\n";
   displayBits( number1 );
   displayBits( number2 );
   cout << "using the bitwise exclusive OR operator ^ is\n";
   displayBits( number1 ^ number2 );

   // demonstrate bitwise complement
   number1 = 21845;
   cout << "\nThe one's complement of\n";
   displayBits( number1 );
   cout << "is" << endl;
   displayBits( ~number1 );

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