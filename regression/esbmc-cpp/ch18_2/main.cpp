// Exercise 18.20: ex18_20.cpp
#include <iostream>

using std::cout;
using std::cin;
using std::endl;
using std::boolalpha;

bool mystery( unsigned );

int main()
{
   unsigned x;

   cout << "Enter an integer: ";
   cin >> x;
   cout << boolalpha 
        << "The result is " << mystery( x ) << endl;

   return 0;

} // end main

// What does this function do?
bool mystery( unsigned bits )
{
   const int SHIFT = 8 * sizeof( unsigned ) - 1;
   const unsigned MASK = 1 << SHIFT;
   unsigned total = 0;

   for ( int i = 0; i < SHIFT + 1; i++, bits <<= 1 )

      if ( ( bits & MASK ) == MASK ) 
         ++total;

   return !( total % 2 );

} // end function mystery

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
