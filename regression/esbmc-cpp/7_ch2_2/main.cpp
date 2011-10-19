// Ex. 2.8: ex02_08.cpp
// Raise x to the y power. 
#include <iostream>

using std::cout;
using std::cin;
using std::endl;

// function main begins program execution
int main()
{
   int x;      // base 
   int y;      // exponent
   int i;      // counts from 1 to y
   int power;  // used to calculate x raised to power y

   i = 1;      // initialize i to begin counting from 1
   power = 1;  // initialize power

   cout << "Enter base as an integer: ";  // prompt for base
   cin >> x;                              // input base

   // prompt for exponent
   cout << "Enter exponent as an integer: ";  
   cin >> y;                              // input exponent

   // count from 1 to y and multiply power by x each time
   while ( i <= y ) {
      power *= x;
      ++i;

   } // end while

   cout << power << endl;  // display result

   return 0;               // indicate successful termination

} // end function main



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
