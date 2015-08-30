// Ex. 2.15: ex02_15.cpp
// What does this program print?
#include <iostream>

using std::cout;
using std::endl;

// function main begins program execution
int main()
{
   int y;          // declare y
   int x = 1;      // initialize x
   int total = 0;  // initialize total

   while ( x <= 10 ) {    // loop 10 times
      y = x * x;          // perform calculation
      cout << y << endl;  // output result
      total += y;         // add y to total
      ++x;                // increment counter x

   } // end while

   cout << "Total is " << total << endl;  // display result

   return 0;  // indicate successful termination

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
