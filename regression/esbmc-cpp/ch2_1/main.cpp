// Ex. 2.5: ex02_05.cpp
// Calculate the sum of the integers from 1 to 10.
#include <iostream>

using std::cout;
using std::endl;

// function main begins program execution
int main()
{
   int sum;              // stores sum of integers 1 to 10
   int x;                // counter

   x = 1;                // count from 1
   sum = 0;              // initialize sum

   while ( x <= 10 ) { 
      sum += x;          // add x to sum
      ++x;               // increment x

   } // end while

   cout << "The sum is: " << sum << endl;

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
