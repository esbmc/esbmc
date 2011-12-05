// Fig. 1.6: fig01_06.cpp
// Addition program.
#include <iostream>

// function main begins program execution
int main()
{
   int integer1;  // first number to be input by user
   int integer2;  // second number to be input by user
   int sum;       // variable in which sum will be stored

   std::cout << "Enter first integer\n";  // prompt
   std::cin >> integer1;                  // read an integer

   std::cout << "Enter second integer\n"; // prompt
   std::cin >> integer2;                  // read an integer

   sum = integer1 + integer2;  // assignment result to sum

   std::cout << "Sum is " << sum << std::endl; // print sum

   return 0;   // indicate that program ended successfully

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
