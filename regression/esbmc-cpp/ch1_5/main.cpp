// Fig. 1.14: fig01_14.cpp
// Using if statements, relational
// operators, and equality operators.
#include <iostream>

using std::cout;  // program uses cout
using std::cin;   // program uses cin
using std::endl;  // program uses endl

// function main begins program execution
int main()
{
   int num1;  // first number to be read from user
   int num2;  // second number to be read from user

   cout << "Enter two integers, and I will tell you\n"
        << "the relationships they satisfy: ";
   cin >> num1 >> num2;   // read two integers

   if ( num1 == num2 )
      cout << num1 << " is equal to " << num2 << endl;

   if ( num1 != num2 )
      cout << num1 << " is not equal to " << num2 << endl;

   if ( num1 < num2 )
      cout << num1 << " is less than " << num2 << endl;

   if ( num1 > num2 )
      cout << num1 << " is greater than " << num2 << endl;

   if ( num1 <= num2 )
      cout << num1 << " is less than or equal to "
           << num2 << endl;

   if ( num1 >= num2 )
      cout << num1 << " is greater than or equal to "
           << num2 << endl;

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
