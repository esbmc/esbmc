// Ex. 2.24: ex02_24.cpp
// What does this program print?
#include <iostream>

using std::cout;
using std::endl;

// function main begins program execution
int main()
{
   int count = 1;            // initialize count

   while ( count <= 10 ) {   // loop 10 times

      // output line of text
      cout << ( count % 2 ? "****" : "++++++++" ) 
           << endl;
      ++count;               // increment count
   }

   return 0;   // indicate successful termination

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
