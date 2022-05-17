// Fig. 5.13: fig05_13.cpp
// break statement exiting a for statement.
#include <iostream>
using namespace std;

int main()
{
   unsigned int count; // control variable also used after loop terminates

   for ( count = 1; count <= 10; ++count ) // loop 10 times
   {
      if ( 5 == count ) // if count is 5,
         break; // break loop only if x is 5

      cout << count << " ";
   } // end for 

   cout << "\nBroke out of loop at count = " << count << endl;
} // end main



/**************************************************************************
 * (C) Copyright 1992-2014 by Deitel & Associates, Inc. and               *
 * Pearson Education, Inc. All Rights Reserved.                           *
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
 **************************************************************************/
