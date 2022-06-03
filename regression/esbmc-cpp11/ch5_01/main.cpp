// Fig. 5.1: fig05_01.cpp
// Counter-controlled repetition.
#include <iostream>
using namespace std;

int main()
{
   unsigned int counter = 1; // declare and initialize control variable

   while ( counter <= 10 ) // loop-continuation condition
   {    
      cout << counter << " ";
      ++counter; // increment control variable by 1
   } // end while 

   cout << endl; // output a newline
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
