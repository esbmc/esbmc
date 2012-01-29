// Fig. 12.21: fig12_21.cpp 
// Demonstrating the flags member function.
#include <iostream>

using std::cout;
using std::endl;
using std::oct;
using std::scientific;
using std::showbase;
using std::ios_base;

int main()
{
   int integerValue = 1000;
   double doubleValue = 0.0947628;

   // display flags value, int and double values (original format)
   cout << "The value of the flags variable is: " << cout.flags()
        << "\nPrint int and double in original format:\n"
        << integerValue << '\t' << doubleValue << endl << endl;

   // use cout flags function to save original format
   ios_base::fmtflags originalFormat = cout.flags();
   cout << showbase << oct << scientific; // change format

   // display flags value, int and double values (new format)
   cout << "The value of the flags variable is: " << cout.flags()
        << "\nPrint int and double in a new format:\n"
        << integerValue << '\t' << doubleValue << endl << endl;

   cout.flags( originalFormat ); // restore format

   // display flags value, int and double values (original format)
   cout << "The restored value of the flags variable is: " 
        << cout.flags()
        << "\nPrint values in original format again:\n"
        << integerValue << '\t' << doubleValue << endl;

   return 0;

} // end main

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