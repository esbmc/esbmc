// Fig. 12.16: fig12_16.cpp 
// Using member-function fill and stream-manipulator setfill 
// to change the padding character for fields larger the 
// printed value.
#include <iostream>

using std::cout;
using std::endl;
using std::showbase;
using std::left;
using std::right;
using std::internal;
using std::hex;
using std::dec;

#include <iomanip>

using std::setw;
using std::setfill;

int main()
{
   int x = 10000;

   // display x
   cout << x << " printed as int right and left justified\n"
        << "and as hex with internal justification.\n"
        << "Using the default pad character (space):" << endl;

   // display x with plus sign
   cout << showbase << setw( 10 ) << x << endl;

   // display x with left justification
   cout << left << setw( 10 ) << x << endl;

   // display x as hex with internal justification
   cout << internal << setw( 10 ) << hex << x << endl << endl;

   cout << "Using various padding characters:" << endl;

   // display x using padded characters (right justification)
   cout << right;
   cout.fill( '*' );
   cout << setw( 10 ) << dec << x << endl;

   // display x using padded characters (left justification)
   cout << left << setw( 10 ) << setfill( '%' ) << x << endl;

   // display x using padded characters (internal justification)
   cout << internal << setw( 10 ) << setfill( '^' ) << hex
        << x << endl;

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