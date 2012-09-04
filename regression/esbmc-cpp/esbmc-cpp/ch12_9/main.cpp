// Fig. 12.11: fig12_12.cpp
// Creating and testing user-defined, nonparameterized 
// stream manipulators.
#include <iostream>

using std::ostream;
using std::cout;
using std::flush;

// bell manipulator (using escape sequence \a)
ostream& bell( ostream& output )
{
   return output << '\a';  // issue system beep
}

// carriageReturn manipulator (using escape sequence \r)
ostream& carriageReturn( ostream& output )
{
   return output << '\r';  // issue carriage return
}

// tab manipulator (using escape sequence \t)
ostream& tab( ostream& output )
{
   return output << '\t';  // issue tab
}

// endLine manipulator (using escape sequence \n and member
// function flush)
ostream& endLine( ostream& output ) 
{ 
   return output << '\n' << flush;  // issue end of line
}

int main()
{
   // use tab and endLine manipulators
   cout << "Testing the tab manipulator:" << endLine
        << 'a' << tab << 'b' << tab << 'c' << endLine;

   cout << "Testing the carriageReturn and bell manipulators:"
        << endLine << "..........";

   cout << bell; // use bell manipulator

   // use ret and endLine manipulators
   cout << carriageReturn << "-----" << endLine;

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