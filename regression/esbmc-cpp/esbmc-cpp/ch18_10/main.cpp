// Fig. 18.19: fig18_19.cpp
// Using functions isspace, iscntrl, ispunct, isprint, isgraph.
#include <iostream>

using std::cout;
using std::endl;

#include <cctype>  // character-handling function prototypes

int main()
{
   cout << "According to isspace:\nNewline " 
        << ( isspace( '\n' ) ? "is a" : "is not a" )
        << " whitespace character\nHorizontal tab " 
        << ( isspace( '\t' ) ? "is a" : "is not a" )
        << " whitespace character\n"
        << ( isspace( '%' ) ? "% is a" : "% is not a" )
        << " whitespace character\n";

   cout << "\nAccording to iscntrl:\nNewline " 
        << ( iscntrl( '\n' ) ? "is a" : "is not a" )
        << " control character\n"
        << ( iscntrl( '$' ) ? "$ is a" : "$ is not a" )
        << " control character\n";

   cout << "\nAccording to ispunct:\n"
        << ( ispunct( ';' ) ? "; is a" : "; is not a" )
        << " punctuation character\n"
        << ( ispunct( 'Y' ) ? "Y is a" : "Y is not a" )
        << " punctuation character\n"
        << ( ispunct('#') ? "# is a" : "# is not a" )
        << " punctuation character\n";

   cout << "\nAccording to isprint:\n"
        << ( isprint( '$' ) ? "$ is a" : "$ is not a" )
        << " printing character\nAlert " 
        << ( isprint( '\a' ) ? "is a" : "is not a" )
        << " printing character\n";

   cout << "\nAccording to isgraph:\n"
        << ( isgraph( 'Q' ) ? "Q is a" : "Q is not a" )
        << " printing character other than a space\nSpace " 
        << ( isgraph(' ') ? "is a" : "is not a" )
        << " printing character other than a space" << endl;

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