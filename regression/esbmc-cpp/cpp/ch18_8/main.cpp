// Fig. 18.17: fig18_17.cpp
// Using functions isdigit, isalpha, isalnum and isxdigit.
#include <iostream>

using std::cout;
using std::endl;

#include <cctype>  // character-handling function prototypes

int main()
{
   cout << "According to isdigit:\n"
        << ( isdigit( '8' ) ? "8 is a" : "8 is not a" ) 
        << " digit\n"
        << ( isdigit( '#' ) ? "# is a" : "# is not a" ) 
        << " digit\n";

   cout << "\nAccording to isalpha:\n"
        << ( isalpha( 'A' ) ? "A is a" : "A is not a" ) 
        << " letter\n"
        << ( isalpha( 'b' ) ? "b is a" : "b is not a" ) 
        << " letter\n"
        << ( isalpha( '&' ) ? "& is a" : "& is not a" ) 
        << " letter\n"
        << ( isalpha( '4' ) ? "4 is a" : "4 is not a" ) 
        << " letter\n";

   cout << "\nAccording to isalnum:\n"
        << ( isalnum( 'A' ) ? "A is a" : "A is not a" )
        << " digit or a letter\n"
        << ( isalnum( '8' ) ? "8 is a" : "8 is not a" )
        << " digit or a letter\n"
        << ( isalnum( '#' ) ? "# is a" : "# is not a" )
        << " digit or a letter\n";

   cout << "\nAccording to isxdigit:\n"
        << ( isxdigit( 'F' ) ? "F is a" : "F is not a" )
        << " hexadecimal digit\n"
        << ( isxdigit( 'J' ) ? "J is a" : "J is not a" )
        << " hexadecimal digit\n"
        << ( isxdigit( '7' ) ? "7 is a" : "7 is not a" )
        << " hexadecimal digit\n"
        << ( isxdigit( '$' ) ? "$ is a" : "$ is not a" )
        << " hexadecimal digit\n"
        << ( isxdigit( 'f' ) ? "f is a" : "f is not a" )
        << " hexadecimal digit" << endl;

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