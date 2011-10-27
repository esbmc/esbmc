// Fig. 16.6: fig16_06.cpp
// Demonstrating the string find member functions.
#include <iostream>

using std::cout;
using std::endl;

#include <string>

using std::string;

int main()
{
   string string1( "noon is 12 p.m." );
   int location;
   
   // find "is" at location 5
   cout << "Original string:\n" << string1 
        << "\n\n(find) \"is\" was found at: " 
        << string1.find( "is" ) 
        << "\n(rfind) \"is\" was found at: " 
        << string1.rfind( "is" );

   // find 'o' at location 1
   location = string1.find_first_of( "misop" );

   cout << "\n\n(find_first_of) found '" << string1[ location ]
        << "' from the group \"misop\" at: "
        << location;
   
   // find 'm' at location 13
   location = string1.find_last_of( "misop" );
   cout << "\n\n(find_last_of) found '" << string1[ location ] 
        << "' from the group \"misop\" at: "
        << location;
   
   // find '1' at location 8 
   location = string1.find_first_not_of( "noi spm" );
   cout << "\n\n(find_first_not_of) '" << string1[ location ]
        << "' is not contained in \"noi spm\" and was found at:" 
        << location;
   
   // find '.' at location 12
   location = string1.find_first_not_of( "12noi spm" );
   cout << "\n\n(find_first_not_of) '" << string1[ location ]
        << "' is not contained in \"12noi spm\" and was " 
        << "found at:" << location << endl;

   // search for characters not in string1
   location = string1.find_first_not_of( "noon is 12 p.m." );
   cout << "\nfind_first_not_of(\"noon is 12 p.m.\")"  
        << " returned: " << location << endl;

   return 0;

}  // end main

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