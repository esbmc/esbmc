// Fig. 20.3: fig20_03.cpp
// Using command-line arguments
#include <iostream>

using std::cout;
using std::endl;
using std::ios;

#include <fstream>

using std::ifstream;
using std::ofstream;

int main( int argc, char *argv[] )
{
   // check number of command-line arguments
   if ( argc != 3 )
      cout << "Usage: copyFile infile_name outfile_name" << endl;

   else {
      ifstream inFile( argv[ 1 ], ios::in );

      // input file could not be opened
      if ( !inFile ) {
         cout << argv[ 1 ] << " could not be opened" << endl;
         return -1;

      }  // end if

      ofstream outFile( argv[ 2 ], ios::out );

      // output file could not be opened
      if ( !outFile ) {
         cout << argv[ 2 ] << " could not be opened" << endl;
         inFile.close();
         return -2;

      } // end if

      char c = inFile.get(); // read first character

      while ( inFile ) {
         outFile.put( c );   // output character
         c = inFile.get();   // read next character

      }  // end while
   }  // end else

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
