// Fig. 14.14: fig14_14.cpp
// Reading a random access file.
#include <iostream>

using std::cout;
using std::endl;
using std::ios;
using std::cerr;
using std::left;
using std::right;
using std::fixed;
using std::showpoint;

#include <iomanip>

using std::setprecision;
using std::setw;

#include <fstream>

using std::ifstream;
using std::ostream;

#include <cstdlib>
#include "clientData.h"  // ClientData class definition
 
void outputLine( ostream&, const ClientData & );

int main()
{
   ifstream inCredit( "credit.dat", ios::in );

   // exit program if ifstream cannot open file
   if ( !inCredit ) {
      cerr << "File could not be opened." << endl;
      exit( 1 );

   } // end if

   cout << left << setw( 10 ) << "Account" << setw( 16 )
        << "Last Name" << setw( 11 ) << "First Name" << left
        << setw( 10 ) << right << "Balance" << endl;

   ClientData client; // create record

   // read first record from file
   inCredit.read( reinterpret_cast< char * >( &client ), 
      sizeof( ClientData ) );

   // read all records from file
   while ( inCredit && !inCredit.eof() ) {

      // display record
      if ( client.getAccountNumber() != 0 )
         outputLine( cout, client );

      // read next from file
      inCredit.read( reinterpret_cast< char * >( &client ),
         sizeof( ClientData ) );

   } // end while

   return 0;

} // end main

// display single record
void outputLine( ostream &output, const ClientData &record )
{
   output << left << setw( 10 ) << record.getAccountNumber()
          << setw( 16 ) << record.getLastName().data()
          << setw( 11 ) << record.getFirstName().data()
          << setw( 10 ) << setprecision( 2 ) << right << fixed 
          << showpoint << record.getBalance() << endl;

} // end outputLine

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
