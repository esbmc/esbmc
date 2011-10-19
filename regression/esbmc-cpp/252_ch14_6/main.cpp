// Fig. 14.13: fig14_13.cpp
// Writing to a random access file.
#include <iostream>

using std::cerr;
using std::endl;
using std::cout;
using std::cin;
using std::ios;

#include <iomanip>

using std::setw;

#include <fstream>

using std::ofstream;

#include <cstdlib>
#include "clientData.h"  // ClientData class definition

int main()
{
   int accountNumber;
   char lastName[ 15 ];
   char firstName[ 10 ];
   double balance;

   ofstream outCredit( "credit.dat", ios::binary );

   // exit program if ofstream cannot open file
   if ( !outCredit ) {
      cerr << "File could not be opened." << endl;
      exit( 1 );

   } // end if

   cout << "Enter account number "
        << "(1 to 100, 0 to end input)\n? ";

   // require user to specify account number
   ClientData client;
   cin >> accountNumber;
   client.setAccountNumber( accountNumber );

   // user enters information, which is copied into file
   while ( client.getAccountNumber() > 0 && 
      client.getAccountNumber() <= 100 ) {

      // user enters last name, first name and balance
      cout << "Enter lastname, firstname, balance\n? ";
      cin >> setw( 15 ) >> lastName;
      cin >> setw( 10 ) >> firstName;
      cin >> balance;

      // set record lastName, firstName and balance values
      client.setLastName( lastName );
      client.setFirstName( firstName );
      client.setBalance( balance );

      // seek position in file of user-specified record
      outCredit.seekp( ( client.getAccountNumber() - 1 ) * 
         sizeof( ClientData ) );

      // write user-specified information in file
      outCredit.write( 
         reinterpret_cast< const char * >( &client ),
         sizeof( ClientData ) );

      // enable user to specify another account number
      cout << "Enter account number\n? ";
      cin >> accountNumber;
      client.setAccountNumber( accountNumber );

   } // end while

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