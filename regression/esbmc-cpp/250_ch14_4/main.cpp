// Fig. 14.8: fig14_08.cpp
// Credit inquiry program.
#include <iostream>

using std::cout;
using std::cin;
using std::ios;
using std::cerr;
using std::endl;
using std::fixed;
using std::showpoint;
using std::left;
using std::right;

#include <fstream>

using std::ifstream;

#include <iomanip>

using std::setw;
using std::setprecision;

#include <cstdlib>    
 
enum RequestType { ZERO_BALANCE = 1, CREDIT_BALANCE, 
   DEBIT_BALANCE, END };
int getRequest();
bool shouldDisplay( int, double );
void outputLine( int, const char * const, double );

int main()
{
   // ifstream constructor opens the file
   ifstream inClientFile( "clients.dat", ios::in );

   // exit program if ifstream could not open file
   if ( !inClientFile ) {
      cerr << "File could not be opened" << endl;
      exit( 1 );

   } // end if

   int request;
   int account;
   char name[ 30 ];
   double balance;

   // get user's request (e.g., zero, credit or debit balance)
   request = getRequest();

   // process user's request
   while ( request != END ) {

      switch ( request ) {

         case ZERO_BALANCE:
            cout << "\nAccounts with zero balances:\n";
            break;

         case CREDIT_BALANCE:
            cout << "\nAccounts with credit balances:\n";
            break;

         case DEBIT_BALANCE:
            cout << "\nAccounts with debit balances:\n";
            break;

      } // end switch

      // read account, name and balance from file
      inClientFile >> account >> name >> balance;

      // display file contents (until eof)
      while ( !inClientFile.eof() ) {

         // display record
         if ( shouldDisplay( request, balance ) )
            outputLine( account, name, balance );

         // read account, name and balance from file
         inClientFile >> account >> name >> balance;

      } // end inner while
      
      inClientFile.clear();    // reset eof for next input
      inClientFile.seekg( 0 ); // move to beginning of file
      request = getRequest();  // get additional request from user

   } // end outer while

   cout << "End of run." << endl;

   return 0; // ifstream destructor closes the file

} // end main

// obtain request from user
int getRequest()
{ 
   int request;

   // display request options
   cout << "\nEnter request" << endl
        << " 1 - List accounts with zero balances" << endl
        << " 2 - List accounts with credit balances" << endl
        << " 3 - List accounts with debit balances" << endl
        << " 4 - End of run" << fixed << showpoint;

   // input user request
   do {
      cout << "\n? ";
      cin >> request;

   } while ( request < ZERO_BALANCE && request > END );

   return request;

} // end function getRequest

// determine whether to display given record
bool shouldDisplay( int type, double balance )
{
   // determine whether to display credit balances
   if ( type == CREDIT_BALANCE && balance < 0 )
      return true;

   // determine whether to display debit balances
   if ( type == DEBIT_BALANCE && balance > 0 )
      return true;

   // determine whether to display zero balances
   if ( type == ZERO_BALANCE && balance == 0 )
      return true;

   return false;

} // end function shouldDisplay

// display single record from file
void outputLine( int account, const char * const name, 
   double balance )
{
   cout << left << setw( 10 ) << account << setw( 13 ) << name
        << setw( 7 ) << setprecision( 2 ) << right << balance
        << endl;

} // end function outputLine

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
