// Fig. 14.15: fig14_15.cpp
// This program reads a random access file sequentially, updates
// data previously written to the file, creates data to be placed
// in the file, and deletes data previously in the file.
#include <iostream>

using std::cout;
using std::cerr;
using std::cin;
using std::endl;
using std::ios;
using std::left;
using std::right;
using std::fixed;
using std::showpoint;

#include <fstream>

using std::ofstream;
using std::ostream;
using std::fstream;

#include <iomanip>

using std::setw;
using std::setprecision;

#include <cstdlib>
#include "clientData.h"  // ClientData class definition

int enterChoice();
void printRecord( fstream& );
void updateRecord( fstream& );
void newRecord( fstream& );
void deleteRecord( fstream& );
void outputLine( ostream&, const ClientData & );
int getAccount( const char * const );

enum Choices { PRINT = 1, UPDATE, NEW, DELETE, END };

int main()
{
   // open file for reading and writing
   fstream inOutCredit( "credit.dat", ios::in | ios::out );

   // exit program if fstream cannot open file
   if ( !inOutCredit ) {
      cerr << "File could not be opened." << endl;
      exit ( 1 );

   } // end if
   
   int choice;

   // enable user to specify action
   while ( ( choice = enterChoice() ) != END ) {

      switch ( choice ) {

         // create text file from record file
         case PRINT:
            printRecord( inOutCredit );
            break;

         // update record
         case UPDATE:
            updateRecord( inOutCredit );
            break;

         // create record
         case NEW:
            newRecord( inOutCredit );
            break;

         // delete existing record
         case DELETE:
            deleteRecord( inOutCredit );
            break;

         // display error if user does not select valid choice
         default:
            cerr << "Incorrect choice" << endl;
            break;

      } // end switch

      inOutCredit.clear(); // reset end-of-file indicator

   } // end while

   return 0;

} // end main

// enable user to input menu choice
int enterChoice()
{
   // display available options
   cout << "\nEnter your choice" << endl
        << "1 - store a formatted text file of accounts" << endl
        << "    called \"print.txt\" for printing" << endl
        << "2 - update an account" << endl
        << "3 - add a new account" << endl
        << "4 - delete an account" << endl
        << "5 - end program\n? ";

   int menuChoice;
   cin >> menuChoice; // receive choice from user

   return menuChoice;

} // end function enterChoice

// create formatted text file for printing
void printRecord( fstream &readFromFile )
{
   // create text file
   ofstream outPrintFile( "print.txt", ios::out );

   // exit program if ofstream cannot create file
   if ( !outPrintFile ) {
      cerr << "File could not be created." << endl;
      exit( 1 );

   } // end if

   outPrintFile << left << setw( 10 ) << "Account" << setw( 16 )
       << "Last Name" << setw( 11 ) << "First Name" << right
       << setw( 10 ) << "Balance" << endl;

   // set file-position pointer to beginning of record file
   readFromFile.seekg( 0 );

   // read first record from record file
   ClientData client;
   readFromFile.read( reinterpret_cast< char * >( &client ),
      sizeof( ClientData ) );

   // copy all records from record file into text file
   while ( !readFromFile.eof() ) {

      // write single record to text file
      if ( client.getAccountNumber() != 0 )
         outputLine( outPrintFile, client );

      // read next record from record file
      readFromFile.read( reinterpret_cast< char * >( &client ), 
         sizeof( ClientData ) );

   } // end while

} // end function printRecord

// update balance in record
void updateRecord( fstream &updateFile )
{
   // obtain number of account to update
   int accountNumber = getAccount( "Enter account to update" );

   // move file-position pointer to correct record in file
   updateFile.seekg( 
      ( accountNumber - 1 ) * sizeof( ClientData ) );

   // read first record from file
   ClientData client;
   updateFile.read( reinterpret_cast< char * >( &client ), 
      sizeof( ClientData ) );

   // update record
   if ( client.getAccountNumber() != 0 ) {
      outputLine( cout, client );

      // request user to specify transaction
      cout << "\nEnter charge (+) or payment (-): ";
      double transaction; // charge or payment
      cin >> transaction;

      // update record balance
      double oldBalance = client.getBalance();
      client.setBalance( oldBalance + transaction );
      outputLine( cout, client );

      // move file-position pointer to correct record in file
      updateFile.seekp(
         ( accountNumber - 1 ) * sizeof( ClientData ) );

      // write updated record over old record in file
      updateFile.write( 
         reinterpret_cast< const char * >( &client ), 
         sizeof( ClientData ) );

   } // end if

   // display error if account does not exist
   else
      cerr << "Account #" << accountNumber 
         << " has no information." << endl;

} // end function updateRecord

// create and insert record
void newRecord( fstream &insertInFile )
{
   // obtain number of account to create
   int accountNumber = getAccount( "Enter new account number" );

   // move file-position pointer to correct record in file
   insertInFile.seekg( 
      ( accountNumber - 1 ) * sizeof( ClientData ) );

   // read record from file
   ClientData client;
   insertInFile.read( reinterpret_cast< char * >( &client ), 
      sizeof( ClientData ) );

   // create record, if record does not previously exist
   if ( client.getAccountNumber() == 0 ) {

      char lastName[ 15 ];
      char firstName[ 10 ];
      double balance;

      // user enters last name, first name and balance
      cout << "Enter lastname, firstname, balance\n? ";
      cin >> setw( 15 ) >> lastName;
      cin >> setw( 10 ) >> firstName;
      cin >> balance;

      // use values to populate account values
      client.setLastName( lastName );
      client.setFirstName( firstName );
      client.setBalance( balance );
      client.setAccountNumber( accountNumber );

      // move file-position pointer to correct record in file
      insertInFile.seekp( ( accountNumber - 1 ) * 
         sizeof( ClientData ) );

      // insert record in file
      insertInFile.write( 
         reinterpret_cast< const char * >( &client ), 
         sizeof( ClientData ) );

   } // end if

   // display error if account previously exists
   else
      cerr << "Account #" << accountNumber
           << " already contains information." << endl;

} // end function newRecord

// delete an existing record
void deleteRecord( fstream &deleteFromFile )
{
   // obtain number of account to delete
   int accountNumber = getAccount( "Enter account to delete" );

   // move file-position pointer to correct record in file
   deleteFromFile.seekg( 
      ( accountNumber - 1 ) * sizeof( ClientData ) );

   // read record from file
   ClientData client;
   deleteFromFile.read( reinterpret_cast< char * >( &client ), 
      sizeof( ClientData ) );

   // delete record, if record exists in file
   if ( client.getAccountNumber() != 0 ) {
      ClientData blankClient;

      // move file-position pointer to correct record in file
      deleteFromFile.seekp( ( accountNumber - 1 ) * 
         sizeof( ClientData ) );

      // replace existing record with blank record
      deleteFromFile.write( 
         reinterpret_cast< const char * >( &blankClient ), 
         sizeof( ClientData ) );

      cout << "Account #" << accountNumber << " deleted.\n";

   } // end if

   // display error if record does not exist
   else
      cerr << "Account #" << accountNumber << " is empty.\n";

} // end deleteRecord

// display single record
void outputLine( ostream &output, const ClientData &record )
{
   output << left << setw( 10 ) << record.getAccountNumber()
          << setw( 16 ) << record.getLastName().data()
          << setw( 11 ) << record.getFirstName().data()
          << setw( 10 ) << setprecision( 2 ) << right << fixed 
          << showpoint << record.getBalance() << endl;

} // end function outputLine

// obtain account-number value from user
int getAccount( const char * const prompt )
{
   int accountNumber;

   // obtain account-number value
   do {
      cout << prompt << " (1 - 100): ";
      cin >> accountNumber;

   } while ( accountNumber < 1 || accountNumber > 100 );

   return accountNumber;

} // end function getAccount

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