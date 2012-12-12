class Base1 {
public:
   Base1( int parameterValue )
   {
      value = parameterValue;
   }
protected:
   int value;
};
class Base2
{
public:
   Base2( char characterData )
   {
      letter = characterData;
   }
protected:
   char letter;
};
class Derived : public Base1, public Base2
{
public:
   Derived( int integer, char character )
      : Base1( integer ), Base2( character ) { }
};
int main()
{
   Derived derived( 7, 'A');
   return 0;
}
