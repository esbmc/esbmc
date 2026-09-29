// A constructor's throw(int) is still unresolved when the converter stores
// its type, so its declared types have to survive the IREP2 round trip.
class X
{
public:
  X() throw(int)
  {
    throw 'c';
  }
};

int main()
{
  try
  {
    X x;
  }
  catch (char)
  {
    return 1;
  }
  return 0;
}
