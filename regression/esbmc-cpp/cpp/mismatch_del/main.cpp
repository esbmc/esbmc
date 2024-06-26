int main()
{
  int* IntArray = new int[5];
    
  delete IntArray; //should be IntArray[]

  return 0;
}
