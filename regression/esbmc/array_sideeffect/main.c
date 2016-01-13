
unsigned int calc_size(unsigned int size)
{
  return size*2;
}

void VLA_size(unsigned int size)
{
  int arr[calc_size(size)];
  arr[calc_size(size)-1] = 1;

  assert(arr[calc_size(size) - 1] == 1);
}

int main() 
{
  int arr1[calc_size(2)];
  int arr2[calc_size(2)];

  arr1[3] = 1;
  arr2[3] = 1;

  assert(arr1[calc_size(2) - 1] == 1);
  assert(arr2[calc_size(1) + 1] == 1);

  VLA_size(1);
  VLA_size(0); // ERROR
} 
