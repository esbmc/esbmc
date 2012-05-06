//#include <stdio.h>
//#include <assert.h>

#define SIZE 5

int array[SIZE];
int n=SIZE;

void SelectionSort()
{
   int lh, rh, i, temp;

   for (lh = 0; lh < n; lh++) {
      rh = lh;
      for (i = lh + 1; i < n; i++) 
         if (array[i] < array[rh]) rh = i;
      temp = array[lh];
      array[lh] = array[rh];
      array[rh] = temp;
   }
}

int main(void){

	int array[SIZE],i;

	for(i=SIZE-1; i>=0; i--)
		array[i]=i;

	SelectionSort();

	for(i=0; i<SIZE; i++)
		assert(array[i]==i);

}

