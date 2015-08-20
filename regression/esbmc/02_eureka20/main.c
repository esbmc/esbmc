//#include <stdio.h>
//#include <assert.h>

#define SIZE 5

int main(void) {

	int a[SIZE], k;
    int j, i, chave;

	for(k=SIZE-1; k>=0; k--)
	{
		a[SIZE-k-1]=k;
	}

	for(j=2; j<=SIZE; j++) {
		chave = a[j-1];
		i=j-2;
		while(i>=0 && a[i]>chave) {
			a[i+1] = a[i];
			i=i-1;
		}
		a[i+1]=chave;

	}


	for(k=0; k<SIZE; k++) 
	{ 
		assert(a[k]!=k);
	}

}
