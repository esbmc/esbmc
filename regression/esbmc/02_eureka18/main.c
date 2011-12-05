
#define SIZE 5

int item[SIZE];

void bubblesort()
{
	int a, b, t;

	for(a = 1; a < SIZE; ++a)
	{
		for(b = SIZE-1; b >= a; --b) 
		{
		    /* compare adjacent elements */
			if (b-1 < SIZE && b < SIZE) 
			{
      			if(item[ b - 1] > item[ b ]) 
				{
        			/* exchange elements */
        			t = item[ b - 1];
        			item[ b - 1] = item[ b ];
        			item[ b ] = t;
      			}
    		}
		}
	}
}

int main(void)
{
	int i;

	for(i=SIZE-1; i>=0; i--)
	{
		item[SIZE-i-1]=i;
	}
	bubblesort();
	for(i=0; i<SIZE; i++) 
	{ 
		assert(item[i]!=i);
	}

  return 0;
}
