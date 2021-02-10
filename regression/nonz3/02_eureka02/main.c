int INFINITY = 899;

int Source[8] = {3,2,1,0, 0,3,2,1};
int Dest[8] =   {0,3,2,1, 3,2,1,0};
int Weight[8] = {4,3,2,1, 4,3,2,1};
int nodecount = 4;
int edgecount = 8;

int ResultNodes[4];
/*archi da aggiungere all'albero*/
int ResultEdges[3];


int main(){
  /*descizione degli archi del grafo*/
  int i,j,k,k_1,h, sourceflag, destflag, min;
  int visited[edgecount];

  /*all'inizio nell'albero c'e` solo il nodo 0*/
  ResultNodes[0]=0;

  for( i=1; i < nodecount; i++ )
    ResultNodes[i] = INFINITY;


  for( i=0; i < nodecount-1; i++ )
    ResultEdges[i] = INFINITY;
  
  k=0;
  /* fin quando ci sono archi da aggiungere o da considerare*/
  while( k!= nodecount-1)
    {

      /*seleziono l'arco da considerare*/
      min=0;
      for(i=0; i< edgecount; i++)
	visited[i]= Weight[i];
  
  
      for( k_1=0; k_1< edgecount; k_1++)
	{
	  for( h=0; h < edgecount; h++)
	    {
	  
	      if( visited[h]< visited[min] )
		min = h;
	    }

	  sourceflag=0;
	  for( j=0; j < nodecount; j++)
	    {
	      if(  Source[min]== ResultNodes[j])
		sourceflag=1;
	    }
	  
	  destflag=1;
	  for( j=0; j < nodecount; j++)
	    {
	      if(  Dest[min]== ResultNodes[j])
		{
		  destflag=0;
		}
      
	    }
      
	  if( !sourceflag && !destflag)
	    {
	      visited[min]=889;
	    }
	}
      ResultEdges[k]=min;
      ResultNodes[k+1]=Dest[min];
      Weight[min]= INFINITY;
      k++;
	  
    }
  assert(k==3);
}

