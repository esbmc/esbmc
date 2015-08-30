int main() 
{
  int M=2;
  int K=4;
  int N=4;

  int n[K+1];
  int g[K+1];
  int u[K+1];
  int t[K+1];
  int i, j, h, c, diff;

  for(i=0; i<=K; i++)
  {
    g[i]=0;
    u[i]=1;
    n[i]=M;
  }

  c=0;
  while(c<N && g[K]==0)
  {

    for(h=0; h<=K; h++)
      t[h]=g[h];

    i=0;
    j=g[0]+u[0];
    while(j>=n[i] || j<0)
    {
      u[i]=-u[i];
	  i++;
      j=g[i]+u[i];
    }
    g[i]=j;

    diff=0;
    for(h=0; h<=K-1; h++)
      diff+=g[h]-t[h];

    assert(diff==1 || diff==-1);
    c++;
  }
}

