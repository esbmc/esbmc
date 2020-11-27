#define DIM 2 //space dimension
#define n 1 // number of points that compose the path
#define p 1 //precision of points localization
#define J_c 25 // candidate value of cost function
#define no 1 //number of obstacles

//obstacles information
float xo[no] = {5}; // coordinates of center 'x'
float yo[no] = {5}; // coordinates of center 'y'
float r[no] = {2.5}; // obstacles radius

//points of the path information
int x[n][DIM];
int nondet_int();

// helper functions declaration
void rest_points(int P1[DIM], int i, float xo, float yo, float r);
float distance(float aux1[DIM], float aux2[DIM]);

int main(){
  int i, j;
  int A[DIM] = {1*p, 1*p}; //start point
  int B[DIM] = {9*p, 9*p}; //target point

  //environmental limits
  int lim[DIM][2] = {0*p, 10*p, 0*p, 10*p};
  // states declaration, x = x[i][0], y = x[i][1]
  // set x[i][j] to non-deterministic
  for(i = 0; i < n; i++){
    for(j = 0; j < DIM; j++)
      x[i][j] = nondet_int();
  } // end for

  // constraints on environment limits and obstacles
  for(i = 0; i < n; i++){
    __ESBMC_assume(x[i][0] >= lim[0][0]);
    __ESBMC_assume(x[i][0] <= lim[0][1]);
    __ESBMC_assume(x[i][1] >= lim[1][0]);
    __ESBMC_assume(x[i][1] <= lim[1][1]);
  } // end for
  // rest_points function must be executed for each obstacle
  for(j = 0; j < no; j++)
    rest_points (A, 0, xo[j]*p, yo[j]*p, r[j]*p);
  for(i = 1; i < n; i++){
    for(j = 0; j < no; j++)
      rest_points (x[i-1], i, xo[j]*p, yo[j]*p, r[j]*p);
  } // end for
  for(j = 0; j < no; j++)
    rest_points (B, n-1, xo[j]*p, yo[j]*p, r[j]*p);

  //compute the cost function
  float aux1[DIM], aux2[DIM];
  float J = 0.0;

  for(j = 0; j < DIM; j++)
    aux1[j] = A[j]/p;
  for(i = 0; i < n; i++){
    for(j = 0; j < DIM; j++){
      aux2[j] = (float)x[i][j]/p;
    }
    J = J + distance(aux1, aux2);
    for(j = 0; j < DIM; j++){
      aux1[j] = aux2[j];
    }
  }
  for(j = 0; j < DIM; j++)
    aux2[j] = B[j]/p;
  J = J + distance(aux1, aux2);

  __ESBMC_assume(J < J_c);

  // test the literal J_optimal
  assert(J > J_c);
  return 0;
}

void rest_points(int P1[DIM], int i, float xo, float yo, float r){
  float sigma = 0.5; // safety margin
  __ESBMC_assume((x[i][0] - xo)*(x[i][0] - xo) + (x[i][1] - yo)*(x[i][1] - yo) > (r + sigma)*(r + sigma));

  // get the equation of the segemnt line between i-1 and i points
  float a, b, c;
  if(P1[0] - x[i][0] == 0){
    a = 1;
    b = 0;
    c = -P1[0];
  }
  else{
    a = (float)(P1[1]-x[i][1])/(P1[0]-x[i][0]);
    b = -1;
    c = (float)-a*P1[0] + P1[1];
  }

  float Py = (a*a*yo - a*b*xo - b*c)/(a*a + b*b);
  if((((Py - x[i][1])/(P1[1] - x[i][1])) >= 0) && (((Py - x[i][1])/(P1[1] - x[i][1])) <= 1))
  {
    float d = (float)fabsf(a*xo + b*yo + c)/sqrt(a*a + b*b);
    __ESBMC_assume(d >= r + sigma);
  }
}

// This function is only avaiable for 2D case
float distance(float aux1[DIM], float aux2[DIM]){
  int j;
  float dist = 0;
  for(j = 0; j < DIM; j++){
    dist = dist + (aux2[j] - aux1[j])*(aux2[j] - aux1[j]);
  }
  return sqrt(dist);
}
