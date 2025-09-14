#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
//#include "utils.h"
//#include "__fc_builtin.h" 

int main(){
float norm = (float)1/(float)100;
int x0 = nondet_int();
__ESBMC_assume((x0 >= 441)&&(x0 <=539));
int x1 = nondet_int();
__ESBMC_assume((x1 >= 270)&&(x1 <=330));
int x2 = nondet_int();
__ESBMC_assume((x2 >= 126)&&(x2 <=154));
int x3 = nondet_int();
__ESBMC_assume((x3 >= 18)&&(x3 <=22));
//unsigned int x0 = Frama_C_interval(235, 255);
//unsigned int x1 = Frama_C_interval(235, 255);
//unsigned int x2 = Frama_C_interval(130, 150);
//unsigned int x3 = Frama_C_interval(10, 30);
float i[4];
i[0] = x0*norm;
i[1] = x1*norm;
i[2] = x2*norm;
i[3] = x3*norm;


float layer1[7];
layer1[0]= (-0.259293f)*i[0] + (-0.147145f)*i[1] + (-0.070675f)*i[2] + (-0.396830f)*i[3] + (-0.019240f);
layer1[0] = tanhFunctionLUT(layer1[0]);
layer1[1]= (0.195045f)*i[0] + (0.474309f)*i[1] + (-0.148397f)*i[2] + (-0.079880f)*i[3] + (-0.016677f);
layer1[1] = tanhFunctionLUT(layer1[1]);
layer1[2]= (0.461122f)*i[0] + (-0.476375f)*i[1] + (0.248677f)*i[2] + (-0.553619f)*i[3] + (-0.017581f);
layer1[2] = tanhFunctionLUT(layer1[2]);
layer1[3]= (-0.324553f)*i[0] + (-0.542259f)*i[1] + (1.332844f)*i[2] + (0.068103f)*i[3] + (-0.118043f);
layer1[3] = tanhFunctionLUT(layer1[3]);
layer1[4]= (0.405520f)*i[0] + (0.221390f)*i[1] + (-0.391765f)*i[2] + (0.337188f)*i[3] + (-0.018501f);
layer1[4] = tanhFunctionLUT(layer1[4]);
layer1[5]= (0.651486f)*i[0] + (-0.367779f)*i[1] + (-0.224814f)*i[2] + (-0.037613f)*i[3] + (0.077769f);
layer1[5] = tanhFunctionLUT(layer1[5]);
layer1[6]= (0.981716f)*i[0] + (0.727559f)*i[1] + (-1.363340f)*i[2] + (-1.402215f)*i[3] + (0.674765f);
layer1[6] = tanhFunctionLUT(layer1[6]);
float layer2[3];
layer2[0]= (-0.228491f)*layer1[0] + (-0.172476f)*layer1[1] + (-0.608707f)*layer1[2] + (-1.003181f)*layer1[3] + (0.519332f)*layer1[4] + (0.686671f)*layer1[5] + (-0.028897f)*layer1[6] + (0.342278f);
layer2[0] = tanhFunctionLUT(layer2[0]);
int r = 0;
layer2[1]= (-0.782173f)*layer1[0] + (-0.306530f)*layer1[1] + (-0.409494f)*layer1[2] + (0.967240f)*layer1[3] + (-0.508822f)*layer1[4] + (0.187891f)*layer1[5] + (0.998079f)*layer1[6] + (0.046970f);
layer2[1] = tanhFunctionLUT(layer2[1]);
if (layer2[1] > layer2[r]) r = 1;
layer2[2]= (-0.182530f)*layer1[0] + (0.064892f)*layer1[1] + (0.407630f)*layer1[2] + (-0.205405f)*layer1[3] + (-0.140221f)*layer1[4] + (0.204790f)*layer1[5] + (-0.901339f)*layer1[6] + (0.131860f);
layer2[2] = tanhFunctionLUT(layer2[2]);
if (layer2[2] > layer2[r]) r = 2;
__ESBMC_assert(r == 0, "Classification is not a 0 anymore.");
//@assert(r == 0);
}

