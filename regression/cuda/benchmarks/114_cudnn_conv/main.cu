#include <cudnn.h>
#include <assert.h>

/*
 * The purpose of this TC is to verify the correctness of the results 
 * of the convolution operation in OM, and we use the precomputed 
 * results for validation.
 *
 * IMG:
 * 1 2 3
 * 4 5 6
 * 7 8 9
 *
 * Convolution kernel:
 * 1 2
 * 3 4
 *
 * OUT:
 * 37 47
 * 67 77
 */

int main()
{
  cudnnHandle_t handle;
  cudnnCreate(&handle);

  cudnnTensorDescriptor_t xDesc;
  cudnnCreateTensorDescriptor(&xDesc);
  cudnnSetTensor4dDescriptor(
    xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 3, 3);

  cudnnFilterDescriptor_t wDesc;
  cudnnCreateFilterDescriptor(&wDesc);
  cudnnSetFilter4dDescriptor(
    wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, 2, 2);

  cudnnConvolutionDescriptor_t convDesc;
  cudnnCreateConvolutionDescriptor(&convDesc);
  cudnnSetConvolution2dDescriptor(
    convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

  cudnnTensorDescriptor_t yDesc;
  cudnnCreateTensorDescriptor(&yDesc);
  int n, c, h, w;
  cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc, &n, &c, &h, &w);
  cudnnSetTensor4dDescriptor(
    yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

  float alpha = 1.0f, beta = 0.0f;
  float img[1 * 1 * 3 * 3] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  float kernel[1 * 1 * 2 * 2] = {1, 2, 3, 4};
  float out[1 * 1 * 2 * 2] = {};
  float dev[1 * 1 * 2 * 2] = {37, 47, 67, 77};

  cudnnConvolutionForward(
    handle,
    &alpha,
    xDesc,
    img,
    wDesc,
    kernel,
    convDesc,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
    0,
    0,
    &beta,
    yDesc,
    out,
    1,
    3,
    2,
    1,
    0);

  // We use the validation set to validate the result of the convolution
  for(int i = 0; i < 4; i++)
    assert(out[i] == dev[i]);

  return 0;
}
