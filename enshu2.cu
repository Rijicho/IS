#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>

__device__ void device_strcpy(char *dst, const char *src)
{
	while(*dst++ = *src++);
}

__global__ void kernel(char *A)
{
	device_strcpy(A, "testtest");
}

int main(){
	char* d_hello;
	char hello[32];

	cudaMalloc((void**)&d_hello, 32);

	kernel<<<1,1>>>(d_hello);

	cudaMemcpy(hello, d_hello, 32, cudaMemcpyDeviceToHost);

	cudaFree(d_hello);

	puts(hello);



	return 0;
}
