#include <iostream>
#include <iomanip>
#include <string>
#include <string.h>
#include <sys/time.h>
#include <stdio.h>

int n = 32;

void printall(float u[])
{
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            std::cout << std::setw(5) << (int)(u[i*n+j]*10);
        }
        std::cout << std::endl << std::endl;
    }
}

void init(float* u){
    for(int i=0; i<n; i++)
    {
        u[i]=0;
        u[i*n]=0;
        u[n*(n-1)+i]=0;
        u[i*n+n-1]=0;

    }
    for(int i=1; i<n-1; i++)
    {
        for(int j=1; j<n-1; j++)
        {
	    u[i*n+j]=1;
		}
    }
}
__global__ void kernel(float* u2, float* u1, int N, float R)
{

	int id = threadIdx.x;
	for(int i=0; i<100; i++){
		if(id/N!=0 && id/N!=N-1 && id%N!=0 && id%N!=N-1)
		{
			//u2[id] = id;
			//u2[id] = u1[id]+1;
			//u2[id] = (1-4*R)*u1[id];
			u2[id] = (1-4*R)*u1[id] + R*(u1[id+N]+u1[id-N]+u1[id+1]+u1[id-1]);
		}
		__syncthreads();
		u1[id] = u2[id];
		__syncthreads();
	}
}

int main()
{
    float r = 0.1f;
    float u1[n*n];
    float u2[n*n];
    init(u1);
    init(u2);
	int size = sizeof(float)*n*n;

	float* d_u1;
	float* d_u2;
	cudaMalloc((void**)&d_u1,size);
	cudaMalloc((void**)&d_u2,size);
	cudaMemcpy(d_u1, u1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_u2, u2, size, cudaMemcpyHostToDevice);


    struct timeval t0, t1;
    gettimeofday(&t0,NULL);
	kernel<<<1,1024>>>(d_u2,d_u1,n,r);
	cudaThreadSynchronize();
    gettimeofday(&t1,NULL);
	cudaMemcpy(u1, d_u1, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(u2, d_u2, size, cudaMemcpyDeviceToHost);


	std::cout << "100steps later:" << std::endl;
    printall(u1);
	printall(u2);
    std::cout << "100 steps, u:" << n << "x" << n << std::endl;
    std::cout << "time: " << (double)(t1.tv_sec - t0.tv_sec)+(double)(t1.tv_usec - t0.tv_usec)*1.0e-6 << std::endl;

	cudaFree(d_u1);
	cudaFree(d_u2);
	return 0;
}
