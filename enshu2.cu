#include <iostream>
#include <iomanip>
#include <string>
#include <string.h>
#include <sys/time.h>

int n = 32;

void printall(float u[])
{
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            std::cout << std::setw(2) << (int)(u[i*n+j]*10);
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

__device__ void device_step(float* u2, float* u1, int N, int R)
{
	if(threadIdx.x==0 || threadIdx.x==N-1)
		return;
	u2[threadIdx.x] = (1-4*R)*u1[threadIdx.x] + R*(u1[threadIdx.x+N]+u1[threadIdx.x-N]+u1[threadIdx.x+1]+u1[threadIdx.x-1]);
}

__global__ void kernel(float* u2, float* u1, int N, int R)
{
	device_step(u2,u1, N, R);
}

int main()
{
    float r = 0.25f;
    float u1[n*n];
    float u2[n*n];
    init(u1);
    init(u2);

	float* d_u1;
	float* d_u2;
	cudaMalloc((void**)&d_u1,sizeof(float)*n*n);
	cudaMalloc((void**)&d_u2,sizeof(float)*n*n);
	cudaMemcpy(d_u1, u1, sizeof(float)*n*n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_u2, u2, sizeof(float)*n*n, cudaMemcpyHostToDevice);

    struct timeval t0, t1;
    gettimeofday(&t0,NULL);
    for(int i=0; i<100; i++){
        kernel<<<1,1024>>>(d_u2,d_u1,n,r);
	cudaMemcpy(u2, d_u2, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
        memcpy(u1, u2, sizeof(float)*n*n);
    }
    gettimeofday(&t1,NULL);
	cudaFree(d_u1);
	cudaFree(d_u2);
    std::cout << "100steps later:" << std::endl;
    printall(u1);
    std::cout << "100 steps, u:" << n << "x" << n << std::endl;
    std::cout << "time: " << (double)(t1.tv_sec - t0.tv_sec)+(double)(t1.tv_usec - t0.tv_usec)*1.0e-6 << std::endl;

    return 0;
}
