#include <iostream>
#include <iomanip>
#include <string>
#include <string.h>
#include <sys/time.h>

int n = 1024;

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

__device__ void device_step(float* u2, float* u1)
{
	if(threadIdx==0 || threadIdx==n-1)
		return;

	for(int k=1; k<n-1; k++)
	{
		u2[k*n+threadIdx] = (1-4*r)*u1[k*n+threadIdx]+r*(u1[(k+1)*n+threadIdx]+u1[(k-1)*n+threadIdx]+u1[k*n+threadIdx+1]+u1[k*n+threadIdx-1]);
	}
}

__global__ void kernel(float* u2, float* u1)
{
	device_step(u2,u1);
}

int main()
{
    float r = 0.25f;
    float u1[n*n];
    float u2[n*n];
    init(u1);
    memcpy(u2, u1, sizeof(float)*n*n);

	float* d_u1;
	float* d_u2;
	cudaMalloc((void**)&d_u1,sizeof(float)*n*n);

	cudaMemcpy(d_u1, u1, sizeof(float)*n*n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_u2, u2, sizeof(float)*n*n, cudaMemcpyHostToDevice);

    struct timeval t0, t1;
    gettimeofday(&t0,NULL);
    for(int i=0; i<100; i++){
        kernel<<<1,1024>>>(d_u2,d_u1);
		cudaMemcpy(u2, d_u2, sizeof(float)*n*n, cudaMemcpyDeviceToHost)
        memcpy(u, u2, sizeof u2);
    }
    gettimeofday(&t1,NULL);
	cudaFree(d_u1);
	cudaFree(d_u2);
    std::cout << "100steps later:" << std::endl;
    //printall(u);
    std::cout << "100 steps, u:" << n << "x" << n << std::endl;
    std::cout << "time: " << (double)(t1.tv_sec - t0.tv_sec)+(double)(t1.tv_usec - t0.tv_usec)*1.0e-6 << std::endl;

    return 0;
}
