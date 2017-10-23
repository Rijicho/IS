#include <iostream>
#include <iomanip>
#include <string>
#include <string.h>
#include <sys/time.h>
#include <stdlib.h>

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

int main()
{
    int size = sizeof(float)*n*n;
    float r = 0.25f;
    float* u = (float*)malloc(size);
    float* u2 = (float*)malloc(size);
    init(u);
    init(u2);
    struct timeval t0, t1;
    gettimeofday(&t0,NULL);
    for(int i=0; i<100; i++){
        #pragma omp parallel for
        for(int k=1; k<n-1; k++){
            #pragma omp parallel for
            for(int l=1; l<n-1; l++){
                u2[k*n+l] = (1-4*r)*u[k*n+l]+r*(u[(k+1)*n+l]+u[(k-1)*n+l]+u[k*n+l+1]+u[k*n+l-1]);
            }
        }
        memcpy(u, u2, sizeof(float)*n*n);
    }
    gettimeofday(&t1,NULL);
    std::cout << "100steps later:" << std::endl;
    //printall(u);
    std::cout << "100 steps, u:" << n << "x" << n << std::endl;
    std::cout << "time: " << (double)(t1.tv_sec - t0.tv_sec)+(double)(t1.tv_usec - t0.tv_usec)*1.0e-6 << std::endl;

    return 0;
}
