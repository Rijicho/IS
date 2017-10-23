#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <sys/time.h>

int n = 100;

void printall(std::vector<std::vector<float> > u)
{
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            std::cout << std::setw(2) << (int)(u[i][j]*10);
        }
        std::cout << std::endl << std::endl;
    }
}

int main()
{
    struct timeval t0, t1;


    float r = 0.25f;
    std::vector<std::vector<float> > u(100,std::vector<float>(100,1));
    for(int i=0; i<n; i++){
        u[0][i]=0;
        u[n-1][i]=0;
        u[i][0]=0;
        u[i][n-1]=0;
    }
    std::vector<std::vector<float> > u2 = u;
    printall(u);
    gettimeofday(&t0,NULL);
    for(int i=0; i<100; i++){
        #pragma omp parallel for
        for(int k=1; k<n-1; k++){
            #pragma omp parallel for
            for(int l=1; l<n-1; l++){
                u2[k][l] = (1-4*r)*u[k][l]+r*(u[k+1][l]+u[k-1][l]+u[k][l+1]+u[k][l-1]);
            }
        }
        u=u2;
    }
    gettimeofday(&t1,NULL);
    std::cout << "100steps later:" << std::endl;
    printall(u);
    std::cout << "100 steps, u:" << n << "x" << n << std::endl;
    std::cout << "time: " << (double)(t1.tv_sec - t0.tv_sec)+(double)(t1.tv_usec - t0.tv_usec)*1.0e-6 << std::endl;

    return 0;
}
