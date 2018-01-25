#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include "mkl.h"
#include <omp.h>

#define M 3000
#define N 4000
#define K 2000

// ----- matrix access ----- //
// mat[i,j]
// rowsize: n
// row    : i
// column : j
#define IDX(mat,n, i,j) (mat)[(i)+(j)*(n)]
//-------------------------//

//C=AB  A:m*k B:k*n C:m*n
void gemm(int m, int n, int k, double* A, double* B, double* C)
{
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            for(int a=0; a<k; a++){
                IDX(C,m,i,j) += IDX(A,m,i,a) * IDX(B,k,a,j);
            }
        }
    }
}

void gemm2(int m, int n, int k, double* A, double* B, double* C)
{
    #pragma omp parallel for
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            for(int a=0; a<k; a++){
                IDX(C,m,i,j) += IDX(A,m,i,a) * IDX(B,k,a,j);
            }
        }
    }
}


void print_matrix(double *a, int m, int n){
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      printf("%.2f ", IDX(a,m, i,j));
    }
    printf("\n");
  }
}

double uniform(){
  static int init_flg = 0;
  if(!init_flg){
    init_flg = 1;
    srand((unsigned)time(NULL));
  }
  return ((double)rand()+1.0)/((double)RAND_MAX+2.0);
}

double get_dtime(){
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)(tv.tv_usec)*0.001*0.001;
}

void run(){
    printf("M,DGEMM,GEMM1,GEMM2\n");
  double t1,t2;
  for(int size = 100; size<=4000; size+=100)
  {
      // c[1:M][1:N] = a[1:M][1:K] * b[1:K][1:N]
      double *a,*b,*c,*c2;
      int m=size, n=size, k=size;
      a = (double*)malloc(sizeof(double)*size*size);
      b = (double*)malloc(sizeof(double)*size*size);
      c = (double*)malloc(sizeof(double)*size*size);
      c2 =(double*)malloc(sizeof(double)*size*size);

      for(int i=0; i<m; i++){
        for(int j=0; j<k; j++){
          IDX(a, m, i, j) = uniform();
        }
      }
      for(int i=0; i<k; i++){
        for(int j=0; j<n; j++){
          IDX(b, k, i, j) = uniform();
        }
      }

      double alpha=1.0, zero=0.0;

      //printf("matrix A:\n");
      //print_matrix(a, m, k);
      //printf("matrix B:\n");
      //print_matrix(b, k, n);

      t1 = get_dtime();
      dgemm("N","N", &m,&n,&k, &alpha,a,&m, b,&k, &zero,c,&m);
      t2 = get_dtime();

      double tdgemm = t2-t1;

      t1 = get_dtime();
      gemm(m,n,k,a,b,c2);
      t2 = get_dtime();

      double tgemm = t2-t1;

      t1 = get_dtime();
      gemm2(m,n,k,a,b,c2);
      t2 = get_dtime();

      double tgemm2 = t2-t1;

      printf("%d,%.10e,%.10e, %.10e\n", size, tdgemm, tgemm, tgemm2);


      free(c2);
      free(c);
      free(b);
      free(a);
  }
}

int main(){
  run();

  return 0;
}
