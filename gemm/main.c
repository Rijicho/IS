#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include "mkl.h"

#define M 3
#define N 4
#define K 2

// ----- matrix access ----- //
// mat[i,j]
// rowsize: n
// row    : i
// column : j
#define IDX(mat,n, i,j) (mat)[(i)+(j)*(n)]
//-------------------------//

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
  double t1,t2;

  // c[1:M][1:N] = a[1:M][1:K] * b[1:K][1:N]
  double *a,*b,*c;
  int m=M, n=N, k=K;
  a = (double*)malloc(sizeof(double)*M*K);
  b = (double*)malloc(sizeof(double)*K*N);
  c = (double*)malloc(sizeof(double)*M*N);

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

  printf("matrix A:\n");
  print_matrix(a, m, k);
  printf("matrix B:\n");
  print_matrix(b, k, n);

  t1 = get_dtime();
  dgemm("N","N", &m,&n,&k, &alpha,a,&m, b,&k, &zero,c,&m);
  t2 = get_dtime();

  printf("matrix C:\n");
  print_matrix(c, m, n);

  printf("DGEMM-elapsed: %.10e\n", t2-t1);

  free(c);
  free(b);
  free(a);
}

int main(){
  run();

  return 0;
}
