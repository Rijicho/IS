#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int comp(const void *a, const void *b)
{
    	return *(int*)a - *(int*)b;
}

int comp_rev(const void *a, const void *b)
{
	return *(int*)b - *(int*)a;
}

void Send(int id, void* data, int count, int tag, MPI_Datatype type)
{
	MPI_Send(data, count, type, id, tag, MPI_COMM_WORLD); 
}

void Recv(int id, void* data, int count, int tag, MPI_Datatype type)
{
	MPI_Recv(data, count, type, id, tag, MPI_COMM_WORLD, NULL);
}

void SwapForEach(int* big, int* small, int size){
	int i, tmp;
	for(i=0; i<size; i++){
		if(big[i]<small[i])
		{
			tmp = big[i];
			big[i]=small[i];
			small[i] = tmp;
		}
	}
}

void SortStep(int myid, int* nums, int npp, int step, int numproc){
	int r[npp];
	if(step%2==0){
		if(myid % 2 == 0){
			qsort(nums, npp, sizeof(int), comp_rev);
			Send(myid+1, nums, npp, step, MPI_INT);
			Recv(myid+1, nums, npp, step, MPI_INT); 
		}else{
			qsort(nums, npp, sizeof(int), comp);
			Recv(myid-1, r, npp, step, MPI_INT);
			SwapForEach(nums, r, npp);
			Send(myid-1, r, npp, step, MPI_INT);
		}		
	}else if (myid != 0 && myid != numproc-1){
		if(myid % 2 == 1){
			qsort(nums, npp, sizeof(int), comp_rev);
			Send(myid+1, nums, npp, step, MPI_INT);
			Recv(myid+1, nums, npp, step, MPI_INT); 
		}else{
			qsort(nums, npp, sizeof(int), comp);
			Recv(myid-1, r, npp, step, MPI_INT);
			SwapForEach(nums, r, npp);
			Send(myid-1, r, npp, step, MPI_INT);
		}
	}
}

int printnums(int* nums, int myid, int npp, int prev){
	int i;
	int ret=0;
	for(i=0; i<npp; i++){
		if((i==0 && prev > nums[0]) || (i!=0 && nums[i-1] > nums[i])){
			ret++;	
		}
		printf("P%d: [%d] %d\n",myid, i, nums[i]);
	}
	return ret;
}

int main(int argc, char **argv){

	int NPP = 370000;
	int nums[NPP];

	int myid, numproc;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	
	int i,j;
	
	//配布  	
	if(myid == 0){
		srand((unsigned)time(NULL));
		for(i=0; i<NPP;i++) nums[i] = /*test[i]*/rand();

		for(i=1; i<numproc; i++){
			int senddata[NPP];
			for(j=0; j<NPP; j++) senddata[j] = /*test[(5*i+j)%20]*/rand();
			Send(i, senddata, NPP, 0, MPI_INT);
		}
	}
	else
	{
		Recv(0, nums, NPP, 0, MPI_INT);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	double t1 = MPI_Wtime();

	//ソート
	for(i=0; i<8; i++)
		SortStep(myid, nums, NPP, i, numproc);	
	qsort(nums, NPP, sizeof(int), comp);
		

	MPI_Barrier(MPI_COMM_WORLD);
	double t2 = MPI_Wtime();
	double time = t2 - t1;
	

	if(myid==0){
		double times[numproc];
		times[0] = time;
		int err = printnums(nums,myid,NPP, -1);


		for(i=1; i<numproc; i++){
			int prev = nums[NPP-1];
			Recv(i, nums, NPP, i, MPI_DOUBLE);
			Recv(i, &(times[i]), 1, i, MPI_DOUBLE);
			err += printnums(nums,i,NPP, prev);
		}

		for(i=0; i<numproc; i++){printf("[%d] time: %lf\n", myid, times[i]);}
		printf("error: %d\n",err);
		
	}else{
		Send(0, nums, NPP, myid, MPI_INT);
		Send(0, &time, 1, myid, MPI_DOUBLE);
	}
	MPI_Finalize();
	return 0;
}
