#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

//qsort用比較器
int comp(const void *a, const void *b)
{
    return *(int*)a - *(int*)b;
}

//qsort用比較器（反転）
int comp_rev(const void *a, const void *b)
{
	return *(int*)b - *(int*)a;
}

//MPI_Sendのエイリアス
void Send(int id, void* data, int count, int tag, MPI_Datatype type)
{
	MPI_Send(data, count, type, id, tag, MPI_COMM_WORLD);
}
//MPI_Recvのエイリアス
void Recv(int id, void* data, int count, int tag, MPI_Datatype type)
{
	MPI_Recv(data, count, type, id, tag, MPI_COMM_WORLD, NULL);
}

//配列の各要素を、大小で分別する（双単調マージに使用）
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

//ブロック奇遇ソート（双単調マージ交換）の1step（step=0～7を順に実行）
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

//ソートされた数値列の出力用（だった。出力を消してエラーチェックのみに流用）
int printnums(int* nums, int myid, int npp, int prev){
	int i;
	int ret=0;
	for(i=0; i<npp; i++){
		if((i==0 && prev > nums[0]) || (i!=0 && nums[i-1] > nums[i])){
			ret++;
		}
	}
	return ret;
}

int main(int argc, char **argv){

	int finish = 0;
	int NPP = 1000;

	int myid, numproc;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	int i,j;
	while(!finish){

    	int nums[NPP];

    	//配布
    	if(myid == 0){
    		srand((unsigned)time(NULL));
    		for(i=0; i<NPP;i++) nums[i] = rand();

    		for(i=1; i<numproc; i++){
    			int senddata[NPP];
    			for(j=0; j<NPP; j++) senddata[j] = rand();
    			Send(i, senddata, NPP, 0, MPI_INT);
    		}
    	}
    	else
    	{
    		Recv(0, nums, NPP, 0, MPI_INT);
    	}

        //-----------------------------------------時間計測ここから
    	MPI_Barrier(MPI_COMM_WORLD);
    	double t1 = MPI_Wtime();

    	//ソート
    	for(i=0; i<8; i++)
    		SortStep(myid, nums, NPP, i, numproc);
    	qsort(nums, NPP, sizeof(int), comp);

    	MPI_Barrier(MPI_COMM_WORLD);
    	double t2 = MPI_Wtime();
        //-----------------------------------------時間計測ここまで

    	double time = t2 - t1;
    	double max = 0;

        //経過時間をプロセス0に送って最大値を計算、出力、送り返す
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

    		for(i=0; i<numproc; i++){
    			if(max<times[i]) max=times[i];
    		}
    		//printf("N=%d: time=%lf\n",NPP*numproc,max);
    		printf("N=%d error= %d time=%lf\n", NPP*numproc,err,max);
            for(i=1; i<numproc; i++)
                Send(i, &max, 1, i, MPI_DOUBLE);
        }else{
    		Send(0, nums, NPP, myid, MPI_INT);
    		Send(0, &time, 1, myid, MPI_DOUBLE);
            Recv(0, &max, 1, myid, MPI_DOUBLE);
    	}

        //経過時間によって各プロセスが処理する数の増分を変更
    	if(max<=0.2f){
            if(max<0.15f)
                NPP+=10000;
            else if(max<0.195f)
                NPP += 1000;
            else if(max<0.199f)
                NPP += 100;
            else
                NPP += 10;
        }
    	else finish = 1; //0.2sを超えたら終了

	}
	MPI_Finalize();
	return 0;
}
