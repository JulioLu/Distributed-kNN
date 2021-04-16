#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cblas.h>
#include<time.h>
#include<sys/resource.h>
#include<mpi.h>



int world_size;
int world_rank;


struct timespec start, finish, t0, t1;

typedef struct knnresult{
  int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      m;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;


knnresult kNN(double *X, double *Y, int n, int m, int d, int k);
void knnSelect(double *distanses, int *idx, int n, int m, int k, knnresult *data);
knnresult distrAllkNN(double * X, int n, int d, int k);
knnresult knnConector(knnresult a, knnresult b);
void insertionSort(double *a, int n);
int binarySearch(double *a, double item, int low, int high);
void work(double distance, int pointId, double *ni,int *id ,int k);

void main(int argc, char *argv[]){

    MPI_Init(&argc,&argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        fprintf(stderr,"Requires at least two processes.\n");
        exit(-1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Request mpirequest;
    MPI_Status  sendstatus;
    
    int n = atoi(argv[1]);
    int d = atoi(argv[2]);
    int k = atoi(argv[3]);
    int distribution = n*d/world_size;
    knnresult result;

    double *Xbuffer = (double*)malloc(distribution * sizeof(double));
    double *X;
    if(world_rank == 0){

        double *X = malloc(n*d * sizeof(double));
        
        for(int i=0;i<n*d;i++) X[i] = ( (double) (rand()) ) / (double) RAND_MAX;
        
        //for(int i=0;i<n*d;i++) printf("X[%d] = %f\n",i,X[i]);

        //double X[16] = { 101,102 ,103,105, 106,108, 108,112,213,215, 217,223, 225,240, 245,260 };//, 280,290 ,483,569,865,974,  1120,1345,   2345,2134, 5678,5678 ,315,340, 7897,87891 };//, 3000, 32000};
        
        for(int i=0; i<world_size; i++){
            MPI_Isend(X+distribution*i, distribution, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &mpirequest);
        }

    }

    MPI_Recv(Xbuffer, distribution, MPI_DOUBLE, 0, world_rank, MPI_COMM_WORLD, &sendstatus);


    clock_gettime(CLOCK_REALTIME, &start);
    result = distrAllkNN(Xbuffer, n, d, k);
    clock_gettime(CLOCK_REALTIME, &finish);
    double duration = ((finish.tv_sec - start.tv_sec) * 1000000 + (finish.tv_nsec - start.tv_nsec) / 1000) / 1000000.0;


   // for(int i=0; i<n*k/world_size; i++){ printf("ndist[%d]=%f from %d\n" ,i ,result.ndist[i], world_rank);}
    
    //printf("\n");
    
    //for(int i=0; i<n*k/world_size; i++){ printf("nidx[%d]=%d from %d\n" ,i ,result.nidx[i], world_rank); }

    printf("Duration: %f seconds , rank: %d\n", duration , world_rank);

    MPI_Finalize();

}

knnresult distrAllkNN(double *X, int n, int d, int k){
    int distribution   = n*d/world_size;
    double *Ybuffer = (double*)malloc(distribution * sizeof(double));
    double *Zbuffer = (double*)malloc(distribution * sizeof(double));

    MPI_Request mpirequest;
    MPI_Status  sendstatus;
    knnresult firstResult, secondResult, finialResult;

    //MPI_Scatter(X,distribution,MPI_DOUBLE,Xbuffer, distribution, MPI_DOUBLE, MPI_COMM_WORLD);
    firstResult = kNN(X, X, n/world_size, n/world_size, d, k);

    Ybuffer = X;

    for(int i=0; i<world_size-1; i++){
       
        if(world_rank == 0)
        {
            MPI_Isend(Ybuffer, distribution, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &mpirequest);
            MPI_Recv (Zbuffer, distribution, MPI_DOUBLE, world_size - 1, world_rank, MPI_COMM_WORLD, &sendstatus); 
            
            world_rank += world_size - 1 - i;
            secondResult = kNN(Zbuffer ,X,  n/world_size, n/world_size, d, k);
            world_rank -= world_size - 1 - i;
                
            firstResult  = knnConector(firstResult, secondResult);
        }
        else if(world_rank != world_size-1)
        {
            MPI_Isend(Ybuffer      , distribution, MPI_DOUBLE, world_rank + 1, world_rank + 1, MPI_COMM_WORLD, &mpirequest);
            MPI_Recv (Zbuffer, distribution, MPI_DOUBLE, world_rank - 1, world_rank, MPI_COMM_WORLD, &sendstatus);
            
            world_rank += + i + 1;
            secondResult = kNN( Zbuffer,X,  n/world_size, n/world_size, d, k);
            world_rank -= + i + 1;

            firstResult = knnConector(firstResult, secondResult);
        }
        else
        {
            MPI_Isend(Ybuffer      , distribution, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &mpirequest);
            MPI_Recv (Zbuffer, distribution, MPI_DOUBLE, world_rank - 1, world_rank, MPI_COMM_WORLD, &sendstatus);
            
            world_rank += 0 + i;
            secondResult = kNN( Zbuffer,X, n/world_size, n/world_size, d, k);
            world_rank -= 0 + i;
            
            firstResult = knnConector(firstResult, secondResult);
        }

        Ybuffer = Zbuffer;

    }

    return firstResult;

}

knnresult knnConector(knnresult a, knnresult b){
    
    #pragma omp for schedule (dynamic) nowait
    for(int l=0; l<a.k * a.m; l+=a.k)
    {
        for(int e=0; e<a.k; e++)
        {
            int i;
            for(i=0; i<a.k - 1; i++)
            {
                if(a.ndist[l+i]>b.ndist[l+e]) break;
            }
            
            if(a.ndist[l+a.k-1]>b.ndist[l+e])
            {
                for(int j=a.k-2; j>=i; j--)
                {
                    a.nidx [l+j+1] = a.nidx [l+j];
                    a.ndist[l+j+1] = a.ndist[l+j];            
                }

                a.nidx [l+i] = b.nidx [l+e];
                a.ndist[l+i] = b.ndist[l+e];
            }
        }
    }
    return a;
}

knnresult kNN(double * X, double * Y, int n, int m, int d, int k){

    knnresult *data = malloc(sizeof(knnresult));

    data->k = k;
    data->m = n;
    data->nidx  = malloc(n*k * sizeof(int));
    data->ndist = malloc(n*k * sizeof(double));

    double *ni =malloc(n*k *sizeof(double));
    int *ids =malloc(n*k *sizeof(int));
    for(int i=0; i<n*k; i++){
        ni[i]=INFINITY;
    }
    
    int    *idx       = malloc(m*n * sizeof(int));
    double *distanses = malloc(n*m * sizeof(double));
    double  xsum[n], ysum[m];
    
    for(int i=0; i<m; i++) ysum[i] = cblas_ddot(d, Y + i*d, 1, Y + i*d, 1);

    for(int i=0; i<n; i++) xsum[i] = cblas_ddot(d, X + i*d, 1, X + i*d, 1);
    

    cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasTrans, m, n ,d , -2, Y, d, X, d, 0, distanses , n);

    
    for(int i=0;  i<m; i++){
        int e=0;
        for(int j=0; j<n; j++){
            idx      [j + i*n] = j; 
            distanses[j + n*i] = sqrt(distanses[j + n*i] + xsum[j] + ysum[i]);
            
            if(distanses[j + n*i] < ni[k-1 + k*i]  ){
                work(distanses[j + n*i],j ,ni + i*k ,ids  + i*k,k);
            }

        }
    }

    for(int i=0; i<n*k; i++){
        data->ndist[i]=ni[i];
        data->nidx[i]=ids[i];
    }

    return *data;

}

void work(double distance, int pointId, double *ni,int *id ,int k){

  

    int i;
    for(i=0;i<k-1;i++){
    if(ni[i] > distance)
        break;
    }

    if(distance < ni[k-1]){
        for(int j=k-2; j>=i; j--){
            id[j+1] = id[j];
            ni[j+1] = ni[j];
        }
        id[i] = pointId;
        ni[i] = distance;
    }
    

}