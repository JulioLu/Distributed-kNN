#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cblas.h>
#include<time.h>
#include<sys/resource.h>
#include<omp.h>
//#include <cilk/cilk.h>



struct timespec star, finish, t0, t1;

// Definition of the kNN result struct
typedef struct knnresult{
  int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      m;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;

//! Compute k nearest neighbors of each point in X [n-by-d]
/*!

  \param  X      Corpus data points              [n-by-d]
  \param  Y      Query data points               [m-by-d]
  \param  n      Number of corpus points         [scalar]
  \param  m      Number of query points          [scalar]
  \param  d      Number of dimensions            [scalar]
  \param  k      Number of neighbors             [scalar]

  \return  The kNN result
*/
knnresult kNN(double *X, double *Y, int n, int m, int d, int k);
void knnSelect(double *distanses, int *idx, int n, int m, int k, knnresult *data);
void insertionSort(double *a, int n);
int binarySearch(double *a, double item, 
                 int low, int high);

void work(double distance, int pointId, double *ni,int *id ,int k);


void main(int argc, char *argv[]){


    int n = 6;
    int m = 6;
    int d = 2;
    int k = 2;
    double X[16] = {1.0,2.0, 1.0,-3.0, 4.0,-1.0, -2.0,-7.0, -8.0,-5.0, 9.0,0.0, 7,-9, 9,-3};
    double Y[16] =  {1.0,2.0, 1.0,-3.0, 4.0,-1.0, -2.0,-7.0, -8.0,-5.0, 9.0,0.0, 7,-9, 9,-3};
    //double *X = malloc(n*d * sizeof(double));
    //double *Y = malloc(m*d * sizeof(double));
    
    clock_gettime(CLOCK_REALTIME, &star);
    knnresult result = kNN( X, Y, n, m, d, k);
    clock_gettime(CLOCK_REALTIME, &finish);

    double duration = ((finish.tv_sec - star.tv_sec) * 1000000 + (finish.tv_nsec - star.tv_nsec) / 1000) / 1000000.0;
    printf("Duration: %f\n", duration);

   
   
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