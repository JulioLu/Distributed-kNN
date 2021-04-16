#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cblas.h>
#include<time.h>
#include<sys/resource.h>
#include<mpi.h>

struct timespec start, finish, t0, t1;

// Definition of the kNN result struct
typedef struct knnresult{
  int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      m;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;

typedef struct node{
    double    mu;
    int pointId;
    double *cordination;
    struct node   *left;
    struct node   *right;
} node;

typedef struct pointData{
    int    *id;
    double *cordination;
    double *ndist;
} pointData;

node      *makeTree(double *X, int n, int  d, int *nidx);
pointData *work(double *VPcordination,int VPpointId, pointData *p, int d, int k);
pointData *searchVPT(node *root, pointData *p, int d,int k);
int    *   insertionSort1(double *arr, int n, int *idxHelp);
int    *dis1(double * X, double * VpCord, int n, int d, int *idxHelp);
double *sortX(double *X, int n, int d, int *idxHelp);
void       insertionSort(double *arr, int n);
double    *dis(double * X, double * VpCord, int n, int d);
double     distance(double *pointA, double *pointB, int d);
void fill(pointData **p ,int n, int d, int k, double *Xbuffer);
knnresult V2(double *Xbuffer,int n, int d, int k, int *idxHelp, pointData **p);
void encoder(node *root, double *data, int d, int *i);
node *decoder(node *root1, double *data, int d, int *i);
void insertionSort(double *a, int n);
int binarySearch(double *a, double item, 
                 int low, int high);



int world_size, world_rank;

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


    double *Xbuffer = (double*)malloc(distribution * sizeof(double));
    if(world_rank == 0){

        double *X = malloc(n*d * sizeof(double));

        for(int i=0;i<n*d;i++) X[i] = ( (double) (rand()) ) / (double) RAND_MAX;
        
        for(int i=0; i<world_size; i++){
            MPI_Isend(X+distribution*i, distribution, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &mpirequest);
        }

    }

    MPI_Recv(Xbuffer, distribution, MPI_DOUBLE, 0, world_rank, MPI_COMM_WORLD, &sendstatus);
    n= n/world_size;
    pointData *(*p) =  malloc(n * sizeof(pointData*));
    int *idxHelp = (int *)malloc(n* sizeof(int));
   

    clock_gettime(CLOCK_REALTIME, &start);
    knnresult data = V2(Xbuffer, n, d, k, idxHelp, p);
    clock_gettime(CLOCK_REALTIME, &finish);
    
    double duration = ((finish.tv_sec - start.tv_sec) * 1000000 + (finish.tv_nsec - start.tv_nsec) / 1000) / 1000000.0;


    printf("Duration: %f seconds , rank: %d\n", duration , world_rank);

    MPI_Finalize();

}



knnresult V2(double *Xbuffer,int n, int d, int k, int *idxHelp, pointData **p){
   
    MPI_Request mpirequest;
    MPI_Status  sendstatus;
   
    Xbuffer = sortX( Xbuffer, n, d, idxHelp);
    
    fill(p, n, d, k, Xbuffer);
    node *root;
    root = makeTree(Xbuffer, n, d, idxHelp);
    int size = d*n + 2*n + 2*n;
    double *datasend = malloc(size * sizeof(double));
    double *dataRec  = malloc(size * sizeof(double));
    int    *i        = malloc(sizeof(int));
    
    *i=0;
    encoder(root, datasend, d, i);
    for(int j=0; j<n; j++){
       p[j] = searchVPT(root, p[j], d, k);
    }

    node *root1;
        
    for(int j=0; j<world_size-1; j++){
       
        if(world_rank == 0)
        {
            MPI_Isend(datasend, size, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &mpirequest);
            MPI_Recv (dataRec, size, MPI_DOUBLE, world_size - 1, world_rank, MPI_COMM_WORLD, &sendstatus);
        }
        else if(world_rank != world_size-1)
        {
            MPI_Isend(datasend      , size, MPI_DOUBLE, world_rank + 1, world_rank + 1, MPI_COMM_WORLD, &mpirequest);
            MPI_Recv (dataRec, size, MPI_DOUBLE, world_rank - 1, world_rank, MPI_COMM_WORLD, &sendstatus);
        }
        else
        {
            MPI_Isend(datasend      , size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &mpirequest);
            MPI_Recv (dataRec, size, MPI_DOUBLE, world_rank - 1, world_rank, MPI_COMM_WORLD, &sendstatus);
        }
        *i=0;
        
        root1 = decoder( root1, dataRec, d, i);
        for(int z=0; z<n; z++){
            p[z] = searchVPT(root1, p[z], d, k);
        }
    }

    knnresult *data = malloc(sizeof(knnresult));
    data->m=n;
    data->k=k;
    data->ndist = malloc(k*n * sizeof(double));
    data->nidx  = malloc(k*n * sizeof(double));
    for(int j=0; j<n; j++){
        for(int e=0; e<k; e++){
            data->ndist[k*j + e] = p[j]->ndist[e];
            data->nidx[k*j + e] = p[j]->id[e];
        }
    }

    return *data;

}


void encoder(node *root, double *data, int d, int *i){
    
    int hasBoth  = 0;
    int hasLeft  = 0;
    int HasRight = 0;
    
    if(!(root->left  == NULL)) hasLeft  = 1;
    if(!(root->right == NULL)) HasRight = 1;
    
    data[(*i)++] = hasLeft;
    data[(*i)++] = HasRight;
    data[(*i)++] = root->mu;
    data[(*i)++] = root->pointId;
    
    for(int e=0; e<d; e++) data[(*i)++] = root->cordination[e];

    if(root->left  == NULL && root->right == NULL) {return;}

    if(!(root->left  == NULL)) encoder(root->left , data, d, i);
    if(!(root->right == NULL)) encoder(root->right, data, d,i );

    return;
}

node *decoder(node *root1, double *data, int d, int *i){

    int hasLeft  = 0;
    int hasRight = 0;
    
    hasLeft  = data[(*i)++];
    hasRight = data[(*i)++];
    
   
    root1 = malloc(sizeof(node));
    root1->mu          = data[(*i)++];
    root1->pointId     = data[(*i)++];

    root1->cordination = malloc(d * sizeof(double));
    for(int e=0; e<d; e++) root1->cordination[e] = data[(*i)++];

    if(hasLeft == 0 && hasRight == 0) return root1;
    if(hasLeft  == 1) root1->left  = decoder(root1->left , data, d, i);
    if(hasRight == 1) root1->right = decoder(root1->right, data, d, i);

    return root1;
}

node *makeTree(double *X, int n, int d, int *nidx){
    
    node *root = malloc(sizeof(node));
    root->cordination = malloc(d*sizeof(double));
    double *leftSide;  
    double *rightSide; 
    double *result;
    int    *idsleft;
    int    *idsright;
    root->pointId = *nidx;

    idsleft   = nidx + 1;
    idsright  = nidx + n/2 + 1;
    leftSide  = X + d;
    rightSide = X + d*(n/2) +d;
    

    for(int i=0; i<d; i++) { root->cordination[i] = X[i]; }
    result = dis(X, root->cordination, n, d);
    root->mu = result[(n-1)/2];

    if(n == 1){

        return root;

    }
    else{

        root->left = makeTree(leftSide, n/2, d, idsleft);

        if((n-1)/2 != 0) 
            root->right = makeTree(rightSide, (n-1)/2, d, idsright);

    }
    
    return root;

}

pointData *work(double *VPcordination, int VPpointId, pointData *p, int d, int k){

    double distance = 0;
    for(int i=0; i<d; i++){
        distance += (VPcordination[i] - p->cordination[i]) * (VPcordination[i] - p->cordination[i]);
    }

    if(distance == 0) return p;
    distance = sqrt(distance);

    int i;
    for(i=0;i<k-1;i++){
    if(p->ndist[i]>distance)
        break;
    }

    if(distance < p->ndist[k-1]){
        for(int j=k-2; j>=i; j--){
            p->id[j+1] = p->id[j];
            p->ndist[j+1] = p->ndist[j];
        }
        p->id[i] = VPpointId;
        p->ndist[i] = distance;
    }
    
    return p;

}

double distance(double *pointA, double *pointB, int d){
    double dis = 0;
    for(int i=0; i<d; i++){
        dis += (pointA[i] - pointB[i]) * (pointA[i] - pointB[i]);
    } 
    return sqrt(dis);
}

pointData *searchVPT(node *root, pointData *p, int d,int k){

    p = work(root->cordination, root->pointId, p,d,k);

    if ((root->right == NULL && root->left == NULL)) return p;

    if(distance(root->cordination, p->cordination, d) < root->mu ){

        if(!(root->left == NULL)) p = searchVPT(root->left, p, d,k);

        if( (distance(root->cordination, p->cordination, d) >  fabs( (root->mu - p->ndist[k-1]) ) ) &&  (distance(root->cordination, p->cordination, d) < root->mu + p->ndist[k-1]) )
            if(!(root->right == NULL)) p = searchVPT(root->right, p, d,k);

    }
    else {//if(distance(root->cordination, p->cordination, d) >= root->mu){

        if(!(root->right == NULL)) p = searchVPT(root->right, p, d,k);
        
        if((distance(root->cordination, p->cordination, d) >  fabs( (root->mu - p->ndist[k-1]) ) ) &&  (distance(root->cordination, p->cordination, d) < root->mu + p->ndist[k-1]) )
            if(!(root->left == NULL)) p = searchVPT(root->left, p, d,k);

    }
    
    return p;
}


void fill(pointData **p ,int n, int d, int k, double *Xbuffer){
    for(int j=0; j<n; j++){
        p[j] = malloc(sizeof(pointData));
        p[j]->cordination = malloc(d*sizeof(double));
    }

    for(int j=0; j<d*n; j+=d){
        for(int i=0; i<d; i++){
            p[j/d]->cordination[i]= Xbuffer[j+i];
        }
    }
    for(int j=0; j<n; j++){
        p[j]->id = malloc(k*sizeof(int));
        p[j]->ndist = malloc(k*sizeof(double));
    }

    for(int j=0; j<n; j++){
        for(int i=0; i<k; i++){
            p[j]->id[i] = -1;
            p[j]->ndist[i] = INFINITY;
        }
    }
}


double *sortX(double *X, int n, int d, int *idxHelp){
    
    
    double *zero = malloc(d*sizeof(double));
    for(int i=0; i<d; i++) zero[i]=0;
    for(int i=0; i<n; i++) idxHelp[i] = i;
    
    idxHelp = dis1(X, zero, n, d, idxHelp);
   
    double *Xtemp = malloc(n*d * sizeof(double));
    for(int i=0,e=0; i<n; i++,e+=2){
        for(int j=0; j<d; j++){
            Xtemp[e+j] = X[d*idxHelp[i]+j]; 
        }
    }

     for(int i=0; i<n; i++){idxHelp[i]+=world_rank*n;}

   
    return Xtemp;
}

int *dis1(double * X, double * VpCord, int n, int d, int *idxHelp){
    
    double *D = (double *) malloc(n*sizeof(double));

    for(int i=0; i<n; i++)
        D[i] = distance(X+d*i, VpCord, d);

    insertionSort1(D,  n, idxHelp);

    return idxHelp;
}

int binarySearch(double *a, double item, 
                 int low, int high)
{
    if (high <= low)
        return (item > a[low]) ? 
                (low + 1) : low;
 
    int mid = (low + high) / 2;
 
    if (item == a[mid])
        return mid + 1;
 
    if (item > a[mid])
        return binarySearch(a, item, 
                            mid + 1, high);
    return binarySearch(a, item, low, 
                        mid - 1);
}
 
// Function to sort an array a[] of size 'n'
int *insertionSort1(double *a, int n, int *idxHelp)
{
    int i, loc, j, key;
    double selected;
 
    for (i = 1; i < n; ++i) 
    {
        j = i - 1;
        selected = a[i];
        key = idxHelp[i];
        // find location where selected sould be inseretd
        loc = binarySearch(a, selected, 0, j);
 
        // Move all elements after location to create space
        while (j >= loc) 
        {
            idxHelp[j+1]=idxHelp[j];
            a[j + 1] = a[j];
            j--;
        }
        idxHelp[j+1] = key;
        a[j + 1] = selected;
    }
    return idxHelp;
}

double *dis(double * X, double * VpCord, int n, int d){
    
    double *D = (double *) malloc((n-1)*sizeof(double));

    for(int i=1; i<n; i++)
        D[i-1] = distance(X+d*i, VpCord, d);

    insertionSort(D,  n-1);

    return D;
}


 
// Function to sort an array a[] of size 'n'
void insertionSort(double *a, int n)
{
    int i, loc, j, k;
    double selected;
 
    for (i = 1; i < n; ++i) 
    {
        j = i - 1;
        selected = a[i];
 
        // find location where selected sould be inseretd
        loc = binarySearch(a, selected, 0, j);
 
        // Move all elements after location to create space
        while (j >= loc) 
        {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = selected;
    }
}