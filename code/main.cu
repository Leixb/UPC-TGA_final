#include<stdio.h>

#define N 1237
#define M 2311

enum modo{FILA, COLUMNA, ELEMENTO};

void CheckCudaError(char sms[], int line);

void Examen21(float *mA, float *mB, float *vC, float *vD) {
    int i, j;
    for (i=0; i<N; i++)
        for (j=0; j<M; j++)
            mA[i*M + j] = mA[i*M + j]*vC[i] - mB[i*M + j]*vD[j] + mA[i*M]*mB[7*M + j];
}

__global__ void kernel_columna(float mA[N*M], float mB[N*M], float vC[N], float vD[M], int *lock) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= M) return;

    if (j == 0) {
        for (int i=0; i<N; i++)
            mA[i*M] = mA[i*M]*vC[i] - mB[i*M]*vD[0] + mA[i*M]*mB[7*M];

        *lock = 1;
        return;
    }

    while(!*lock) __syncthreads();

    for (int i=0; i<N; i++)
        mA[i*M + j] = mA[i*M + j]*vC[i] - mB[i*M + j]*vD[j] + mA[i*M]*mB[7*M + j];
}

__global__ void kernel_fila(float mA[N*M], float mB[N*M], float vC[N], float vD[M]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    for (int j=0; j<M; j++)
        mA[i*M + j] = mA[i*M + j]*vC[i] - mB[i*M + j]*vD[j] + mA[i*M]*mB[7*M + j];
}

__global__ void kernel_elemento(float mA[N*M], float mB[N*M], float vC[N], float vD[M], int *lock) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= M) return;

    if (j == 0) {
        mA[i*M] = mA[i*M]*vC[i] - mB[i*M]*vD[0] + mA[i*M]*mB[7*M];
        lock[i] = 1;
        return;
    }

    while(!lock[i]) __syncthreads();

    mA[i*M + j] = mA[i*M + j]*vC[i] - mB[i*M + j]*vD[j] + mA[i*M]*mB[7*M + j];
}

int verify(float *mA_ref, float *mA, float tol) {
    for (int i=0; i<N*M; i++)
        if (fabs(mA_ref[i] - mA[i]) > tol) return 0;
    return 1;
}

int main(int argc, char **argv) {
    enum modo modo = FILA;

    if (argc > 1) {
        if ((strcmp("fila", argv[1])) == 0) modo = FILA;
        else if ((strcmp("col", argv[1])) == 0) modo = COLUMNA;
        else if ((strcmp("ele", argv[1])) == 0) modo = ELEMENTO;
        else {
            fprintf(stderr, "Parámetro inválido\n");
            return 1;
        }
    }

    float *mA, *mB, *vC, *vD;

    mA = (float*)malloc(sizeof(float)*N*M);
    mB = (float*)malloc(sizeof(float)*N*M);
    vC = (float*)malloc(sizeof(float)*N);
    vD = (float*)malloc(sizeof(float)*M);

    // Rellenamos las matrices con valores de prueba
    for (int i=0; i<N; i++) {
        for (int j=0; j<M; j++) {
            mA[i*M + j] = 1.0 + (i*3)%7;
            mB[i*M + j] = 2.0 + j%11;
        }
        vC[i] = i*0.3;
    }
    for (int j=0; j<M; j++) vD[j] = j*0.75;

    float *mA_dev, *mB_dev, *vC_dev, *vD_dev;

    cudaMalloc((float**)&mA_dev, sizeof(float)*N*M);
    cudaMalloc((float**)&mB_dev, sizeof(float)*N*M);
    cudaMalloc((float**)&vC_dev, sizeof(float)*N);
    cudaMalloc((float**)&vD_dev, sizeof(float)*M);
    CheckCudaError((char *) "Obtener Memoria en el device", __LINE__);

    cudaMemcpy(mA_dev, mA, sizeof(float)*N*M, cudaMemcpyHostToDevice);
    cudaMemcpy(mB_dev, mB, sizeof(float)*N*M, cudaMemcpyHostToDevice);
    cudaMemcpy(vC_dev, vC, sizeof(float)*N,   cudaMemcpyHostToDevice);
    cudaMemcpy(vD_dev, vD, sizeof(float)*M,   cudaMemcpyHostToDevice);
    CheckCudaError((char *) "Memcpy H -> D", __LINE__);

    int *lock;
    dim3 dimGrid, dimBlock;

    switch (modo) {
        case FILA:
            dimBlock = dim3(1024, 1, 1);
            dimGrid = dim3((N*M + dimBlock.x - 1)/dimBlock.x, 1, 1);

            kernel_fila<<<dimGrid, dimBlock>>>(mA_dev, mB_dev, vC_dev, vD_dev);
            break;
        case COLUMNA:
            cudaMalloc((int**)&lock, sizeof(int));
            cudaMemset(lock, 0, sizeof(int));
            CheckCudaError((char *) "Crear lock", __LINE__);

            dimBlock = dim3(1024, 1, 1);
            dimGrid = dim3((N*M + dimBlock.x - 1)/dimBlock.x, 1, 1);

            kernel_columna<<<dimGrid, dimBlock>>>(mA_dev, mB_dev, vC_dev, vD_dev, lock);
            break;
        case ELEMENTO:
            cudaMalloc((int**)&lock, N*sizeof(int));
            cudaMemset(lock, 0, N*sizeof(int));
            CheckCudaError((char *) "Crear lock", __LINE__);

            dimBlock = dim3(32, 32, 1);
            dimGrid = dim3((N + dimBlock.x - 1)/dimBlock.x, (M + dimBlock.y - 1)/dimBlock.y , 1);

            kernel_elemento<<<dimGrid, dimBlock>>>(mA_dev, mB_dev, vC_dev, vD_dev, lock);
            break;
        default:
            fprintf(stderr, "ERROR\n");
    }
    CheckCudaError((char *) "Kernel", __LINE__);

    float *mA_cuda = (float*)malloc(sizeof(float)*N*M);

    cudaMemcpy(mA_cuda, mA_dev, sizeof(float)*N*M, cudaMemcpyDeviceToHost);
    CheckCudaError((char *) "Memcpy D -> H", __LINE__);

    cudaFree(mA_dev);
    cudaFree(mB_dev);
    cudaFree(vC_dev);
    cudaFree(vD_dev);

    cudaDeviceSynchronize();

    Examen21(mA, mB, vC, vD);

    // Comprobación con tolerancia alta debido a errores de float
    if (!verify(mA, mA_cuda, 1e-2)) {
        fprintf(stderr, "FAIL\n");
        return 1;
    }

    fprintf(stderr, "OK\n");
}

void CheckCudaError(char sms[], int line) {
    cudaError_t error;

    error = cudaGetLastError();
    if (error) {
        fprintf(stderr, "(ERROR) %s - %s in %s at line %d\n", sms, cudaGetErrorString(error), __FILE__, line);
        exit(EXIT_FAILURE);
    } // else fprintf(stderr, "(OK) %s \n", sms);
}
