#include<stdio.h>

#define N 100
#define M 100

enum modo{FILA, COLUMNA, ELEMENTO};

void CheckCudaError(char sms[], int line);

void Examen21(float *mA, float *mB, float *vC, float *vD) {
    int i, j;
    for (i=0; i<N; i++)
        for (j=0; j<M; j++)
            mA[i*M + j] = mA[i*M + j]*vC[i] - mB[i*M + j]*vD[j]; //+ mA[i*M]*mB[7*M + j];
}

__global__ void kernel_fila(float mA[N*M], float mB[N*M], float vC[N], float vD[M]) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= M) return;

    for (int i=0; i<N; i++)
        mA[M*i + j] = mA[M*i + j]*vC[i] - mB[M*i + j]*vD[j];

    /* for (int i=0; i<N; i++) */
        /* mA[M*i + j] += mA[M*i]*mB[M*7 + j]; */
}

__global__ void kernel_columna(float mA[N*M], float mB[N*M], float vC[N], float vD[M]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) return;

    for (int j=0; j<N; j++)
        mA[M*i + j] = mA[M*i + j]*vC[i] - mB[M*i + j]*vD[j];

    /* for (int j=0; j<N; i++) */
        /* mA[M*i + j] += mA[M*i]*mB[M*7 + j]; */
}

__global__ void kernel_elemento(float mA[N*M], float mB[N*M], float vC[N], float vD[M]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= N || j >= M) return;

    mA[M*i + j] = mA[M*i + j]*vC[i] - mB[M*i + j]*vD[j];

    /* mA[M*i + j] += mA[M*i]*mB[M*7 + j]; */
}

int verify(float *mA_ref, float *mA) {
    for (int i=0; i<N*M; i++)
        if (mA_ref[i] != mA[i]) return 0;
    return 1;
}

int main(int argc, char **argv) {
    CheckCudaError((char *) "Start", __LINE__);
    enum modo modo = FILA;

    if (argc > 1) {
        if ((strcmp("row", argv[1])) == 0) modo = FILA;
        else if ((strcmp("col", argv[1])) == 0) modo = COLUMNA;
        else if ((strcmp("ele", argv[1])) == 0) modo = ELEMENTO;
        else {
            fprintf(stderr, "Invalid argument\n");
            return 1;
        }
    }

    float *mA, *mB, *vC, *vD;

    mA = (float*)malloc(sizeof(float)*N*M);
    mB = (float*)malloc(sizeof(float)*N*M);
    vC = (float*)malloc(sizeof(float)*N);
    vD = (float*)malloc(sizeof(float)*M);

    // Fill matrices
    for (int i=0; i<N; i++) {
        for (int j=0; j<M; j++) {
            mA[i*M + j] = 1.0;
            mB[i*M + j] = 5.0;
        }
        vC[i] = 2.5;
    }
    for (int j=0; j<M; j++) vD[j] = 0.75;

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
    CheckCudaError((char *) "Memcpy", __LINE__);

    dim3 dimBlock(1, 1, 1);
    if (modo == ELEMENTO) dimBlock.x = dimBlock.y = 32;
    else dimBlock.x = 1024;

    dim3 dimGrid(1, 1, 1);
    dimGrid.x = (N + dimBlock.x - 1)/dimBlock.x;
    dimGrid.y = (M + dimBlock.y - 1)/dimBlock.y;

    switch (modo) {
        case FILA:
            kernel_fila<<<dimGrid, dimBlock>>>(mA_dev, mB_dev, vC_dev, vD_dev);
            break;
        case COLUMNA:
            kernel_columna<<<dimGrid, dimBlock>>>(mA_dev, mB_dev, vC_dev, vD_dev);
            break;
        case ELEMENTO:
            kernel_elemento<<<dimGrid, dimBlock>>>(mA_dev, mB_dev, vC_dev, vD_dev);
            break;
        default:
            fprintf(stderr, "ERROR\n");
    }
    CheckCudaError((char *) "Kernel", __LINE__);

    float *mA_cuda = (float*)malloc(sizeof(float)*N*M);

    cudaMemcpy(mA_cuda, mA_dev, sizeof(float)*N*M, cudaMemcpyDeviceToHost);

    cudaFree(mA_dev);
    cudaFree(mB_dev);
    cudaFree(vC_dev);
    cudaFree(vD_dev);

    cudaDeviceSynchronize();

    Examen21(mA, mB, vC, vD);

    if (verify(mA, mA_cuda)) {
        fprintf(stderr, "OK\n");
        return 0;
    } else {
        fprintf(stderr, "FAIL\n");
    }

    for (int i=0; i<N; i++) {
        for (int j=0; j<M; j++) {
            printf("%g %g ; ", mA[i*M + j], mA_cuda[i*M + j]);
        }
        printf("\n");
    }

}

void CheckCudaError(char sms[], int line) {
    cudaError_t error;

    error = cudaGetLastError();
    if (error) {
        fprintf(stderr, "(ERROR) %s - %s in %s at line %d\n", sms, cudaGetErrorString(error), __FILE__, line);
        exit(EXIT_FAILURE);
    } // else fprintf(stderr, "(OK) %s \n", sms);
}
