
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

struct Entry {
    int key;
    int value;
    int valid;
};


__global__ void insert_atomic_correct(
    Entry* table,
    int table_size,
    int n,
    int* cas_conflict,
    int* probe_count,
    int* failed_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int idx = (tid * 7) % table_size;

    int attempts = 0;
    int max_attempts = table_size;

    while (attempts < max_attempts) {

        //  尝试占位
        int old = atomicCAS(&table[idx].valid, 0, 1);

        if (old == 0) {
            table[idx].key = tid;
            table[idx].value = tid;
            return;
        }
        else {
            //  真正冲突（CAS失败）
            atomicAdd(cas_conflict, 1);

            idx = (idx + 1) % table_size;
            attempts++;

            //  探测步数
            atomicAdd(probe_count, 1);
        }
    }

    atomicAdd(failed_count, 1);
}

//--------------------------------------
//  非atomic版本（对照）
//--------------------------------------
__global__ void insert_no_atomic_correct(
    Entry* table,
    int table_size,
    int n,
    int* probe_count,
    int* failed_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int idx = (tid * 7) % table_size;

    int attempts = 0;
    int max_attempts = table_size;

    while (attempts < max_attempts) {

        if (table[idx].valid == 0) {
            table[idx].valid = 1;
            table[idx].key = tid;
            table[idx].value = tid;
            return;
        }
        else {
            idx = (idx + 1) % table_size;
            attempts++;

            atomicAdd(probe_count, 1);
        }
    }

    atomicAdd(failed_count, 1);
}

int count_valid(Entry* table, int size) {
    int c = 0;
    for (int i = 0; i < size; i++) {
        if (table[i].valid == 1) c++;
    }
    return c;
}


void run(int THREADS, int TABLE_SIZE) {

    Entry* d_table;
    Entry* h_table = (Entry*)malloc(TABLE_SIZE * sizeof(Entry));

    int* d_cas_conflict, * d_probe, * d_failed;
    int h_cas, h_probe, h_failed;

    cudaMalloc(&d_table, TABLE_SIZE * sizeof(Entry));
    cudaMalloc(&d_cas_conflict, sizeof(int));
    cudaMalloc(&d_probe, sizeof(int));
    cudaMalloc(&d_failed, sizeof(int));

    cudaEvent_t start, stop;
    float t1, t2;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int block = 256;
    int grid = (THREADS + block - 1) / block;

    // 非 atomic
    cudaMemset(d_table, 0, TABLE_SIZE * sizeof(Entry));
    cudaMemset(d_probe, 0, sizeof(int));
    cudaMemset(d_failed, 0, sizeof(int));

    cudaEventRecord(start);
    insert_no_atomic_correct << <grid, block >> > (d_table, TABLE_SIZE, THREADS, d_probe, d_failed);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&t1, start, stop);

    cudaMemcpy(h_table, d_table, TABLE_SIZE * sizeof(Entry), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_probe, d_probe, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_failed, d_failed, sizeof(int), cudaMemcpyDeviceToHost);

    int success1 = count_valid(h_table, TABLE_SIZE);

    // atomic
    cudaMemset(d_table, 0, TABLE_SIZE * sizeof(Entry));
    cudaMemset(d_cas_conflict, 0, sizeof(int));
    cudaMemset(d_probe, 0, sizeof(int));
    cudaMemset(d_failed, 0, sizeof(int));

    cudaEventRecord(start);
    insert_atomic_correct << <grid, block >> > (d_table, TABLE_SIZE, THREADS,
        d_cas_conflict, d_probe, d_failed);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&t2, start, stop);

    cudaMemcpy(h_table, d_table, TABLE_SIZE * sizeof(Entry), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_cas, d_cas_conflict, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_probe, d_probe, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_failed, d_failed, sizeof(int), cudaMemcpyDeviceToHost);

    int success2 = count_valid(h_table, TABLE_SIZE);

    printf("线程数=%d, 表大小=%d\n"
        "  非原子: 成功=%d, 失败=%d, 探测=%d, 耗时=%.4f ms\n"
        "  原子  : 成功=%d, 失败=%d, CAS冲突=%d, 探测=%d, 耗时=%.4f ms\n\n",
        THREADS,
        TABLE_SIZE,
        success1,
        h_failed,     // 注意：这里其实是 non-atomic 的 failed
        h_probe,      // non-atomic probe
        t1,           //  non-atomic 时间

        success2,
        h_failed,     // atomic failed
        h_cas,
        h_probe,
        t2            //  atomic 时间
    );

    cudaFree(d_table);
    cudaFree(d_cas_conflict);
    cudaFree(d_probe);
    cudaFree(d_failed);
    free(h_table);
}

//--------------------------------------
int main() {

    int thread_list[] = { 500,1000 };//线程大小
    int table_list[] = { 1024, 2048 };//哈希表大小

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            run(thread_list[i], table_list[j]);
        }
    }

    cudaDeviceReset();
    return 0;
}
