
__kernel void matrix_mult(__global float* A,
                          __global float* B,
                          __global float* C,
                          __local float* Bwrk,  /* local memory of b column for work group */
                          const int ncol) {
    int i = get_global_id(0);
    int iloc = get_local_id(0);
    int nloc = get_local_size(0);

    float Awrk[AWRK_SIZE];  /* private memory of a column for work item */

    if (i < ncol) {
        /* copy elements in private memory */
        for (int k = 0 ; k < ncol ; k++) {
            Awrk[k] = A[i * ncol + k];
        }

        for (int j = 0 ; j < ncol ; j++) {
            /* copy elements in local memory */
            for (int k = iloc ; k < ncol ; k = k + nloc) {
                Bwrk[k] = B[k * ncol + j];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            float tmp = 0.0;
            for (int k = 0 ; k < ncol ; k++) {
                tmp += Awrk[k] * Bwrk[k];
            }
            C[i * ncol + j] = tmp;
        }
    }
}
