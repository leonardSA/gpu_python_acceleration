__kernel void matrix_mult(__global float *a,
                          __global float* b,
                          __global float* c,
                          const unsigned int a_ncol,
                          const unsigned int b_ncol) {
    int rows = get_global_id(0);    /* iterate over rows */
    int columns = get_global_id(1); /* then iterate over columns */

    /* compute value */
    float value = 0;
    for (unsigned int i = 0 ; i < a_ncol ; i++) {
        value += a[rows * a_ncol + i] * b[i * b_ncol + columns];
    }

    c[rows * b_ncol + columns] = value;
}
