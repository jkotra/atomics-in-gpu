// OpenCL Kernels: Emulated CAS Spin-Loop vs. Native Hardware FP32 Atomics
// Supported on Mesa Rusticl with cl_ext_float_atomics

// Function declaration for cl_ext_float_atomics
float __attribute__((overloadable)) atomic_fetch_add(volatile __global atomic_float *, float);

// -----------------------------------------------------------------------------
// Method 1: Emulated CAS Spin-Loop
// Reinterprets float bits as int and spins on atomic_cmpxchg until successful.
// -----------------------------------------------------------------------------
void atomic_add_f32_cas(volatile __global int* addr, float val) {
    int old_val, new_val, read_val;
    do {
        old_val = *addr;
        new_val = as_int(as_float(old_val) + val);
        read_val = atomic_cmpxchg(addr, old_val, new_val);
    } while (read_val != old_val);
}

__kernel void fruit_histogram_cas(
    __global int* out_bins,
    __global const int* category_ids,
    __global const float* weights,
    int N
) {
    int gid = get_global_id(0);
    if (gid < N) {
        int cat = category_ids[gid];
        atomic_add_f32_cas(&out_bins[cat], weights[gid]);
    }
}

// -----------------------------------------------------------------------------
// Method 2: Native Hardware Float Atomics (cl_ext_float_atomics)
// Issues single native hardware atomic float addition instruction directly at L2.
// -----------------------------------------------------------------------------
__kernel void fruit_histogram_native(
    volatile __global atomic_float* out_bins,
    __global const int* category_ids,
    __global const float* weights,
    int N
) {
    int gid = get_global_id(0);
    if (gid < N) {
        int cat = category_ids[gid];
        atomic_fetch_add(&out_bins[cat], weights[gid]);
    }
}
