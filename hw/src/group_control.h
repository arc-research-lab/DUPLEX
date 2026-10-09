#ifndef GROUP_CONTROL_H
#define GROUP_CONTROL_H

// includes
#include <hls_vector.h>
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include "assert.h"
#include <iostream>
#include <cmath>
#include "etc/ap_utils.h"

#define IMG_HEIGHT 100
#define IMG_WIDTH 100
#define IMG_SIZE IMG_HEIGHT * IMG_WIDTH
#define TOTAL_ITERS IMG_SIZE

#define M 16
#define N 16
#define N_k 1
#define N_w 16
#define M_w (M / N_k)
#define D 8
#define RESP_D D

#define BRAM_LATENCY 1
#define ADJ_LATENCY 1
#define RESP_LAT BRAM_LATENCY
#define MAX_CYCLES (64 * TOTAL_ITERS + 1024)

#define B_DEPTH (IMG_SIZE / N_w)
#define PE_LEN (IMG_SIZE / M)

struct resp_t {
    bool valid;
    int  pe;
    int  output_idx;
    int  data;
};

struct req_t {
    int local_addr;
    int output_idx;
};

template<int DEPTH>
struct reg_queue_t {
    req_t data[DEPTH];
    int8_t   head;
    int8_t   tail;
    int8_t   count;

    void init() {
        head = 0; tail = 0; count = 0;
        for (int i = 0; i < DEPTH; i++) {
    #pragma HLS unroll
            data[i] = {0, 0};
        }
    }

    bool full()  const { return count >= DEPTH; }
    bool empty() const { return count == 0; }

    void push(req_t r) {
        data[tail] = r;
        tail = (tail + 1 == DEPTH) ? 0 : tail + 1;
        count++;
    }

    req_t pop() {
        req_t r = data[head];
        head = (head + 1 == DEPTH) ? 0 : head + 1;
        count--;
        return r;
    }

    req_t front() const { return data[head]; }

    void step(bool do_push, req_t in, bool do_pop) {
        for (int d = 0; d < DEPTH; d++) {
        #pragma HLS unroll
            bool w = do_push & (tail == d);
            data[d].local_addr = w ? in.local_addr : data[d].local_addr;
            data[d].output_idx = w ? in.output_idx : data[d].output_idx;
        }
        int8_t tail_inc = (tail + 1 == DEPTH) ? 0 : tail + 1;
        int8_t head_inc = (head + 1 == DEPTH) ? 0 : head + 1;
        tail  = do_push ? tail_inc : tail;
        head  = do_pop  ? head_inc : head;
        count = count + (do_push ? 1 : 0) - (do_pop ? 1 : 0);
    }
};

typedef reg_queue_t<D> req_queue_t;

typedef ap_uint<512> block_t;

typedef union{
    float data_float;
    unsigned int data_uint;
} fp_int;

#endif