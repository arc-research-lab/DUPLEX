#include "group_control.h"


// ------------------------------------------------------ //
// --------------------- load ------------------------- //
// ------------------------------------------------------ //

static void load_trans(int *x_trans, int x_local[M][IMG_SIZE / M]) {
    #pragma HLS inline off

    load_trans_i: for (int i = 0; i < IMG_SIZE; i++) {
    #pragma HLS pipeline II=1
        x_local[i % M][i / M] = x_trans[i];
    }
}

static void load_img(int *ext_data, int img_mem[N_k][N_w][B_DEPTH]) {
    #pragma HLS inline off

    load_img_i: for (int address = 0; address < IMG_SIZE; address++) {
    #pragma HLS pipeline II=1
        int bank_id = address % N_w;
        int local_addr = address / N_w;
        int val = ext_data[address];

        load_img_k: for (int k = 0; k < N_k; k++) {
    #pragma HLS unroll
            img_mem[k][bank_id][local_addr] = val;
        }
    }
}

// ------------------------------------------------------ //
// ----------- executes one cycle for one group --------- //
// ------------------------------------------------------ //

static void group_step(int k,
                       int bank_mem[N_w][B_DEPTH],
                       int x_local[M][IMG_SIZE / M],
                       int y_local[M][IMG_SIZE / M],
                       int output_local[M][IMG_SIZE / M],
                       int pe_iteration[M],
                       req_queue_t req_arr[M_w][N_w],
                       unsigned int rr[N_w],
                       resp_t resp_arr[N_w][RESP_LAT],
                       int slot_arr[N_w][RESP_LAT],
                       int resp_data[M_w][D],
                       int resp_idx[M_w][D],
                       bool resp_valid[M_w][D],
                       int resp_wr_ptr[M_w],
                       int resp_rd_ptr[M_w],
                       int resp_cnt[M_w],
                       int &completed) {
#pragma HLS inline

    // ---------------------------------------------------------------
    // request queue states
    // ---------------------------------------------------------------
    bool         q_full[M_w][N_w];
    req_t        q_head[M_w][N_w];
    unsigned int req_vec[N_w];
    #pragma HLS array_partition variable=q_full  dim=0 complete
    #pragma HLS array_partition variable=q_head  dim=0 complete
    #pragma HLS array_partition variable=req_vec dim=0 complete

    snap: for (int n = 0; n < N_w; n++) {
    #pragma HLS unroll
        unsigned int v = 0;
        for (int m = 0; m < M_w; m++) {
        #pragma HLS unroll
            q_full[m][n] = req_arr[m][n].full();
            q_head[m][n] = req_arr[m][n].front();
            v |= (req_arr[m][n].empty() ? 0u : 1u) << m;
        }
        req_vec[n] = v;
    }

    // ---------------------------------------------------------------
    // rr arbitration - candidate choice
    // ---------------------------------------------------------------
    unsigned int cand[N_w];
    #pragma HLS array_partition variable=cand dim=0 complete

    choose_candidate: for (int n = 0; n < N_w; n++) {
    #pragma HLS unroll
        unsigned int req = req_vec[n];
        unsigned int hi  = req & rr[n];
        unsigned int v   = (hi != 0) ? hi : req;  
        cand[n] = v & (~v + 1u); 
    }

    // ---------------------------------------------------------------
    // rank
    // ---------------------------------------------------------------
    int rank[N_w];
    #pragma HLS array_partition variable=rank dim=0 complete

    rank_compute: for (int n = 0; n < N_w; n++) {
    #pragma HLS unroll
        int r = 0;
        for (int np = 0; np < n; np++) {
        #pragma HLS unroll
            r += ((cand[n] & cand[np]) != 0) ? 1 : 0;
        }
        rank[n] = r;
    }

    // ---------------------------------------------------------------
    // admission
    // ---------------------------------------------------------------
    unsigned int adm[N_w];
    bool         admit[N_w];
    #pragma HLS array_partition variable=adm   dim=0 complete
    #pragma HLS array_partition variable=admit dim=0 complete

    admit_decide: for (int n = 0; n < N_w; n++) {
    #pragma HLS unroll
        unsigned int okm = 0;
        for (int m = 0; m < M_w; m++) {
        #pragma HLS unroll
            okm |= ((resp_cnt[m] + rank[n] < D) ? 1u : 0u) << m;
        }
        admit[n] = (cand[n] & okm) != 0;
        adm[n]   = admit[n] ? cand[n] : 0u;
    }

    int admit_count[M_w];
    #pragma HLS array_partition variable=admit_count dim=0 complete

    commit_count: for (int m = 0; m < M_w; m++) {
    #pragma HLS unroll
        int c = 0;
        for (int n = 0; n < N_w; n++) {
        #pragma HLS unroll
            c += (int)((adm[n] >> m) & 1u);
        }
        admit_count[m] = c;
    }

    // ---------------------------------------------------------------
    // land
    // ---------------------------------------------------------------
    bool land_hit [M_w][D];
    int  land_data[M_w][D];
    int  land_idx [M_w][D];
    #pragma HLS array_partition variable=land_hit  dim=0 complete
    #pragma HLS array_partition variable=land_data dim=0 complete
    #pragma HLS array_partition variable=land_idx  dim=0 complete

    land: for (int m = 0; m < M_w; m++) {
    #pragma HLS unroll
        for (int d = 0; d < D; d++) {
        #pragma HLS unroll
            int hits = 0;
            int data = 0;
            int idx  = 0;
            for (int n = 0; n < N_w; n++) {
            #pragma HLS unroll
                int sel = (resp_arr[n][RESP_LAT - 1].valid ? 1 : 0)
                        & ((resp_arr[n][RESP_LAT - 1].pe == m) ? 1 : 0)
                        & ((slot_arr[n][RESP_LAT - 1] == d) ? 1 : 0);
                int msk = -sel;
                hits += sel;
                data |= resp_arr[n][RESP_LAT - 1].data       & msk;
                idx  |= resp_arr[n][RESP_LAT - 1].output_idx & msk;
            }
            land_hit[m][d]  = (hits != 0);
            land_data[m][d] = data;
            land_idx[m][d]  = idx;
        }
    }

    // ---------------------------------------------------------------
    // drain
    // ---------------------------------------------------------------
    bool do_drain[M_w];
    #pragma HLS array_partition variable=do_drain dim=0 complete

    int done = 0;

    drain: for (int p = 0; p < M_w; p++) {
    #pragma HLS unroll
        int  rd     = resp_rd_ptr[p];
        bool v_head = resp_valid[p][rd] | land_hit[p][rd];
        int  d_head = land_hit[p][rd] ? land_data[p][rd] : resp_data[p][rd];
        int  i_head = land_hit[p][rd] ? land_idx[p][rd]  : resp_idx[p][rd];
        bool fire   = (resp_cnt[p] > 0) & v_head;
        do_drain[p] = fire;
        if (fire) { 
            output_local[k * M_w + p][i_head] = d_head;
        }
        done += fire ? 1 : 0;
    }
    completed = done;

    resp_update: for (int p = 0; p < M_w; p++) {
    #pragma HLS unroll
        for (int d = 0; d < D; d++) {
        #pragma HLS unroll
            resp_data[p][d] = land_hit[p][d] ? land_data[p][d] : resp_data[p][d];
            resp_idx[p][d]  = land_hit[p][d] ? land_idx[p][d]  : resp_idx[p][d];
            bool drained_here = do_drain[p] & (resp_rd_ptr[p] == d);
            resp_valid[p][d]  = (resp_valid[p][d] | land_hit[p][d]) & !drained_here;
        }
        int rd_inc = (resp_rd_ptr[p] == D - 1) ? 0 : resp_rd_ptr[p] + 1;
        resp_rd_ptr[p] = do_drain[p] ? rd_inc : resp_rd_ptr[p];
    }

    apply_credit: for (int m = 0; m < M_w; m++) {
    #pragma HLS unroll
        resp_cnt[m] = resp_cnt[m] + admit_count[m] - (do_drain[m] ? 1 : 0);
    }

    // ---------------------------------------------------------------
    // encode
    // ---------------------------------------------------------------
    resp_t issue[N_w];
    int    issue_slot[N_w];
    #pragma HLS array_partition variable=issue dim=0 complete
    #pragma HLS array_partition variable=issue_slot dim=0 complete

    encode_winner: for (int n = 0; n < N_w; n++) {
    #pragma HLS unroll
        int la = 0, ix = 0, wr = 0, pe = 0;
        for (int m = 0; m < M_w; m++) {
        #pragma HLS unroll
            int bit = (int)((adm[n] >> m) & 1u);
            int msk = -bit; 
            la |= q_head[m][n].local_addr & msk;
            ix |= q_head[m][n].output_idx & msk;
            wr |= resp_wr_ptr[m]          & msk;
            pe |= m                       & msk;
        }

        int slot = wr + rank[n];
        slot = (slot >= D) ? slot - D : slot;

        int bdata = bank_mem[n][la];

        unsigned int below   = cand[n] - 1u;
        unsigned int above   = ~(below | cand[n]) & PE_ALL;
        unsigned int rr_next = (above != 0) ? above : PE_ALL;

        issue[n].valid      = admit[n];
        issue[n].pe         = pe;
        issue[n].output_idx = ix;
        issue[n].data       = admit[n] ? bdata : 0;
        issue_slot[n]       = admit[n] ? slot  : 0;
        rr[n]               = admit[n] ? rr_next : rr[n];
    }

    advance_wr_ptr: for (int m = 0; m < M_w; m++) {
    #pragma HLS unroll
        int w = resp_wr_ptr[m] + admit_count[m];
        resp_wr_ptr[m] = (w >= D) ? w - D : w;
    }

    delay_shift: for (int n = 0; n < N_w; n++) {
    #pragma HLS unroll

        for (int i = RESP_LAT - 1; i > 0; i--) {
        #pragma HLS unroll
            resp_arr[n][i].valid      = resp_arr[n][i - 1].valid;
            resp_arr[n][i].pe         = resp_arr[n][i - 1].pe;
            resp_arr[n][i].output_idx = resp_arr[n][i - 1].output_idx;
            resp_arr[n][i].data       = resp_arr[n][i - 1].data;
            slot_arr[n][i] = slot_arr[n][i - 1];
        }
        resp_arr[n][0].valid      = issue[n].valid;
        resp_arr[n][0].pe         = issue[n].pe;
        resp_arr[n][0].output_idx = issue[n].output_idx;
        resp_arr[n][0].data       = issue[n].data;
        slot_arr[n][0] = issue_slot[n];
    }

    // ---------------------------------------------------------------
    // decode
    // ---------------------------------------------------------------
    bool  do_push[M_w][N_w];
    req_t new_req[M_w];
    #pragma HLS array_partition variable=do_push dim=0 complete
    #pragma HLS array_partition variable=new_req dim=0 complete

    decode: for (int m = 0; m < M_w; m++) {
    #pragma HLS unroll
        int  g   = k * M_w + m;
        int  j   = pe_iteration[g];
        bool ask = (g + j * M) < TOTAL_ITERS;
        int  jr  = ask ? j : 0;                

        int x = x_local[g][jr];
        int y = y_local[g][jr];

        int addr = y * IMG_WIDTH + x;
        int bank  = addr % N_w;
        int local_addr = addr / N_w;

        new_req[m].local_addr = local_addr;
        new_req[m].output_idx = j;

        int pushed = 0;
        for (int n = 0; n < N_w; n++) {
        #pragma HLS unroll
            bool p = ask & (bank == n) & !q_full[m][n];
            do_push[m][n] = p;
            pushed += p ? 1 : 0;
        }
        pe_iteration[g] = j + pushed;
    }

    req_update: for (int m = 0; m < M_w; m++) {
    #pragma HLS unroll
        for (int n = 0; n < N_w; n++) {
        #pragma HLS unroll
            bool pop = ((adm[n] >> m) & 1u) != 0;
            req_arr[m][n].step(do_push[m][n], new_req[m], pop);
        }
    }
}

// ------------------------------------------------------ //
// ------------------------ top ------------------------- //
// ------------------------------------------------------ //

extern "C" void top(int *ext_data, int *x_trans, int *y_trans, int *output) {

    #pragma HLS INTERFACE m_axi port=ext_data bundle=gmem0 depth=IMG_SIZE
    #pragma HLS INTERFACE m_axi port=x_trans bundle=gmem1 depth=IMG_SIZE
    #pragma HLS INTERFACE m_axi port=y_trans bundle=gmem2 depth=IMG_SIZE
    #pragma HLS INTERFACE m_axi port=output bundle=gmem0 depth=IMG_SIZE

    static int bank_mem[N_k][N_w][B_DEPTH];
    #pragma HLS array_partition variable=bank_mem dim=1 complete
    #pragma HLS array_partition variable=bank_mem dim=2 complete
    #pragma HLS bind_storage variable=bank_mem impl=BRAM type=RAM_1P latency=1

    static int x_local[M][IMG_SIZE / M];
    static int y_local[M][IMG_SIZE / M];
    static int output_local[M][IMG_SIZE / M];
    #pragma HLS array_partition variable=x_local dim=1 type=complete
    #pragma HLS array_partition variable=y_local dim=1 type=complete
    #pragma HLS array_partition variable=output_local dim=1 type=complete
    #pragma HLS bind_storage variable=x_local impl=BRAM type=RAM_T2P
    #pragma HLS bind_storage variable=y_local impl=BRAM type=RAM_T2P
    #pragma HLS bind_storage variable=output_local impl=BRAM type=RAM_T2P

    load_trans(x_trans, x_local);
    load_trans(y_trans, y_local);
    load_img(ext_data, bank_mem);

    int pe_iteration[M];
    unsigned int rr_ptr[N_k][N_w];
    resp_t resp_arr[N_k][N_w][RESP_LAT];
    req_queue_t req_arr[N_k][M_w][N_w];
    #pragma HLS array_partition variable=pe_iteration dim=0 complete
    #pragma HLS array_partition variable=rr_ptr dim=0 complete
    #pragma HLS array_partition variable=resp_arr dim=0 complete
    #pragma HLS array_partition variable=req_arr dim=0 complete

    static int  slot_arr[N_k][N_w][RESP_LAT];
    static int  resp_data[N_k][M_w][D];
    static int  resp_idx [N_k][M_w][D];
    static bool resp_valid[N_k][M_w][D];
    static int  resp_wr_ptr[N_k][M_w];
    static int  resp_rd_ptr[N_k][M_w];
    static int  resp_cnt   [N_k][M_w];
    #pragma HLS array_partition variable=slot_arr    dim=0 complete
    #pragma HLS array_partition variable=resp_data   dim=0 complete
    #pragma HLS array_partition variable=resp_idx    dim=0 complete
    #pragma HLS array_partition variable=resp_valid  dim=0 complete
    #pragma HLS array_partition variable=resp_wr_ptr dim=0 complete
    #pragma HLS array_partition variable=resp_rd_ptr dim=0 complete
    #pragma HLS array_partition variable=resp_cnt    dim=0 complete

    static int done_reg[N_k];
    #pragma HLS array_partition variable=done_reg dim=0 complete

    init_pe: for (int p = 0; p < M; p++) {
    #pragma HLS unroll
        pe_iteration[p] = 0;
    }

    init_k: for (int k = 0; k < N_k; k++) {
    #pragma HLS unroll
        for (int n = 0; n < N_w; n++) {
        #pragma HLS unroll
            rr_ptr[k][n] = PE_ALL;
            for (int i = 0; i < RESP_LAT; i++) {
            #pragma HLS unroll
                resp_arr[k][n][i].valid      = false;
                resp_arr[k][n][i].pe         = 0;
                resp_arr[k][n][i].output_idx = 0;
                resp_arr[k][n][i].data       = 0;
                slot_arr[k][n][i] = 0;
            }
        }
        for (int p = 0; p < M_w; p++) {
        #pragma HLS unroll
            for (int n = 0; n < N_w; n++) {
        #pragma HLS unroll
                req_arr[k][p][n].init();
            }
            for (int d = 0; d < D; d++) {
            #pragma HLS unroll
                resp_valid[k][p][d] = false;
            }
            resp_wr_ptr[k][p] = 0;
            resp_rd_ptr[k][p] = 0;
            resp_cnt[k][p]    = 0;
        }
        done_reg[k] = 0;
    }

    int completed = 0;

    cycle_loop: for (int cycle = 0; cycle < MAX_CYCLES; cycle++) {
    #pragma HLS pipeline II=1
        if (completed >= TOTAL_ITERS) break;

        int done_this[N_k];
        #pragma HLS array_partition variable=done_this dim=0 complete

        group_loop: for (int k = 0; k < N_k; k++) {
        #pragma HLS unroll
            group_step(k, bank_mem[k], x_local, y_local, output_local,
                       pe_iteration, req_arr[k], rr_ptr[k], resp_arr[k],
                       slot_arr[k], resp_data[k], resp_idx[k], resp_valid[k],
                       resp_wr_ptr[k], resp_rd_ptr[k], resp_cnt[k], done_this[k]);
        }

        int completed_incr = 0;
        fold_done: for (int k = 0; k < N_k; k++) {
        #pragma HLS unroll
            completed_incr += done_reg[k];
            done_reg[k] = done_this[k];
        }

        completed += completed_incr;
    }

    store_loop: for (int i = 0; i < IMG_SIZE; i++) {
    #pragma HLS pipeline II=1
        output[i] = output_local[i % M][i / M];
    }
}