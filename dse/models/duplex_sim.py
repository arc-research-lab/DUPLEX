from collections import deque


class ReqT:
    def __init__(self, local_addr, output_idx):
        self.local_addr = local_addr
        self.output_idx = output_idx


class RespT:
    def __init__(self, valid=False, pe=0, output_idx=0, data=0):
        self.valid = valid
        self.pe = pe
        self.output_idx = output_idx
        self.data = data


class HardwareQueueHLS:

    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = deque()
        self.push_buffer = None

    def empty(self):
        return len(self.queue) == 0

    def full(self):
        count = len(self.queue) + (1 if self.push_buffer is not None else 0)
        return count >= self.capacity

    def push(self, item):
        if not self.full():
            self.push_buffer = item

    def update_cycle(self):
        if self.push_buffer is not None:
            self.queue.append(self.push_buffer)
            self.push_buffer = None

    def pop(self):
        if not self.empty():
            return self.queue.popleft()
        return None


def simulate_duplex(M_w, N_w, D, RESP_LAT, B_DEPTH, IMG_WIDTH, total_iters, k,
                    x_trans, y_trans):
    """Simulate one DUPLEX group on an index trace.

    Args:
        M_w:         PEs in the group.
        N_w:         banks in the group's replica.
        D:           request-queue depth and number of response slots per PE.
        RESP_LAT:    bank read latency in cycles.
        B_DEPTH:     words per bank (ceil(V / N_w)).
        IMG_WIDTH:   row width used to form addresses: addr = y * IMG_WIDTH + x.
        total_iters: number of reads in the trace (before padding).
        k:           group index; only k = 0 is supported (one group is simulated).
        x_trans, y_trans: numpy arrays of length M_w * ceil(total_iters / M_w)
                     (pad the trace to a multiple of M_w).

    Returns:
        (total_cycles, useful reads per cycle for this group)
    """
    if k != 0:
        raise ValueError("simulate_duplex models a single group; pass k=0")

    M = M_w
    iters_per_pe = (total_iters + M - 1) // M

    x_local = x_trans.reshape(M, iters_per_pe)
    y_local = y_trans.reshape(M, iters_per_pe)
    bank_mem = [[(b * 1000 + a) for a in range(B_DEPTH)] for b in range(N_w)]
    output_local = [[0] * iters_per_pe for _ in range(M)]

    pe_iteration = [0] * M
    req_arr = [[HardwareQueueHLS(D) for _ in range(N_w)] for _ in range(M_w)]
    rr = [0] * N_w

    resp_arr = [[RespT() for _ in range(RESP_LAT)] for _ in range(N_w)]
    slot_arr = [[0] * RESP_LAT for _ in range(N_w)]

    resp_data = [[0] * D for _ in range(M_w)]
    resp_idx = [[0] * D for _ in range(M_w)]
    resp_valid = [[False] * D for _ in range(M_w)]

    resp_wr_ptr = [0] * M_w
    resp_rd_ptr = [0] * M_w
    resp_cnt = [0] * M_w

    total_cycles = 0
    total_completed_reads = 0

    while total_completed_reads < total_iters:
        total_cycles += 1

        q_nonempty = [[not req_arr[m][n].empty() for n in range(N_w)] for m in range(M_w)]
        q_full = [[req_arr[m][n].full() for n in range(N_w)] for m in range(M_w)]

        cand_pe = [0] * N_w
        cand_valid = [False] * N_w
        for n in range(N_w):
            win = -1
            for offset in range(M_w):
                m = (rr[n] + offset) % M_w
                if win < 0 and q_nonempty[m][n]:
                    win = m
            cand_valid[n] = (win >= 0)
            cand_pe[n] = win if win >= 0 else 0

        rank = [0] * N_w
        for n in range(N_w):
            r = 0
            for nb in range(n):
                if cand_valid[n] and cand_valid[nb] and cand_pe[n] == cand_pe[nb]:
                    r += 1
            rank[n] = r

        admit = [cand_valid[n] and (resp_cnt[cand_pe[n]] + rank[n] < D) for n in range(N_w)]

        admit_count = [0] * M_w
        for m in range(M_w):
            admit_count[m] = sum(1 for n in range(N_w) if admit[n] and cand_pe[n] == m)

        for n in range(N_w):
            resp = resp_arr[n][RESP_LAT - 1]
            if resp.valid:
                slot = slot_arr[n][RESP_LAT - 1]
                resp_data[resp.pe][slot] = resp.data
                resp_idx[resp.pe][slot] = resp.output_idx
                resp_valid[resp.pe][slot] = True

        do_drain = [False] * M_w
        done = 0
        for p in range(M_w):
            rd = resp_rd_ptr[p]
            fire = (resp_cnt[p] > 0) and resp_valid[p][rd]
            do_drain[p] = fire
            if fire:
                if resp_idx[p][rd] < len(output_local[p]):
                    output_local[p][resp_idx[p][rd]] = resp_data[p][rd]
                resp_valid[p][rd] = False
                resp_rd_ptr[p] = (rd + 1) % D
                done += 1

        total_completed_reads += done

        next_resp_cnt = [resp_cnt[m] + admit_count[m] - (1 if do_drain[m] else 0)
                         for m in range(M_w)]

        issue = [RespT() for _ in range(N_w)]
        issue_slot = [0] * N_w
        for n in range(N_w):
            if admit[n]:
                win = cand_pe[n]
                r = req_arr[win][n].pop()
                issue[n] = RespT(valid=True, pe=win, output_idx=r.output_idx,
                                 data=bank_mem[n][r.local_addr % B_DEPTH])
                issue_slot[n] = (resp_wr_ptr[win] + rank[n]) % D
                rr[n] = (win + 1) % M_w

        for m in range(M_w):
            resp_wr_ptr[m] = (resp_wr_ptr[m] + admit_count[m]) % D

        for n in range(N_w):
            for i in range(RESP_LAT - 1, 0, -1):
                resp_arr[n][i] = resp_arr[n][i - 1]
                slot_arr[n][i] = slot_arr[n][i - 1]
            resp_arr[n][0] = issue[n]
            slot_arr[n][0] = issue_slot[n]

        for m in range(M_w):
            j = pe_iteration[m]
            if (m + j * M) >= total_iters:
                continue
            addr = y_local[m][j] * IMG_WIDTH + x_local[m][j]
            bank = addr % N_w
            if not q_full[m][bank]:
                req_arr[m][bank].push(ReqT(addr // N_w, j))
                pe_iteration[m] += 1

        for m in range(M_w):
            resp_cnt[m] = next_resp_cnt[m]
            for n in range(N_w):
                req_arr[m][n].update_cycle()

    return total_cycles, total_completed_reads / total_cycles