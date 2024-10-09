#define FPS 60
#define N_BEADS 8 

#define DT (1.0f / FPS)
#define N_STEPS 100
#define SDT (DT / N_STEPS)
#define STS (N_STEPS * FPS)

typedef struct bead {
    float radius;
    float mass;
    float2 pos;
    float2 prev_pos;
    float2 vel;
} bead;

typedef struct wire{
    float2 pos;
    float radius;
} wire;

__constant float2 gravity = {0.0F, -10.0F};

void start_step(__local bead *a) {
    a->vel += gravity * SDT; 
    a->prev_pos = a->pos;
    a->pos += a->vel * SDT; 
}

void end_step(__local bead *a) {
    a->vel = a->pos - a->prev_pos;
    a->vel = a->vel * STS;
}

void bead_col(__local bead *a, __local bead *b) {
    float2 dir = b->pos - a->pos;
    float d = length(dir);
    if (d == 0.0f || d > a->radius + b->radius)
        return;
    dir = dir / d;
    float corr = (a->radius + b->radius - d) / 2.0f;
    a->pos = a->pos - dir * corr;
    b->pos = b->pos + dir * corr;
    float v0a = dot(a->vel, dir); 
    float v0b = dot(b->vel, dir); 
    float ma = a->mass;
    float mb = b->mass;
    float mt = ma + mb;
    float vc = ma * v0a + mb * v0b;
    float v1a = (vc - mb * (v0a - v0b)) / mt;
    float v1b = (vc - ma * (v0b - v0a)) / mt; 
    a->vel = a->vel + dir * (v1a - v0a); 
    b->vel = b->vel + dir * (v1b - v0b); 
}

void keep_on_wire(__local bead *a, __local wire *b) {
    float2 dir = a->pos - b->pos;
    float len = length(dir); 
    if (len == 0.0f)
        return;
    dir = dir / len;
    float lambda = b->radius - len;
    a->pos = a->pos + dir * lambda;
}

__kernel void update_sim(
    __global bead(*groups)[N_BEADS], 
    __global wire *gwire
) {
    __global bead *group;
    __local bead beads[N_BEADS];
    __local wire lwire;
    
    group = groups[get_global_id(0)];
    for (int i = 0; i < N_BEADS; i++) 
        beads[i] = group[i];
    lwire = *gwire;
    for (int s = 0; s < N_STEPS; s++) {
        int i, j;
        for (i = 0; i < N_BEADS; i++)
            start_step(beads + i);
        for (i = 0; i < N_BEADS; i++)
            keep_on_wire(beads + i, &lwire);
        for (i = 0; i < N_BEADS; i++)
            end_step(beads + i);
        for (i = 0; i < N_BEADS; i++) {
            for (j = 0; j < i; j++)
                bead_col(beads + i, beads + j);
        }
    }
    for (int i = 0; i < N_BEADS; i++) 
        group[i] = beads[i];
}
