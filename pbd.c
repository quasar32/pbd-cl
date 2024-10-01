#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <CL/cl.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>

#define FPS 60
#define DT (1.0f / FPS)
#define N_STEPS 100
#define SDT (DT / N_STEPS)
#define N_BEADS 8 
#define N_GROUPS 8 

typedef struct __attribute__((aligned(8))) float2 {
  float x;
  float y;
} float2;

static float2 gravity = {0.0F, -10.0F};

typedef struct bead {
  float radius;
  float mass;
  float2 pos;
  float2 prev_pos;
  float2 vel;
} bead;

typedef struct wire {
  float2 pos;
  float radius;
} wire;

static bead bds[N_BEADS * N_GROUPS]; 
static wire wr = {{0.0f, 0.0f}, 0.8f};
static FILE *fps[N_GROUPS];

static void print_sim(int f) {
  bead *bd = bds;
  for (int i = 0; i < N_GROUPS; i++) {
    for (int j = 0; i < N_BEADS; i++) {
      fprintf(fps[i], "%d,%d,%f,%f,%f\n", f, 0, 
          bd->pos.x, bd->pos.y, bd->radius); 
      bd++;
    }
    fprintf(fps[i], "%d,%d,%f,%f,%f\n", f, 1, 
        wr.pos.x, wr.pos.y, wr.radius); 
  }
}

static void die(const char *fn, int err) {
  fprintf(stderr, "%s(%d)\n", fn, err);
  exit(EXIT_FAILURE);
}

static void notify(const char *err, const void *info, size_t sz, void *user) {
    fprintf(stderr, "opencl-err: %s\n", err);
}

static char *read_all(const char *path) {
  int fd = open(path, O_RDONLY);
  if (fd < 0)
    die("open", errno);
  struct stat st;
  if (fstat(fd, &st) < 0)
    die("stat", errno);
  char *str = malloc(st.st_size + 1);
  if (!str)
    die("malloc", errno);
  if (read(fd, str, st.st_size) < 0)
    die("read", errno);
  close(fd);
  str[st.st_size] = '\0';
  return str;
}

static cl_kernel kernel;
static cl_command_queue cq;
static cl_mem bds_mem;

static void create_cl(void) {
  int err;
  cl_platform_id pid;
  err = clGetPlatformIDs(1, &pid, NULL);
  if (err != CL_SUCCESS)
    die("clGetPlatformIDs", err);
  cl_device_id did;
  err = clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, 1, &did, NULL);
  if (err != CL_SUCCESS)
    die("clGetDeviceIDs", err);
  cl_context ctx = clCreateContext(NULL, 1, &did, notify, NULL, &err); 
  if (!ctx)
    die("clCreateContext", err);
  bds_mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE | 
      CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_READ_ONLY, sizeof(bds), bds, &err);
  if (err != CL_SUCCESS)
    die("clCreateBuffer", err);
  cl_mem wr_mem = clCreateBuffer(ctx, CL_MEM_READ_ONLY | 
      CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, sizeof(wr), &wr, &err);
  if (err != CL_SUCCESS)
    die("clCreateBuffer", err);
  const char *str = read_all("pbd.cl");
  cl_program prog = clCreateProgramWithSource(ctx, 1, &str, NULL, &err);
  free((void *) str);
  if (err != CL_SUCCESS)
    die("clCreateProgramWithSource", err);
  err = clBuildProgram(prog, 1, &did, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_sz;
    clGetProgramBuildInfo(prog, did, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
    char *log = malloc(log_sz);
    if (!log)
      die("malloc", errno);
    err = clGetProgramBuildInfo(prog, did, 
        CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
    if (err == CL_SUCCESS)
      fprintf(stderr, "cl-log: %s\n", log);
    free(log);
  }
  if (err != CL_SUCCESS)
    die("glBuildProgram", err);
  kernel = clCreateKernel(prog, "frame", &err);
  if (err != CL_SUCCESS)
    die("clCreateKernel", err);
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bds_mem);
  if (err != CL_SUCCESS)
    die("clSetKernelArg", err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &wr_mem);
  if (err != CL_SUCCESS)
    die("clSetKernelArg", err);
  cq = clCreateCommandQueue(ctx, did, 0, &err);
  if (err != CL_SUCCESS)
    die("clCreateCommandQueue", err);
}

static void frame(void) {
  cl_event ev;
  cl_int err = clEnqueueNDRangeKernel(cq, kernel, 1, NULL,
      (size_t[]) {8}, (size_t[]) {8} , 0, NULL, &ev);
  if (err != CL_SUCCESS)
    die("clEnqueueTask", err);
  clWaitForEvents(1, &ev);
  clReleaseEvent(ev);
  err = clEnqueueReadBuffer(cq, bds_mem, CL_TRUE, 0, 
      sizeof(bds), bds, 0, NULL, NULL);
  if (err != CL_SUCCESS)
    die("clEnqueueReadBuffer", err);
}


int main(int argc, char **argv) {
  srand48(time(NULL));
  for (int i = 0; i < N_GROUPS; i++) {
    char buf[64]; 
    sprintf(buf, "out%d.csv", i); 
    fps[i] = fopen(buf, "wb");
    if (!fps[i])
      die("errno", errno);
  }
  bead *bd = bds;
  for (int i = 0; i < N_GROUPS; i++) { 
    float r = 0.1f;
    float rot = 0.0f;
    for (int j = 0; j < N_BEADS; j++) {
      bd->radius = r;
      bd->mass = (float) M_PI * r * r; 
      bd->pos.x = wr.pos.x + wr.radius * cosf(rot);
      bd->pos.y = wr.pos.y + wr.radius * sinf(rot);
      rot += (float) M_PI / N_BEADS;
      r = 0.05f + drand48() * 0.1f;
      bd++;
    }
  }
  create_cl();
  for (int i = 0; i < N_GROUPS; i++) 
    fprintf(fps[i], "f,t,x,y,r\n");
  int f;
  for (f = 0; f < 10 * FPS; f++) {
    print_sim(f);
    frame();
  }
  print_sim(f);
  for (int i = 0; i < N_GROUPS; i++) 
    fclose(fps[i]);
}
