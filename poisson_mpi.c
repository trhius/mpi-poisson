#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

double L = 1.0; /* kích thước miền tính toán */
int N = 32;     /* số điểm nội bộ trên mỗi chiều của lứoi */

double *u, *u_new; /* con trỏ tới mảng chứa giá trị lời giải */

/* định nghĩa macro để tính chỉ số của phần tử mảng 2D lưu dưới dạng mảng 1D */
#define INDEX(i, j) ((N + 2) * (i) + (j))

int my_rank; /* rank của tiến trình */

int *proc;                   /* mảng lưu trữ tiến trình quản lý từng điểm lứoi */
int *i_min, *i_max;          /* chỉ số đầu cuối của điểm lưỡi mỗi tiến trình quản lý */
int *left_proc, *right_proc; /* tiến trình trái phải */

double comm_time, non_comm_time;
double comm_start, comm_end;

int main(int argc, char *argv[]);
void allocate_arrays();
void jacobi(int num_procs, double f[]);
void make_domains(int num_procs);
double *make_source();
void timestamp();

int main(int argc, char *argv[]) {
  double change;
  double epsilon = 1.0E-03;
  double *f;
  char file_name[100];
  int i;
  int j;
  double my_change;
  int my_n;
  int n;
  int num_procs;
  int step;
  double *swap;
  double wall_time;
  comm_time = 0.0;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (1 < argc) {
    sscanf(argv[1], "%d", &N);
  }
  else {
    N = 32;
  }

  if (2 < argc) {
    sscanf(argv[2], "%lf", &epsilon);
  }
  else {
    epsilon = 1.0E-03;
  }

  if (3 < argc) {
    strcpy(file_name, argv[3]);
  }
  else {
    strcpy(file_name, "poisson_mpi.out");
  }

  if (my_rank == 0) {
    timestamp();
    printf("\n");
    printf("POISSON_MPI:\n");
    printf("  C version\n");
    printf("  2-D Poisson equation using Jacobi algorithm\n");
    printf("  MPI version: 1-D domains, non-blocking send/receive\n");
    printf("  Number of processes         = %d\n", num_procs);
    printf("  Number of interior vertices = %d\n", N);
    printf("  Desired fractional accuracy = %f\n", epsilon);
    printf("\n");
  }

  allocate_arrays(); 
  f = make_source();
  make_domains(num_procs);

  step = 0;
  wall_time = MPI_Wtime();

  do {
    jacobi(num_procs, f); // thực hiện bước lặp jacobi
    ++step;

    // tính sai số và kiểm tra điều kiện hội tụ
    change = 0.0;
    n = 0;
    my_change = 0.0;
    my_n = 0;

    for (i = i_min[my_rank]; i <= i_max[my_rank]; i++) {
      for (j = 1; j <= N; j++) {
        if (u_new[INDEX(i, j)] != 0.0) {
          my_change = my_change + fabs(1.0 - u[INDEX(i, j)] / u_new[INDEX(i, j)]);
          my_n = my_n + 1;
        }
      }
    }

    // sử dụng MPI_Allreduce để tính sai số toàn cục
    MPI_Allreduce(&my_change, &change, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&my_n, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (n != 0) {
      change = change / n;
    }
    if (my_rank == 0 && (step % 10) == 0) {
      printf("  N = %d, n = %d, my_n = %d, Step %4d  Error = %g\n",
             N, n, my_n, step, change);
    }

    // hoán đổi con trỏ u và u_new
    swap = u;
    u = u_new;
    u_new = swap;

  } while (epsilon < change);

  wall_time = MPI_Wtime() - wall_time;
  if (my_rank == 0) {
    printf("\n");
    printf("  T_w_com: %f secs\n", comm_time);
    printf("  T_wo_com: %f secs\n", wall_time - comm_time);
  }

  MPI_Finalize();
  free(f);

  if (my_rank == 0) {
    printf("\n");
    printf("POISSON_MPI:\n");
    printf("  Normal end of execution.\n");
    printf("\n");
    timestamp();
  }

  return 0;
}

// phân bố bộ nhứo cho mảng u và u_new, khởi tạo chúng vs giá trị 0
void allocate_arrays() {
  int i;
  int ndof = (N + 2) * (N + 2);

  u = (double *)malloc(ndof * sizeof(double));
  for (i = 0; i < ndof; i++) {
    u[i] = 0.0;
  }

  u_new = (double *)malloc(ndof * sizeof(double));
  for (i = 0; i < ndof; i++) {
    u_new[i] = 0.0;
  }

  return;
}

/*
  thực hiện bước lặp jacobi cho miền đc phân chia
  gồm cập nhật giá trị 
  và giao tiếp lớp biên giữa các tiến trình bằng MPI_Isend và MPI_Irecv ko chặn
*/ 
void jacobi(int num_procs, double f[]) {
  double h;
  int i, j;
  MPI_Request request[4];
  int requests;
  MPI_Status status[4];
  /*
    H is the lattice spacing.
  */
  h = L / (double)(N + 1);

  /*
    Update ghost layers using non-blocking send/receive
  */
  requests = 0;

  comm_start = MPI_Wtime();

  if (left_proc[my_rank] >= 0 && left_proc[my_rank] < num_procs) {
    MPI_Irecv(u + INDEX(i_min[my_rank] - 1, 1), N, MPI_DOUBLE,
              left_proc[my_rank], 0, MPI_COMM_WORLD,
              request + requests++);

    MPI_Isend(u + INDEX(i_min[my_rank], 1), N, MPI_DOUBLE,
              left_proc[my_rank], 1, MPI_COMM_WORLD,
              request + requests++);
  }

  if (right_proc[my_rank] >= 0 && right_proc[my_rank] < num_procs) {
    MPI_Irecv(u + INDEX(i_max[my_rank] + 1, 1), N, MPI_DOUBLE,
              right_proc[my_rank], 1, MPI_COMM_WORLD,
              request + requests++);

    MPI_Isend(u + INDEX(i_max[my_rank], 1), N, MPI_DOUBLE,
              right_proc[my_rank], 0, MPI_COMM_WORLD,
              request + requests++);
  }
  
  // cập nhật jacobi cho các điểm nội bộ
  for (i = i_min[my_rank] + 1; i <= i_max[my_rank] - 1; i++) {
    for (j = 1; j <= N; j++) {
      u_new[INDEX(i, j)] =
          0.25 * (u[INDEX(i - 1, j)] + u[INDEX(i + 1, j)] +
                  u[INDEX(i, j - 1)] + u[INDEX(i, j + 1)] +
                  h * h * f[INDEX(i, j)]);
    }
  }

  comm_end = MPI_Wtime();
  comm_time += comm_end - comm_start;

  // chờ hoàn tất giao tiếp
  MPI_Waitall(requests, request, status);

  // cập nhật Jacobi cho các điểm biên 
  i = i_min[my_rank];
  for (j = 1; j <= N; j++) {
    u_new[INDEX(i, j)] =
        0.25 * (u[INDEX(i - 1, j)] + u[INDEX(i + 1, j)] +
                u[INDEX(i, j - 1)] + u[INDEX(i, j + 1)] +
                h * h * f[INDEX(i, j)]);
  }

  i = i_max[my_rank];
  if (i != i_min[my_rank]) {
    for (j = 1; j <= N; j++) {
      u_new[INDEX(i, j)] =
          0.25 * (u[INDEX(i - 1, j)] + u[INDEX(i + 1, j)] +
                  u[INDEX(i, j - 1)] + u[INDEX(i, j + 1)] +
                  h * h * f[INDEX(i, j)]);
    }
  }

  return;
}

/*
  chia miền tính toán cho các tiến trình
  xác định chỉ số đầu, cuối
  và các tiến trình lân cận trái, phải
*/
void make_domains(int num_procs) {
  double d;
  double eps = 0.0001;;
  int i, p;
  double x_max, x_min;
  /*
    Allocate arrays for process information.
  */
  proc = (int *)malloc((N + 2) * sizeof(int));
  i_min = (int *)malloc(num_procs * sizeof(int));
  i_max = (int *)malloc(num_procs * sizeof(int));
  left_proc = (int *)malloc(num_procs * sizeof(int));
  right_proc = (int *)malloc(num_procs * sizeof(int));
  /*
    Divide the range [(1-eps)..(N+eps)] evenly among the processes.
  */
  d = (N - 1.0 + 2.0 * eps) / (double)num_procs;

  for (p = 0; p < num_procs; p++) {
    /*
      The I indices assigned to domain P will satisfy X_MIN <= I <= X_MAX.
    */
    x_min = -eps + 1.0 + (double)(p * d);
    x_max = x_min + d;
    /*
      For the node with index I, store in PROC[I] the process P it belongs to.
    */
    for (i = 1; i <= N; i++) {
      if (x_min <= i && i < x_max) {
        proc[i] = p;
      }
    }
  }
  /*
    Now find the lowest index I associated with each process P.
  */
  for (p = 0; p < num_procs; p++) {
    for (i = 1; i <= N; i++) {
      if (proc[i] == p) {
        break;
      }
    }
    i_min[p] = i;
    /*
      Find the largest index associated with each process P.
    */
    for (i = N; 1 <= i; i--) {
      if (proc[i] == p) {
        break;
      }
    }
    i_max[p] = i;
    /*
      Find the processes to left and right.
    */
    left_proc[p] = -1;
    right_proc[p] = -1;

    if (proc[p] != -1) {
      if (1 < i_min[p] && i_min[p] <= N) {
        left_proc[p] = proc[i_min[p] - 1];
      }
      if (0 < i_max[p] && i_max[p] < N) {
        right_proc[p] = proc[i_max[p] + 1];
      }
    }
  }

  return;
}

double *make_source() {
  double *f;
  int i;
  int j;
  int k;
  double q = 10.0;

  f = (double *)malloc((N + 2) * (N + 2) * sizeof(double));

  for (i = 0; i < (N + 2) * (N + 2); i++) {
    f[i] = 0.0;
  }

  i = 1 + N / 4;
  j = i;
  k = INDEX(i, j);
  f[k] = q;

  i = 1 + 3 * N / 4;
  j = i;
  k = INDEX(i, j);
  f[k] = -q;

  return f;
}

void timestamp() {
#define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time(NULL);
  tm = localtime(&now);

  strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

  printf("%s\n", time_buffer);

  return;
#undef TIME_SIZE
}