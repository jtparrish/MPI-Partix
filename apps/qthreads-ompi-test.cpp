#include <mpi.h>
#include <stdio.h>
#include <stdint.h>
#include <qthread/qthread.h>
#include <qthread/barrier.h>

#include <unistd.h>
#include <stdarg.h>
#include <stdlib.h>
#include <assert.h>

#define DEBUG 0 
#define GDB 0


#define NUM_TASKS 100 

int debug(const char *format, ...) {
#if DEBUG
  va_list args;
  va_start(args, format);

  int ret = vprintf(format, args);

  va_end(args);

  return ret;
#else
  return -1;
#endif
}

void attach_loop(int rank) {
  volatile int i = 0;
  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  printf("rank %d PID %d on %s ready for attach\n", rank, getpid(), hostname);
  fflush(stdout);
  while (0 == i) {
    sleep(5);
  }
}

typedef struct args {
  int rank;
  int tag;
  qt_barrier_t *barrier;
} args_t;

int main(void);
void proc0(void);
void proc1(void);
static aligned_t task(void *arg_ptr_);

int main(void) {
  debug("start\n");
  qthread_initialize();
  int thread_support = -1;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &thread_support);
  assert(thread_support == MPI_THREAD_MULTIPLE);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    printf("ERROR: comm size must be 2\n");
    return 1;
  }

  int rank; 
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#if GDB
  attach_loop(rank);
#endif
  qt_barrier_t *barrier = qt_barrier_create(NUM_TASKS + 1, REGION_BARRIER);
  for (size_t i = 0; i < NUM_TASKS; i++) {
    debug("task %lu forked\n", i);
    args_t *args_ptr = (args_t *) malloc(sizeof(args_t)); 
    *args_ptr = (args_t) { rank, (int) i, barrier };
    qthread_fork(&task, (void *) args_ptr, NULL);
  }

  qt_barrier_enter(barrier);

  debug("proc %d ready to finalize\n", rank);

  qthread_finalize();
  MPI_Finalize();

  debug("proc %d exiting\n", rank);

  return 0;
}


static aligned_t task(void *args_ptr_) {
  args_t *args_ptr = (args_t *) args_ptr_;
  switch (args_ptr->rank) {
    case 0:
      {
        debug("sending %d\n", args_ptr->tag);
        MPI_Ssend(&args_ptr->tag, 1, MPI_INT, 1, args_ptr->tag, MPI_COMM_WORLD);
        debug("sent %d\n", args_ptr->tag);
        free(args_ptr);
        break;
      }
    case 1:
      {
        debug("receiving %d\n", args_ptr->tag);
        int dest = -1;
        MPI_Recv(&dest, 1, MPI_INT, 0, args_ptr->tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        assert(dest == args_ptr->tag);
        debug("received %d\n", args_ptr->tag);
        free(args_ptr);
        break;
      }
    default:
      {
        debug("INVALID RANK\n");
        break;
      }
  }
  qt_barrier_enter(args_ptr->barrier);
  return 0;
}
