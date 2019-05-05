/* ========================================================================== */
/*                                                                            */
// Lane Detection
//
// Code written by Mohit Rane and Smitha Bhaskar
// Reference for Sequencer part of this code has been taken from
// Prof. Sam Siewert's seqgen.c code 
//
// Sequencer for Lane Detection
// Sequencer = RT_MAX	@ 50 Hz
// Servcie_1 = RT_MAX-1	@ 2 Hz
// Service_2 = RT_MAX-2	@ 1.667 Hz Hz
// Service_3 = RT_MIN	@ 1 Hz

// This is necessary for CPU affinity macros in Linux
//#define _GNU_SOURCE

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>

#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <semaphore.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>

#include <syslog.h>
#include <sys/time.h>

#include <errno.h>

#define USEC_PER_MSEC (1000)
#define NSEC_PER_MSEC (1000000)
#define NANOSEC_PER_SEC (1000000000)
#define NUM_CPU_CORES (1)
#define TRUE (1)
#define FALSE (0)

#define NUM_THREADS (3+1)

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;

pthread_mutex_t lockS1, lockS2, lockFrame;

int logFlag = 0;

/* Global Variables for Image */
Mat frame;

/* Global Variables */
long wcet_1 = 0, wcet_2 = 0, wcet_3 = 0;
struct timespec start_time, stop_time;
long delta_time_sec;
long delta_time_nsec;

int abortTest=FALSE;
int abortS1=FALSE, abortS2=FALSE, abortS3=FALSE;
sem_t semS1, semS2, semS3;
struct timeval start_time_val;

unsigned long long S1Cnt=0;
unsigned long long S2Cnt=0;
unsigned long long S3Cnt=0;

typedef struct
{
    int threadIdx;
    unsigned long long sequencePeriods;
} threadParams_t;


void *Sequencer(void *threadp);

void *Service_1(void *threadp);
void *Service_2(void *threadp);
void *Service_3(void *threadp);

void print_scheduler(void);

void laneDetect(void);

long delta_time(struct timespec *start_time, struct timespec *stop_time);

// LOGGING DECLARATIONS
int release = 900;

int startSeq_sec[900] = {0}, startSeq_msec[900] = {0};
int stopSeq_sec[900] = {0}, stopSeq_msec[900] = {0};

int startS1_sec[900] = {0}, startS1_msec[900] = {0};
int stopS1_sec[900] = {0}, stopS1_msec[900] = {0};

int startS2_sec[900] = {0}, startS2_msec[900] = {0};
int stopS2_sec[900] = {0}, stopS2_msec[900] = {0};


long delta_time(struct timespec *start_time, struct timespec *stop_time) {
	delta_time_sec = stop_time->tv_sec - start_time->tv_sec;
	
	if((stop_time->tv_sec - start_time->tv_sec) != 0)
	{
		delta_time_nsec = ((stop_time->tv_sec - start_time->tv_sec - 1) * 10 * NANOSEC_PER_SEC) + (NANOSEC_PER_SEC - start_time->tv_nsec) + stop_time->tv_nsec;
	}
	else
		delta_time_nsec = stop_time->tv_nsec - start_time->tv_nsec;

	return delta_time_nsec;
}

int main(void)
{
    struct timeval current_time_val;
    int i, rc, scope;
    cpu_set_t threadcpu;
    pthread_t threads[NUM_THREADS];
    threadParams_t threadParams[NUM_THREADS];
    pthread_attr_t rt_sched_attr[NUM_THREADS];
    int rt_max_prio, rt_min_prio;
    struct sched_param rt_param[NUM_THREADS];
    struct sched_param main_param;
    pthread_attr_t main_attr;
    pid_t mainpid;
    // represents a set of CPUs
    cpu_set_t allcpuset;

    printf("Starting Sequencer Demo\n");
    gettimeofday(&start_time_val, (struct timezone *)0);
    gettimeofday(&current_time_val, (struct timezone *)0);
    syslog(LOG_CRIT, "Sequencer @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);

   //printf("System has %d processors configured and %d available.\n", get_nprocs_conf(), get_nprocs());

    // clears set so that it contains on CPUs
   CPU_ZERO(&allcpuset);

   for(i=0; i < NUM_CPU_CORES; i++)
        // add CPU cpu to set
       CPU_SET(i, &allcpuset);

    // return the number of CPUs in set
   printf("Using CPUS=%d from total available.\n", CPU_COUNT(&allcpuset));


    // initialize the sequencer semaphores
    //
    if (sem_init (&semS1, 0, 0)) { printf ("Failed to initialize S1 semaphore\n"); exit (-1); }
    if (sem_init (&semS2, 0, 0)) { printf ("Failed to initialize S2 semaphore\n"); exit (-1); }
    if (sem_init (&semS3, 0, 0)) { printf ("Failed to initialize S3 semaphore\n"); exit (-1); }
    
	// Set default protocol for mutex
	//
	pthread_mutex_init(&lockS1, NULL);
	pthread_mutex_init(&lockS2, NULL);
	pthread_mutex_init(&lockFrame, NULL);

    // returns the process ID of the calling process
    mainpid=getpid();

    // storing the max (99) and min (1) priorities of SCHED_FIFO in variables
    rt_max_prio = sched_get_priority_max(SCHED_FIFO);
    rt_min_prio = sched_get_priority_min(SCHED_FIFO);

    rc=sched_getparam(mainpid, &main_param);

    //*********************************************************************************************************
    // for every thread, scheduling policy (which is SCHED_FIFO in this case) and thread priority has to be set
    //*********************************************************************************************************

    // for main thread, priority is set to be max
    main_param.sched_priority=rt_max_prio;
    // sets scheduling policy and parameters for main thread
    rc=sched_setscheduler(getpid(), SCHED_FIFO, &main_param);
    if(rc < 0) perror("main_param");
    print_scheduler();

    // returns the contention scope attribute of the thread attributes object referred to by attr in the buffer pointed to by scope
    pthread_attr_getscope(&main_attr, &scope);

    // thread competes for resources with ALL other threads in ALL processes on the system that are in the same scheduling allocation domain
    if(scope == PTHREAD_SCOPE_SYSTEM)
      printf("PTHREAD SCOPE SYSTEM\n");
    // thread competes for resources with all other threads in the same process that were also created with the PTHREAD_SCOPE_PROCESS contention scope
    else if (scope == PTHREAD_SCOPE_PROCESS)
      printf("PTHREAD SCOPE PROCESS\n");
    else
      printf("PTHREAD SCOPE UNKNOWN\n");

    // printing out max and min priorities
    printf("rt_max_prio=%d\n", rt_max_prio);
    printf("rt_min_prio=%d\n", rt_min_prio);

    for(i=0; i < NUM_THREADS; i++)
    {

      CPU_ZERO(&threadcpu);
      CPU_SET(3, &threadcpu);

      // initialize thread attribute
      rc=pthread_attr_init(&rt_sched_attr[i]);
      // thread will have its own scheduling policy and parameters with PTHREAD_EXPLICIT_SCHED
      rc=pthread_attr_setinheritsched(&rt_sched_attr[i], PTHREAD_EXPLICIT_SCHED);
      // setting thread scheduling policy to be SCHED_FIFO
      rc=pthread_attr_setschedpolicy(&rt_sched_attr[i], SCHED_FIFO);
      //rc=pthread_attr_setaffinity_np(&rt_sched_attr[i], sizeof(cpu_set_t), &threadcpu);

      // ignore next 2 lines as they are set correctly later in the code; removing these 2 lines should not create any problem
      //rt_param[i].sched_priority=rt_max_prio-i;
      //pthread_attr_setschedparam(&rt_sched_attr[i], &rt_param[i]);

      threadParams[i].threadIdx=i;
    }
   
    printf("Service threads will run on %d CPU cores\n", CPU_COUNT(&threadcpu));

    // Create Service threads which will block awaiting release for:
    //

    // Servcie_1 = RT_MAX-1	@ 5 Hz
    //
    // setting priority for service thread
    rt_param[1].sched_priority=rt_max_prio-1;
    pthread_attr_setschedparam(&rt_sched_attr[1], &rt_param[1]);
    // after attributes are set, service thread is created. 
    // Upon creating, thread enters Service_1 and waits for semaphore there, which will be given by Sequencer
    // Sequencer thread is created at last after every service thread is created,
    // so until then all 7 service threads will be waiting for their respective semaphroes to be given by Sequencer
    rc=pthread_create(&threads[1],               // pointer to thread descriptor
                      &rt_sched_attr[1],         // use specific attributes
                      //(void *)0,               // default attributes
                      Service_1,                 // thread function entry point
                      (void *)&(threadParams[1]) // parameters to pass in
                     );
    if(rc < 0)
        perror("pthread_create for service 1");
    else
        printf("pthread_create successful for service 1\n");


    // Service_2 = RT_MAX-2	@ 2 Hz
    //
    rt_param[2].sched_priority=rt_max_prio-2;
    pthread_attr_setschedparam(&rt_sched_attr[2], &rt_param[2]);
    rc=pthread_create(&threads[2], &rt_sched_attr[2], Service_2, (void *)&(threadParams[2]));
    if(rc < 0)
        perror("pthread_create for service 2");
    else
        printf("pthread_create successful for service 2\n");


    // Service_3 = RT_MIN	@ 1 Hz
    //
    rt_param[3].sched_priority=rt_min_prio;
    pthread_attr_setschedparam(&rt_sched_attr[3], &rt_param[3]);
    rc=pthread_create(&threads[3], &rt_sched_attr[3], Service_3, (void *)&(threadParams[3]));
    if(rc < 0)
        perror("pthread_create for service 3");
    else
        printf("pthread_create successful for service 3\n");
	
	
	/*** Required Program Initialization Starts */

	/*** Required Program Initialization Ends */

 
    // Create Sequencer thread, which like a cyclic executive, is highest prio
    printf("Start sequencer\n");
    threadParams[0].sequencePeriods=900;

    // Sequencer = RT_MAX	@ 50 Hz
    //
    // Sequencer thread given max priority
    rt_param[0].sched_priority=rt_max_prio;
    pthread_attr_setschedparam(&rt_sched_attr[0], &rt_param[0]);
    rc=pthread_create(&threads[0], &rt_sched_attr[0], Sequencer, (void *)&(threadParams[0]));
    if(rc < 0)
        perror("pthread_create for sequencer service 0");
    else
        printf("pthread_create successful for sequeencer service 0\n");

    // after 2701 iterations of Sequencer, all the services and Sequencer will exit and come here
   for(i=0;i<NUM_THREADS;i++)
	   pthread_join(threads[i], NULL);
   
	/*** Required Program Closing Starts */

	if(pthread_mutex_destroy(&lockS1) != 0)
		perror("mutex S1 destroy");

	if(pthread_mutex_destroy(&lockS2) != 0)
		perror("mutex S2 destroy");
		
	if(pthread_mutex_destroy(&lockFrame) != 0)
		perror("mutex Frame destroy");

	/*** Required Program Closing Ends */

   printf("\nTEST COMPLETE\n");
   
   printf("WCET 1 = %ldms\n", wcet_1/1000000);
   printf("WCET 2 = %ldms\n", wcet_2/1000000);
   
   return 0;
}


void *Sequencer(void *threadp)
{
    struct timeval current_time_val;
    struct timespec delay_time = {0,20000000}; // delay for 20 msec, 50 Hz
    struct timespec remaining_time;

    double residual;
    int rc, delay_cnt=0;
    unsigned long long seqCnt=0;
    threadParams_t *threadParams = (threadParams_t *)threadp;

    gettimeofday(&current_time_val, (struct timezone *)0);
    //syslog(LOG_CRIT, "Sequencer thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);
    printf("Sequencer thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);

    // runs the sequencer for 900 times and then terminates
    do
    {
        delay_cnt=0; residual=0.0;

        //gettimeofday(&current_time_val, (struct timezone *)0);
        //syslog(LOG_CRIT, "Sequencer thread prior to delay @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);
        do
        {
            rc=nanosleep(&delay_time, &remaining_time);

            if(rc == EINTR)
            { 
                residual = remaining_time.tv_sec + ((double)remaining_time.tv_nsec / (double)NANOSEC_PER_SEC);

                if(residual > 0.0) printf("residual=%lf, sec=%d, nsec=%d\n", residual, (int)remaining_time.tv_sec, (int)remaining_time.tv_nsec);
 
                delay_cnt++;
            }
            else if(rc < 0)
            {
                perror("Sequencer nanosleep");
                exit(-1);
            }
           
        } while((residual > 0.0) && (delay_cnt < 100));

        seqCnt++;
        gettimeofday(&current_time_val, (struct timezone *)0);
        //syslog(LOG_CRIT, "Sequencer cycle %llu @ sec=%d, msec=%d\n", seqCnt, (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);


        if(delay_cnt > 1) printf("Sequencer looping delay %d\n", delay_cnt);


        // Release each service at a sub-rate of the generic sequencer rate
	// Servcie_1 = RT_MAX-1	@ 2 Hz - 500 ms
        if((seqCnt % 25) == 0) sem_post(&semS1);
        
	// Service_2 = RT_MAX-2	@ 1.667 Hz - 600 ms   
        if((seqCnt % 30) == 0) sem_post(&semS2);    

        //gettimeofday(&current_time_val, (struct timezone *)0);
        //syslog(LOG_CRIT, "Sequencer release all sub-services @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);

    } while(!abortTest && (seqCnt < threadParams->sequencePeriods));

    // give semaphore for 901th time and then aborts service
    sem_post(&semS1); sem_post(&semS2);
    abortS1=TRUE; abortS2=TRUE; abortS3=TRUE;

    pthread_exit((void *)0);
}


/**************************************************
* SERVICE 1
* Image Capture
**************************************************/
void *Service_1(void *threadp)
{
    struct timeval current_time_val;
    //unsigned long long S1Cnt=0;
    threadParams_t *threadParams = (threadParams_t *)threadp;

	/*** S1 Initialization Starts */

	// CAMERA VIDEO
	/*
	VideoCapture cap;
	// open the default camera using default API
	cap.open(0);
	// OR advance usage: select any API backend
	int deviceID = 0;             // 0 = open default camera
	int apiID = cv::CAP_ANY;      // 0 = autodetect default API
	// open selected camera using selected API
	cap.open(deviceID + apiID);*/

	// PRECAPTURED VIDEO
	VideoCapture cap("lane_white_right.mp4");
	
	
	// Check if camera opened successfully
	if(!cap.isOpened()){
		printf("Error opening video stream or file");
	}
	/*** S1 Initialization Ends */
	
    gettimeofday(&current_time_val, (struct timezone *)0);
    syslog(LOG_CRIT, "Camera Capture thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);
    printf("Camera Capture thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);

    // abortS1 gets true in Sequencer after Sequencer executes for 900 times
    while(!abortS1)
    {
        sem_wait(&semS1);
	clock_gettime(CLOCK_REALTIME, &start_time);
	
	pthread_mutex_lock(&lockS1);
	S1Cnt++;	
	startS1_sec[S1Cnt] = (int)(start_time.tv_sec - start_time_val.tv_sec);
	startS1_msec[S1Cnt] = (int)start_time.tv_nsec/NSEC_PER_MSEC;
	logFlag = 1;
	pthread_mutex_unlock(&lockS1);
	
	//printf("Img Capture release %llu @ sec=%d, msec=%d\n", S1Cnt, startS1_sec[S1Cnt], startS1_msec[S1Cnt]);
	
	/*** Actual service 1 starts here */
	// CAMERA VIDEO
	//cap.read(frame);

	// PRECAPTURED VIDEO
	pthread_mutex_lock(&lockFrame);
	
	cap >> frame;
	
	// If the frame is empty, break immediately
	if (frame.empty())
	{
		printf("no frame\n");
		break;
	}
	
	// CAMERA VIDEO
	//if (waitKey(5) >= 0)
	//break;
	
	// PRECAPTURED VIDEO
	// Press  ESC on keyboard to exit
	
	char c=(char)waitKey(1);
	if(c==27)
		break;
		
	pthread_mutex_unlock(&lockFrame);
	
	
	gettimeofday(&current_time_val, (struct timezone *)0);
	//printf("Frame Sampler end %llu @ sec=%d, msec=%d\n", S1Cnt, (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);
	/*** Actual service 1 ends here */
	
	clock_gettime(CLOCK_REALTIME, &stop_time);
	long run_time = delta_time(&start_time, &stop_time);
	if(run_time > wcet_1){
		wcet_1 = run_time;
	}
	
	sem_post(&semS3);
    }
	
	/*** S1 Closing Ends */
	// When everything done, release the video capture object
	cap.release();
 
	// Closes all the frames
	destroyAllWindows();
	/*** S1 Closing Ends */

    pthread_exit((void *)0);
}

/**************************************************
* SERVICE 2
* Image Processing - Lane Detection and Result
**************************************************/
void *Service_2(void *threadp)
{
    struct timeval current_time_val;
    //unsigned long long S2Cnt=0;
    threadParams_t *threadParams = (threadParams_t *)threadp;

    gettimeofday(&current_time_val, (struct timezone *)0);
    syslog(LOG_CRIT, "Time-stamp with Image Analysis thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);
    printf("Time-stamp with Image Analysis thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec-start_time_val.tv_sec), (int)current_time_val.tv_usec/USEC_PER_MSEC);

    while(!abortS2)
    {
        sem_wait(&semS2);
	clock_gettime(CLOCK_REALTIME, &start_time);
	
	pthread_mutex_lock(&lockS2);
	S2Cnt++;	
	startS2_sec[S2Cnt] = (int)(start_time.tv_sec - start_time_val.tv_sec);
	startS2_msec[S2Cnt] = (int)start_time.tv_nsec/NSEC_PER_MSEC;
	logFlag = 2;
	pthread_mutex_unlock(&lockS2);
	
	//printf("Img Process release %llu @ sec=%d, msec=%d\n", S2Cnt, startS2_sec[S2Cnt], startS2_msec[S2Cnt]);
	
	/*** Actual service 2 starts here */
	laneDetect();
	/*** Actual service 2 ends here */
	
	clock_gettime(CLOCK_REALTIME, &stop_time);
	
	//printf("Img Process end     %llu @ sec=%d, msec=%d\n", S2Cnt, stopS2_sec[S2Cnt], stopS2_msec[S2Cnt]);
	long run_time = delta_time(&start_time, &stop_time);
	if(run_time > wcet_2){
		wcet_2 = run_time;
	}
	
	sem_post(&semS3);
    }

    pthread_exit((void *)0);
}

/**************************************************
* SERVICE 3
* Logging
**************************************************/
void *Service_3(void *threadp)
{
    while(!abortS1)
    {	
	sem_wait(&semS3);
	
	/*** Actual service 3 starts here */
	
	if(logFlag == 1)
	{
		logFlag = 0;
		pthread_mutex_lock(&lockS1);
		printf("Img Capture release %llu @ sec=%d, msec=%d\n", S1Cnt, startS1_sec[S1Cnt], startS1_msec[S1Cnt]);
		syslog(LOG_CRIT, "Img Capture release %llu @ sec=%d, msec=%d\n", S1Cnt, startS1_sec[S1Cnt], startS1_msec[S1Cnt]);
		pthread_mutex_unlock(&lockS1);
	}
	
	else if(logFlag == 2)
	{
		logFlag = 0;
		pthread_mutex_lock(&lockS2);
		printf("Img Process release %llu @ sec=%d, msec=%d\n", S2Cnt, startS2_sec[S2Cnt], startS2_msec[S2Cnt]);
		syslog(LOG_CRIT, "Img Process release %llu @ sec=%d, msec=%d\n", S2Cnt, startS2_sec[S2Cnt], startS2_msec[S2Cnt]);
		pthread_mutex_unlock(&lockS2);
	}
	
	/*** Actual service 3 ends here */
    }

    pthread_exit((void *)0);
}


void print_scheduler(void)
{
   int schedType;

    // returns the scheduling policy of the process specified by pid
   schedType = sched_getscheduler(getpid());

   switch(schedType)
   {
       case SCHED_FIFO:
           printf("Pthread Policy is SCHED_FIFO\n");
           break;
       case SCHED_OTHER:
           printf("Pthread Policy is SCHED_OTHER\n"); exit(-1);
         break;
       case SCHED_RR:
           printf("Pthread Policy is SCHED_RR\n"); exit(-1);
           break;
       default:
           printf("Pthread Policy is UNKNOWN\n"); exit(-1);
   }
}


/**************************************************
* LANE DETECTION
**************************************************/
void laneDetect(void)
{
	// required variables    
	Mat gray_frame, hls_frame, white_frame, yellow_frame, white_yellow_frame, bitwise_and_frame;
	Mat gauss_frame, thresh_frame, edge_frame, cropped_frame;
    
	//** COLOR CONVERSION
	// conversion to gray and hls
	pthread_mutex_lock(&lockFrame);
	cvtColor(frame, gray_frame, CV_BGR2GRAY);
	cvtColor(frame, hls_frame, CV_BGR2HLS);
	pthread_mutex_unlock(&lockFrame);
	
	// white frame
	inRange(gray_frame, 125, 255, white_frame);
	// yellow frame
	inRange(hls_frame, Scalar(0, 62, 50), Scalar(60, 75, 55), yellow_frame);
	
	// white_yellow_frame
	bitwise_or(yellow_frame, white_frame, white_yellow_frame);
	
	// bitwise_and_frame
	bitwise_and(gray_frame, white_yellow_frame, bitwise_and_frame);

	//** BLUR
	GaussianBlur(bitwise_and_frame, gauss_frame, Size(1, 1), 0, 0 );
	
	//** EDGE DETECTION
	Mat kernel;
	Point anchor;
	threshold(gauss_frame, thresh_frame, 140, 255, THRESH_BINARY);
	
	// Kernel [-1 0 1] create
	anchor = Point(-1, -1);
	double delta = 0;
	int ddepth = -1;
	kernel = Mat(1, 3, CV_32F);
	kernel.at<float>(0, 0) = -1;
	kernel.at<float>(0, 1) = 0;
	kernel.at<float>(0, 2) = 1;

	// find edges by filter
	filter2D(thresh_frame, edge_frame, ddepth, kernel, anchor, delta, BORDER_DEFAULT);
	
	//** REGION OF INTEREST
	// in a trapezium shape
	Mat mask = Mat::zeros(edge_frame.size(), edge_frame.type());
	Point pts[4] = {
		Point(90, 540),
		Point(280, 340),
		Point(600, 340),
		Point(900, 540)
	};

	// Create a binary polygon mask
	fillConvexPoly(mask, pts, 4, Scalar(255, 0, 0));
	// Multiply the edges image and the mask to get the output
	bitwise_and(edge_frame, mask, cropped_frame);

	//** HOUGH LINES DETECTION
	vector<Vec4i> hough_lines;
	HoughLinesP(cropped_frame, hough_lines, 1, CV_PI/180, 40, 5, 200);

	//** OVERLAPPING LINES ON SOURCE
	for( size_t i = 0; i < hough_lines.size(); i++ )
	{
		Vec4i hline = hough_lines[i];
		line(frame, Point(hline[0], hline[1]), Point(hline[2], hline[3]), Scalar(0,0,255), 3, CV_AA);
	}
	
	//** SHOW OUTPUT FRAME
	imshow("Output", frame);
}
// CODE ENDS
