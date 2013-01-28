/*

  Copied from Ken Stanley's NEAT C++ code. Originally written by Faustino Gomez
*/
#ifndef CARTPOLE_H
#define CARTPOLE_H    
#include <cmath>
 
   
    
class CartPole {
public:
  CartPole(bool randomize,bool velocity) : nmarkov_long(true), generalization_test(false)
  {
  maxFitness = 100000;

  MARKOV=velocity;

  MIN_INC = 0.001;
  POLE_INC = 0.05;
  MASS_INC = 0.01;

  LENGTH_2 = 0.05;
  MASSPOLE_2 = 0.01;

  // CartPole::reset() which is called here
}

  virtual void simplifyTask()
  {
  if(POLE_INC > MIN_INC) {
    POLE_INC = POLE_INC/2;
    MASS_INC = MASS_INC/2;
    LENGTH_2 -= POLE_INC;
    MASSPOLE_2 -= MASS_INC;
    //cout<<"#SIMPLIFY\n"<<endl;
    //cout<<"#Pole Length %2.4f\n"<<LENGTH_2;
  }
  else
    {
      //cout<<"#NO TASK CHANGE\n"<<endl;
    }
}

  virtual void nextTask()
  {

   LENGTH_2 += POLE_INC;   /* LENGTH_2 * INCREASE;   */
   MASSPOLE_2 += MASS_INC; /* MASSPOLE_2 * INCREASE; */
   //  ++new_task;
   //cout<<"#Pole Length %2.4f\n"<<LENGTH_2<<endl;
}



  double maxFitness;
  bool MARKOV;

  bool last_hundred;
  bool nmarkov_long;  //Flag that we are looking at the champ
  bool generalization_test;  //Flag we are testing champ's generalization

  double state[6];

  double jigglestep[1000];


  virtual void init(bool randomize)
  {
  static int first_time = 1;

  if (!MARKOV) {
    //Clear all fitness records
    cartpos_sum=0.0;
    cartv_sum=0.0;
    polepos_sum=0.0;
    polev_sum=0.0;
  }

  balanced_sum=0; //Always count # balanced

  last_hundred=false;

  if (randomize) {
    state[0] = (lrand48()%4800)/1000.0 - 2.4;
    state[1] = (lrand48()%2000)/1000.0 - 1;
    state[2] = (lrand48()%400)/1000.0 - 0.2;
    state[3] = (lrand48()%400)/1000.0 - 0.2;
    state[4] = (lrand48()%3000)/1000.0 - 1.5;
    state[5] = (lrand48()%3000)/1000.0 - 1.5;
  }
  else {


  if (!generalization_test) {
    state[0] = state[1] = state[3] = state[4] = state[5] = 0;
    state[2] = 0.07; // one_degree;
  }
  else {
    state[4] = state[5] = 0;
  }

    }
  if(first_time){
    //cout<<"Initial Long pole angle = %f\n"<<state[2]<<endl;;
    //cout<<"Initial Short pole length = %f\n"<<LENGTH_2<<endl;
    first_time = 0;
  }
}




  void performAction(double output,int stepnum)
  { 
  
  int i;
  double  dydx[6];

  const bool RK4=true; //Set to Runge-Kutta 4th order integration method
  const double EULER_TAU= TAU/4;
 
  /*random start state for long pole*/
  /*state[2]= drand48();   */
     
  /*--- Apply action to the simulated cart-pole ---*/

  if(RK4){
    for(i=0;i<2;++i){
      dydx[0] = state[1];
      dydx[2] = state[3];
      dydx[4] = state[5];
      step(output,state,dydx);
      rk4(output,state,dydx,state);
    }
  }
  else{
    for(i=0;i<8;++i){
      step(output,state,dydx);
      state[0] += EULER_TAU * dydx[0];
      state[1] += EULER_TAU * dydx[1];
      state[2] += EULER_TAU * dydx[2];
      state[3] += EULER_TAU * dydx[3];
      state[4] += EULER_TAU * dydx[4];
      state[5] += EULER_TAU * dydx[5];
    }
  }

  //Record this state
  cartpos_sum+=fabs(state[0]);
  cartv_sum+=fabs(state[1]);
  polepos_sum+=fabs(state[2]);
  polev_sum+=fabs(state[3]);
  if (stepnum<=1000)
    jigglestep[stepnum-1]=fabs(state[0])+fabs(state[1])+fabs(state[2])+fabs(state[3]);

  if (false) {
    ////cout<<"[ x: "<<state[0]<<" xv: "<<state[1]<<" t1: "<<state[2]<<" t1v: "<<state[3]<<" t2:"<<state[4]<<" t2v: "<<state[5]<<" ] "<<
    //cartpos_sum+cartv_sum+polepos_sum+polepos_sum+polev_sum<<endl;
    if (!(outsideBounds())) {
      if (balanced_sum<1000) {
	//cout<<".";
	++balanced_sum;
      }
    }
    else {
      if (balanced_sum==1000)
	balanced_sum=1000;
      else balanced_sum=0;
    }
  }
  else if (!(outsideBounds()))
    ++balanced_sum;

}

  
  void step(double action, double *st, double *derivs)
  {
    double force,costheta_1,costheta_2,sintheta_1,sintheta_2,
          gsintheta_1,gsintheta_2,temp_1,temp_2,ml_1,ml_2,fi_1,fi_2,mi_1,mi_2;

    force =  (action - 0.5) * FORCE_MAG * 2;
    costheta_1 = cos(st[2]);
    sintheta_1 = sin(st[2]);
    gsintheta_1 = GRAVITY * sintheta_1;
    costheta_2 = cos(st[4]);
    sintheta_2 = sin(st[4]);
    gsintheta_2 = GRAVITY * sintheta_2;
    
    ml_1 = LENGTH_1 * MASSPOLE_1;
    ml_2 = LENGTH_2 * MASSPOLE_2;
    temp_1 = MUP * st[3] / ml_1;
    temp_2 = MUP * st[5] / ml_2;
    fi_1 = (ml_1 * st[3] * st[3] * sintheta_1) +
           (0.75 * MASSPOLE_1 * costheta_1 * (temp_1 + gsintheta_1));
    fi_2 = (ml_2 * st[5] * st[5] * sintheta_2) +
           (0.75 * MASSPOLE_2 * costheta_2 * (temp_2 + gsintheta_2));
    mi_1 = MASSPOLE_1 * (1 - (0.75 * costheta_1 * costheta_1));
    mi_2 = MASSPOLE_2 * (1 - (0.75 * costheta_2 * costheta_2));
    
    derivs[1] = (force + fi_1 + fi_2)
                 / (mi_1 + mi_2 + MASSCART);
    
    derivs[3] = -0.75 * (derivs[1] * costheta_1 + gsintheta_1 + temp_1)
                 / LENGTH_1;
    derivs[5] = -0.75 * (derivs[1] * costheta_2 + gsintheta_2 + temp_2)
                  / LENGTH_2;

}

  void rk4(double f, double y[], double dydx[], double yout[])
  {

	int i;

	double hh,h6,dym[6],dyt[6],yt[6];


	hh=TAU*0.5;
	h6=TAU/6.0;
	for (i=0;i<=5;i++) yt[i]=y[i]+hh*dydx[i];
	step(f,yt,dyt);
	dyt[0] = yt[1];
	dyt[2] = yt[3];
	dyt[4] = yt[5];
	for (i=0;i<=5;i++) yt[i]=y[i]+hh*dyt[i];
	step(f,yt,dym);
	dym[0] = yt[1];
	dym[2] = yt[3];
	dym[4] = yt[5];
	for (i=0;i<=5;i++) {
		yt[i]=y[i]+TAU*dym[i];
		dym[i] += dyt[i];
	}
	step(f,yt,dyt);
	dyt[0] = yt[1];
	dyt[2] = yt[3];
	dyt[4] = yt[5];
	for (i=0;i<=5;i++)
		yout[i]=y[i]+h6*(dydx[i]+dyt[i]+2.0*dym[i]);
}

  bool outsideBounds()
  {
  const double failureAngle = thirty_six_degrees; 

  return 
    state[0] < -2.4              || 
    state[0] > 2.4               || 
    state[2] < -failureAngle     ||
    state[2] > failureAngle      ||
    state[4] < -failureAngle     ||
    state[4] > failureAngle;  
}

  const static int NUM_INPUTS=7;
  const static double MUP = 0.000002;
  const static double MUC = 0.0005;
  const static double GRAVITY= -9.8;
  const static double MASSCART= 1.0;
  const static double MASSPOLE_1= 0.1;

  const static double LENGTH_1= 0.5;		  /* actually half the pole's length */

  const static double FORCE_MAG= 10.0;
  const static double TAU= 0.01;		  //seconds between state updates 

  const static double one_degree= 0.0174532;	/* 2pi/360 */
  const static double six_degrees= 0.1047192;
  const static double twelve_degrees= 0.2094384;
  const static double fifteen_degrees= 0.2617993;
  const static double thirty_six_degrees= 0.628329;
  const static double fifty_degrees= 0.87266;

  double LENGTH_2;
  double MASSPOLE_2;
  double MIN_INC;
  double POLE_INC;
  double MASS_INC;

  //Queues used for Gruau's fitness which damps oscillations
  int balanced_sum;
  double cartpos_sum;
  double cartv_sum;
  double polepos_sum;
  double polev_sum;



};

#endif

