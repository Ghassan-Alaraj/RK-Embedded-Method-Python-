# imports
import numpy as np
from matplotlib import pyplot as plt
from time import time
from mpl_toolkits.mplot3d import Axes3D



class RungeKutta(object):
	"""
	Class that implements explicit RK methods.

	Attributes:
		alpha (numpy array): vector of weights in the Butcher tableau
		beta  (numpy array): vector of nodes in the Butcher tableau
		gamma (numpy array): RK matrix in the Butcher tableau
		n (int): number of derivatives used in explicit RK method
	"""
	def __init__(self, alpha, beta, gamma, order):
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma
		self.n = len(beta)
		self.order = order

	def adaptive_step_solve(self, func, y0, t0, t1, tol, *args):
		"""
		Compute solution to ODE using an explicit RK method with adaptive step size.

		Args:
			func (callable): derivative function that returns an ndarray of derivative values.
			y0 (ndarray): initial conditions for each solution variable.
			t0 (float): start value of independent variable.
			t1 (float):	stop value of independent variable.
			tol (float): error tolerance for determining adaptive step size.
			*args : optional system parameters to pass to derivative function.

		Returns:
			t (ndarray): independent variable values at which dependent variable(s) calculated.
			y (ndarray): dependent variable(s).
		"""
		y = [] #set-up y list
		y.append(y0)

		tk = [] #set-up time list
		tk.append(t0)

		hNew = 0.10 # intial guess for step size
		tolFloat = 1.e-14    # set the floating point error tolerance
		phi = 0.9 #saftey factor 

		while (t1 - tk[-1]) > tolFloat: 
			succ = False #a boolean to indicate if the step was a success
			h = hNew
			y1, y2 = adaptive_step_solve_step(self, func, tk[-1] , y[-1], h,args) # get the value of y1 and y2 using the specified RK method

			epsilon = max(abs(y2 - y1)) #calculate the maxium error 
			
			if (abs(epsilon) < tol) and epsilon > tolFloat: #increase step size if the error is within the limit 
				hNew = phi * h * (abs(tol/epsilon))**(1/(self.order - 1))
				succ = True		
			elif (abs(epsilon) >= tol) and epsilon > tolFloat:#otherwise decrease the step size
				hNew = phi * h * (abs(tol/epsilon))**(1/(self.order))

			#lets add a test for min and max value of h
			if abs(hNew) < (t1/1000000.) or epsilon == 0.0: #our stepsize is to small i.e. we dont want to take more than a million steps 
				succ = True	
				hNew = (t1/1000000.) #generate a new step and calculate new y1, y2 and record
				y1, y2 = adaptive_step_solve_step(self, func,tk[-1] , y[-1], hNew,args)
				h = hNew # record step size

			elif abs(hNew) > (t1/100.): #our stepsize is to large i.e. we want to take at least 100
				succ = True	
				hNew = (t1/1000000.) #generate a new step and calculate new y1, y2 and record
				y1, y2 = adaptive_step_solve_step(self, func,tk[-1] , y[-1], hNew,args)
				h = hNew # record step size
				

			if succ:
				y.append(y2)
				tk.append(tk[-1] + h)

		return np.array(tk), np.array(y)



def adaptive_step_solve_step(self, func, tk, yk, h, *args):
	''' Compute a single step of function using an embedded method.
	
		Parameters
		----------
		self : object
			RungeKutta object.
		func : callable
			Derivative function.
		tk : float
			Independent variable at current step.
		yk : ndarray
			Dependent variable (solution) at current step.	
		h : float
			Step size.
		args : iterable
			Optional parameters to pass into derivative function. 
			
		Returns
		-------
		y1 : ndarray
			Dependent variable (solution) at next step, lower order.
		y2 : ndarray
			Dependent variable (solution) at next step, higher order.	
	'''
	n = len(self.beta)
	m = len(yk)
	fns = np.empty((n,m))
	fns1 = np.empty((n,m))
	fns2 = np.empty((n,m))
	gammaF = np.empty((1,m))

	for i in range(n):  #compute fns for all solutions

		for j in range(m): #compute gamma for all solutions
			gammaF[0,j] = np.dot(fns[:,j], self.gamma[i,:])

		fns[i,:] = func(tk + h * self.beta[i], yk + h * gammaF[0], *args[0])

	for i in range(n): #compute alpha * fn for all solutions
		fns1[i,:] = self.alpha[1,i] * fns[i,:]
		fns2[i,:] = self.alpha[0,i] * fns[i,:]

	#compute next step for both systems	
	y1 = yk + h*(sum(fns1))
	y2 = yk + h*(sum(fns2))
		
	return y1, y2

def derivative_predator_prey_food(t, y, a1, a2, b1, b2, d1, d2):
    """
    Compute the derivatives for the predator-prey-food model, for rabbits (r), foxes (f) and carrots (c)

    Args:
        t (float): independent variable i.e. time (s).
        y (ndarray): current r and f.
        a1 (float): system parameter.
        a2 (float): system parameter.
        b1 (float): system parameter.
        b2 (float): system parameter.
        d1 (float): system parameter.
        d2 (float): system parameter.

    Returns:
        derivative (ndarray): derivatives of r, f and c.
    """
    f_r = y[0]*a1*y[2]/(1+b1*y[2]) - y[1]*a2*y[0]/(1+b2*y[0])-d1*y[0]
    f_f = y[1]*a2*y[0]/(1+b2*y[0])-d2*y[1]
    f_c = y[2] * (1 - y[2]) - y[0] * a1 * y[2] / (1 + b1 * y[2])
    derivative = np.array([f_r, f_f, f_c])
    return derivative


def main():

    # set parameters needed to solve ODE
    t0 = 0.0                          # set the range of independent variable
    t1 = 1500.                      # set the range of independent variable
    tol = 1.e-6                     # set the error tolerance
    y0 = np.array([0.25, 8., 0.8])  # initial conditions
    a1 = 5                          # model parameter
    a2 = 0.1                        # model parameter
    b1 = 2.5                        # model parameter
    b2 = 2                          # model parameter
    d1 = 0.4                        # model parameter
    d2 = 0.01                       # model parameter

    #use the Bogcaki 3rd order method as the embedded method
    alpha = np.array([[2./9., 1./3., 4./9., 0.0 ],
                    [7./24., 1./4., 1./3., 1./8.]])

    beta = np.array([0., 1./2., 3./4., 1.0])

    gamma = np.array([[0., 0., 0., 0.], 
                    [1./2., 0., 0., 0.],
                    [0., 3./4., 0., 0.],
                    [2./9., 1./3., 4./9., 0.]])
    order = 3

    method = RungeKutta(alpha, beta, gamma,order)

    #get solution data and record time for comparsion with fixed step size later
    tk, yk = method.adaptive_step_solve(derivative_predator_prey_food, y0,0.0,t1,tol,a1,a2,b1,b2,d1,d2)
    t0 = time()
    b1 = 3.5
    tk1, yk1 = method.adaptive_step_solve(derivative_predator_prey_food, y0,0.0,t1,tol,a1,a2,b1,b2,d1,d2)
    t1 = time()
    print(t1-t0)

    #---------Graph One---------------#
    #b1 = 2.5 graphs

    #plot 3-D phase graph
    ax2 = plt.axes(projection='3d')
    ax2.scatter3D(yk[:,0], yk[:,1], yk[:,2], c=tk, cmap='Reds')
    ax2.set_xlabel('r(t)')
    ax2.set_ylabel('f(t)')
    ax2.set_zlabel('c(t)')
    ax2.set_title('3-D phase plot b1 = 2.5')
    plt.savefig("Q4_32_3D_phase_b25")

    f, ax = plt.subplots(1, 2, sharex='col', sharey='row', gridspec_kw={'hspace': 0.5, 'wspace': 0.2})
    rgraph, = ax[0].plot(tk,yk[:,0],'-b',label = 'r(t)')
    fgraph, = ax[0].plot(tk,yk[:,1],'-r',label = 'f(t)')
    cgraph, = ax[0].plot(tk,yk[:,2],'-g',label = 'c(t)')
    ax[0].set_ylabel('solution',fontsize = 9)  #set labels
    ax[0].set_xlabel('t',fontsize = 9)
    ax[0].set_title('r(t), f(t) and c(t) for b1 = 2.5',fontsize = 8)
    ax[0].legend(loc = 0, handles = [rgraph,fgraph,cgraph ],fontsize = 9)  #add legend

    phasegraph, = ax[1].plot(yk[:,0],yk[:,1],'-b',label = 'r(t) vs f(t)')
    ax[1].set_ylabel('f(t)',fontsize = 9)  #set labels
    ax[1].set_xlabel('r(t)',fontsize = 9)
    ax[1].set_title('phase graph of r(t) vs f(t) \n for b1 = 2.5',fontsize = 8)
    ax[1].legend(loc = 0, handles = [phasegraph],fontsize = 9)  #add legend

    plt.savefig("Q4_31_b1_25")

    #---------Graph Two---------------#
    #b1 = 3.5 graphs

    #plot 3-D phase graph
    ax2 = plt.axes(projection='3d')
    ax2.scatter3D(yk1[:,0], yk1[:,1], yk1[:,2], c=tk1, cmap='Blues')
    ax2.set_xlabel('r(t)')
    ax2.set_ylabel('f(t)')
    ax2.set_zlabel('c(t)')
    ax2.set_title('3-D phase plot b1 = 3.5')

    plt.savefig("Q4_42_3D_phase_b35")

    f, ax1 = plt.subplots(1, 2, sharex='col', sharey='row', gridspec_kw={'hspace': 0.5, 'wspace': 0.2})
    rgraph, = ax1[0].plot(tk1,yk1[:,0],'-b',label = 'r(t)')
    fgraph, = ax1[0].plot(tk1,yk1[:,1],'-r',label = 'f(t)')
    cgraph, = ax1[0].plot(tk1,yk1[:,2],'-g',label = 'c(t)')
    ax1[0].set_ylabel('solution',fontsize = 9)  #set labels
    ax1[0].set_xlabel('t',fontsize = 9)
    ax1[0].set_title('r(t), f(t) and c(t) = 3.5',fontsize = 8)
    ax1[0].legend(loc = 0, handles = [rgraph,fgraph,cgraph ],fontsize = 9)  #add legend

    phasegraph, = ax1[1].plot(yk1[:,0],yk1[:,1],'-b',label = 'r(t) vs f(t)')
    ax1[1].set_ylabel('f(t)',fontsize = 9)  #set labels
    ax1[1].set_xlabel('r(t)',fontsize = 9)
    ax1[1].set_title('phase graph of r(t) vs f(t) \n for b1 = 3.5',fontsize = 8)
    ax1[1].legend(loc = 0, handles = [phasegraph],fontsize = 9)  #add legend

    plt.savefig("Q4_41_b1_35")

    #------------Graph 3------------#
    #histograms
    stepsize1 = []
    for i in range(1,len(tk)):
        stepsize1.append(tk[i] - tk[i-1])

    f, ax2 = plt.subplots(1, 1, sharex='col', sharey='row', gridspec_kw={'hspace': 0.5, 'wspace': 0.2})

    _ = ax2.hist(stepsize1,bins = 90)
    ax2.set_ylabel('count',fontsize = 9)  #set labels
    ax2.set_xlabel('t',fontsize = 9)
    ax2.set_title('r(t) and f(t) for b1 = 2.5',fontsize = 8)
    plt.savefig("Q4_51_hhist_25") 

    stepsize2 = []
    for i in range(1,len(tk1)):
        stepsize2.append(tk1[i] - tk1[i-1])
    f, ax3 = plt.subplots(1, 1, sharex='col', sharey='row', gridspec_kw={'hspace': 0.5, 'wspace': 0.2})

    _ = ax3.hist(stepsize1,bins = 90)
    ax3.set_ylabel('count',fontsize = 9)  #set labels
    ax3.set_xlabel('t',fontsize = 9)
    ax3.set_title('r(t) and f(t) for b1 = 3.5',fontsize = 8)
    plt.savefig("Q4_52_hhist_35")

    #90 bins are not enough to showcase the difference between the step size of the two solution, but they are different !


    plt.show()

if __name__ == "__main__":
    #Author: Ghassan Al-A'raj
    main()

