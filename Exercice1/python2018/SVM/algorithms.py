import time
import numpy as np
import scipy.sparse.linalg as spla
from numpy.random import randint
from scipy.sparse.linalg.dsolve import linsolve

def GD(fx, gradf, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lips']       - Lipschitz constant for gradient.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               fx                      - objective function
               gradf                 - gradient mapping
               
    :return: x, info
    """
    print(68*'*')
    print('Gradient Descent')

    # Initialize x and alpha.

    maxit = parameter['maxit']
    x = parameter['x0']
    #alpha -- step size
    #alpha = 1/L with L:= Lipschitz constant
    alpha = 1/parameter['Lips']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        x_next = x - alpha*gradf(x)
        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        #### YOUR CODES HERE

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter %  5 ==0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
    info['iter'] = maxit
    return x, info


# gradient with strong convexity
def GDstr(fx, gradf, parameter) :
    """
    Function:  GDstr(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lips']       - Lipschitz constant for gradient.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               fx                      - objective function
               gradf                  - gradient mapping
               
    :return: x, info
    """

    print(68*'*')
    print('Gradient Descent  with strong convexity')

    # Initialize x and alpha.
    x = parameter['x0']
    #alpha = 2/(L+mu)
    alpha = 2/(parameter['Lips'] + parameter['strcnvx'])

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start timer
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
        x_next = x - alpha*gradf(x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info

# accelerated gradient
def AGD(fx, gradf, parameter):
    """
    *******************  EE556 - Mathematics of Data  ************************
    Function:  AGD (fx, gradf, parameter)
    Purpose:   Implementation of the accelerated gradient descent algorithm.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lips']       - Lipschitz constant for gradient.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               fx                      - objective function
               gradf                  - gradient mapping
               
    :return: x, info
    """
    print(68 * '*')
    print('Accelerated Gradient')

    # Initialize x, y and t.
    #### YOUR CODES HERE

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):

        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        #### YOUR CODES HERE

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))


        # Prepare next iteration
        x = x_next
        t = t_next

    return x, info

# accelerated gradient with strong convexity
def AGDstr(fx, gradf, parameter):
    """
    *******************  EE556 - Mathematics of Data  ************************
    Function:  AGDstr(fx, gradf, parameter)
    Purpose:   Implementation of the accelerated gradient descent algorithm.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lips']       - Lipschitz constant for gradient.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               fx                      - objective function
               gradf                  - gradient mapping
               
    :return: x, info
    """
    print(68 * '*')
    print('Accelerated Gradient with strong convexity')

    # Initialize x, y and t.
    #### YOUR CODES HERE

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):

        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        #### YOUR CODES HERE

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))


        # Prepare next iteration
        x = x_next

    return x, info

# LSGD
def LSGD(fx, gradf, parameter):
    """
    Function:  [x, info] = LSGD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent with line-search.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lips']       - Lipschitz constant for gradient.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               fx                      - objective function
               gradf                  - gradient mapping
               
    :return: x, info
    """
    print(68 * '*')
    print('Gradient Descent with line search')

    # Initialize x, y and t.
    #### YOUR CODES HERE

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()
        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
        #### YOUR CODES HERE

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare next iteration
        x = x_next

    return x, info

# LSAGD
def LSAGD(fx, gradf, parameter):
    """
    Function:  [x, info] = LSAGD (fx, gradf, parameter)
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lips']       - Lipschitz constant for gradient.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               fx                      - objective function
               gradf                  - gradient mapping
               
    :return: x, info
        """
    print(68 * '*')
    print('Accelerated Gradient with line search')

    # Initialize x, y and t.
    #### YOUR CODES HERE

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
        #### YOUR CODES HERE

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next

    return x, info


# AGDR
def AGDR(fx, gradf, parameter):
    """
    Function:  [x, info] = AGDR (fx, gradf, parameter)
    Purpose:   Implementation of the AGD with adaptive restart.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lips']       - Lipschitz constant for gradient.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               fx                      - objective function
               gradf                  - gradient mapping
               
    :return: x, info
    """
    print(68 * '*')
    print('Accelerated Gradient with restart')

    # Initialize x, y, t and find the initial function value (fval).
    
    #### YOUR CODES HERE

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
        #### YOUR CODES HERE

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next

    return x, info


# LSAGDR
def LSAGDR(fx, gradf, parameter):
    """
    Function:  [x, info] = LSAGDR (fx, gradf, parameter)
    Purpose:   Implementation of AGD with line search and adaptive restart.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lips']       - Lipschitz constant for gradient.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               fx                      - objective function
               gradf                  - gradient mapping
               
    :return: x, info
    """
    print(68 * '*')
    print('Accelerated Gradient with line search + restart')

    # Initialize x, y, t and find the initial function value (fval).
    #### YOUR CODES HERE

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
        #### YOUR CODES HERE

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next

    return x, info

def QNM(fx, gradf, parameter):
    """
    Function:  [x, info] = QNM (fx, gradf, hessf, parameter)
    Purpose:   Implementation of the quasi-Newton method with BFGS update.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lips']       - Lipschitz constant for gradient.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               fx                      - objective function
               gradf                  - gradient mapping
               
    :return: x, info
    """
    print(68 * '*')
    print('Quasi Newton Method')
    # Initialize x, B0, alpha, grad (and any other)
    #### YOUR CODES HERE

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
        #### YOUR CODES HERE

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info

# Newton
def NM(fx, gradf, hessf, parameter):
    """
    Function:  [x, info] = NM (fx, gradf, hessf, parameter)
    Purpose:   Implementation of the Newton method.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lips']       - Lipschitz constant for gradient.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               fx                      - objective function
               gradf                  - gradient mapping
               
    :return: x, info
    """

    print(68 * '*')
    # Initialize x, alpha (and any other)
    print('Newton Method')
    
    #### YOUR CODES HERE

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}


    # Main loop.
    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        

    	#### YOUR CODES HERE

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info

def SGD(fx, gradfsto, parameter):
    """
    Function:  [x, info] = SGD(fx, gradf, parameter)
    Purpose:   Implementation of the stochastic gradient descent algorithm.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lmax']       - Maximum of Lipschitz constant of all f_i.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               parameter['no0functions']    - Number of functions.
               fx                      - objective function
               gradfsto               - stochastic gradient mapping
               
    :return: x, info
    """
    print(68*'*')
    print('Stochastic Gradient Descent')

    # Initialize x and alpha and other.
    
    #### YOUR CODES HERE

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}


    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        

    	#### YOUR CODES HERE

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info


def SAG(fx, gradfsto, parameter):
    """
    Function:  [x, info] = SAG(fx, gradf, parameter)
    Purpose:   Implementation of the stochastic averaging gradient algorithm.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lmax']       - Maximum of Lipschitz constant of all f_i.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               parameter['no0functions']    - Number of functions.
               fx                      - objective function
               gradfsto               - stochastic gradient mapping
    """
    print(68*'*')
    print('Stochastic Averaging Gradient')

    # Initialize.
    
    #### YOUR CODES HERE


    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}
    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
        #### YOUR CODES HERE

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info

def SVR(fx, gradf, gradfsto, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the stochastic gradient with variance reduction algorithm.
    Parameter: parameter['x0']         - Initial estimate.
               parameter['maxit']      - Maximum number of iterations.
               parameter['Lmax']       - Maximum of Lipschitz constant of all f_i.
               parameter['strcnvx']    - Strong convexity parameter of f(x).
               parameter['no0functions']    - Number of functions.
               fx                      - objective function
               gradf                  - gradient mapping
               gradfsto               - stochastic gradient mapping
    """
    print(68*'*')
    print('Stochastic Gradient Descent with variance reduction')

    # Initialize.
    
    #### YOUR CODES HERE

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
        #### YOUR CODES HERE

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info



