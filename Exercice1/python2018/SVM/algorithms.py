import time
import numpy as np
import math
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


        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        x_next = x - alpha * gradf(x)


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
    maxit = parameter['maxit']
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
    maxit = parameter['maxit']
    x = parameter['x0']
    t = 0
    y = parameter['x0']
    alpha = 1/parameter['Lips']


    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):

        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        x_next = y -alpha*gradf(y)
        t_next = 0.5*(1 + np.sqrt(1 + 4*(t**2)))
        y_next = x_next + (t-1)*(x_next-x)/(t_next)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))


        # Prepare next iteration
        x = x_next
        t = t_next
        y = y_next

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
    maxit = parameter['maxit']
    x = parameter['x0']
    y = parameter['x0']
    alpha = 1/parameter['Lips']
    sq_mu = math.sqrt(parameter['strcnvx'])
    sq_l = math.sqrt(parameter['Lips'])
    beta = (sq_l - sq_mu)/(sq_l + sq_mu)


    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):

        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        x_next = y -alpha*gradf(y)

        y = x_next + beta*(x_next - x)

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
    maxit = parameter['maxit']
    x = parameter['x0']
    L = parameter['Lips']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()
        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        #Compute d
        d = -gradf(x)
        L_k0 = 0.5*L
        #Compute i
        i = 0
        term_1 = fx(x + d/(L_k0))
        term_2 = (1/L_k0)*(np.linalg.norm(d))**2
        while term_1 > fx(x) - term_2:
            i += 1
            term_1 = fx(x + d/(2**i * L_k0))
            term_2 = (1/(L_k0*(2**(i+1))))*(np.linalg.norm(d))**2

        L = (2**i)*L_k0
        x_next = x +(1/L)*d

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
    maxit = parameter['maxit']
    x = parameter['x0']
    y = parameter['x0']
    t = 0

    L = parameter['Lips']


    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        d = -gradf(y)
        L_k0 = 0.5 * L
        # Compute i
        i = 0
        term_1 = fx(y + d / (L_k0))
        term_2 = (1 / L_k0) * (np.linalg.norm(d)) ** 2
        while term_1 > fx(y) - term_2:
            i += 1
            term_1 = fx(y + d / (2 ** i * L_k0))
            term_2 = (1 / (L_k0 * (2 ** (i + 1)))) * (np.linalg.norm(d)) ** 2

        L = ((2 ** i) * L_k0)
        x_next = y + (1 / L) * d
        #4L/(L_k-1) == 2^(i+1)
        t_next = 0.5*(1 + np.sqrt(1 + (t**2)*(2**(i+1))))
        y_next = x_next + (t/t_next)*(x_next -x)


        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next
        y = y_next

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
    
    maxit = parameter['maxit']
    x = parameter['x0']
    Alpha = 1/parameter['Lips']
    t = 0
    y = parameter['x0']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        x_next = y - Alpha*gradf(y)
        # MAL --> NO ES EL GRADIENTE! ES LA FUNCION EN SI!
        if np.linalg.norm(gradf(x)) < np.linalg.norm(gradf(x_next)):
            y_next = x
            t_next = 1
            x_next = y - Alpha * gradf(y)
        else:
            t_next = 0.5*(1 + np.sqrt(1 +4*(t**2)))
            y_next = x_next + ((t -1)/t_next)*(x_next - x)



        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next
        y = y_next

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
    ###YOU ARE MISSING SOMETHING! WHAT'S THE INITIAL FUNCTION VALUE???
    maxit = parameter['maxit']
    x = parameter['x0']
    y = parameter['x0']
    t = 0
    L = parameter['Lips']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.


        d = -gradf(y)
        L_k0 = 0.5 * L

        # Compute i
        i = 0
        term_1 = fx(y + d / (L_k0))
        term_2 = (1 / L_k0) * (np.linalg.norm(d)) ** 2
        while term_1 > fx(y) - term_2:
            i += 1
            term_1 = fx(y + d / (2 ** i * L_k0))
            term_2 = (1 / (L_k0 * (2 ** (i + 1)))) * (np.linalg.norm(d)) ** 2

        #L = ((2 ** i) * L_k0)
        x_next = y + (1 / ((2 ** i) * L_k0)) * d
        if np.linalg.norm(gradf(x)) < np.linalg.norm(gradf(x_next)):
            y_next = x
            t_next = 1
            x_next = x
            iter -= 1
        else:
            L = ((2 ** i) * L_k0)

        # 4L/(L_k-1) == 2^(i+1)
            t_next = 0.5 * (1 + np.sqrt(1 + (t ** 2) * (2 ** (i + 1))))
            y_next = x_next + (t / t_next) * (x_next - x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next
        y = y_next

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
    maxit = parameter['maxit']
    x = parameter['x0']
    B = np.eye(len(x))
    Alpha = 10
    k = 0.1

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.

        d = -1*np.dot(B, gradf(x))


        i = 0
        term_2 = np.dot(gradf(x).T, d)

        while fx(x + 64 * Alpha / (2 ** i) * d) > (fx(x) + k * (64 * Alpha / (2 ** (i +1))) * term_2):
            i += 1

        #Compute new step size and new point
        Alpha = 64 * Alpha / (2 ** i)
        x_next = x + Alpha * d

        #Update Hessian
        v = gradf(x_next) - gradf(x)
        s = x_next - x
        B_aux = np.dot(B, v)



        B = B - np.dot(B_aux, B_aux.T)/np.dot(v.T, B_aux) + np.dot(s, s.T)/np.dot(s.T, v)


        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))
            print('Update on x:', np.dot(s.T, s))

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
    print('Newton Method')

    # Initialize x, alpha (and any other)
    maxit = parameter['maxit']
    x = parameter['x0']
    Alpha = 10
    k = 0.1

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}


    # Main loop.
    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #Compute d
        d = spla.spsolve(hessf(x), -gradf(x))

        i = 0
        term_2 = np.dot(gradf(x).T, d)

        while fx(x + 64*Alpha/(2**i)*d) > (fx(x) + k*(64*Alpha/(2**(i+1)))*term_2):
            i += 1

        Alpha = 64*Alpha/(2**i)
        x_next = x + Alpha*d

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



