import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ode 1. Euler's method
def euler(f, t0, tn, n, y0):
    h = abs(tn-t0)/n
    t = np.linspace(t0, tn, n + 1) #Linespace will help us to make tSol [the array with all the intervals]
    y = np.zeros(n + 1) #*remember that tSol and ySol need to have the same size
    y[0] = y0
    for k in range(0, n):
        y[k + 1] = y[k] + h*f(t[k], y[k])
    return y 


# ode 2. Runge-Kutta's method
def RK4(f, t0, tn, n, y0):
    h = abs(tn-t0)/n
    t = np.linspace(t0, tn, n + 1) #Linespace will help us to make tSol [the array with all the intervals]
    y = np.zeros(n + 1) #*remember that tSol and ySol need to have the same size
    y[0] = y0
    for k in range(0, n):
        s1 = f(t[k], y[k])
        s2 = f(t[k] + (h/2), y[k]+s1*(h/2))
        s3 = f(t[k] + (h/2), y[k]+s2*(h/2))
        s4 = f(t[k] + h, y[k]+s3*h)
        y[k+1] = y[k] + (h/6)*(s1+(2*s2)+(2*s3)+s4)#### Método de RK4
    return y

def plot_results(x, y_exact, ye, yrk, fig_title):
    fig = plt.figure()
    fig.suptitle(fig_title, fontsize=12)
    plt.plot(x, ye, 'bo', label='Euler')
    plt.plot(x, yrk, 'g-o', label='Runge-Kutta')
    plt.plot(x, y_exact, 'r', label='Sol. analítica')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    return fig;

def plot_resultsLabels(x, y_exact, ye, yrk, fig_title, label_x, label_y):
    fig = plt.figure()
    fig.suptitle(fig_title, fontsize=12)
    plt.plot(x, ye, 'bo', label=f'Euler: {round(ye[-1],2)}')
    plt.plot(x, yrk, 'g-o', label=f'Runge-Kutta: {round(yrk[-1],2)}')
    plt.plot(x, y_exact, 'r', label=f'Sol. analítica: {round(y_exact[-1],2)}')
    plt.grid()
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend()
    return fig;

def plot_resultsLast(x_good, y_exact, x, ye, yrk, fig_title, label_x, label_y):
    fig = plt.figure()
    fig.suptitle(fig_title, fontsize=12)
    plt.plot(x, ye, 'bo', label=f'Euler: {round(ye[-1],2)}')
    plt.plot(x, yrk, 'g-o', label=f'Runge-Kutta: {round(yrk[-1],2)}')
    plt.plot(x_good, y_exact, 'r', label=f'Sol. analítica: {round(y_exact[-1],2)}')
    plt.grid()
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend()
    return fig;

def plot_2resultsLabels(x, ye, yrk, fig_title, label_x, label_y):
    fig = plt.figure()
    fig.suptitle(fig_title, fontsize=12)
    plt.plot(x, ye, 'bo', label=f'Euler: {round(ye[-1],2)}')
    plt.plot(x, yrk, 'g-o', label=f'Runge-Kutta: {round(yrk[-1],2)}')
    plt.grid()
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend()
    return fig;


def multipage(filename, figs=None, dpi=200):
    
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


### Función tomada del código del doc Martín para graficar una curva
def XYplot(x,y, xmin, xmax, xlab, ylab, fig_title, symbol_color, scale=True): 
    
    #        import numpy as np

    fig = plt.figure()  
    fig.suptitle(fig_title, fontsize=12)
    plt.plot(x, y,  symbol_color)
    plt.xlim(xmin, xmax)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(True)
    if scale:
        plt.xscale('log')
    

    return fig

