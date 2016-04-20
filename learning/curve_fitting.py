from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pdb
import numpy as np


def linear_hip(x, a, b):
    return a * x + b


def exp_hip(x, a, b, c):
    return a * np.exp(b * x) + c


def quadratic_hip(x, a, b):
    return a * np.power(x, 2) + b



# For beta = 0.5
x = [2, 3, 10, 100]
x = np.asarray(x)
y = [650, 738, 3185, 6205]
y = np.asarray(y)

popt_lin, pcov_lin = curve_fit(linear_hip, x, y)
#popt_exp, pcov_exp = curve_fit(exp_hip, x, y)
popt_quad, pcov_quad = curve_fit(quadratic_hip, x, y)

print "Linear Reg:"
print "\tPOpt:", popt_lin
print "\tPCov:", pcov_lin, "\n"
#print "Exp Reg:"
#print "\tPOpt:", popt_exp
#print "\tPCov:", pcov_exp, "\n"
print "Quad Reg:"
print "\tPOpt:", popt_quad
print "\tPCov:", pcov_quad, "\n"


x_ = xrange(0,100)

y_lin = []
y_quad = []

for i in x_:
    y_lin.append(linear_hip(i, popt_lin[0], popt_lin[1]))

for i in x_:
    y_quad.append(quadratic_hip(i, popt_quad[0], popt_quad[1]))

fq = plt.figure()

plt.plot(x_, y_lin, color='r')
plt.plot(x_, y_quad, color='y')

plt.title('Function')
plt.ylabel('Iterations to Opt. Pol.')
plt.xlabel('Number of Patterns')

plt.scatter(x, y, color='b')

fq.show()

pdb.set_trace()