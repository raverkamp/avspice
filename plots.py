import matplotlib as mp
import matplotlib.pyplot as plt
from spice import *


def drange(start, end, step=None):
    x = float(start)
    if step is None:
        s = 1.0
    else:
        s = step
    s = float(s)
    if s <=0:
        raise Exception("step <=0")        
    while x < end:
        yield x
        x += s
    if x < end + s/2.0:
        yield end
    

def main():
    t = NPNTransistor(None, "", 1e-12, 25e-3, 100, 10)
    #(2, <XNode T1.B/rb.n: {<Port rb.n>, <Port T1.B>}>, 0.49979218996132607)
    #(3, <XNode T1.E/re.p: {<Port re.p>, <Port T1.E>}>, 0.020988815016071836)
    #(4, <XNode T1.C/v1.p: {<Port T1.C>, <Port v1.p>}>, 2.0)    
    vb = 0.5
    ve = 0.02
    x = list(drange(-3, 3, 0.1))
    ie = [t.IE(vb-ve, vb-vc) for vc in x]
    ib = [t.IB(vb-ve, vb-vc) for vc in x]

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    plt.ion()
    plt.plot(x,ie)
    plt.plot(x,ib)
    plt.show()


    
    vc = 2
    ve = 0
    x = list(drange(-0.5,3.5, 0.01)    )
    ie = [t.IE(vb-ve, vb-vc) for vb in x]
    ib = [t.IB(vb-ve, vb-vc) for vb in x]
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    plt.plot(x,ie, color="black")
    plt.plot(x,ib, color="green")
    plt.show()

main()

input()
