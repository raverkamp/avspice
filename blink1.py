import matplotlib as mp
import matplotlib.pyplot as plt
from spice import *
import argparse
import sys
from util import *

from blinker_circuit import create_blinker

    
def blinker(args):
    net = create_blinker(transistor_gain=100, r_ca=0.01)
    ana = Analysis(net)

    base_vca = args.vca   #0.397 #-5 # -8.25


    sol = None
    res = ana.analyze(maxit=50, start_solution_vec=sol, capa_voltages={"ca": base_vca},
                      variables={"vc": 9},
                       start_voltages= {
                          "t1.C": args.t1c,
                           "t1.B": args.t1b
                       })
    if isinstance(res, str):
        raise Exception("can not find inital solution")
    sol = res.solution_vec
    print("---------------------------------------------------")
    ch = 0

    xs = []
    ca_v = []
  
    t1b_v = []
    t2c = []
    t1c = []
    t1b = []
    t2b = []
    
    it1b = []
    it1c = []
    it2c = []
    it2b = []
    iC =[]
    itd  = []
    iters = []
    cond = []
    s = 1e-4 #0.001/1
    x = 0
    v_ca = base_vca
    switched = False
    ca_cu = 0
    printstep = 0.1
    i = 0
    while x < 4.5:
        if x >= i*printstep:
            sys.stderr.write("time={0}\n".format(x))
            i = i+1
        ok = False
        maxit = 50
        capa = net.get_object("ca").capa
        vca_new = v_ca + s*ca_cu/capa
        v_ca = vca_new
        res = ana.analyze(maxit=maxit, start_solution_vec=sol, capa_voltages={"ca": v_ca },
                              variables={"vc": 9})
        if isinstance(res,str):
            if s < 0.001 * 1e-4:
                print(x, res)
                break
            s = s/2
            print("############### step ", x, s, v_ca)
            continue
        ca_cu = res.get_current("ca.p")

#        capa = net.get_object("ca").capa
#        vca_new = v_ca + s*ca_cu/capa
#        print(("OK", x, ca_cu,v_ca, vca_new))
#        v_ca = vca_new
        xs.append(x)

        ca_v.append(v_ca)
        iC.append(res.get_current("ca.p"))
        t1b_v.append(res.get_voltage("t1.B"))
        t2c.append(res.get_voltage("t2.C"))
        t1c.append(res.get_voltage("t1.C"))
        t1b.append(res.get_voltage("t1.B"))
        t2b.append(res.get_voltage("t2.B"))
        it2c.append(res.get_current("t2.C"))
        it1c.append(res.get_current("t1.C"))
        it2b.append(res.get_current("t2.B"))
        it1b.append(res.get_current("t1.B"))
        cond.append(math.log10(res.mat_cond))
        iters.append(res.iterations)
        itd.append(res.get_current("d1.p"))
        #   print((x, res.iterations, ch, ica))
        sol = res.solution_vec

        x += s
    fig, ((a1,a2, a3), (a4, a5, a6), (a7,a8,a9)) = plt.subplots(3,3)

    a1.plot(xs, it1c, label="it1c")
    a1.legend()
    a2.plot(xs, it1b, label="it1b")
    a2.legend()
    """
    a2.set_title("capa voltage=blue and t1.B voltage=red")
    a2.plot(xs, ca_v, label="capa voltage")
    a2.plot(xs, t1b_v, label="t1b voltage")
    a2.legend()
    """
    a3.plot(xs, t2c, label="t2c")
    a3.plot(xs, t1b, label="t1b")
    a3.plot(xs, t2b, label="t2b,t1c")
    a3.set_title("Voltages")
    a3.legend()
    
    a4.plot(xs, it2c)
    a4.set_title("it2c")

    a5.plot(xs, it2b)
    a5.set_title("it2b")

    a6.set_title("iterations")
    a6.plot(xs, iters)

    a7.set_title("CA current")
    a7.plot(xs,iC, label="CA current")
    
    a7.legend()

    a8.plot(xs, ca_v, label="capa voltage")
    a8.legend()

    a9.set_title("log(cond)")
    a9.plot(xs, cond)
    a9.legend()
    
    plt.ion()
    plt.show()
    input()

def catest(args):
    net = Network()
    vc = net.addV("vc",  9) 
    r1 = net.addR("r1", 1e3)
    c = net.addCapa("capa", 1e-6)

    connect(vc.p,r1.p)
    connect(r1.n, c.p)
    connect(c.n,vc.n)
    connect(vc.n, net.ground)

    ana = Analysis(net)
    base_vca = 0
    v_ca = base_vca
    ca_cu = 0
    sol = None
    res = ana.analyze(maxit=50, start_solution_vec=sol, capa_voltages={"capa": base_vca},
                  variables={"vc": 9})
    if isinstance(res, str):
        raise Exception("can not find inital solution for time sweep")
    sol = res.solution_vec

    x = 0
    s = 0.0001
    ca_v = []
    ca_i = []
    xs =[]
    while x < 0.01:
        print("///", x)
        ok = False
        maxit = 50
        capa = net.get_object("capa").capa
        vca_new = v_ca + s*ca_cu/capa
        v_ca = vca_new
        res = ana.analyze(maxit=maxit, start_solution_vec=sol, capa_voltages={"capa": v_ca })
        if isinstance(res,str):
            break
        ca_cu = res.get_current("capa.p")
        xs.append(x)
        x = x + s
        ca_v.append(v_ca)
        ca_i.append(ca_cu)

    fig, ((a1,a2, a3)) = plt.subplots(1,3)
    
    a1.plot(xs,ca_v, label="v(ca)")
    a1.legend()
    a2.plot(xs, ca_i, label="i(ca)")
    a2.legend()
    
    plt.ion()
    plt.show()
    input()

def blinker_static(args):
    net = create_blinker()
    ana = Analysis(net)

    base_vca = 0.0   #0.397 #-5 # -8.25


    sol = None
    res = ana.analyze(maxit=50, start_solution_vec=sol, capa_voltages={"ca": base_vca},
                      variables={"vc": 9},
                      start_voltages= {
                          "t1.C": 4,
                          "t2.C": 0.1,
                          "t1.B": 0.1})
    print(res)
    vn = res.get_voltage("ca.n")
    vp = res.get_voltage("ca.p")
    print("voltage ca ={0}".format(vp-vn))
    
    for s in ["t1.C", "t1.B", "t2.C", "t2.B"]:
            print("{0} V = {1}   I={2}".format(s, res.get_voltage(s), res.get_current(s)))



def try_random_start(args):
    import random
    net = create_blinker(r_ca=1e-1)
    ana = Analysis(net)
    base_vca = 0.0
    res = ana.analyze(maxit=50, start_solution_vec=None, capa_voltages={"ca": base_vca},
                      variables={"vc": 9},
                      start_voltages= {
                          "t1.C": args.t1c,
                          #"t2.C": random.random() * 9,
                          "t1.B": args.t1b
                      })
    if isinstance(res, str):
        print("fail")
    else:
        for s in ["t1.C", "t1.B", "t2.C", "t2.B"]:
            print("{0} V = {1}   I={2}".format(s, res.get_voltage(s), res.get_current(s)))

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    
    parser_b = subparsers.add_parser('b')
    parser_b.set_defaults(func=blinker)
    parser_b.add_argument('-t1c', type=float, default=0)
    parser_b.add_argument('-t1b', type=float, default=0)
    parser_b.add_argument('-vca', type=float, default=0)

    parser_c = subparsers.add_parser('c')
    parser_c.set_defaults(func=catest)

    parser_s = subparsers.add_parser('s')
    parser_s.set_defaults(func=blinker_static)

    parser_sol = subparsers.add_parser('sol')
    parser_sol.set_defaults(func=try_random_start)
    parser_sol.add_argument('-t1c', type=float, default=0)
    parser_sol.add_argument('-t1b', type=float, default=0)
    
    
    args = parser.parse_args()
    args.func(args)

sys.exit(main())

