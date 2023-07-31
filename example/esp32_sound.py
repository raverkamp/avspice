import matplotlib as mp
import matplotlib.pyplot as plt
from avspice import *
import argparse
import sys
from avspice.util import *

tchip = NPNTransistor("", 1e-12, 25e-3, 100, 10)

tnpn = NPNTransistor("", 1e-12, 25e-3, 100, 10)
tpnp = PNPTransistor("", 1e-12, 25e-3, 100, 10)



def create_speaker(ohm, indu):
    """  create a a  loud speaker """
    sc =  SubCircuit(("i1", "i2"))
    sc.addR("r", ohm, "i1","x")
    sc.addInduc("l", indu,"x","i2")
    return sc


def mk_circuit(pullup=1e3):
    v5 =  5
    v33 = 3.3
    net = Circuit()
    net.addV("VC5",v5,"V5","0")
    net.addV("VC3",v33,"V33","0")
    # open collector output
    net.add_component("VPWM", PwmVoltage("A",3.3, 440,0.5), ("RSIG","0"))
    net.addR("RSIG", 10e3,"RSIG", "VSIG")
    net.add_component("TCHIP", tchip, ("VSIG","SIG","00"))
    
    net.addR("R0",5, "00", "0")

    net.addR("Pullup", pullup, "V33", "SIG")

    speaker = create_speaker(7,0.05e-3)
    net.add_component("T",tnpn,("B","C","000"))
    net.addR("R00",5, "000", "0")
    net.addR("RB",1e3,"SIG", "B")
    net.addR("C",10,"V5","X")
    net.add("speaker",speaker,("X", "C"))
    return net

def sound1(args):
    net = mk_circuit(pullup=args.pullup)
    ana = Analysis(net)
    swings = args.swings
    freq =  440.0

    (fig, ((p11, p12, p13),(p21,p22, p23), (p31,p32,p33), (p41,p42,p43))) = plt.subplots(4,3)
    res = ana.transient(1.0/freq * swings, 1e-5/freq, induc_currents={"speaker/l":0.3},capa_voltages={"capa":3})
    time = res.get_time()

    p11.set_title("volt TCHIP.B")
    p11.plot(time, res.get_voltage("TCHIP.B"))
    p12.set_title("volt TCHIP.C")
    p12.plot(time, res.get_voltage("TCHIP.C"))
    p13.set_title("curr TCHIP.B")
    p13.plot(time, res.get_current("TCHIP.B"))

    p21.set_title("Voltage T.B")
    p21.plot(time, res.get_voltage("T.B"))
    p22.set_title("Curr T.B")
    p22.plot(time, res.get_current("T.B"))

    p23.set_title("curr T.C")
    p23.plot(time, res.get_current("T.C"))

    p31.set_title("voltage TCHIP.E")
    p31.plot(time, res.get_voltage("TCHIP.E"))

    p32.set_title("curr TCHIP.C")
    p32.plot(time, res.get_current("TCHIP.C"))

    p33.set_title("curr speaker.i1")
    p33.plot(time, res.get_current("speaker/r.p"))

    p41.set_title("v(t1.c, t1.e)")
    p41.plot(time, res.get_voltage2("T.C", "T.E"))

    
    plt.show()



    
def mk_circuit2(pullup=1e3):
    v5 =  5
    v33 = 5
    net = Circuit()
    net.addV("VC5",v5,"V5","0")
    net.addV("VC3",v33,"V33","0")
    # open collector output
    net.add_component("VPWM", PwmVoltage("A",3.3, 440,0.5), ("RSIG","0"))
    net.addR("RSIG", 10e3,"RSIG", "VSIG")
    net.add_component("TCHIP", tchip, ("VSIG","SIG","00"))
    
    net.addR("RECHIP",5, "00", "0")

    net.addR("Pullup", pullup, "V33", "SIG")

    speaker = create_speaker(7,0.05e-3)
    net.add_component("T1",tnpn,("B","C1","X"))
    net.add_component("T2",tpnp,("B","X","000"))
    
    net.addR("R00",5, "000", "0")
    net.addR("RB",1e3,"SIG", "B")
    net.addR("C",10,"V5","C1")
    net.addCapa("capa",10e-6,"X","SIN")
    #net.addR("rcapa",1,"X","SIN")
    net.add("speaker",speaker,("SIN", "0"))
    return net
    
    
def sound2(args):
    net = mk_circuit2(pullup=args.pullup)
    ana = Analysis(net)
    swings = args.swings
    freq =  440.0

    (fig, ((p11, p12, p13),(p21,p22, p23), (p31,p32,p33))) = plt.subplots(3,3)
    res = ana.transient(1.0/freq * swings, 1e-5/freq, induc_currents={"speaker/l":0.3},capa_voltages={"capa":3})
    time = res.get_time()

    p11.set_title("volt TCHIP.B")
    p11.plot(time, res.get_voltage("TCHIP.B"))
    p12.set_title("volt TCHIP.C")
    p12.plot(time, res.get_voltage("TCHIP.C"))
    p13.set_title("curr TCHIP.B")
    p13.plot(time, res.get_current("TCHIP.B"))

    p21.set_title("Voltage X")
    p21.plot(time, res.get_voltage("X"))

    p33.set_title("curr speaker.i1")
    p33.plot(time, res.get_current("speaker/r.p"))
    
    plt.show()

def sound2_initial(args):
    net = mk_circuit2(pullup=args.pullup)
    ana = Analysis(net)
    swings = args.swings
    freq =  440.0
    
    for i in range(100):
        res = ana.analyze(capa_voltages={"xcapa":3})
        if isinstance(res,str):
            print(res)
        else:
            res.display()
            break


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    parser_s1 = subparsers.add_parser('s1')
    parser_s1.set_defaults(func=sound1)
    parser_s1.add_argument("-pullup",type=float, default=1e3)
    parser_s1.add_argument("-swings",type=float, default=2)

    parser_s2 = subparsers.add_parser('s2')
    parser_s2.set_defaults(func=sound2)
    parser_s2.add_argument("-pullup",type=float, default=1e3)
    parser_s2.add_argument("-swings",type=float, default=2)

    parser_s2i = subparsers.add_parser('s2i')
    parser_s2i.set_defaults(func=sound2_initial)
    parser_s2i.add_argument("-pullup",type=float, default=1e3)
    parser_s2i.add_argument("-swings",type=float, default=2)
    
    
    args = parser.parse_args()
    args.func(args)
    return 0

sys.exit(main())
    
