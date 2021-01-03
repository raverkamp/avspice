
import pprint as pp
import numpy as np

class Component(object):

    def __init__(self, name):
        self.name = name

    def ports(self):
        raise NotImplementedError("ports method not implemented")

class Port(object):

    def __init__(self, component, name):
        self.component = component
        self.name = name

    def __repr__(self):
        return "<Port {0}.{1}>".format(self.component.name, self.name)

    def pname(self):
        return self.component.name + "." + self.name

        
class Node2(Component):

    def __init__(self, parent, name):
        self.name = name
        self.parent = parent
        self.p1 = Port(self, "p1")
        self.p2 = Port(self, "p2")

    def ports(self):
        return [self.p1, self.p2]

class Node(Component):
    def __init__(self, parent, name):
        self.name = name
        self.parent = parent
        self.port = Port(self, "port")

    def ports(self):
        return [self.port]

    def __repr__(self):
        return "<Node {0}>".format(self.name)


class Resistor(Node2):

    def __init__(self, parent, name, ohm):
        super().__init__(parent, name)
        self.ohm = ohm

class Current(Node2):
    def __init__(self, parent, name, amp):
        super().__init__(parent, name)
        self.amp = amp

 # only the topology
class Network(object):

    def __init__(self):
      
        self.connections = []
        self.ground = Node(self, "ground")
        self.components = { self.ground.name : self.ground}

    def addR(self, name, ohm):
        if name in self.components:
            raise Exception("Name {0} already exists".format(name))
        r = Resistor(self, name , ohm)
        self.components[name] = r
        return r

    def addC(self, name, amp):
        if name in self.components:
            raise Exception("Name {0} already exists".format(name))
        c = Current(self, name , amp)
        self.components[name] = c
        return c

    def addN(self,name):
        if name in self.components:
            raise Exception("Name {0} already exists".format(name))
        node = Node(self, name)
        self.components[name] = node
        return node

    def addConnection(self, p1, p2):
        if isinstance(p2, Node):
            p2 = p2.port
        if isinstance(p1, Node):
            p1 = p1.port
    
        c1 = p1.component
        c2 = p2.component
        if c1.parent != self or c2.parent != self:
            raise Exception("wrong network for addConnection")
        
        self.connections.append((p1, p2))


def connect(p1,p2):
    if isinstance(p1, Node):
        compo = p1
    else:
        compo = p1.component        
    compo.parent.addConnection(p1,p2)
 
class XNode():
    def __init__(self, name, ports):
        self.ports = ports
        self.name = name

    def __repr__(self):
        return "<XNode {0}: {1}>".format(self.name, repr(self.ports))

def mk_xnode(nodes):
    def key(x):
        if isinstance(x.component, Node):
            return "a" +  x.component.name
        else:
            return "b" + x.component.name

    l = list(nodes)
    l.sort(key=key)
    if isinstance(l[0].component, Node):
        s = None
        for x in l:
            if not isinstance(x.component, Node):
                break
            if s:
                s= s+ "/" + x.component.name
            else:
                s= x.component.name
    else:
        s = None
        for p in l:
            
            if s:
                s= s+ "/" + p.pname()
            else:
                s= p.pname()
    return XNode(s, nodes)

                            
def compute_nodes(nw):
    ## adjancency
    adj = dict()
    def add(p1 ,p2):
        if p1 in adj:
            l = adj[p1]
            l.append(p2)
        else:
            adj[p1] = [p2]
    
    for (p1, p2) in nw.connections:
        add(p1, p2)
        add(p2, p1)

    done = set()
    nodes = []
    todo = [nw.ground.port]
    ground_node = None
    while todo:
        pp.pprint(("todo", todo))
        port = todo.pop()
        if port in done:
            continue
        
        stack = [port]
        node = set()
        done.add(port)
        
        while stack:
            x = stack.pop()
            for p in x.component.ports():
                if not p in done:
                    todo.append(p)
            node.add(x)
            for p1 in adj[x]:
                if p1 in done:
                    continue
                stack.append(p1)
                done.add(p1)
        # per algorithm ground is the first node
        nodes.append(mk_xnode(node))
        
    return nodes

        
    
def analyze(net):
    nodes = compute_nodes(net)
    ground = nodes[0]

    nodes.pop(0)
    
    n = len(nodes)


    def p_2_n_index(port):
        for i in range(n):
            if port in nodes[i].ports:
                return i
        raise Exception("BUG")
    
    mat = []
    
    for _ in range(n):
        mat.append([0]*n)
    v = [0]*n

    mat = np.zeros((n,n))
    v = np.zeros(n)
    
    row = 0
    for i in range(n):
        node = nodes[i]
        I = 0
        GG = 0
        for port in node.ports:
            comp = port.component
            if isinstance(comp, Current):
                if comp.p1 == port:
                    I = I - comp.amp
                else:
                    I = I + comp.amp
            if isinstance(comp, Resistor):
                o = comp.ohm
                GG = GG + 1/o
                if port == comp.p1:
                    op = comp.p2
                else:
                    op = comp.p1
                if not op in ground.ports:
                    oi = p_2_n_index(op)
                    mat[i][oi] = - 1/o
            mat[i][i] = GG
            v[i] = I 
    pp.pprint(mat)
    pp.pprint(v)
    voltages = np.linalg.solve(mat, v)
    pp.pprint(voltages)
    
            
        

    
    
