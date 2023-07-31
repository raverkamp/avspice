
* -*- mode: text -*-
.model BC546B npn ( IS=7.59E-15 VAF=73.4 BF=480 IKF=0.0962 NE=1.2665
+ ISE=3.278E-15 IKR=0.03 ISC=2.00E-13 NC=1.2 NR=1 BR=5 RC=0.25 CJC=6.33E-12
+ FC=0.5 MJC=0.33 VJC=0.65 CJE=1.25E-11 MJE=0.55 VJE=0.65 TF=4.26E-10
+ ITF=0.6 VTF=3 XTF=20 RB=100 IRB=0.0001 RBM=10 RE=0.5 TR=1.50E-07)


* 1e-12 25e-3 100 10
.model TRAV npn(IS=1e-12 bf=100 br=10)

               
                                 
*   net.addD("d1", 1e-8, 25e-3, "v", "da")
.model DRAV D(IS=1e-8)                       
                        


.model T2N5551  NPN (Is=2.511f Xti=3 Eg=1.11 Vaf=100 Bf=242.6 Ne=1.249 Ise=2.511f Ikf=.3458 Xtb=1.5 Br=3.197 Nc=2 Isc=0
+ Ikr=0 Rc=1 Cjc=4.883p Mjc=.3047 Vjc=.75 Fc=.5 Cje=18.79p Mje=.3416 Vje=.75 Tr=1.202n Tf=560p Itf=50m
+ Vtf=5 Xtf=8 Rb=10)



       *SRC=2N5401;2N5401;BJTs PNP; Si;  160.0V  0.60A  300MHz   Central Semi Central Semi
.MODEL 2N5401  PNP (
+ IS=20.743E-15
+ BF=516.02
+ VAF=100
+ IKF=.23172
+ ISE=30.077E-15
+ NE=1.2905
+ BR=.1001
+ VAR=100
+ IKR=2.7895
+ ISC=4.1177E-12
+ NC=2.0459
+ NK=.80454
+ RB=1.0110
+ CJE=50.447E-12
+ VJE=.70813
+ MJE=.32821
+ CJC=9.0325E-12
+ VJC=.41518
+ MJC=.33181
+ TF=48.913E-12
+ XTF=57.737
+ VTF=18.593
+ ITF=23.057E-3
+ TR=10.000E-9)
******
