
* -*- mode:spice -*-
.model BC546B npn ( IS=7.59E-15 VAF=73.4 BF=480 IKF=0.0962 NE=1.2665
+ ISE=3.278E-15 IKR=0.03 ISC=2.00E-13 NC=1.2 NR=1 BR=5 RC=0.25 CJC=6.33E-12
+ FC=0.5 MJC=0.33 VJC=0.65 CJE=1.25E-11 MJE=0.55 VJE=0.65 TF=4.26E-10
+ ITF=0.6 VTF=3 XTF=20 RB=100 IRB=0.0001 RBM=10 RE=0.5 TR=1.50E-07)


* 1e-12 25e-3 100 10
.model TRAV npn(IS=1e-12 bf=100 br=10)

               
                                 
*   net.addD("d1", 1e-8, 25e-3, "v", "da")
.model DRAV D(IS=1e-8)                       
                        
