'''Edits'''
from multiprocessing import Pool
import random as rand
from random import randint
import numpy as np
import matplotlib.pyplot as plt


def Ising_Energy(JT):
        N = 16
        s = [[-1 for a in range(N)] for a in range(N)]
        Groups = 2**3

        steps = 2**23
#        Groups = 2**5
        Measurements = 2**6
        C_Time = 2**14
        #print s
        Z = 0.0   #Number of steps
        Ising_Energy = -2*N**2
        EI = 0.
        E_I = [0. for i in range(Measurements)]
        for i in range(steps):
            dE1 = 0.0
            k1 = rand.randint(0 , N-1)
            k2 = rand.randint(0 , N-1)

            for n in range(Measurements):         #Take Measurements
                if i == ((n+1)*C_Time):
                    Z += 1
                    EI += Ising_Energy
                    E_I[n] = Ising_Energy


            I = k1-1
            if I >= 0:
                dE1 += s[I][k2]
            else:
                dE1 += s[N-1][k2]

            I = k1 + 1
            if I <= N-1:
                dE1 += s[I][k2]
            else:
                dE1 += s[0][k2]

            I = k2 + 1
            if I <= N-1:
                dE1 += s[k1][I]
            else:
                dE1 += s[k1][0]

            I = k2 - 1
            if I >= 0:
                dE1 += s[k1][I]
            else:
                dE1 += s[k1][N-1]

            dE_I = 2*JT*dE1*s[k1][k2]


            R = np.exp(-dE_I)           ##Acceptance Ratio


            '''Whether or not to accept the state and change Magnetizations and Energy'''
            if (R >= 1) or (rand.random() < R):
                Ising_Energy += dE_I/JT
                s[k1][k2] = (-1)*s[k1][k2]
        '''Calculate Jackknife error'''
        B_Block2 = [0.0 for i in range(Groups)]   # group averaging of Standard Energies
        for a in range(Groups):                 #G is number of groups
            for c in range(Measurements/Groups):       #steps/G is number of elements per group
                B_Block2[a] += E_I[(a*Measurements/Groups) + c]/(Measurements/Groups)
        B_Block_Div2 = [0.0 for a in range(Groups)]
        st_dev2 = [0.0 for a in range(Groups)]
        st_devR = 0.0
        st_devS = 0.0
        jack_knifeS = 0.0
        for i in range(Groups):              ## implementing jackknife method
            for a in range(Groups):
                if a <> i:
                    B_Block_Div2[i] += B_Block2[a]
                else:
                    B_Block_Div2[i] += 0
        for i in range(Groups):
            B_Block_Div2[i] = B_Block_Div2[i]/(Groups-1)
        for a in range(Groups):
            st_dev2[a] =(float(B_Block_Div2[a]-EI/Z))**2
        for a in range(Groups):
            jack_knifeS += st_dev2[a]
        jack_knife_s = np.sqrt(jack_knifeS)/Groups**2



        '''bootstrap = 0.0
        B_Block2 = [0.0 for i in range(Groups)]
        for a in range(Groups):
            for b in range(Measurements/Groups):
                B_Block2[a] += M1_est[randint(0,Measurements-1)]
        for a in range(Groups):
            B_Block2[a] = B_Block2[a]/(Measurements/Groups)
        print B_Block2
        for a in range(Groups):
            bootstrap += (B_Block2[a]-A/Z)**2
        bootstrap = np.sqrt(bootstrap/Groups)'''


        Avg_S_Energy = EI/Z
#        print(E_Replica)
#        print(E_s)
#        print("Replica")
#        print(Avg_Replica_Energy)
#        print("Avg_S_Energy")
        return Avg_S_Energy, jack_knife_s, 1/JT

def A_Energy(JT):
    N = 20
    s = [[-1 for a in range(N)] for a in range(N)]
    s2 = [[-1 for a in range(N)] for a in range(N)]

    Groups = 2**3
    steps = 2**23
    Measurements = 2**6
    C_Time = 2**14

    Energy_A = -4*N**2
    EA = 0.
    Z = 0.

    for i in range(steps):
        dE_A1 = 0.
        dE_A2 = 0.
        dEA = 0.

        k1 = rand.randint(0,N-1)
        k2 = rand.randint(0,N-1)

        for n in range(Measurements):         #Take Measurements
            if i == ((n+1)*C_Time):
                Z += 1
                EA += Energy_A

        I = k1 + 1
        if I <= N-1:
            dE_A1 += s[I][k2]
            dE_A2 += s2[I][k2]
        else:
            dE_A1 += s[0][k2]
            dE_A2 += s2[0][k2]

        I = k1 - 1
        if I >= 0:
            dE_A1 += s[I][k2]
            dE_A2 += s2[I][k2]
        else:
            dE_A1 += s[N-1][k2]
            dE_A2 += s2[N-1][k2]

        I = k2 + 1
        if I <= N-1:
            dE_A1 += s[k1][I]
            dE_A2 += s2[k1][I]
        else:
            dE_A1 += s[k1][0]
            dE_A2 += s2[k1][0]

        I = k2-1
        if I >= 0:
            dE_A1 += s[k1][I]
            dE_A2 += s2[k1][I]
        else:
            dE_A1 += s[k1][N-1]
            dE_A2 += s2[k1][N-1]
        dEA += 2*JT*dE_A1*s[k1][k2]
        dEA += 2*JT*dE_A2*s2[k1][k2]

        R = np.exp(-dEA)

        if (R >= 1) or (rand.random() < R):
            Energy_A += dEA/JT
            s[k1][k2] = -s[k1][k2]
            s2[k1][k2] = -s2[k1][k2]
    return EA/Z


def Replica_Energy(JT):        #N is Dimension of Matrix
        N = 16
        s = [[-1 for a in range(N)] for a in range(N)]
        Groups = 2**3
        s2 = [[-1 for a in range(N)] for a in range(N)]

        steps = 2**23
#        Groups = 2**5
        Measurements = 2**6
        C_Time = 2**14
        #print s
        M1 = N**2  #Magnetization of Replica 1
        M2 = N**2  #Magnetization of Replica 2
        A1 = 0.0   #Absolute Value of Total Magnetization
        A2 = 0.0
        Z = 0.0   #Number of steps
        M1_est = [0.0 for a in range(Measurements)]  # List of all Magnetizations for first system
        M2_est = [0.0 for a in range(Measurements)]  # List of all Magnetizations for second system
        Replica_Energy = -4*N**2
        Replica_Energy2 = -4*N**2
        Energy_A = -N**2
        Energy_A2 = -N**2
        Energy_s = -2*N**2
        Energy_s2 = -2*N**2
        Ising_Energy = -2*N**2
        E_Replica = [0 for a in range(Measurements)]     # List of all Energies for combined system
        E_s = [0 for a in range(Measurements)]
        E_A = [0 for a in range(Measurements)]
        E_A2 = [0 for a in range(Measurements)]
        ER = 0.0
        ER2 = 0.0
        ES = 0.0
        EA = 0.0
        EA2 = 0.0
        EI = 0.
        E_I = [0. for i in range(Measurements)]
        for i in range(steps):
            dE1 = 0.0
            dE2 = 0.0

            dE_A = 0.
            dE_A2 = 0.

            dEA = 0.
            dEA2 = 0.

            dE_s2 = 0.0
            dE_s = 0.0

            dE1_R = 0.
            dE2_R = 0.

            dER1 = 0.
            dER2 = 0.

            k1 = rand.randint(0 , N-1)
            k2 = rand.randint(0 , N-1)
            s_pend = s[k1][k2]
            s2_pend = s2[k1][k2]
            dE_I = 0.
            dE2_I = 0.
            dE_Ising = 0.
            dE2_Ising = 0.


            for n in range(Measurements):         #Take Measurements
                if i == ((n+1)*C_Time):
                    Z += 1
                    EA += Energy_A
                    EA += Energy_A2
                    ER += Replica_Energy
                    ER2 += Replica_Energy2
                    ES += Energy_s
                    EI += Ising_Energy
                    M1_est[n] = M1
                    M2_est[n] = M2
                    E_s[n] = Energy_s
                    E_I[n] = Ising_Energy
                    E_Replica[n] = Replica_Energy

            if k1 <= N/2-1:


                if rand.random() < 0.5:
                    s_pend = (-1)*s_pend
                if rand.random() < 0.5:
                    s2_pend = (-1)*s2_pend


                I = k1 + 1
                if I <= N/2-1:
                    dE1 += s[I][k2]
                    dE2 += s2[I][k2]

                    dE1_R += s[I][k2]
                    dE2_R += s2[I][k2]

                    dE_I += s[I][k2]
                    dE2_I += s2[I][k2]
                else:
                    dE1 += s[I][k2]
                    dE2 += s2[I][k2]

                    dE1_R += s[I][k2]
                    dE2_R += s2[I][k2]

                    dE_I += s[0][k2]
                    dE2_I += s2[0][k2]

                I = k1 - 1
                if I >= 0:
                    dE1 += s[I][k2]
                    dE2 += s2[I][k2]

                    dE_I += s[I][k2]
                    dE2_I += s2[I][k2]

                    dE1_R += s[I][k2]
                    dE2_R += s2[I][k2]
                else:
                    dE1 += s[N-1][k2]
                    dE2 += s2[N-1][k2]

                    dE1_R += s[N-1][k2]
                    dE2_R += s2[N-1][k2]

                    dE_I += s[N/2-1][k2]
                    dE2_I += s2[N/2-1][k2]

                I = k2 + 1
                if I <= N-1:
                    dE1 += s[k1][I]
                    dE2 += s2[k1][I]

                    dE1_R += s[k1][I]
                    dE2_R += s2[k1][I]

                    dE_I += s[k1][I]
                    dE2_I += s2[k1][I]
                else:
                    dE1 += s[k1][0]
                    dE2 += s2[k1][0]

                    dE1_R += s2[k1][0]
                    dE2_R += s[k1][0]

                    dE_I += s[k1][0]
                    dE2_I += s2[k1][0]

                I = k2 - 1
                if I >= 0:
                    dE1 += s[k1][I]
                    dE2 += s2[k1][I]

                    dE1_R += s[k1][I]
                    dE2_R += s2[k1][I]

                    dE_I += s[k1][I]
                    dE2_I += s2[k1][I]
                else:
                    dE1 += s[k1][N-1]
                    dE2 += s2[k1][N-1]

                    dE1_R += s2[k1][N-1]
                    dE2_R += s[k1][N-1]

                    dE_I += s[k1][N-1]
                    dE2_I += s2[k1][N-1]


                if s[k1][k2] == s_pend:
                    dE_s += 0.
                else:
                    dE_s += -2*JT*dE1*s_pend
                    dE_Ising += -2*JT*dE_I*s_pend
                    dER1 += -2*JT*dE1_R*s_pend

                if s2[k1][k2] == s2_pend:
                    dE_s2 += 0.
                else:
                    dE_s2 += -2*JT*dE2*s2_pend
                    dE2_Ising += -2*JT*dE2_I*s2_pend
                    dER2 += -2*JT*dE2_R*s2_pend

            else:
                s_pend = (-1)*s_pend
                s2_pend = (-1)*s2_pend

                I = k1-1
                if I >= N/2:
                    dE_A += s[I][k2]
                    dE_A2 += s2[I][k2]

                    dE1_R += s[I][k2]
                    dE2_R += s2[I][k2]

                    dE1 += s[I][k2]
                    dE2 += s2[I][k2]
                else:
                    dE_A += s[N-1][k2]
                    dE_A2 += s2[N-1][k2]

                    dE1_R += s[I][k2]
                    dE2_R += s2[I][k2]

                    dE1 += s[I][k2]
                    dE2 += s2[I][k2]

                I = k1 + 1
                if I <= N-1:
                    dE_A += s[I][k2]
                    dE_A2 += s2[I][k2]

                    dE1_R += s[I][k2]
                    dE2_R += s2[I][k2]

                    dE1 += s[I][k2]
                    dE2 += s2[I][k2]
                else:
                    dE_A += s[N/2][k2]
                    dE_A2 += s2[N/2][k2]

                    dE1_R += s[0][k2]
                    dE2_R += s2[0][k2]

                    dE1 += s[0][k2]
                    dE2 += s2[0][k2]

                I = k2 + 1
                if I <= N-1:
                    dE_A += s[k1][I]
                    dE_A2 += s2[k1][I]

                    dE1_R += s[k1][I]
                    dE2_R += s2[k1][I]

                    dE1 += s[k1][I]
                    dE2 += s2[k1][I]
                else:
                    dE_A += s[k1][0]
                    dE_A2 += s2[k1][0]

                    dE1_R += s2[k1][0]
                    dE2_R += s[k1][0]

                    dE1 += s[k1][0]
                    dE2 += s2[k1][0]

                I = k2 - 1
                if I >= 0:
                    dE_A += s[k1][I]
                    dE_A2 += s2[k1][I]

                    dE1_R += s[k1][I]
                    dE2_R += s2[k1][I]

                    dE1 += s[k1][I]
                    dE2 += s2[k1][I]

                else:
                    dE_A += s[k1][N-1]
                    dE_A2 += s2[k1][N-1]

                    dE1_R += s2[k1][N-1]
                    dE2_R += s[k1][N-1]

                    dE1 += s[k1][N-1]
                    dE2 += s2[k1][N-1]


                dE_s += -2*JT*dE1*s_pend
                dE_s2 += -2*JT*dE2*s2_pend
                dEA += -2*JT*dE_A*s_pend
                dEA2 += -2*JT*dE_A2*s2_pend
                dER1 += -2*JT*dE1_R*s_pend
                dER2 += -2*JT*dE2_R*s2_pend


            R = np.exp(-dE_s-dE_s2)           ##Acceptance Ratio


            '''Whether or not to accept the state and change Magnetizations and Energy'''
            if (R >= 1) or (rand.random() < R):
                if k1 > N/2-1:
                    Energy_A += dEA/JT
                    Energy_A2 += dEA2/JT

                    Replica_Energy += dE_s/JT
                    Replica_Energy += dE_s2/JT

                    Replica_Energy2 += dER1/JT
                    Replica_Energy2 += dER2/JT

                    Energy_s += dE_s/JT
                    Energy_s2 += dE_s2/JT
                    s[k1][k2] = s_pend
                    s2[k1][k2] = s2_pend
#                    print "Replica E:"
#                    print Replica_Energy
#                    print "A_Energy"
#                    print Energy_A + Energy_A2
                else:
                    if s[k1][k2] == s_pend:
                        Replica_Energy += 0.
                    else:
                        Energy_s += dE_s/JT
                        Ising_Energy += dE_Ising/JT

                        Replica_Energy += dE_s/JT
                        Replica_Energy2 += dER1/JT

                        s[k1][k2] = s_pend
#                        print "Replica E:"
#                        print Replica_Energy
#                    if (dE_s != dE_I):
 #                       print "dE, dE_I is:"
  #                      print dE_s/JT, dE_I/JT
   #                     print "Energy Difference is:"
    #                    print Replica_Energy-Ising_Energy
     #                   print "Replica:"
      #                  print Replica_Energy
       #                 print "Ising:"
        #                print Ising_Energy
                    if s2[k1][k2] == s2_pend:
                        Replica_Energy += 0.
                    else:
                        Energy_s2 += dE_s2/JT
                        Ising_Energy += dE2_Ising/JT

                        Replica_Energy += dE_s2/JT
                        Replica_Energy2 += dER2/JT

                        s2[k1][k2] = s2_pend
#                        print "Replica E:"
#                        print Replica_Energy
#                    if (dE_s2 != dE2_I):
 #                       print "dE2, dE2_I is:"
  #                      print dE_s2/JT, dE2_I/JT
   #                     print "Energy Difference is:"
    #                    print Replica_Energy-Ising_Energy
     #                   print "Replica:"
      #                  print Replica_Energy
       #                 print "Ising:"
        #                print Ising_Energy

        '''Calculate Jackknife error'''
        B_Block = [0.0 for i in range(Groups)]    # group averaging of Replica Energies
        B_Block2 = [0.0 for i in range(Groups)]   # group averaging of Standard Energies
        for a in range(Groups):                 #G is number of groups
            for c in range(Measurements/Groups):       #steps/G is number of elements per group
                B_Block[a] += E_Replica[(a*Measurements/Groups) + c]/(Measurements/Groups)
                B_Block2[a] += E_s[(a*Measurements/Groups) + c]/(Measurements/Groups)
        B_Block_Div = [0.0 for a in range(Groups)]
        B_Block_Div2 = [0.0 for a in range(Groups)]
        st_dev = [0.0 for a in range(Groups)]
        st_dev2 = [0.0 for a in range(Groups)]
        st_devR = 0.0
        st_devS = 0.0
        jack_knifeR = 0.0
        jack_knifeS = 0.0
        for i in range(Groups):              ## implementing jackknife method
            for a in range(Groups):
                if a <> i:
                    B_Block_Div[i] += B_Block[a]
                    B_Block_Div2[i] += B_Block2[a]
                else:
                    B_Block_Div[i] += 0
                    B_Block_Div2[i] += 0
        for i in range(Groups):
            B_Block_Div[i] = B_Block_Div[i]/(Groups-1)
            B_Block_Div2[i] = B_Block_Div2[i]/(Groups-1)
        for a in range(Groups):
            st_dev[a] = (float(B_Block_Div[a]-ER/Z))**2
            st_dev2[a] =(float(B_Block_Div2[a]-ES/Z))**2
        for a in range(Groups):
            jack_knifeR += st_dev[a]
            jack_knifeS += st_dev2[a]
        jack_knife_replica = np.sqrt(jack_knifeR)/Groups**2
        jack_knife_s = np.sqrt(jack_knifeS)/Groups**2



        '''bootstrap = 0.0
        B_Block2 = [0.0 for i in range(Groups)]
        for a in range(Groups):
            for b in range(Measurements/Groups):
                B_Block2[a] += M1_est[randint(0,Measurements-1)]
        for a in range(Groups):
            B_Block2[a] = B_Block2[a]/(Measurements/Groups)
        print B_Block2
        for a in range(Groups):
            bootstrap += (B_Block2[a]-A/Z)**2
        bootstrap = np.sqrt(bootstrap/Groups)'''


        Avg_Replica_Energy = ER/Z
        Avg_Replica_Energy2 = ER2/Z
        Avg_B_Energy = EI/Z
        Avg_S_Energy = ES/Z
        Avg_A_Energy = EA/Z
#        print(E_Replica)
#        print(E_s)
#        print("Replica")
#        print(Avg_Replica_Energy)
#        print("Avg_S_Energy")
#        print(Avg_S_Energy)
        return Avg_Replica_Energy, Avg_Replica_Energy2, Avg_A_Energy, Avg_B_Energy, Avg_S_Energy,1/JT, jack_knife_replica, jack_knife_s


def Renyi_Entropy(T_lower, T_upper, dT):
    N = 20
    Beta2 = [2/(i*dT + dT) for i in range(int((T_upper-T_lower)/dT))]
    Beta = [1/(i*dT + dT) for i in range(int((T_upper-T_lower)/dT))]
    Temps2 = [i*dT +dT for i in range(int((T_upper-T_lower)/dT))]
    Replica_Energies = [0. for i in range(int((T_upper-T_lower)/dT))]
    S_Energies = [0. for i in range(int((T_upper-T_lower)/dT))]
    S_Error = [0. for i in range(int((T_upper-T_lower)/dT))]
    I_AB = [0. for i in range(int((T_upper-T_lower)/dT))]
    I_AB2 = [0. for i in range(int((T_upper-T_lower)/dT))]
    I_AB3 = [0. for i in range(int((T_upper-T_lower)/dT))]
    I_AB4 = [0. for i in range(int((T_upper-T_lower)/dT))]
    I_AB5 = [0. for i in range(int((T_upper-T_lower)/dT))]
    I_AB6 = [0. for i in range(int((T_upper-T_lower)/dT))]
    R_Error = [0. for i in range(int((T_upper-T_lower)/dT))]
    pool = Pool(processes = 8)
    print "Starting Pool"
#    Results = pool.map(Replica_Energy, Beta)
#    Results = [(-1024.0, -1024.0, -512.0, -512.0, -512.0, 0.2, 0.0, 0.0), (-1024.0, -1024.0, -512.0, -512.0, -512.0, 0.4, 0.0, 0.0), (-1024.0, -1024.0, -512.0, -512.0, -512.0, 0.6000000000000001, 0.0, 0.0), (-1024.0, -1024.0, -512.0, -512.0, -512.0, 0.8, 0.0, 0.0), (-1023.375, -1023.375, -512.0, -511.375, -511.625, 1.0, 0.0054103569837223517, 0.0043939776082655355), (-1021.4375, -1021.4375, -512.0, -509.4375, -510.875, 1.2, 0.015896673586210091, 0.0091694624582341552), (-1016.3125, -1016.3125, -512.0, -504.3125, -508.125, 1.4000000000000001, 0.014801101796735184, 0.013827580014915131), (-1008.875, -1008.8125, -511.75, -497.125, -504.0625, 1.6, 0.020011630666340454, 0.014417422637013087), (-990.9375, -990.625, -511.625, -479.3125, -494.0625, 1.7999999999999998, 0.019614776753149963, 0.013057320264282961), (-965.9375, -965.3125, -511.5, -454.75, -482.4375, 2.0, 0.042686996586022645, 0.032209345760520772), (-927.6875, -926.5625, -510.5, -417.5625, -460.75, 2.2, 0.075858998158703853, 0.056491297923409484), (-861.6875, -859.75, -508.375, -353.875, -433.0625, 2.4000000000000004, 0.13115560772266449, 0.083256796638254651), (-832.5625, -829.5625, -504.5, -328.5625, -416.25, 2.6000000000000005, 0.072646520154411767, 0.064924283804683611), (-765.3125, -760.5, -504.5, -263.875, -385.75, 2.8000000000000003, 0.099861110849321935, 0.072621868236596138), (-710.4375, -706.0, -491.5, -220.625, -356.625, 3.000000000000001, 0.068170635035229091, 0.040694541637286842), (-681.0625, -675.0625, -488.5, -195.625, -340.4375, 3.2, 0.11346047684143611, 0.068933788359084369), (-649.0, -644.4375, -462.625, -185.25, -321.6875, 3.4000000000000004, 0.07315165003649117, 0.065248431784731012), (-620.125, -613.9375, -455.875, -165.4375, -312.625, 3.6000000000000005, 0.08210144894777413, 0.043160373968012276), (-588.5, -584.1875, -430.375, -157.5625, -288.6875, 3.8000000000000003, 0.073287746247241714, 0.035078659417659891), (-550.375, -547.8125, -408.75, -138.25, -275.125, 4.0, 0.11273702620473545, 0.055909497679764081), (-501.0625, -497.625, -358.875, -135.5625, -251.0, 4.2, 0.10661184329304695, 0.053803441041044446), (-458.0, -454.3125, -331.375, -117.8125, -228.3125, 4.4, 0.12038060456467742, 0.080169944092695261), (-427.0, -424.25, -304.375, -110.875, -213.5, 4.6000000000000005, 0.15065724155312929, 0.079999825613649253), (-408.8125, -405.9375, -282.125, -111.875, -200.1875, 4.800000000000001, 0.07130530484000365, 0.024346606426756148), (-371.125, -367.375, -254.125, -103.75, -185.1875, 5.000000000000001, 0.15800000542394579, 0.079106360280613991), (-350.6875, -348.125, -239.75, -97.5, -178.25, 5.2, 0.083949606442170185, 0.039269223405712463), (-326.1875, -322.625, -219.375, -95.4375, -163.375, 5.4, 0.059955737337242326, 0.049905992221677228), (-323.25, -321.1875, -215.625, -93.375, -161.125, 5.6000000000000005, 0.043483838702629618, 0.043847547501257106), (-301.6875, -298.375, -204.625, -84.625, -154.3125, 5.800000000000001, 0.059789301657684438, 0.046713622615535993), (-297.625, -294.25, -196.125, -91.375, -148.375, 6.000000000000002, 0.080770730056529294, 0.048049435025958181), (-284.875, -281.875, -186.125, -86.25, -144.375, 6.2, 0.094659044058822625, 0.032490907651807233), (-267.1875, -264.9375, -182.0, -73.4375, -133.4375, 6.4, 0.15221952107159117, 0.079444179777339585), (-258.9375, -255.5, -172.25, -75.6875, -132.5625, 6.600000000000001, 0.13263359705535102, 0.078940853255707488), (-240.375, -238.375, -159.75, -71.4375, -119.125, 6.800000000000001, 0.12316039113107211, 0.069979547565608341), (-236.5, -234.5625, -158.0, -66.6875, -119.25, 7.0, 0.047560883376478826, 0.023675450262942443), (-231.125, -230.125, -152.125, -67.6875, -116.625, 7.199999999999999, 0.11561713552266713, 0.054965809685430904), (-224.75, -222.0, -144.125, -71.5625, -114.625, 7.400000000000001, 0.075909268183624728, 0.032182747444857662), (-218.9375, -217.0625, -145.5, -64.6875, -110.9375, 7.6000000000000005, 0.13287754915858524, 0.04254084535124248), (-215.8125, -213.0, -142.625, -63.9375, -108.3125, 7.800000000000001, 0.084157077265896912, 0.047085451787251251), (-198.875, -195.375, -123.75, -64.875, -101.125, 8.0, 0.05781530088361362, 0.036396593435536514), (-194.4375, -192.0, -130.375, -56.375, -98.625, 8.2, 0.087658689862457076, 0.051478605606021362), (-192.6875, -190.125, -127.25, -59.8125, -95.0, 8.4, 0.10884915389277437, 0.073083506891718331), (-178.9375, -177.0, -114.875, -53.625, -88.5, 8.6, 0.068480554947016825, 0.042440074423731086), (-181.3125, -180.0, -116.375, -59.375, -90.75, 8.799999999999999, 0.12126656632402741, 0.075942079625910749), (-171.25, -170.4375, -112.875, -52.125, -83.5, 9.0, 0.077789458005476, 0.043196434107954561), (-177.5625, -175.0625, -113.375, -57.5, -90.125, 9.2, 0.090207815217194787, 0.044215307565938208), (-165.8125, -164.875, -109.625, -50.75, -81.75, 9.4, 0.11030419690199106, 0.064500818199999546), (-159.0, -158.1875, -99.75, -49.3125, -82.75, 9.6, 0.063743215675418005, 0.041729730124642657), (-161.375, -158.9375, -99.875, -52.375, -81.25, 9.8, 0.063562203070062662, 0.043655373249124767)]
    Results = [(-1600.0, -1600.0, -800.0, -800.0, -800.0, 0.2, 0.0, 0.0), (-1600.0, -1600.0, -800.0, -800.0, -800.0, 0.4, 0.0, 0.0), (-1600.0, -1600.0, -800.0, -800.0, -800.0, 0.6000000000000001, 0.0, 0.0), (-1599.75, -1599.75, -800.0, -799.75, -799.875, 0.8, 0.0027338055164984609, 0.0020879784524403412), (-1598.9375, -1598.9375, -800.0, -798.9375, -799.3125, 1.0, 0.0045847312291194837, 0.0035072000809101651), (-1595.9375, -1595.9375, -800.0, -795.9375, -798.25, 1.2, 0.011751919469114651, 0.0063134534034520792), (-1588.875, -1588.875, -800.0, -788.875, -794.625, 1.4000000000000001, 0.02972817098631083, 0.013827580014915292), (-1572.5625, -1572.5, -800.0, -772.5625, -786.75, 1.6, 0.025027761809809125, 0.012626906806901884), (-1544.75, -1544.4375, -799.25, -745.5, -772.6875, 1.7999999999999998, 0.058121502528087753, 0.052036643347519611), (-1501.9375, -1501.1875, -799.5, -702.75, -751.25, 2.0, 0.067046777686912057, 0.042674228335503203), (-1435.375, -1433.25, -796.25, -639.3125, -715.6875, 2.2, 0.087762537760253348, 0.08900538442270764), (-1359.1875, -1355.0625, -794.5, -565.5625, -679.0, 2.4000000000000004, 0.11973670255878921, 0.062579668611055211), (-1257.375, -1251.0, -788.625, -469.375, -629.5, 2.6000000000000005, 0.14432102575221362, 0.11741580651640489), (-1161.5, -1154.0625, -778.875, -384.25, -579.0625, 2.8000000000000003, 0.14733833777418803, 0.085689831833046778), (-1114.0625, -1106.75, -769.625, -347.0, -550.9375, 3.000000000000001, 0.12796855240478289, 0.10186788736399838), (-1067.3125, -1060.1875, -759.125, -312.25, -533.1875, 3.2, 0.097857732204218423, 0.064092776491005005), (-1022.5625, -1016.625, -739.0, -284.0625, -510.4375, 3.4000000000000004, 0.067785836521298687, 0.052048610608839301), (-966.0, -959.25, -713.25, -252.3125, -481.25, 3.6000000000000005, 0.059330508303580588, 0.051120457547316187), (-926.5625, -923.25, -689.0, -236.125, -459.625, 3.8000000000000003, 0.11055798726780715, 0.069227923420095455), (-868.1875, -862.9375, -651.625, -213.8125, -431.875, 4.0, 0.11176258706270373, 0.06692241994857144), (-781.125, -776.8125, -563.875, -202.9375, -391.0, 4.2, 0.20645058997668711, 0.08906222028241062), (-715.3125, -709.75, -508.625, -190.5625, -356.25, 4.4, 0.23718520136066593, 0.11085685305213416), (-686.6875, -682.9375, -489.125, -179.5, -343.0625, 4.6000000000000005, 0.1952642638139801, 0.10296250411710872), (-630.1875, -625.0625, -438.375, -173.75, -309.1875, 4.800000000000001, 0.22982367403633713, 0.11509004657117144), (-583.0, -579.1875, -404.75, -157.1875, -288.9375, 5.000000000000001, 0.19396537979411077, 0.12760789498492628), (-557.5, -553.375, -394.5, -149.9375, -281.6875, 5.2, 0.15506566457998583, 0.079874192010600698), (-511.9375, -508.4375, -347.375, -149.25, -256.375, 5.4, 0.16236010507788692, 0.081736518022898649), (-488.375, -484.4375, -333.625, -140.125, -246.25, 5.6000000000000005, 0.1156171355226672, 0.050433587177209742), (-462.5625, -457.375, -313.0, -137.5625, -228.6875, 5.800000000000001, 0.12663784192182356, 0.079499037783673482), (-457.4375, -453.125, -307.375, -136.4375, -228.0, 6.000000000000002, 0.1711228141666106, 0.088782061613175492), (-432.5, -429.6875, -292.5, -121.75, -216.875, 6.2, 0.12372798260638944, 0.056242691914716994), (-417.0625, -412.9375, -278.375, -123.9375, -209.5, 6.4, 0.081349856159540726, 0.03748405273157749), (-400.25, -396.625, -263.375, -124.8125, -200.0, 6.600000000000001, 0.12500996452630236, 0.061939450560319596), (-383.9375, -383.125, -254.375, -116.25, -188.9375, 6.800000000000001, 0.13750287620919152, 0.064721317983448753), (-372.6875, -369.9375, -247.75, -112.75, -185.5625, 7.0, 0.1208240753356877, 0.057136698799711408), (-356.0625, -353.5, -239.875, -107.75, -175.0, 7.199999999999999, 0.15919918821094797, 0.098442308353571137), (-346.25, -342.875, -228.0, -107.1875, -176.25, 7.400000000000001, 0.11089055668191276, 0.056930583682396507), (-347.25, -345.0, -229.625, -105.625, -168.6875, 7.6000000000000005, 0.10536092608219598, 0.065534161422703283), (-339.625, -336.25, -223.25, -103.0, -171.625, 7.800000000000001, 0.13178861407354531, 0.08720725331240152), (-302.8125, -302.625, -199.625, -91.0, -152.875, 8.0, 0.094661511333369613, 0.042899846191479005), (-317.25, -314.5, -209.75, -95.375, -157.3125, 8.2, 0.074367550218561077, 0.048800215237788318), (-310.0, -307.8125, -199.125, -97.625, -155.0625, 8.4, 0.10775725995358272, 0.065047674257113788), (-290.9375, -290.0, -188.875, -93.625, -147.8125, 8.6, 0.11051291157706053, 0.064248064914553482), (-292.125, -288.125, -193.375, -88.0625, -147.0625, 8.799999999999999, 0.12229769956885614, 0.058023640141671481), (-279.875, -276.25, -186.125, -86.3125, -139.1875, 9.0, 0.082721148353279414, 0.045360786969577252), (-271.4375, -270.1875, -176.25, -84.4375, -135.125, 9.2, 0.080611537732171362, 0.065397405082448262), (-251.125, -248.5, -171.375, -72.375, -125.125, 9.4, 0.08922640329142488, 0.0490754260400015), (-252.875, -249.875, -162.625, -79.375, -128.0, 9.6, 0.098489746622336671, 0.073202715766993873), (-250.5, -248.9375, -160.375, -81.25, -125.4375, 9.8, 0.065649290936842547, 0.038287513153826731)]

#    Results2 = pool.map(Ising_Energy, 2Beta)
#    Results2 = [(-512.0, 0.0), (-512.0, 0.0), (-512.0, 0.0), (-512.0, 0.0), (-512.0, 0.0), (-512.0, 0.0), (-512.0, 0.0), (-512.0, 0.0), (-511.75, 0.0027338055164984609), (-511.625, 0.0030564874860784575), (-510.75, 0.0075695647133096292), (-509.25, 0.010228963604812095), (-507.0625, 0.011962026289188578), (-504.0, 0.02057934030645733), (-502.0, 0.014973669492186065), (-493.8125, 0.02549618705705536), (-486.0625, 0.030664049805827622), (-475.4375, 0.027719830819240534), (-463.6875, 0.061978401847093065), (-449.0, 0.05465332386917536), (-425.9375, 0.076797352666056556), (-397.4375, 0.1000791586707198), (-360.875, 0.070139561983900187), (-321.0625, 0.067121049765835272), (-297.0, 0.069877124296868431), (-266.8125, 0.10797594527140965), (-247.4375, 0.087899925103195178), (-238.625, 0.11139209192100785), (-220.6875, 0.065799722989688683), (-202.625, 0.058158995092249602), (-200.9375, 0.055969340837279585), (-187.625, 0.046441179177596478), (-183.1875, 0.069786822851696295), (-172.9375, 0.05309116710408477), (-169.25, 0.055892785828343033), (-165.8125, 0.040944385107060645), (-158.5, 0.066310048113897124), (-148.875, 0.048049435025957973), (-145.3125, 0.053755670437575363), (-143.8125, 0.025642333104660669), (-138.3125, 0.065761851280093153), (-136.1875, 0.055488788508234889), (-135.875, 0.072411450531310789), (-123.4375, 0.065979316749717604), (-118.8125, 0.042526202602215436), (-117.5625, 0.043439057080996726), (-116.0625, 0.052785282259341412), (-109.5625, 0.052096452163705703), (-113.8125, 0.058440754762438249), (-111.375, 0.057922924396127416), (-107.5, 0.033925634271363081), (-106.0, 0.059330508303580422), (-112.875, 0.051430189319169387), (-97.5625, 0.039818497524857839), (-95.75, 0.060782360732991536), (-99.5625, 0.045347054800347252), (-90.5, 0.033556465130296663), (-88.9375, 0.06072982461013636), (-90.8125, 0.059496918916619394), (-86.8125, 0.060287230269186996), (-84.5, 0.066778014739516001), (-83.5, 0.054858061283234352), (-90.5625, 0.056655055119974422), (-79.875, 0.070863944434772924), (-81.375, 0.030999875920382017), (-80.125, 0.056020782618063224), (-78.875, 0.056793671878981919), (-80.625, 0.056595937024589767), (-76.625, 0.04474736651055685), (-75.4375, 0.045974489545993528), (-76.6875, 0.053102896718611674), (-72.5625, 0.054777110893056084), (-73.5, 0.061616847061795876), (-73.875, 0.053332568312367733), (-76.3125, 0.059810131445943111), (-72.625, 0.048539484883889789), (-61.625, 0.035354986744324017), (-67.5, 0.068617973879193647), (-65.9375, 0.037993586112636667), (-61.5625, 0.033868216848308583), (-64.9375, 0.064740560952018425), (-69.25, 0.07076280126851496), (-58.125, 0.028487162986830895), (-63.875, 0.066549122266383154), (-58.0625, 0.036794405477923522), (-57.4375, 0.049973025149435329), (-58.25, 0.046070909465513996), (-58.375, 0.055304689650925741), (-58.625, 0.063405234895717547), (-62.3125, 0.065932102663203401), (-57.375, 0.039921985764553365), (-60.5, 0.04255731242173573), (-53.8125, 0.056600063546598176), (-52.875, 0.04611819980221965), (-51.5625, 0.050935854232677984), (-51.875, 0.018989620008006212), (-49.625, 0.032758169913926695), (-56.6875, 0.069393036859993609)]
    Results2 = [(-800.0, 0.0, 0.2), (-800.0, 0.0, 0.4), (-800.0, 0.0, 0.6000000000000001), (-800.0, 0.0, 0.8), (-799.125, 0.0037847823566550436, 1.0), (-795.3125, 0.0094619558916370708, 1.2), (-787.5625, 0.020965365395277817, 1.4000000000000001), (-771.5625, 0.038287513153826704, 1.6), (-742.4375, 0.036556663666182751, 1.7999999999999998), (-697.8125, 0.086492835499326806, 2.0), (-623.1875, 0.11935117258561208, 2.2), (-488.0, 0.15492100624518407, 2.4000000000000004), (-418.625, 0.09083134825012916, 2.6000000000000005), (-381.5, 0.10033475935561438, 2.8000000000000003), (-329.1875, 0.053546718166355388, 3.000000000000001), (-289.125, 0.063991880252118924, 3.2), (-270.625, 0.094698512737699192, 3.4000000000000004), (-262.75, 0.094872636327470256, 3.6000000000000005), (-237.25, 0.069376206553214664, 3.8000000000000003), (-220.5625, 0.040623696873723074, 4.0), (-208.75, 0.052538403106874437, 4.2), (-202.6875, 0.066196067811721632, 4.4), (-186.8125, 0.043567903541358893, 4.6000000000000005), (-180.5, 0.063095062612065198, 4.800000000000001), (-170.25, 0.050086619804278837, 5.000000000000001), (-165.3125, 0.071705960758328066, 5.2), (-155.125, 0.05809470731089271, 5.4), (-153.9375, 0.089040364629770177, 5.6000000000000005), (-144.3125, 0.052418240572862269, 5.800000000000001), (-137.5625, 0.079725904548688178, 6.000000000000002), (-135.5, 0.072053626748076055, 6.2), (-124.5625, 0.045566272900078857, 6.4), (-120.625, 0.047027547272712883, 6.600000000000001), (-118.625, 0.038524677665386586, 6.800000000000001), (-122.5625, 0.034918502368187429, 7.0), (-115.9375, 0.069024077913955623, 7.199999999999999), (-113.6875, 0.080819871518423403, 7.400000000000001), (-111.1875, 0.088105162622424418, 7.6000000000000005), (-110.9375, 0.067564967716147617, 7.800000000000001), (-102.125, 0.084979147305218078, 8.0), (-98.75, 0.031527783240681853, 8.2), (-98.4375, 0.071784088398320789, 8.4), (-91.6875, 0.064961445391528233, 8.6), (-82.5, 0.098934554518662038, 8.799999999999999), (-92.5625, 0.049194258666010404, 9.0), (-87.8125, 0.070320250068913057, 9.2), (-82.8125, 0.066045359767324133, 9.4), (-87.5625, 0.064547873115297458, 9.6), (-80.3125, 0.055735170132084594, 9.8)]
#    Results3 = pool.map(A_Energy, Beta)
    Results3 = [-1600,-1600,-1600, -1600, -1600, -1600, -1600, -1599.75, -1599.25, -1598.125, -1594.5, -1594.375, -1582.25, -1574.75, -1562.875, -1532.125, -1519.125, -1481.375, -1455.125, -1412.375, -1318.75, -1229.875, -1156.875, -975.125, -900., -835.5, -756.375, -721.75, -704.5, -643., -633.5, -591.25, -566.375, -567., -525.875, -518.375, -503.375, -474.625, -467.875, -462.875, -423.125, -419.75, -424., -389.875, -400.625, -380., -364.25, -370., -342.]
#    print Results3
 #   print Results2
    I_Error = [0. for i in range(int((T_upper-T_lower)/dT))]
    I_plot = [0. for i in range(int((T_upper-T_lower)/dT))]
    I_plot2 = [0. for i in range(int((T_upper-T_lower)/dT))]
    I_plot3 = [0. for i in range(int((T_upper-T_lower)/dT))]
    I_plot4 = [0. for i in range(int((T_upper-T_lower)/dT))]
    I_plot5 = [0. for i in range(int((T_upper-T_lower)/dT))]
    I_plot6 = [0. for i in range(int((T_upper-T_lower)/dT))]



#    I_AB2= 0.
    SError = 0.
    RError = 0.

    for i in range(int((T_upper-T_lower)/dT)):
        I_AB[i] += dT/N*(2*Results[i][0]-2*Results2[i][0] - Results3[i])/(T_lower + i*dT)**2
#        I_AB2[i] +=2*dT/N*(-.5*Results[i][0] + Results[i][1] + Results2[i][0])/(T_lower +i*dT)**2
 #       I_AB3[i] += 2*dT/N*(-Results[i][0] + Results[i][1] + Results[i][2])/(T_lower +i*dT)**2
  #      I_AB4[i] += 2*dT/N*(-.5*Results[i][0] + Results[i][1] + Results[i][2])/(T_lower +i*dT)**2
   #     I_AB5[i] += dT/N*(-Results[i][0] + Results[i][1] + Results[i][2])/(T_lower +i*dT)**2
 #       I_AB6[i] = 2*dT/N*(Results[i][0] - Results2[i][0] + 2*N**2)/(T_lower +i*dT)**3
        Replica_Energies[i] = Results[i][0]
        S_Energies[i] = Results[i][1]
        R_Error[i] = Results[i][2]
        S_Error[i] = Results[i][3]

    print I_AB

    for i in range(int((T_upper-T_lower)/dT)):
        for j in range(i, int((T_upper-T_lower)/dT)):
            I_plot[i] += I_AB[j]
            I_plot2[i] += I_AB2[j]
            I_plot3[i] += I_AB3[j]
            I_plot4[i] += I_AB4[j]
        #    I_plot5[i] += I_AB5[j]
         #   I_plot6[i] += I_AB6[j]
            I_Error[i] += np.sqrt(4*S_Error[j]**2 + 4*R_Error[j]**2)/N**2


#    plt.errorbar(Temps2, Replica_Energies, xerr = 0, yerr = Replica_Error, ecolor = 'red')

#    plt.plot(Temps2, I_plot2)
#    plt.title("Replica/2 (Two Models)")
#    plt.show()

#    plt.plot(Temps2, I_AB2)
#    plt.title("Unsummed Negative")
#    plt.show()

#    plt.plot(Temps2, I_AB)z x
#    plt.title("Unsummed Positive")
#    plt.show()

    plt.plot(Temps2, I_plot)
    plt.title("Replica (Three Simulations)")
    plt.show()

#    plt.plot(Temps2, I_plot3)
#    plt.title("Replica (One Model)")
#    plt.show()

#    plt.plot(Temps2, I_plot4)
#    plt.title("Replica/2 (One Model)")
#    plt.show()

#    plt.plot(Temps2, I_AB4)
#    plt.title("Unsummed Negative I_AB/T (Two Models)")
#    plt.show()

#    plt.plot(Temps2, I_plot4)
#    plt.title("Summed Negative I_AB/T (Two Models)")
#    plt.show()

#    plt.plot(Temps2, I_AB5)
#    plt.title("Unsummed Positive I_AB/T (One Model)")
#    plt.show()

#    plt.plot(Temps2, I_plot5)
#    plt.title("Summed Positive I_AB/T (One Model)")
#    plt.show()

#    plt.plot(Temps2, I_AB6)
#    plt.title("Unsummed Positive I_AB/T (Two Models)")
#    plt.show()

#    plt.plot(Temps2, I_plot6)
#    plt.title("Summed Positive I_AB/T (Two Models)")
#    plt.show()



Renyi_Entropy(.2, 10.0, .2)

def Renyi_Entropy2(T_lower, T_upper, dT):
    N = 16
    Temps = [1/(i*dT + dT) for i in range(int((T_upper-T_lower)/dT))]
    Temps2 = [i*dT +dT for i in range(int((T_upper-T_lower)/dT))]
    Replica_Energies = [0. for i in range(int((T_upper-T_lower)/dT))]
    S_Energies = [0. for i in range(int((T_upper-T_lower)/dT))]
    S_Error = [0. for i in range(int((T_upper-T_lower)/dT))]
    I_AB = [0. for i in range(int((T_upper-T_lower)/dT))]
    R_Error = [0. for i in range(int((T_upper-T_lower)/dT))]
    pool = Pool(processes = 8)
    print "Starting Pool"
    Results = pool.map(Replica_Energy, Temps)
    print "Pool 2"
    Results2 = pool.map(Ising_Energy, Temps)
    print Results2
    I_Error = [0. for i in range(int((T_upper-T_lower)/dT))]
    I_plot = [0. for i in range(int((T_upper-T_lower)/dT))]
#    I_AB2= 0.
    SError = 0.
    RError = 0.

    for i in range(int((T_upper-T_lower)/dT)):
        I_AB[i] += 2*dT/N*((0.5)*Results[i][0] - 2*Results2[i][0] + 2*N**2)/(T_lower + i*dT)**2
        Replica_Energies[i] = Results[i][0]
        S_Energies[i] = Results2[i][0]
        R_Error[i] = Results[i][2]
        S_Error[i] = Results[i][3]

    print I_AB

    for i in range(int((T_upper-T_lower)/dT)):
        for j in range(i, int((T_upper-T_lower)/dT)):
            I_plot[i] += I_AB[j]
            I_Error[i] += np.sqrt(4*S_Error[j]**2 + 4*R_Error[j]**2)/N**2


#    plt.errorbar(Temps2, Replica_Energies, xerr = 0, yerr = Replica_Error, ecolor = 'red')
    plt.errorbar(Temps2, I_plot, xerr = 0, yerr = I_Error, ecolor = 'red')
    plt.title("Entropy vs Temperature")
    plt.show()



#Renyi_Entropy(0.2, 10.0, 0.2)




def Plot_Error(Points):
    Bootstrap = [0 for a in range(Points)]
    Jack_Knife = [0 for a in range(Points)]
    Results = [0 for a in range(Points)]
    for a in range(Points):
        Groups = 2**(a+3)
        Results[a] = Monte_Carlo(.6, Groups)
    for a in range(points):
        Jack_Knife[a] = Results[a][1]
        Bootstrap[a] = Results[a][2]
    plt.plot(Jack_Knife)
    plt.plot(Bootstrap)
    plt.show()



def Vary_Temp(T_upper, T_lower, dT):
#    Magnetizations = []
#    Temperatures = []

    y_err = [0 for a in range(int((T_upper-T_lower)/dT))]
    JT = [1/(i*dT + dT) for i in range(int((T_upper-T_lower)/dT))]
    Temperatures = [(i*dT + dT) for i in range(int((T_upper-T_lower)/dT))]
    Magnetization = [0 for i in range(int((T_upper-T_lower)/dT))]
    pool = Pool(processes = 8)
    Results = pool.map(Monte_Carlo, JT)
    print Results
    for i in range(int((T_upper-T_lower)/dT)):
        Magnetization[i] = Results[i][0]
        y_err[i] = Results[i][1]
#    for i in range(int((T_upper-T_lower)/dT)):
#        Magnetizations[i], y_err[i] = Monte_Carlo2(2**20, 2**8,1/(i*dT + dT))
#        Temperatures[i] = i*dT + dT
#        print i
    plt.errorbar(Temperatures, Magnetization, xerr = 0, yerr = y_err, ecolor = 'red')
    plt.title("Magnetization vs Temperature")
    plt.show()








