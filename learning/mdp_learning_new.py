#! /usr/bin/env python
import PyQt4
import matplotlib
from numpy.matlib import rand
from _random import Random
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfinv
import time
import pdb
from sequence_generator import SeqGenerator
from agent import Agent
import drawing_tools as dt



def reward_function(s, a_name, pat):
    if a_name == 'prime':
        if s[0] == 'win':
            return 1
            return 1-s[1]/float(len(pat))
        elif s[0] == 'fail':
            return -1
            return -(1-s[1]/float(len(pat)))
    elif a_name == 'inhibit':
        if s[0] == 'win':
            print "aaaaaa"
            return 1
            return 1-s[1]/float(len(pat))
        elif s[0] == 'fail':
            return -1
            return -(1-s[1]/float(len(pat)))
    elif a_name == 'idle':
        return 0
    
def update(Q, s_now, a, s_next, r, alpha):
    Q[s_now,a] = Q[s_now,a] + alpha*(r + np.max(Q[s_next,:]) - Q[s_now,a])
    return Q 

def update_sarsa(Q, et, s_now, a_now, r, s_next, a_next, alpha, lbd):
    delta = r + Q[s_next, a_next] - Q[s_now, a_now]
    et[s_now, a_now] += 1
    
    Q = Q + alpha * delta * et
    et = lbd * et
    
    return Q, et

def update_monte_carlo ( Q, N, traj, r):
    B = {}
    for sa in traj:
        s = sa[0]
        a = sa[1]
        if not B.has_key((s,a)):
            B[(s,a)] = 1
            Q[s,a] = (Q[s,a]*N[s,a] + r)/float(N[s,a]+1)
            N[s,a] += 1
            
    return Q

def diff_Qmc(Qmc, Qmc_prev, traj):
    B = {}
    traj_diff = []
    for sa in traj:
        s = sa[0]
        a = sa[1]
        if not B.has_key((s,a)):
            traj_diff.append(np.abs(Qmc[s, a] - Qmc_prev[s, a]))

    return np.sum(traj_diff)/len(traj_diff)

#def get_ie(x, n, z):
#    return ((x/float(n) + (z**2)/(2.0*n) + z/np.sqrt(n)*
#           np.sqrt((x/float(n))*(1-(x/float(n))) + z**2/(4.0*n)))/
#           (1 + (z**2)/float(n)))
#
#def get_ie_action(x0, n0, x1, n1, z):
#    if n0 == 0 or n1 == 0:
#        return int(np.random.uniform(0,1) > 0.5)
#    
#    ub0 = get_ie(x0, n0, z)
#    ub1 = get_ie(x1, n1, z)
#    
#    print "x0, n0, ub0", x0, n0, ub0
#    print "x1, n1, ub1", x1, n1, ub1
#    print "a:", int(ub0 < ub1)
#    return int(ub0 < ub1)

def joint( s, S ):
    m = S['match']
    return m[s[0]]*len(S['time']) + s[1]

def update_state( s, a_name, y, pat, t_seq, t_mdp):
    s_next = ('ndef', 0)
    
    # If the previous state was either a win or a loss
    if s[0] != 'ndef':
        return s_next
        
    if a_name == 'prime':
        s_next = (s[0], t_mdp)
    elif a_name == 'inhibit':
        s_next = (s[0], t_mdp)
    elif a_name == 'idle':
        if t_mdp < len(pat) - 1:
            s_next = (s[0], s[1] + 1)
     
    if y == pat[t_seq]:
        # Win
        if t_seq == len(pat) - 1:
            if a_name == 'prime':
                s_next = ('win', s_next[1])
            elif a_name == 'inhibit':
                print "Lost by inhibitng"
                s_next = ('fail', s_next[1])
            elif a_name == 'idle':
                s_next = ('fail', s_next[1])
    else:
        # Loss
        if a_name == 'prime':
            s_next = ('fail', s_next[1])
        elif a_name == 'inhibit':
            print "Won by inhibitng"
            s_next = ('win', s_next[1])
        elif a_name == 'idle':
            s_next = ('fail', s_next[1])
    
    #print "s_next =", s_next    
#    print "s_next:", s_next, "\n\n"    
    return s_next    

if __name__ == '__main__':
    plt.close("all")
    alpha = 0.05
    lbd = 0.9
    Tw = 10000  # data width
    Yh = []
    Lh = []
    Ah = []
    L = []  # labels
    Rh = []
    Sh = []
    Q0h = []
    Q1h = []
    Q0hmc = []
    Q1hmc = []
    Q0hq = []
    Q1hq = []
    Alphah = []
    Epsh = []
    traj = []
    primed = False
    seq_gen = SeqGenerator()
    #agent = Agent()
    total_r = 0
    pattern = [1, 1, 1, 2]
    #pattern = [1, 1, 1, 1, 1]
    S = {'match': {'ndef' : 0,'fail' : 1, 'win': 2},
         'time': range(0,len(pattern))}
    A = ['idle', 'prime', 'inhibit'] # maybe also something like !prime (inhibit)
    
    Q = np.zeros([len(S['match'])*len(S['time']),len(A)])
    Q_q = np.zeros([len(S['match'])*len(S['time']),len(A)])
    et = np.zeros([len(S['match'])*len(S['time']),len(A)])
    
    alpha_test = 0.9
    visits = np.zeros([len(S['match'])*len(S['time']),len(A)])
    
    Qmc = np.zeros([len(S['match'])*len(S['time']),len(A)])
    Qmc_prev = np.zeros([len(S['match'])*len(S['time']),len(A)])
    Nmc = np.zeros([len(S['match'])*len(S['time']),len(A)])
    
    #Q[:,1] = np.squeeze(np.ones([len(S['match'])*len(S['time']), 1]))
    eps = 0.5
    s0 = ('ndef',0)
    s = s0
    s_next = s0
    t_mdp = 0
    t_seq = 0
    t_to_update = 1
    t_lifetime = 1
    a = 0
    a_next = 0
    seq_active = False
    x0 = 0
    n0 = 0
    x1 = 0
    n1 = 0
    ie_z = np.sqrt(2)*erfinv(2*0.975-1) # 95% confidence
    np.random.seed()
    
    # for debugging purposes - not known to the agent
    opt_pol = np.asarray([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    found_opt_pol = False
    found_opt_q_pol = False
    found_opt_mc_pol = False
    greedy = False
    eps_print = False
    
    for t in range(Tw):
        (y, l) = seq_gen.step()
        Yh.append(y)
        Lh.append(l)
        
        r = 0
        
        if seq_active is False and y == pattern[0]:
            seq_active = True
        
        if seq_active is True:
            t_to_update -= 1
            # pdb.set_trace()
            if t_to_update == 0 or y != pattern[t_seq]:
                #print "\n\nSTEP outsied:", len(Sh) + 1
                    
                if t_seq > 0:
                    #print "\n\nSTEP:", len(Sh) + 1
                    s_next = update_state(s, A[a], y, pattern, t_seq, t_mdp)
                    
                    js = joint(s, S)
                    jn = joint(s_next, S)
                    
                    r = reward_function(s_next, A[a], pattern)
                    #alpha = beta * np.abs(Q[js, a] - Qmc[js, a])
                    #Alphah.append(alpha_test)
#                    if s_next[0] == "win":
#                        print "WON"
#                        print "\ts =", s
#                        print "\ta =", A[a]
#                        print "\tr =", r
#                        print "\ts_next =", s_next, "\n\n"
#                    if s_next[0] == "fail":
#                        print "FAILED"
#                        print "\ts =", s
#                        print "\ta =", A[a]
#                        print "\tr =", r
#                        print "\ts_next =", s_next, "\n\n"
                    Q_q = update(Q_q, js, a, jn, r, alpha)
                    #Q_q = update(Q_q, js, a, jn, r, alpha_test)
                    #alpha = 1.0/np.power(t_lifetime, 1/3.0)
                    
                    t_mdp += 1
                    
                if s_next[0] == 'ndef':
                    #choose an action
                    ran = np.random.uniform(0,1)
                    if ran > eps:
                        greedy = True
                        a_next = np.argmax(Q_q[joint(s_next,S),:])
                    else:
                        ran = np.random.randint(3)
                        greedy = False
                        a_next = ran

                if t_seq > 0:
                    Q, et = update_sarsa(Q, et, js, a, r, jn, a_next, alpha, lbd)
                    #Q, et = update_sarsa(Q, et, js, a, r, jn, a_next, alpha_test, lbd)
                
                if A[a] == 'idle' and (A[a_next] == 'prime' or A[a_next] == 'inhibit'):
                    t_to_update = (len(pattern) - 1) - t_seq
                    primed = True
                else:                   
                    t_to_update = 1
                
                a = a_next
                # rewards should also be given if the agent successfully predicts a wrong pattern
                s = s_next
                traj.append((joint(s, S),a))
                
  
            t_seq += 1
            t_lifetime += 1
        
        Ah.append(a)
        Sh.append(joint(s_next,S))
        
#        if s_next[0] != 'win' and s_next[0] != 'fail':
#            if t_seq == len(pattern):
#                pdb.set_trace()
#        
#        if t_seq == len(pattern):
#            print "FINAL S:", s_next
#        
#        if t_seq == len(pattern) and not s_next[0] == 'win' and not s_next[0] == 'fail':
#            pdb.set_trace()
#            pass
#        else:
        # IF TSEQ FINAL MAS NAO WIN E NAO FAIL
        if s_next[0] == 'win' or s_next[0] == 'fail' or t_seq == len(pattern):
            """
            Learning alpha
            """
            #alpha_test = (np.sum(np.abs(Qmc-Qmc_prev)))/24#/(2.0*24))
            alpha_test = np.linalg.norm(np.abs(Qmc - Qmc_prev))
            #alpha_test = diff_Qmc(Qmc, Qmc_prev, traj)
            # AND WHEN INHIBITED???????????????
            if s_next[0] == "win" and primed and greedy:
                #alpha = 0.9 * alpha
                eps = 0.9 * eps
            elif s_next[0] == "fail" and primed and greedy:
                #alpha = 1 - ((1 - alpha) * 0.9)
                eps = 1 - ((1 - eps) * 0.9)
          
            if eps < 0.01 and eps_print is False:
                print "Eps is now less than 1%"
                print 'Occurences of pattern:', Lh.count(0) / 4.0
                eps_print = True
                
          
            Alphah.append(alpha)
            Epsh.append(eps)
            
            
            # Update the IE values
            if primed == False:
                x0 += (s_next[0] == 'fail')
                n0 += 1
            else:
                x1 += (s_next[0] == 'win')
                n1 += 1
            
            a = 0
            s_next = s0
            t_mdp = 0
            t_seq = 0
            seq_active = False
            primed = False
            Qmc_prev = Qmc.copy()
            Qmc = update_monte_carlo(Qmc, Nmc, traj, r)
                
            traj = []
        
        Q0h.append(Q[0,1])
        Q1h.append(Q[2,1])
        Q0hmc.append(Qmc[0,1])
        Q1hmc.append(Qmc[2,1])
        Q0hq.append(Q_q[0,1])
        Q1hq.append(Q_q[2,1])
        Rh.append(r)
        
        #get the policy for debugging and visualization purposes
        pol = np.argmax(Q, 1)
        q_pol = np.argmax(Q_q, 1)
        mc_pol = np.argmax(Qmc, 1)
        
        if np.all(pol == opt_pol) and not found_opt_pol:
            found_opt_pol = True
            print 'Found optimal policy after', t, 'iterations'
            print 'Occurences of pattern:', Lh.count(0) / 4.0
         
        if np.all(q_pol == opt_pol) and not found_opt_q_pol:
            found_opt_q_pol = True
            print 'Found optimal Q policy after', t, 'iterations'
            print 'Occurences of pattern:', Lh.count(0) / 4.0         
         
        if np.all(mc_pol[:3] == opt_pol[:3]) and not found_opt_mc_pol:
            found_opt_mc_pol = True
            print 'Found MC optimal policy after', t, 'iterations'
            print 'Occurences of pattern:', Lh.count(0) / 4.0
         
    # should input be discrete or continuous?
    print("Simulation finished.")
    print("Average Reward:", total_r / float(Tw), " ", total_r)
    print "Final SARSA policy:", pol
    if np.all(pol == opt_pol):
        print "\tCorrect!"
    else:
        print "\tIncorrect!"
    print "Final Q policy:", q_pol
    if np.all(q_pol == opt_pol):
        print "\tCorrect!"
    else:
        print "\tIncorrect!"
    print "Final MC policy:", mc_pol
    if np.all(mc_pol == opt_pol):
        print "\tCorrect!"
    else:
        print "\tIncorrect!"
    
#    t = []
#    c_reward = 0
#    for i, a in enumerate(Ah):
#        if a == 2:
#            t.append(i)
#    
#    for i in t:
#        c_reward += Rh[i]
#        print "State:", Sh[i]
#        print "Action:", Ah[i]
#        print "Reward:", Rh[i]
#        print "Next State:", Sh[i+1]
#        print "\n\n"
#    print "C Reward:", c_reward
    dt.draw_sequences(Yh, Ah, Rh, Sh)
    #dt.draw_history(Q0h, Q0hmc, Q0hq, Q1h, Q1hmc, Q1hq, Alphah, Epsh)
    
    pdb.set_trace()
