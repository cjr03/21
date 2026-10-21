import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core, sim, json
from collections import Counter

def trace(P, seed):
    piles=[0,0,0,0]; b,H,score=0,2,0
    D=[0,4,4,4,4,4,4,4,4,4,16]; ev=[]
    for k,v in enumerate(sim.deck(seed)):
        D[v]-=1
        mv=core.score_moves(P,piles,b,H,D,v)
        _,idx,hold=mv[0]
        s0=piles[idx]
        before=[core.STATES[s] for s in piles]
        if hold: H-=1; s0=0
        if core.BUST[s0][v]:
            b+=1
            forced = all(core.BUST[s][v] for s in piles)
            ev.append(("bust",k,v,forced,before,H))
        elif hold:
            ev.append(("hold",k,v,all(core.BUST[s][v] for s in piles),before,H))
        score+=core.REW[s0][v]; piles[idx]=core.NXT[s0][v]
        if b>=3: break
    return score,b,k,ev

if __name__=="__main__":
    pd = json.load(open(sys.argv[1])) if len(sys.argv)>1 else {}
    P=core.Params(pd)
    deaths=[]; bc=Counter(); hc=Counter()
    for s in range(200):
        sc,b,k,ev=trace(P,s)
        for e in ev:
            if e[0]=="bust": bc["forced" if e[3] else "chosen"]+=1
            else: hc["forced" if e[3] else "chosen"]+=1
        if b>=3:
            deaths.append(k)
            if len(deaths)<=6:
                print(s,sc,k,[(e[0],e[1],e[2],e[3],e[4]) for e in ev])
    print("deaths",len(deaths),sorted(deaths)); print("busts",bc,"holds",hc)
