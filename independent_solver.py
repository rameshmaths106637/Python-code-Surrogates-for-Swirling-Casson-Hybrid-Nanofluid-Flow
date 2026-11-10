import numpy as np
from scipy.integrate import solve_bvp

# --- property ratios (Table ratios of main.tex)
rho_f,cp_f,k_f,s_f,b_f = 997.1,4179.,0.613,5.5e-6,21e-5
P1=dict(rho=3970.,cp=765.,k=40.,s=3.5e7,b=0.85e-5)   # Al2O3 as in manuscript
P2=dict(rho=8933.,cp=385.,k=401.,s=5.96e7,b=1.67e-5) # Cu
def ratios(phi1,phi2,s1=None):
    P1_=dict(P1); 
    if s1 is not None: P1_['s']=s1
    def vavg(a1,a2,af):
        return (1-phi2)*((1-phi1)*af+phi1*a1)+phi2*a2
    Phi_v=(1-phi1)**-2.5*(1-phi2)**-2.5
    Phi_rho=vavg(P1_['rho'],P2['rho'],rho_f)/rho_f
    Phi_cp=vavg(P1_['rho']*P1_['cp'],P2['rho']*P2['cp'],rho_f*cp_f)/(rho_f*cp_f)
    Phi_b=vavg(P1_['rho']*P1_['b'],P2['rho']*P2['b'],rho_f*b_f)/(rho_f*b_f)
    k1,k2=P1_['k'],P2['k']
    knf=k_f*(k1+2*k_f-2*phi1*(k_f-k1))/(k1+2*k_f+phi1*(k_f-k1))
    khnf=knf*(k2+2*knf-2*phi2*(knf-k2))/(k2+2*knf+phi2*(knf-k2))
    phi=phi1+phi2
    if phi>0:
        sp=phi1*P1_['s']+phi2*P2['s']; a=sp/s_f-phi
        Phi_s=1+3*a/(sp/(s_f*phi)+2-a)
    else: Phi_s=1.0
    return dict(Phi_v=Phi_v,Phi_rho=Phi_rho,Phi_cp=Phi_cp,Phi_b=Phi_b,Phi_k=khnf/k_f,Phi_s=Phi_s)

REF=dict(Re=2.,A1=.3,A2=.3,Om=.5,M=1.,chi=.5,beta=1.5,Lam=.1,Rd=1.,gt=.02,Ec=.02,em=.2,phi1=.02,phi2=.02,
         Pr=6.2,ek=.2,z1=.5,z2=.5,tw=.1,visc_diss=0.,s1=None,Phi_s=None)
def solve(**kw):
    p=dict(REF); p.update(kw)
    R=ratios(p['phi1'],p['phi2'],p['s1'])
    if p['Phi_s'] is not None: R['Phi_s']=p['Phi_s']
    Re,Pr=p['Re'],p['Pr']; c=1+1/p['beta']; em,ek,tw=p['em'],p['ek'],p['tw']
    pv=R['Phi_v']/R['Phi_rho']*c
    Lor=R['Phi_s']/R['Phi_rho']*p['M']/(1+p['chi']**2)
    def rhs(x,y):
        F,F1,F2,F3,G,G1,T,T1=y
        v=1+em*T
        D=R['Phi_k']*(1+ek*T)+p['Rd']*(1+tw*T)**3-4*R['Phi_cp']*p['gt']*Re*Pr*F**2
        joule=R['Phi_s']*p['M']*Re*Pr*p['Ec']/(1+p['chi']**2)*(F1**2+G**2)
        vd=p['visc_diss']*R['Phi_v']*c/v*Pr*p['Ec']*(F2**2+G1**2)
        N=(4*R['Phi_cp']*p['gt']*Re*Pr*F*F1*T1 - 3*tw*p['Rd']*(1+tw*T)**2*T1**2
           -2*R['Phi_cp']*Re*Pr*F*T1 - R['Phi_k']*ek*T1**2 - joule - vd)
        T2=N/D
        pref=pv/(Re*v)
        G2=(-(2*(F*G1-F1*G)-Lor*(G+p['chi']*F1))/pref)+em/v*G1*T1
        F4=(-(2*F*F3+2*G*G1-Lor*(F2-p['chi']*G1)+R['Phi_b']/R['Phi_rho']*p['z1']*(1+2*p['z2']*T)*T1)/pref
            +2*em/v*F3*T1+em/v*F2*T2-2*em**2/v**2*F2*T1**2)
        return np.vstack([F1,F2,F3,F4,G1,G2,T1,T2])
    L=p['Lam']
    def bc(a,b):
        return np.array([a[0],a[1]-p['A1']-L*a[2],a[4]-1-L*a[5],a[6]-1,
                         b[0],b[1]-p['A2']+L*b[2],b[4]-p['Om']+L*b[5],b[6]])
    x=np.linspace(0,1,101); y0=np.zeros((8,x.size))
    y0[1]=p['A1']+(p['A2']-p['A1'])*x; y0[4]=1+(p['Om']-1)*x; y0[6]=1-x; y0[7]=-1
    s=solve_bvp(rhs,bc,x,y0,tol=1e-8,max_nodes=100000)
    if s.status!=0:   # fall back to continuation in Re
        raise RuntimeError(s.message)
    def out(xi):
        F,F1,F2,F3,G,G1,T,T1=s.sol(xi)
        cf=R['Phi_v']*c/(1+em*T)*np.hypot(F2,G1)
        nu=-(R['Phi_k']*(1+ek*T)+p['Rd']*(1+tw*T)**3)*T1
        return cf,nu
    cf1,nu1=out(0.); cf2,nu2=out(1.)
    return dict(Cf1=cf1,Cf2=cf2,Nu1=nu1,Nu2=nu2,sol=s,R=R)

if __name__=="__main__":
    r=solve()
    print({k:round(v,4) for k,v in r['R'].items()})
    print("ref:",[round(r[k],3) for k in ('Cf1','Cf2','Nu1','Nu2')],"manuscript: 2.106 2.195 2.433 2.493")
