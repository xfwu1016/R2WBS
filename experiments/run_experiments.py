#!/usr/bin/env python3
"""Reproduce the JOTA numerical experiments for R2WBS (Randomized Two-Way Block Splitting).

No network or distributed runtime is required.

Python >=3.10; NumPy, SciPy, scikit-learn, threadpoolctl.  All methods use
identical data, rho=theta=1, and zero initial states.  Output includes every
planned case and seed, even if a run reaches the iteration limit.

The implementation executes the incremental row/column cache protocol.  Its
reported communication is an exact count of real-valued unicast payloads in
the stated logical deployment, not measured network traffic.  Timings are
single-threaded sequential Python timings, not federated wall-clock speedups.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import PolynomialFeatures
from threadpoolctl import threadpool_limits, threadpool_info


def shrink(x, threshold):
    return np.sign(x)*np.maximum(np.abs(x)-threshold, 0.0)


def slices(lengths):
    ends=np.r_[0,np.cumsum(lengths)]
    return [slice(int(ends[k]),int(ends[k+1])) for k in range(len(lengths))]


def normalize_columns(A):
    A=A-A.mean(axis=0)
    norms=np.linalg.norm(A,axis=0)
    if np.any(norms==0):
        raise ValueError('Unexpected constant design column.')
    return A/norms


def make_case(case, repeat):
    seed=20261122+100*repeat
    rng=np.random.default_rng(seed)
    if case=='correlated_lasso':
        m,n=240,120
        Z=rng.normal(size=(m,n))
        A=np.empty_like(Z)
        A[:,0]=Z[:,0]
        for j in range(1,n):
            A[:,j]=0.6*A[:,j-1]+0.8*Z[:,j]
        row_sizes=[60]*4; col_sizes=[30]*4
        loss='squared'; delta=None; rank=None
    elif case=='rank_deficient_lasso':
        m,n,rank=180,240,60
        A=rng.normal(size=(m,rank))@rng.normal(size=(rank,n))/np.sqrt(rank)
        row_sizes=[45]*4; col_sizes=[60]*4
        loss='squared'; delta=None
    elif case=='uneven_huber':
        m,n=360,180
        Z=rng.normal(size=(m,n))
        A=np.empty_like(Z); A[:,0]=Z[:,0]
        for j in range(1,n):
            A[:,j]=0.6*A[:,j-1]+0.8*Z[:,j]
        row_sizes=[30,60,90,180]; col_sizes=[15,30,45,90]
        loss='huber'; delta=0.2; rank=None
    elif case=='diabetes_quadratic':
        raw=load_diabetes()
        # Fit one optimization problem using all observations. This is not a
        # predictive validation experiment and has no train/test accuracy claim.
        A=normalize_columns(PolynomialFeatures(degree=2,include_bias=False).fit_transform(raw.data))
        b=(raw.target-raw.target.mean())/raw.target.std(ddof=0)
        permutation=rng.permutation(len(b))
        A=A[permutation]; b=b[permutation]
        m,n=A.shape
        row_sizes=[110,110,111,111]; col_sizes=[16,16,16,17]
        loss='squared'; delta=None; rank=None
        return A,b,dict(case=case,repeat=repeat,data_seed=seed,m=m,n=n,
            row_sizes=row_sizes,col_sizes=col_sizes,loss=loss,delta=delta,
            construction_rank=rank,source='scikit-learn load_diabetes, 442 observations and 10 original covariates; all 65 degree-1/2 monomials',
            response_transform='center and divide by population standard deviation',
            feature_transform='degree-2 PolynomialFeatures without intercept; center each of 65 columns and scale to Euclidean norm 1',
            partition='seeded row permutation; fixed contiguous feature groups',
            repetition='same objective, different seeded sample-to-row assignments and activation seeds')
    else:
        raise ValueError(case)
    A=normalize_columns(A)
    support=rng.choice(n,size=max(6,n//10),replace=False)
    truth=np.zeros(n); truth[support]=rng.choice([-1.,1.],size=len(support))*(1+rng.random(len(support)))
    b=A@truth+0.04*rng.normal(size=m)
    outliers=[]
    if loss=='huber':
        outliers=rng.choice(m,size=m//10,replace=False)
        b[outliers]+=rng.choice([-1.,1.],size=len(outliers))*2.0
    return A,b,dict(case=case,repeat=repeat,data_seed=seed,m=m,n=n,
        row_sizes=row_sizes,col_sizes=col_sizes,loss=loss,delta=delta,
        construction_rank=rank,support_size=len(support),noise_sd=0.04,
        outlier_count=len(outliers),outlier_magnitude=2.0 if len(outliers) else 0.0,
        feature_transform='center each column and scale to Euclidean norm 1',
        repetition='independent seeded problem instance and activation seed')


class Problem:
    def __init__(self,A,b,meta):
        self.A=A; self.b=b; self.meta=meta
        self.rows=slices(meta['row_sizes']); self.cols=slices(meta['col_sizes'])
        self.M=len(self.rows); self.N=len(self.cols)
        self.mi=np.array(meta['row_sizes']); self.nj=np.array(meta['col_sizes'])
        self.loss=meta['loss']; self.delta=meta['delta']
        self.rho=1.0; self.theta=1.0
        self.scale=max(1.,float(np.linalg.norm(b)))
        grad0=-b if self.loss=='squared' else np.clip(-b,-self.delta,self.delta)
        self.tau=0.08*float(np.max(np.abs(A.T@grad0)))
        self.block=[[A[r,c].copy() for c in self.cols] for r in self.rows]
        self.factors=[]
        start=time.perf_counter()
        for i in range(self.M):
            fs=[]
            for j in range(self.N):
                a=self.block[i][j]
                if a.shape[1]<=a.shape[0]:
                    fs.append(('column',cho_factor(np.eye(a.shape[1])+a.T@a,lower=True,check_finite=False)))
                else:
                    fs.append(('row',cho_factor(np.eye(a.shape[0])+a@a.T,lower=True,check_finite=False)))
            self.factors.append(fs)
        self.setup_seconds=time.perf_counter()-start
        self.graph_payload=2*(self.mi[:,None]+self.nj[None,:])
        self.row_payload=2*int(self.mi.sum())
        # Compute tau through the hub: rows -> hub (m), hub -> all graph
        # clients (N*m), graph products -> hub (M*n), plus M norm scalars.
        self.initialization_payload=int((self.N+1)*self.mi.sum()+self.M*self.nj.sum()+self.M)
        # Full fixed-point audit: graph_payload.sum()+2m.
        # Original primal-dual gap: 2*sum(mi+nj)+2m+3M.
        self.audit_payload=int(2*self.graph_payload.sum()+2*self.row_payload+3*self.M)

    def loss_value(self,r):
        if self.loss=='squared':return float(0.5*r@r)
        a=np.abs(r)
        return float(np.where(a<=self.delta,0.5*r*r,self.delta*a-0.5*self.delta**2).sum())

    def gradient(self,r):
        return r if self.loss=='squared' else np.clip(r,-self.delta,self.delta)

    def prox_loss(self,q,i):
        b=self.b[self.rows[i]]
        if self.loss=='squared':return (q+b)/2
        z=q-b
        # prox_{h_delta}(z) with rho=1.
        return b+np.where(np.abs(z)<=2*self.delta,z/2,z-self.delta*np.sign(z))

    def graph(self,i,j,qy,qx):
        a=self.block[i][j]; direction,factor=self.factors[i][j]
        if direction=='column':
            x=cho_solve(factor,qx+a.T@qy,check_finite=False)
        else:
            w=cho_solve(factor,qy-a@qx,check_finite=False)
            x=qx+a.T@w
        return a@x,x

    def certificate(self,x,distributed=True):
        if distributed:
            xs=[x[c] for c in self.cols]
            rs=[sum((self.block[i][j]@xs[j] for j in range(self.N)),start=np.zeros(self.mi[i]))-self.b[r] for i,r in enumerate(self.rows)]
            gradients=[self.gradient(r) for r in rs]
            at=[sum((self.block[i][j].T@gradients[i] for i in range(self.M)),start=np.zeros(self.nj[j])) for j in range(self.N)]
            max_at=max(float(np.max(np.abs(v))) for v in at)
            r=np.concatenate(rs); grad=np.concatenate(gradients)
        else:
            r=self.A@x-self.b; grad=self.gradient(r)
            max_at=float(np.max(np.abs(self.A.T@grad)))
        alpha=min(1.,self.tau/max(max_at,1e-300))*(1-16*np.finfo(float).eps)
        q=alpha*grad
        primal=self.loss_value(r)+self.tau*float(np.abs(x).sum())
        dual=float(-0.5*q@q-self.b@q)
        gap=primal-dual
        return dict(primal=primal,dual=dual,gap=gap,relative_gap=gap/max(1.,abs(primal)),
            dual_constraint_ratio=alpha*max_at/self.tau,
            dual_box_ratio=float(np.max(np.abs(q))/self.delta) if self.loss=='huber' else 0.)


class CacheState:
    def __init__(self,p):
        self.p=p
        self.y=[np.zeros(m) for m in p.mi]
        self.x=[np.zeros(n) for n in p.nj]
        self.yg=[[np.zeros(p.mi[i]) for j in range(p.N)] for i in range(p.M)]
        self.xg=[[np.zeros(p.nj[j]) for j in range(p.N)] for i in range(p.M)]
        self.C=[np.zeros(n) for n in p.nj]
        self.E=[np.zeros(m) for m in p.mi]

    def vector(self):
        return np.concatenate(self.y+self.x+[self.yg[i][j] for i in range(self.p.M) for j in range(self.p.N)]+[self.xg[i][j] for i in range(self.p.M) for j in range(self.p.N)])

    def snapshot(self):
        return [v/(self.p.M+1) for v in self.C],[v/(self.p.N+1) for v in self.E]

    def step(self,active):
        p=self.p; a,t=self.snapshot()
        # These snapshots remain unchanged while all states and caches commit.
        for i in range(p.M):
            dy=p.prox_loss(self.y[i]-2*t[i],i)-self.y[i]+t[i]
            self.y[i]+=dy; self.E[i]+=dy
        for j in range(p.N):
            dx=shrink(2*a[j]-self.x[j],p.tau)-a[j]
            self.x[j]+=dx; self.C[j]+=dx
        for i,j in np.argwhere(active):
            yp,xp=p.graph(i,j,self.yg[i][j]+2*t[i],2*a[j]-self.xg[i][j])
            dy=yp-self.yg[i][j]-t[i]; dx=xp-a[j]
            self.yg[i][j]+=dy; self.xg[i][j]+=dx
            self.E[i]-=dy; self.C[j]+=dx

    def audit(self):
        p=self.p; a,t=self.snapshot(); sq=0.
        for i in range(p.M):
            d=p.prox_loss(self.y[i]-2*t[i],i)-self.y[i]+t[i]
            sq+=float(d@d)
        for j in range(p.N):
            d=shrink(2*a[j]-self.x[j],p.tau)-a[j]
            sq+=float(d@d)
        for i in range(p.M):
            for j in range(p.N):
                yp,xp=p.graph(i,j,self.yg[i][j]+2*t[i],2*a[j]-self.xg[i][j])
                dy=yp-self.yg[i][j]-t[i]; dx=xp-a[j]
                sq+=float(dy@dy+dx@dx)
        cert=p.certificate(np.concatenate(a))
        cert.update(fixed_point_residual=float(np.sqrt(sq)),normalized_residual=float(np.sqrt(sq)/p.scale))
        return cert

    def cache_error(self):
        p=self.p
        errs=[np.max(np.abs(self.C[j]-self.x[j]-sum(self.xg[i][j] for i in range(p.M)))) for j in range(p.N)]
        errs +=[np.max(np.abs(self.E[i]-self.y[i]+sum(self.yg[i][j] for j in range(p.N)))) for i in range(p.M)]
        return float(max(errs))


def validate_cache(p):
    import copy
    rng=np.random.default_rng(314159)
    s=CacheState(p)
    for v in s.y+s.x+[s.yg[i][j] for i in range(p.M) for j in range(p.N)]+[s.xg[i][j] for i in range(p.M) for j in range(p.N)]:
        v[:]=rng.normal(size=len(v))
    s.C=[s.x[j]+sum(s.xg[i][j] for i in range(p.M)) for j in range(p.N)]
    s.E=[s.y[i]-sum(s.yg[i][j] for j in range(p.N)) for i in range(p.M)]
    max_error=0.
    for k in range(12):
        active=np.ones((p.M,p.N),bool) if k==0 else rng.random((p.M,p.N))<0.35
        old=copy.deepcopy(s); expected=copy.deepcopy(s)
        a=[(old.x[j]+sum(old.xg[i][j] for i in range(p.M)))/(p.M+1) for j in range(p.N)]
        t=[(old.y[i]-sum(old.yg[i][j] for j in range(p.N)))/(p.N+1) for i in range(p.M)]
        for i in range(p.M):expected.y[i]+=p.prox_loss(old.y[i]-2*t[i],i)-old.y[i]+t[i]
        for j in range(p.N):expected.x[j]+=shrink(2*a[j]-old.x[j],p.tau)-a[j]
        for i,j in np.argwhere(active):
            yp,xp=p.graph(i,j,old.yg[i][j]+2*t[i],2*a[j]-old.xg[i][j])
            expected.yg[i][j]+=yp-old.yg[i][j]-t[i]
            expected.xg[i][j]+=xp-a[j]
        s.step(active)
        max_error=max(max_error,float(np.max(np.abs(s.vector()-expected.vector()))),s.cache_error())
    assert max_error<1e-10,max_error
    return dict(random_state_steps=12,max_absolute_cache_or_direct_update_error=max_error,passed=True)


def central_reference(p):
    # FISTA is solely a centralized high-accuracy objective reference.
    L=float(np.linalg.norm(p.A,2)**2)
    x=np.zeros(p.A.shape[1]); y=x.copy(); t=1.
    start=time.perf_counter()
    for k in range(1,200001):
        xn=shrink(y-p.A.T@p.gradient(p.A@y-p.b)/L,p.tau/L)
        tn=(1+np.sqrt(1+4*t*t))/2
        y=xn+(t-1)/tn*(xn-x); x=xn; t=tn
        if k%50==0:
            cert=p.certificate(x,distributed=False)
            if cert['relative_gap']<=1e-10:break
    cert.update(iterations=k,runtime_seconds=time.perf_counter()-start,lipschitz_constant=L,converged=cert['relative_gap']<=1e-10)
    return x,cert


def run_method(p,probability,activation_seed,max_rounds,check_every,gap_tol,residual_tol,mask_type='bernoulli'):
    state=CacheState(p); rng=np.random.default_rng(activation_seed)
    count=0; update_payload=0; audit_count=0; hist=[]
    active_counts=np.zeros((p.M,p.N),int)
    start=time.perf_counter()
    for k in range(1,max_rounds+1):
        if mask_type=='rectangle_2x2':
            assert p.M==p.N==4 and probability==0.25
            active=np.zeros((p.M,p.N),bool)
            active[np.ix_(rng.choice(p.M,2,replace=False),rng.choice(p.N,2,replace=False))]=True
        else:
            active=np.ones((p.M,p.N),bool) if probability==1 else rng.random((p.M,p.N))<probability
        state.step(active); active_counts+=active
        count+=int(active.sum())
        update_payload+=int(p.graph_payload[active].sum())+p.row_payload
        if k%check_every==0 or k==max_rounds:
            cert=state.audit(); audit_count+=1
            elapsed=time.perf_counter()-start
            hist.append(dict(round=k,update_graph_evaluations=count,audit_graph_evaluations=audit_count*p.M*p.N,
                update_scalar_payload=update_payload,audit_scalar_payload=audit_count*p.audit_payload,
                initialization_scalar_payload=p.initialization_payload,
                total_scalar_payload=update_payload+audit_count*p.audit_payload+p.initialization_payload,
                sequential_seconds=elapsed,**cert))
            if cert['relative_gap']<=gap_tol and cert['normalized_residual']<=residual_tol:break
    elapsed=time.perf_counter()-start
    cache_error=state.cache_error()  # Offline implementation validation, outside timed run.
    result=dict(case=p.meta['case'],repeat=p.meta['repeat'],participation=probability,mask_type=mask_type,
        data_seed=p.meta['data_seed'],activation_seed=activation_seed,rounds=k,
        converged=bool(cert['relative_gap']<=gap_tol and cert['normalized_residual']<=residual_tol),
        update_graph_evaluations=count,audit_graph_evaluations=audit_count*p.M*p.N,
        total_graph_evaluations=count+audit_count*p.M*p.N,
        loss_prox_evaluations=(k+audit_count)*p.M,regularizer_prox_evaluations=(k+audit_count)*p.N,
        audits=audit_count,update_scalar_payload=update_payload,audit_scalar_payload=audit_count*p.audit_payload,
        initialization_scalar_payload=p.initialization_payload,total_scalar_payload=update_payload+audit_count*p.audit_payload+p.initialization_payload,
        sequential_seconds=elapsed,setup_seconds=p.setup_seconds,
        sequential_plus_setup_seconds=elapsed+p.setup_seconds,
        cache_max_absolute_error=cache_error,tau=p.tau,**cert)
    assert np.isfinite(result['primal']) and result['gap']>=-1e-9
    assert result['dual_constraint_ratio']<=1+1e-12
    assert result['dual_box_ratio']<=1+1e-12
    assert cache_error<1e-9
    return result,hist,active_counts


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def validate_dual_bias():
    """Independent scalar reduced recursion, not the cached-state solver."""
    records=[]
    def step(v):
        x,a,b=v
        xp=(x-a)/2
        xg=x-b
        xn=(xp+xg)/2
        return xn,a+xp-xn,b+xg-xn
    for c in [-2.,-1.,0.5,1.25,2.]:
        v=(0.,c,0.); repaired=(0.,c/2,-c/2)
        for _ in range(200):
            v=step(v); repaired=step(repaired)
        records.append(dict(column_dual_sum=c,predicted_x=-c,actual_x=v[0],
            repaired_x=repaired[0],bias_error=abs(v[0]+c),repair_error=abs(repaired[0])))
    assert max(max(r['bias_error'],r['repair_error']) for r in records)<1e-12
    return dict(model='M=N=1, A=0, rho=1, l(y)=y^2/2, r(x)=x^2/2',
        iterations=200,records=records,passed=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'reproduced'/'results')
    parser.add_argument('--cases',nargs='+',choices=['correlated_lasso','rank_deficient_lasso','uneven_huber','diabetes_quadratic'],default=['correlated_lasso','rank_deficient_lasso','uneven_huber','diabetes_quadratic'])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--probabilities',nargs='+',type=float,default=[1.,0.5,0.25])
    parser.add_argument('--max-rounds',type=int,default=40000)
    parser.add_argument('--check-every',type=int,default=25)
    parser.add_argument('--gap-tol',type=float,default=1e-6)
    parser.add_argument('--residual-tol',type=float,default=1e-5)
    parser.add_argument('--no-rectangle',action='store_true',help='Omit the prespecified correlated-mask comparison.')
    args=parser.parse_args()
    if args.repeats < 1 or args.max_rounds < 1 or args.check_every < 1:
        parser.error('--repeats, --max-rounds and --check-every must be positive.')
    if not all(0 < p <= 1 for p in args.probabilities):
        parser.error('--probabilities must lie in (0, 1].')
    if args.gap_tol <= 0 or args.residual_tol <= 0:
        parser.error('Both stopping tolerances must be positive.')
    if len(set(args.cases)) != len(args.cases) or len(set(args.probabilities)) != len(args.probabilities):
        parser.error('Duplicate cases or probabilities would overwrite run records.')
    args.output=args.output.expanduser().resolve()
    root=Path(__file__).resolve().parents[1]
    for protected in [root/'results',root/'reference_figures']:
        if args.output == protected or protected in args.output.parents:
            parser.error('Reference archives are read-only inputs; choose a different --output directory.')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('Output must be a new or empty directory; choose a fresh --output path.')
    args.output.mkdir(parents=True,exist_ok=True)
    for sub in ['data','trajectories']:(args.output/sub).mkdir(exist_ok=True)
    environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
        platform=platform.platform(),processor=platform.processor(),cpu_count=os.cpu_count(),blas_threads=1,threadpools_before_limit=threadpool_info())
    manifest=dict(cases=args.cases,repeats=args.repeats,probabilities=args.probabilities,include_rectangle=not args.no_rectangle,rho=1.,theta=1.,initial_state='zero',
        max_rounds=args.max_rounds,check_every=args.check_every,gap_tolerance=args.gap_tol,
        normalized_residual_tolerance=args.residual_tol,normalized_residual='||D(s)-s|| / max(1,||b||_2)',
        relative_gap='(primal-dual)/max(1,abs(primal))',
        lambda_rule='0.08 * ||A^T grad loss(-b)||_infinity',
        architecture='remote row-loss nodes and graph clients; column regularizers colocated with the coordinator',
        graph_payload='2*(m_i+n_j) real scalars per activated graph client',
        row_payload='2*m real scalars per round; all row-loss nodes active',
        audit_payload='4*sum_ij(m_i+n_j)+4*m+3*M real scalars; full fixed-point residual plus a separate distributed original-objective/primal-dual-gap computation',
        initialization_payload='(N+1)*m+M*n+M real scalars: row owners send g0=grad loss(-b) to hub (m), hub forwards to all graph clients (N*m), graph clients return Aij^T*g0_i (M*n), and row response norms contribute M scalars; zero state needs no additional vector exchange',
        exclusions='Packet headers, control flags, transport latency, feature construction and data standardization excluded; no matrix-block transfer; response-dependent gradients and intermediate values are exchanged and no label-privacy claim is made; regularizer arithmetic counted but has no network payload.',
        audits='At fixed audit rounds all graph clients respond; audit predictions do not alter iteration states. This is controlled partial updating, not a claim about unavailable audit clients.',
        timing='Single-threaded sequential Python; includes cache updates, masks, all audit matrix-vector products and proximal calls; setup_seconds measures only local Cholesky factorization, separately reported. Data generation, feature construction and normalization are outside the timed solve. No network wall-clock claim.',
        reference='Centralized FISTA to relative primal-dual gap 1e-10; used only as correctness reference, outside federated communication counts.',
        correlated_mask_case='correlated_lasso: independently choose two of four rows and two of four columns uniformly, activate their 2x2 Cartesian product; each graph marginal p=1/4',
        planned_run_count=len(args.cases)*args.repeats*len(args.probabilities)+(args.repeats if 'correlated_lasso' in args.cases and not args.no_rectangle else 0))
    (args.output/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'bias_validation.json').write_text(json.dumps(validate_dual_bias(),indent=2)+'\n')
    results=[]; problems=[]; validations=[]
    with threadpool_limits(limits=1):
        environment['threadpools_during_run']=threadpool_info()
        (args.output/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
        for case in args.cases:
            for repeat in range(args.repeats):
                A,b,meta=make_case(case,repeat); p=Problem(A,b,meta)
                check=validate_cache(p); check.update(case=case,repeat=repeat); validations.append(check)
                xr,ref=central_reference(p)
                meta.update(tau=p.tau,rho=1.,theta=1.,numerical_rank=int(np.linalg.matrix_rank(A)),
                    local_factorizations=p.M*p.N,setup_seconds=p.setup_seconds,reference=ref)
                fname=f'{case}_r{repeat}.npz'
                np.savez_compressed(args.output/'data'/fname,A=A,b=b,reference_x=xr,row_sizes=p.mi,col_sizes=p.nj)
                meta.update(data_file='data/'+fname,sha256=hashlib.sha256((args.output/'data'/fname).read_bytes()).hexdigest())
                problems.append(meta)
                print(json.dumps({'reference':case,'repeat':repeat,'reference_gap':ref['relative_gap'],'reference_iterations':ref['iterations']}),flush=True)
                configurations=[(prob,'bernoulli') for prob in args.probabilities]
                if case=='correlated_lasso' and not args.no_rectangle:
                    configurations.append((0.25,'rectangle_2x2'))
                for probability,mask_type in configurations:
                    class_index={'correlated_lasso':0,'rank_deficient_lasso':1,'uneven_huber':2,'diabetes_quadratic':3}[case]
                    activation_seed=73000+1000*class_index+repeat
                    res,hist,counts=run_method(p,probability,activation_seed,args.max_rounds,args.check_every,args.gap_tol,args.residual_tol,mask_type)
                    res['objective_minus_reference']=res['primal']-ref['primal']
                    stem=f'{case}_r{repeat}_p{probability:g}_{mask_type}'
                    res['trajectory_file']='trajectories/'+stem+'.csv'
                    write_csv(args.output/res['trajectory_file'],hist)
                    (args.output/'trajectories'/f'{stem}_activation_counts.json').write_text(json.dumps(counts.tolist())+'\n')
                    results.append(res)
                    # Checkpoint all completed records, including nonconvergence.
                    write_csv(args.output/'runs.csv',results)
                    (args.output/'runs.json').write_text(json.dumps(results,indent=2)+'\n')
                    (args.output/'problems.json').write_text(json.dumps(problems,indent=2)+'\n')
                    (args.output/'cache_validation.json').write_text(json.dumps(validations,indent=2)+'\n')
                    print(json.dumps({k:res[k] for k in ['case','repeat','participation','mask_type','rounds','converged','total_scalar_payload','relative_gap','normalized_residual','sequential_seconds']}),flush=True)
    print(json.dumps({'complete':True,'runs':len(results),'converged':sum(r['converged'] for r in results),'output':str(args.output.resolve())}),flush=True)


if __name__=='__main__':main()
